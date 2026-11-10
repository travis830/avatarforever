"""Streaming generation: the models stay loaded, and video comes out one chunk at a time.

inference.py takes a whole audio file, allocates latents for the whole run and decodes at the end.
Here a Session holds only what the model looks at when it draws the next chunk: the first chunk,
the previous chunk and the audio that goes with them. Each call to ``Session.next`` takes the next
slice of audio and returns that slice's frames.

One chunk is four latent frames. The first chunk is 25 frames (1.00 s at 25 fps) because the first
latent frame stands for a single frame; every later chunk is 32 frames (1.28 s).
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.audio_vae import encode_audio
from ltx_core.model.audio_vae.ops import AudioProcessor
from ltx_core.model.transformer.ar_feature_cache import ARFeatureCache
from ltx_core.model.video_vae import decode_video
from ltx_core.types import Audio, LatentState, VideoLatentShape, VideoPixelShape
from ltx_pipelines import ARA2VidDistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.helpers import (
    combined_image_conditionings,
    noise_audio_state,
    noise_video_state,
    post_process_latent,
    simple_denoising_func,
)

logger = logging.getLogger("avatar.engine")

FPS = 25.0
CHUNK_LATENT_FRAMES = 4
AUDIO_LATENTS_PER_SECOND = 25
DEFAULT_SIGMAS = (1.0, 0.98125, 0.909375, 0.421875, 0.0)
THREE_STEP_SIGMAS = (1.0, 0.909375, 0.421875, 0.0)

DEFAULT_PROMPT = (
    "Natural audio-driven speaking motion with accurate lip synchronization, smooth and continuous facial "
    "animation, subtle head movement, natural blinking, gentle breathing, and relaxed upper-body motion. "
    "Expressions and gestures should respond naturally to the rhythm, tone, and emotion of the speech while "
    "remaining restrained and realistic.\n\n"
    "Maintain strong temporal consistency across all frames. Keep facial appearance, identity, pose, body "
    "structure, clothing details, lighting, and background stable throughout the video. Avoid sudden motion "
    "changes, excessive gestures, unnatural expression shifts, frame-to-frame appearance variation, flickering, "
    "jitter, ghosting, texture instability, temporal artifacts, or deformation. All motion should be coherent, "
    "fluid, stable, and naturally driven by the audio."
)


def chunk_frames(index: int) -> int:
    """Pixel frames in chunk ``index``."""
    return 25 if index == 0 else 32


def chunk_seconds(index: int) -> float:
    return chunk_frames(index) / FPS


@dataclass
class Timing:
    audio_ms: float = 0.0
    denoise_ms: float = 0.0
    decode_ms: float = 0.0
    steps_ms: list[float] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return self.audio_ms + self.denoise_ms + self.decode_ms


@dataclass
class Chunk:
    index: int
    frames: torch.Tensor  # uint8 [f, h, w, 3] on the CPU
    seconds: float
    timing: Timing


class Engine:
    """The loaded models. One Engine serves one Session at a time."""

    def __init__(
        self, checkpoint: str, gemma_root: str, *, cache_dir: str | None = None, live_prompts: bool = True
    ) -> None:
        started = time.time()
        self.pipeline = ARA2VidDistilledPipeline(
            distilled_checkpoint_path=checkpoint, gemma_root=gemma_root, spatial_upsampler_path=None, loras=()
        )
        self.ledger = self.pipeline.model_ledger
        self.components = self.pipeline.pipeline_components
        self.device = self.pipeline.device
        self.dtype = self.pipeline.dtype
        self.cache_dir = Path(cache_dir or Path(checkpoint).parent / "prompt-cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._contexts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._positions: dict[tuple[int, int, int], torch.Tensor] = {}
        self._default_key = hashlib.sha256(DEFAULT_PROMPT.encode()).hexdigest()[:24]

        # The text encoder is 24 GB and only needed to turn a prompt into embeddings. Do that first,
        # keep the embeddings on disk, and it never has to share the card with the transformer.
        self.context(DEFAULT_PROMPT)
        with torch.inference_mode():
            self.audio_encoder = self.ledger.audio_encoder()
            self.video_encoder = self.ledger.video_encoder()
            self.video_decoder = self.ledger.video_decoder()
            self.transformer = self.ledger.transformer()
        self.audio_processor = AudioProcessor(
            target_sample_rate=self.audio_encoder.sample_rate,
            mel_bins=self.audio_encoder.mel_bins,
            mel_hop_length=self.audio_encoder.mel_hop_length,
            n_fft=self.audio_encoder.n_fft,
        ).to(device=self.device)
        self.stepper = EulerDiffusionStep()
        # With the text encoder kept on the card (about 24 GB more), a new prompt costs a fraction of a
        # second instead of ten: that is what lets an action be asked for in the middle of a stream.
        self.text_encoder = None
        self.embeddings_processor = None
        if live_prompts:
            with torch.inference_mode():
                self.text_encoder = self.ledger.text_encoder()
                self.embeddings_processor = self.ledger.gemma_embeddings_processor()
        logger.info("models loaded in %.1fs (live prompts: %s)", time.time() - started, live_prompts)

    def context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Text embeddings for ``prompt``: from memory, from disk, or by loading the text encoder once."""
        key = hashlib.sha256(prompt.encode()).hexdigest()[:24]
        if key in self._contexts:
            return self._contexts[key]
        path = self.cache_dir / f"{key}.pt"
        if path.exists():
            saved = torch.load(path, map_location=self.device)
            pair = (saved["video"], saved["audio"])
        elif getattr(self, "text_encoder", None) is not None:
            with torch.inference_mode():
                hidden_states, attention_mask = self.text_encoder.encode(prompt)
                processed = self.embeddings_processor.process_hidden_states(hidden_states, attention_mask)
            pair = (processed.video_encoding.detach(), processed.audio_encoding.detach())
            if len(self._contexts) > 64:  # prompts asked for in passing are not kept for ever
                for old in [k for k in self._contexts if k != self._default_key][:16]:
                    self._contexts.pop(old, None)
        else:
            started = time.time()
            with torch.inference_mode():
                video, audio, _, _ = self.pipeline._encode_guidance_contexts(
                    prompt=prompt, negative_prompt="", images=[], enhance_prompt=False, guidance_scale=1.0
                )
            pair = (video.detach(), audio.detach())
            torch.save({"video": pair[0].cpu(), "audio": pair[1].cpu(), "prompt": prompt}, path)
            logger.info("encoded a new prompt in %.1fs", time.time() - started)
        self._contexts[key] = pair
        return pair

    def positions(self, chunks: int, height: int, width: int) -> torch.Tensor:
        """Positions of a window of ``chunks`` chunks. Every window is laid out as if it started at zero."""
        key = (chunks, height, width)
        if key not in self._positions:
            frames = 8 * (CHUNK_LATENT_FRAMES * chunks - 1) + 1
            shape = VideoPixelShape(batch=1, frames=frames, width=width, height=height, fps=FPS)
            noiser = GaussianNoiser(generator=torch.Generator(device=self.device).manual_seed(0))
            state, _ = noise_video_state(
                output_shape=shape, noiser=noiser, conditionings=[], components=self.components,
                dtype=self.dtype, device=self.device,
            )
            self._positions[key] = state.positions
        return self._positions[key]


class Session:
    """One continuous run: an optional first frame, a prompt, then audio in and frames out."""

    def __init__(
        self,
        engine: Engine,
        *,
        image_path: str | None = None,
        prompt: str = DEFAULT_PROMPT,
        seed: int = 42,
        height: int = 512,
        width: int = 768,
        sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
        feature_cache: bool = True,
        decode_context: int = 2,
        decode_delay: bool = True,
        audio_context_seconds: float = 2.0,
    ) -> None:
        self.e = engine
        self.image_path = image_path
        self.height, self.width = height, width
        self.tokens_per_frame = (height // 32) * (width // 32)
        self.sigmas = torch.tensor(sigmas, device=engine.device, dtype=torch.float32)
        self.feature_cache = feature_cache
        self.decode_delay = decode_delay
        self.decode_context = max(1 if decode_delay else 0, min(CHUNK_LATENT_FRAMES - 1, decode_context))
        self.audio_context = int(round(audio_context_seconds * AUDIO_LATENTS_PER_SECOND))
        self.generator = torch.Generator(device=engine.device).manual_seed(seed)
        self.noiser = GaussianNoiser(generator=self.generator)
        self.prompt = prompt
        self.set_prompt(None)
        self.index = 0
        self.first: torch.Tensor | None = None       # tokens of chunk 0
        self.previous: torch.Tensor | None = None    # tokens of the chunk before this one
        self.first_audio: torch.Tensor | None = None     # audio latents [1, 8, t, 16] of chunk 0
        self.previous_audio: torch.Tensor | None = None
        self.wave = torch.zeros(0, dtype=torch.float32)  # every sample heard so far, 16 kHz mono
        self.wave_consumed = 0                            # samples of ``wave`` already turned into chunks

    def set_prompt(self, prompt: str | None) -> None:
        """Draw the chunks from here on with ``prompt``; None goes back to the one the session began with."""
        video_context, audio_context = self.e.context(prompt or self.prompt)
        self.denoise = simple_denoising_func(
            video_context=video_context, audio_context=audio_context, transformer=self.e.transformer
        )

    # ----- audio -----

    def _audio_latents(self, samples: torch.Tensor) -> torch.Tensor:
        """Audio latents for the next chunk. ``samples`` is exactly that chunk's audio, 16 kHz mono float."""
        need = chunk_frames(self.index)  # 25 latents per second and 25 frames per second: one each
        self.wave = torch.cat([self.wave, samples])
        end = self.wave_consumed + samples.numel()
        # Encode with some audio to the left so the encoder sees what led up to this chunk.
        left = min(self.wave_consumed, self.audio_context * 640)
        window = self.wave[self.wave_consumed - left : end]
        self.wave_consumed = end
        # Keep the buffer from growing: only the left context is ever read again.
        keep = self.audio_context * 640
        if self.wave.numel() > keep:
            drop = self.wave.numel() - keep
            self.wave = self.wave[drop:]
            self.wave_consumed -= drop
        waveform = window.to(self.e.device).view(1, 1, -1).repeat(1, 2, 1)
        latents = encode_audio(Audio(waveform=waveform, sampling_rate=16000), self.e.audio_encoder, self.e.audio_processor)
        skip = left // 640
        latents = latents[:, :, skip : skip + need]
        if latents.shape[2] < need:  # the encoder came up one short at the very end of the window
            pad = latents[:, :, -1:].expand(-1, -1, need - latents.shape[2], -1)
            latents = torch.cat([latents, pad], dim=2)
        return latents.contiguous()

    def _audio_state(self, current: torch.Tensor, frames: int) -> tuple[LatentState, slice]:
        if self.index == 0:
            parts = []
        elif self.index == 1:
            parts = [self.first_audio]
        else:
            parts = [self.first_audio, self.previous_audio]
        window = torch.cat([*parts, current], dim=2)
        shape = VideoPixelShape(batch=1, frames=frames, width=self.width, height=self.height, fps=FPS)
        state, _ = noise_audio_state(
            output_shape=shape, noiser=self.noiser, conditionings=[], components=self.e.components,
            dtype=self.e.dtype, device=self.e.device, noise_scale=0.0, initial_latent=window.to(self.e.dtype),
        )
        state = replace(state, denoise_mask=torch.zeros_like(state.denoise_mask))
        total = window.shape[2]
        return state, slice(total - current.shape[2], total)

    # ----- video -----

    def _first_state(self) -> LatentState:
        shape = VideoPixelShape(batch=1, frames=25, width=self.width, height=self.height, fps=FPS)
        conditionings = []
        if self.image_path:
            conditionings = combined_image_conditionings(
                images=[ImageConditioningInput(path=self.image_path, frame_idx=0, strength=1.0, crf=0)],
                height=self.height, width=self.width, video_encoder=self.e.video_encoder,
                dtype=self.e.dtype, device=self.e.device,
            )
        state, _ = noise_video_state(
            output_shape=shape, noiser=self.noiser, conditionings=conditionings, components=self.e.components,
            dtype=self.e.dtype, device=self.e.device, noise_scale=float(self.sigmas[0]),
        )
        return state

    def _window_state(self) -> tuple[LatentState, slice]:
        history = [self.first] if self.index == 1 else [self.first, self.previous]
        count = CHUNK_LATENT_FRAMES * self.tokens_per_frame
        noise = torch.randn(
            1, count, self.first.shape[-1], device=self.e.device, dtype=self.e.dtype, generator=self.generator
        ) * float(self.sigmas[0])
        latent = torch.cat([*history, noise], dim=1)
        mask = torch.zeros(1, latent.shape[1], 1, device=self.e.device, dtype=torch.float32)
        mask[:, -count:] = 1.0
        state = LatentState(
            latent=latent,
            denoise_mask=mask,
            positions=self.e.positions(len(history) + 1, self.height, self.width),
            clean_latent=latent,
        )
        return state, slice(latent.shape[1] - count, latent.shape[1])

    def _decode(self, current: torch.Tensor) -> torch.Tensor:
        """Frames to show after generating ``current``.

        The decoder is not causal: a frame comes out right only if the decoder also sees the latent
        frames on both sides of it. Decoding each chunk on its own leaves a visible step at every
        chunk boundary. So a few latent frames of the previous chunk go in first as left context,
        and with ``decode_delay`` the output runs one latent frame (8 frames, 0.32 s) behind: the
        last latent frame of a chunk is held back and shown with the next chunk, when its right
        neighbour exists. Chunk 0 then yields 17 frames and every later chunk 32.
        """
        per_frame = self.tokens_per_frame
        if self.previous is None:
            decoded = self._decode_latents(current, CHUNK_LATENT_FRAMES)
            return decoded[:17] if self.decode_delay else decoded
        lead = self.decode_context + (1 if self.decode_delay else 0)
        tokens = torch.cat([self.previous[:, -lead * per_frame :], current], dim=1)
        decoded = self._decode_latents(tokens, lead + CHUNK_LATENT_FRAMES)
        if self.decode_delay:
            first = 8 * self.decode_context - 7
            return decoded[first : first + 32]
        return decoded[-32:]

    def _decode_latents(self, tokens: torch.Tensor, frames: int) -> torch.Tensor:
        shape = VideoLatentShape(
            batch=1, channels=tokens.shape[-1], frames=frames, height=self.height // 32, width=self.width // 32
        )
        latent = self.e.components.video_patchifier.unpatchify(tokens, output_shape=shape)
        return next(iter(decode_video(latent, self.e.video_decoder, None, self.generator)))

    # ----- one chunk -----

    @torch.inference_mode()
    def next(self, samples: torch.Tensor) -> Chunk:
        """Generate the next chunk. ``samples``: this chunk's audio, 16 kHz mono float in [-1, 1]."""
        expected = int(round(chunk_seconds(self.index) * 16000))
        if samples.numel() != expected:
            raise ValueError(f"chunk {self.index} needs {expected} samples, got {samples.numel()}")
        timing = Timing()
        clock = _Clock(self.e.device)

        audio = self._audio_latents(samples.float().cpu())
        timing.audio_ms = clock.lap()

        if self.index == 0:
            video_state = self._first_state()
            current = slice(0, video_state.latent.shape[1])
            window_frames = 25
        else:
            video_state, current = self._window_state()
            window_frames = 8 * (video_state.latent.shape[1] // self.tokens_per_frame - 1) + 1
        audio_state, audio_current = self._audio_state(audio, window_frames)

        cache = ARFeatureCache() if self.feature_cache else None
        for step in range(len(self.sigmas) - 1):
            if cache is None:
                denoised, _ = self.denoise(video_state, audio_state, self.sigmas, step)
            else:
                denoised, _ = self.denoise(
                    video_state, audio_state, self.sigmas, step,
                    ar_feature_cache=cache, video_current_slice=current, audio_current_slice=audio_current,
                )
            denoised = post_process_latent(denoised, video_state.denoise_mask, video_state.clean_latent)
            stepped = self.e.stepper.step(video_state.latent, denoised, self.sigmas, step)
            stepped = post_process_latent(stepped, video_state.denoise_mask, video_state.clean_latent)
            video_state = replace(video_state, latent=stepped, clean_latent=stepped)
            timing.steps_ms.append(clock.lap())
        timing.denoise_ms = sum(timing.steps_ms)

        tokens = video_state.latent[:, current].contiguous()
        frames = self._decode(tokens).cpu()
        timing.decode_ms = clock.lap()

        if self.index == 0:
            self.first, self.first_audio = tokens, audio
        self.previous, self.previous_audio = tokens, audio
        chunk = Chunk(index=self.index, frames=frames, seconds=frames.shape[0] / FPS, timing=timing)
        self.index += 1
        return chunk


class _Clock:
    def __init__(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"
        self.at = self._now()

    def _now(self) -> float:
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def lap(self) -> float:
        now = self._now()
        took, self.at = (now - self.at) * 1000.0, now
        return took
