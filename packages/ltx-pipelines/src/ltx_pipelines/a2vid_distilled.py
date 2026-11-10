from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from collections.abc import Iterator
from typing import Literal

import torch

from ltx_core.components.guiders import CFGGuider, STGGuider
from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByGlobalLatentPrefix,
    VideoConditionByGlobalLatentSuffix,
    VideoConditionByLatentIndex,
)
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.audio_vae import AudioEncoder
from ltx_core.model.transformer import X0Model
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.upsampler import LatentUpsampler
from ltx_core.model.video_vae import TilingConfig, VideoDecoder, VideoEncoder, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.text_encoders.gemma import EmbeddingsProcessor, GemmaTextEncoder
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils import ModelLedger, euler_denoising_loop
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    default_2_stage_distilled_arg_parser,
    detect_checkpoint_path,
)
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    detect_params,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    apply_video_channel_condition_to_state,
    cleanup_memory,
    combined_image_conditionings,
    denoise_video_only,
    encode_prompts,
    generate_enhanced_prompt,
    get_device,
    global_image_conditionings_by_appending_latent,
    global_image_conditionings_by_prepending_latent,
    noise_audio_state,
    noise_video_state,
    modality_from_latent_state,
    simple_denoising_func,
)
from ltx_pipelines.utils.autoregressive import AudioAlignment, autoregressive_euler_denoising_loop
from ltx_pipelines.utils.media_io import (
    decode_audio_from_file,
    encode_video,
    ensure_stereo_audio,
    normalize_latent,
    resize_and_center_crop,
)
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()
DistilledStageMode = Literal["one-stage", "two-stage"]
A2VidInferenceBehavior = Literal["default", "ar"]
SigmaSchedule = list[float] | tuple[float, ...] | torch.Tensor | None
STGMode = Literal["stg_av", "stg_v"]
logger = logging.getLogger(__name__)


@dataclass
class _FastA2VidModules:
    text_encoder: GemmaTextEncoder
    embeddings_processor: EmbeddingsProcessor
    audio_encoder: AudioEncoder
    video_encoder: VideoEncoder
    transformer: X0Model
    video_decoder: VideoDecoder
    spatial_upsampler: LatentUpsampler | None = None


@dataclass(frozen=True)
class LatentConditioningInput:
    """Pre-encoded latent conditioning pasted into target latent frame indices."""

    latent: torch.Tensor
    latent_idx: int
    strength: float = 1.0


@dataclass(frozen=True)
class VideoFrameConditioningInput:
    """Decoded video frames encoded as one temporal VAE latent conditioning."""

    frames: torch.Tensor
    latent_idx: int = 0
    strength: float = 1.0


@dataclass(frozen=True)
class GlobalLatentConditioningInput:
    """Pre-encoded latent conditioning appended/prepended as removable global context."""

    latent: torch.Tensor
    strength: float = 1.0
    repeat_to_target_frames: bool = False
    position_mode: Literal["zero", "match_target"] = "zero"


def _resolve_sigmas(
    values: SigmaSchedule,
    default_values: list[float],
    *,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    sigmas = torch.as_tensor(default_values if values is None else values, device=device, dtype=torch.float32)
    if sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError(f"{name} must be a 1D sigma schedule with at least two values.")
    if torch.any((sigmas < 0.0) | (sigmas > 1.0)):
        raise ValueError(f"{name} values must be in [0, 1].")
    if torch.any(sigmas[:-1] < sigmas[1:]):
        raise ValueError(f"{name} must be monotonically non-increasing.")
    return sigmas


def _warn_if_channel_condition_is_inert(transformer: torch.nn.Module) -> None:
    """Say so when the first-frame channel condition cannot change the output.

    The pathway adds ``proj(condition) * gate`` to the video tokens. With an all-zero projection
    it adds zero whatever the gate holds, so turning the condition on does nothing. Weights that
    were zero-filled because the checkpoint lacked them, or that were saved before the pathway
    was trained, look exactly like this.
    """
    velocity_model = getattr(transformer, "velocity_model", transformer)
    projection = getattr(velocity_model, "video_channel_condition_proj", None)
    if projection is None:
        return
    if int(torch.count_nonzero(projection.weight)) == 0:
        logger.warning(
            "video_channel_condition_proj.weight is all zeros in this checkpoint: the first-frame channel "
            "condition adds nothing to the video tokens, so it has no effect on this run."
        )


def _build_stg_perturbation_config(
    *,
    stg_blocks: list[int] | None,
    stg_mode: STGMode,
) -> BatchedPerturbationConfig:
    if stg_mode not in ("stg_av", "stg_v"):
        raise ValueError(f"stg_mode must be 'stg_av' or 'stg_v', got {stg_mode!r}.")
    perturbations = [
        Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=stg_blocks),
    ]
    if stg_mode == "stg_av":
        perturbations.append(Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=stg_blocks))
    return BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=perturbations)])


def guided_denoising_func(
    *,
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    negative_video_context: torch.Tensor | None,
    negative_audio_context: torch.Tensor | None,
    transformer: X0Model,
    guidance_scale: float,
    stg_scale: float,
    stg_blocks: list[int] | None,
    stg_mode: STGMode,
):
    cfg_guider = CFGGuider(guidance_scale)
    stg_guider = STGGuider(stg_scale)
    stg_perturbation_config = (
        _build_stg_perturbation_config(stg_blocks=stg_blocks, stg_mode=stg_mode)
        if stg_guider.enabled()
        else None
    )

    def guided_denoising_step(
        video_state: LatentState,
        audio_state: LatentState,
        sigmas: torch.Tensor,
        step_index: int,
        *,
        ar_feature_cache=None,
        video_current_slice: slice | None = None,
        audio_current_slice: slice | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(
            video_state,
            video_context,
            sigma,
            ar_feature_cache=ar_feature_cache,
            ar_current_slice=video_current_slice,
        )
        pos_audio = modality_from_latent_state(
            audio_state,
            audio_context,
            sigma,
            ar_feature_cache=ar_feature_cache,
            ar_current_slice=audio_current_slice,
        )

        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        pos_denoised_video, pos_denoised_audio = denoised_video, denoised_audio

        if cfg_guider.enabled():
            if negative_video_context is None or negative_audio_context is None:
                raise ValueError("negative_prompt context is required when guidance_scale != 1.0.")
            neg_video = modality_from_latent_state(video_state, negative_video_context, sigma)
            neg_audio = modality_from_latent_state(audio_state, negative_audio_context, sigma)
            neg_denoised_video, neg_denoised_audio = transformer(video=neg_video, audio=neg_audio, perturbations=None)
            denoised_video = denoised_video + cfg_guider.delta(pos_denoised_video, neg_denoised_video)
            denoised_audio = denoised_audio + cfg_guider.delta(pos_denoised_audio, neg_denoised_audio)

        if stg_guider.enabled() and stg_perturbation_config is not None:
            ptb_video = replace(pos_video, ar_feature_cache=None, ar_current_slice=None)
            ptb_audio = replace(pos_audio, ar_feature_cache=None, ar_current_slice=None)
            perturbed_video, perturbed_audio = transformer(
                video=ptb_video,
                audio=ptb_audio,
                perturbations=stg_perturbation_config,
            )
            denoised_video = denoised_video + stg_guider.delta(pos_denoised_video, perturbed_video)
            denoised_audio = denoised_audio + stg_guider.delta(pos_denoised_audio, perturbed_audio)

        return denoised_video, denoised_audio

    return guided_denoising_step


class A2VidDistilledPipeline:
    """
    Distilled audio-to-video generation pipeline.

    The input audio is VAE-encoded and kept clean as conditioning while the
    model denoises only video latents. ``stage_mode='one-stage'`` runs the
    distilled model once at the target resolution. ``stage_mode='two-stage'``
    runs the default distilled two-stage flow: half-resolution generation,
    latent upsampling, then high-resolution refinement.
    """

    def __init__(
        self,
        distilled_checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str | None,
        loras: list[LoraPathStrengthAndSDOps] | tuple[LoraPathStrengthAndSDOps, ...] = (),
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=distilled_checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root_path=gemma_root,
            loras=loras,
            quantization=quantization,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        self._fast_modules: _FastA2VidModules | None = None
        self.last_audio_latent: torch.Tensor | None = None
        self.last_stage1_video_latent: torch.Tensor | None = None
        self.last_final_video_latent: torch.Tensor | None = None

    def clear_fast_cache(self) -> None:
        """Release cached fast-inference modules from GPU memory."""
        cached = self._fast_modules
        if cached is None:
            return

        self._fast_modules = None
        del cached
        cleanup_memory()

    def _get_fast_modules(
        self,
        *,
        stage_mode: DistilledStageMode,
        transformer_extra_config: dict | None = None,
    ) -> _FastA2VidModules:
        cached = self._fast_modules
        if (
            cached is not None
            and transformer_extra_config is not None
            and not self._has_channel_condition_projection(cached.transformer)
        ):
            self.clear_fast_cache()
            cached = None
        if cached is None:
            cached = _FastA2VidModules(
                text_encoder=self.model_ledger.text_encoder(),
                embeddings_processor=self.model_ledger.gemma_embeddings_processor(),
                audio_encoder=self.model_ledger.audio_encoder(),
                video_encoder=self.model_ledger.video_encoder(),
                transformer=self.model_ledger.transformer(extra_config=transformer_extra_config),
                video_decoder=self.model_ledger.video_decoder(),
                spatial_upsampler=(
                    self.model_ledger.spatial_upsampler()
                    if stage_mode == "two-stage" and self.model_ledger.spatial_upsampler_path is not None
                    else None
                ),
            )
            self._fast_modules = cached
            logger.info("Loaded fast-infer A2V distilled modules onto GPU")
        elif stage_mode == "two-stage" and cached.spatial_upsampler is None:
            cached.spatial_upsampler = self.model_ledger.spatial_upsampler()
        return cached

    @staticmethod
    def _has_channel_condition_projection(transformer: torch.nn.Module) -> bool:
        velocity_model = getattr(transformer, "velocity_model", transformer)
        return hasattr(velocity_model, "video_channel_condition_proj")

    def _encode_prompt_context(
        self,
        *,
        prompt: str,
        images: list[ImageConditioningInput],
        enhance_prompt: bool,
        text_encoder: GemmaTextEncoder | None = None,
        embeddings_processor: EmbeddingsProcessor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text_encoder is None or embeddings_processor is None:
            (ctx_p,) = encode_prompts(
                [prompt],
                self.model_ledger,
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            )
            return ctx_p.video_encoding, ctx_p.audio_encoding

        if enhance_prompt:
            prompt = generate_enhanced_prompt(
                text_encoder,
                prompt,
                images[0][0] if len(images) > 0 else None,
            )
        hidden_states, attention_mask = text_encoder.encode(prompt)
        processed = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
        return processed.video_encoding, processed.audio_encoding

    def _encode_guidance_contexts(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        images: list[ImageConditioningInput],
        enhance_prompt: bool,
        guidance_scale: float,
        text_encoder: GemmaTextEncoder | None = None,
        embeddings_processor: EmbeddingsProcessor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if text_encoder is None or embeddings_processor is None:
            prompts = [prompt]
            if guidance_scale != 1.0:
                prompts.append(negative_prompt)
            ctx = encode_prompts(
                prompts,
                self.model_ledger,
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            )
            v_context_p, a_context_p = ctx[0].video_encoding, ctx[0].audio_encoding
            if guidance_scale == 1.0:
                return v_context_p, a_context_p, None, None
            return v_context_p, a_context_p, ctx[1].video_encoding, ctx[1].audio_encoding

        if enhance_prompt:
            prompt = generate_enhanced_prompt(
                text_encoder,
                prompt,
                images[0][0] if len(images) > 0 else None,
            )
        hidden_states, attention_mask = text_encoder.encode(prompt)
        processed = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
        v_context_p, a_context_p = processed.video_encoding, processed.audio_encoding
        if guidance_scale == 1.0:
            return v_context_p, a_context_p, None, None

        hidden_states, attention_mask = text_encoder.encode(negative_prompt)
        processed = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
        return v_context_p, a_context_p, processed.video_encoding, processed.audio_encoding

    def _build_video_conditionings(
        self,
        *,
        images: list[ImageConditioningInput],
        video_frame_conditionings: list[VideoFrameConditioningInput],
        latent_conditionings: list[LatentConditioningInput],
        global_condition_images: list[ImageConditioningInput],
        global_latent_conditionings: list[GlobalLatentConditioningInput],
        global_condition_position: Literal["prepend", "append"],
        global_condition_repeat_to_target_frames: bool,
        global_condition_position_mode: Literal["zero", "match_target"],
        height: int,
        width: int,
        video_encoder: VideoEncoder,
        dtype: torch.dtype,
    ) -> list[ConditioningItem]:
        conditionings = combined_image_conditionings(
            images=images,
            height=height,
            width=width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        conditionings.extend(
            self._build_video_frame_conditionings(
                video_frame_conditionings=video_frame_conditionings,
                height=height,
                width=width,
                video_encoder=video_encoder,
                dtype=dtype,
            )
        )
        conditionings.extend(
            self._build_latent_conditionings(
                latent_conditionings=latent_conditionings,
                dtype=dtype,
            )
        )
        conditionings.extend(
            self._build_global_latent_conditionings(
                latent_conditionings=global_latent_conditionings,
                position=global_condition_position,
                dtype=dtype,
            )
        )
        if global_condition_position == "prepend":
            conditionings.extend(
                global_image_conditionings_by_prepending_latent(
                    images=global_condition_images,
                    height=height,
                    width=width,
                    video_encoder=video_encoder,
                    dtype=dtype,
                    device=self.device,
                    repeat_to_target_frames=global_condition_repeat_to_target_frames,
                    position_mode=global_condition_position_mode,
                )
            )
        else:
            conditionings.extend(
                global_image_conditionings_by_appending_latent(
                    images=global_condition_images,
                    height=height,
                    width=width,
                    video_encoder=video_encoder,
                    dtype=dtype,
                    device=self.device,
                    repeat_to_target_frames=global_condition_repeat_to_target_frames,
                    position_mode=global_condition_position_mode,
                )
            )
        return conditionings

    def _build_video_frame_conditionings(
        self,
        *,
        video_frame_conditionings: list[VideoFrameConditioningInput],
        height: int,
        width: int,
        video_encoder: VideoEncoder,
        dtype: torch.dtype,
    ) -> list[ConditioningItem]:
        conditionings: list[ConditioningItem] = []
        for conditioning in video_frame_conditionings:
            frames = conditioning.frames
            if frames.ndim != 4:
                raise ValueError(
                    "video_frame_conditionings frames must have shape [F, H, W, C], "
                    f"got {tuple(frames.shape)}."
                )
            if frames.shape[0] <= 0:
                raise ValueError("video_frame_conditionings frames must contain at least one frame.")
            if frames.shape[-1] != 3:
                raise ValueError(
                    "video_frame_conditionings frames must use RGB channel-last layout, "
                    f"got {tuple(frames.shape)}."
                )
            if conditioning.latent_idx < 0:
                raise ValueError(f"video frame latent_idx must be non-negative, got {conditioning.latent_idx}.")
            if conditioning.strength < 0.0 or conditioning.strength > 1.0:
                raise ValueError(f"video frame conditioning strength must be in [0, 1], got {conditioning.strength}.")

            video = resize_and_center_crop(frames.to(device=self.device, dtype=torch.float32), height, width)
            video = normalize_latent(video, self.device, dtype)
            encoded_video = video_encoder(video)
            conditionings.append(
                VideoConditionByLatentIndex(
                    latent=encoded_video,
                    strength=conditioning.strength,
                    latent_idx=conditioning.latent_idx,
                )
            )
        return conditionings

    def _build_global_latent_conditionings(
        self,
        *,
        latent_conditionings: list[GlobalLatentConditioningInput],
        position: Literal["prepend", "append"],
        dtype: torch.dtype,
    ) -> list[ConditioningItem]:
        conditionings: list[ConditioningItem] = []
        conditioning_cls = (
            VideoConditionByGlobalLatentPrefix
            if position == "prepend"
            else VideoConditionByGlobalLatentSuffix
        )
        for conditioning in latent_conditionings:
            latent = conditioning.latent
            if latent.ndim == 4:
                latent = latent.unsqueeze(0)
            if latent.ndim != 5:
                raise ValueError(
                    "global_latent_conditionings must contain latents with shape "
                    "[C, F, H, W] or [B, C, F, H, W]."
                )
            if conditioning.strength < 0.0 or conditioning.strength > 1.0:
                raise ValueError(
                    f"global latent conditioning strength must be in [0, 1], got {conditioning.strength}."
                )
            if conditioning.position_mode not in ("zero", "match_target"):
                raise ValueError(
                    "global latent conditioning position_mode must be 'zero' or 'match_target', "
                    f"got {conditioning.position_mode!r}."
                )
            conditionings.append(
                conditioning_cls(
                    latent=latent.to(device=self.device, dtype=dtype),
                    strength=conditioning.strength,
                    repeat_to_target_frames=conditioning.repeat_to_target_frames,
                    position_mode=conditioning.position_mode,
                )
            )
        return conditionings

    def _build_latent_conditionings(
        self,
        *,
        latent_conditionings: list[LatentConditioningInput],
        dtype: torch.dtype,
    ) -> list[ConditioningItem]:
        conditionings: list[ConditioningItem] = []
        for conditioning in latent_conditionings:
            latent = conditioning.latent
            if latent.ndim == 4:
                latent = latent.unsqueeze(0)
            if latent.ndim != 5:
                raise ValueError(
                    "latent_conditionings must contain latents with shape [C, F, H, W] or [B, C, F, H, W]."
                )
            if conditioning.latent_idx < 0:
                raise ValueError(f"latent_idx must be non-negative, got {conditioning.latent_idx}.")
            if conditioning.strength < 0.0 or conditioning.strength > 1.0:
                raise ValueError(f"latent conditioning strength must be in [0, 1], got {conditioning.strength}.")
            conditionings.append(
                VideoConditionByLatentIndex(
                    latent=latent.to(device=self.device, dtype=dtype),
                    strength=conditioning.strength,
                    latent_idx=conditioning.latent_idx,
                )
            )
        return conditionings

    def _total_video_latent_frames(self, output_shape: VideoPixelShape) -> int:
        return VideoLatentShape.from_pixel_shape(
            shape=output_shape,
            latent_channels=self.pipeline_components.video_latent_channels,
            scale_factors=self.pipeline_components.video_scale_factors,
        ).frames

    def _denoise_video_only_ar(  # noqa: PLR0913
        self,
        *,
        output_shape: VideoPixelShape,
        conditionings: list[ConditioningItem],
        noiser: GaussianNoiser,
        sigmas: torch.Tensor,
        stepper: DiffusionStepProtocol,
        denoise_fn_builder,
        total_video_latent_frames: int,
        ar_video_chunk_size: int,
        ar_history_chunk_count: int | None,
        ar_sink_first_chunk: bool,
        ar_relative_positions: bool,
        ar_history_feature_cache: bool,
        ar_first_frame_prefix_condition: bool,
        ar_first_frame_condition_position: Literal["prepend", "append"],
        ar_rope_max_temporal_index: int | None,
        dtype: torch.dtype,
        ar_audio_alignment: AudioAlignment = "time",
        initial_video_latent: torch.Tensor | None = None,
        initial_audio_latent: torch.Tensor | None = None,
        video_channel_condition: torch.Tensor | None = None,
        derive_channel_condition_from_first_chunk: bool = False,
        channel_condition_current_chunk_only: bool = False,
        noise_scale: float = 1.0,
    ) -> LatentState:
        video_state, video_tools = noise_video_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=conditionings,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=noise_scale,
            initial_latent=initial_video_latent,
        )
        video_state = apply_video_channel_condition_to_state(
            video_state=video_state,
            video_tools=video_tools,
            channel_condition=video_channel_condition,
        )
        audio_state, _ = noise_audio_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=[],
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=0.0,
            initial_latent=initial_audio_latent,
        )
        audio_state = replace(audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask))

        video_state, _ = autoregressive_euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn_builder=denoise_fn_builder,
            total_video_latent_frames=total_video_latent_frames,
            video_chunk_size=ar_video_chunk_size,
            history_chunk_count=ar_history_chunk_count,
            sink_first_chunk=ar_sink_first_chunk,
            relative_positions=ar_relative_positions,
            history_feature_cache=ar_history_feature_cache,
            derive_channel_condition_from_first_chunk=derive_channel_condition_from_first_chunk,
            channel_condition_current_chunk_only=channel_condition_current_chunk_only,
            first_frame_prefix_condition=ar_first_frame_prefix_condition,
            first_frame_condition_position=ar_first_frame_condition_position,
            rope_max_temporal_index=ar_rope_max_temporal_index,
            audio_alignment=ar_audio_alignment,
            video_temporal_scale=self.pipeline_components.video_scale_factors.time,
        )
        video_state = video_tools.clear_conditioning(video_state)
        return video_tools.unpatchify(video_state)

    def _repeat_first_frame_channel_condition_latent(
        self,
        *,
        latent: torch.Tensor | None,
        output_shape: VideoPixelShape,
    ) -> torch.Tensor | None:
        if latent is None:
            return None
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        if latent.ndim != 5:
            raise ValueError(
                "first-frame channel condition latent must have shape [C, F, H, W] or [B, C, F, H, W], "
                f"got {tuple(latent.shape)}."
            )
        if latent.shape[2] <= 0:
            raise ValueError("first-frame channel condition latent must contain at least one latent frame.")
        shape = VideoLatentShape.from_pixel_shape(
            shape=output_shape,
            latent_channels=latent.shape[1],
            scale_factors=self.pipeline_components.video_scale_factors,
        )
        first_latent = latent[:, :, :1].contiguous()
        return first_latent.repeat(1, 1, shape.frames, 1, 1)

    @staticmethod
    def _channel_condition_extra_config(
        *,
        first_frame_channel_condition_latent: torch.Tensor | None,
        stage2_first_frame_channel_condition_latent: torch.Tensor | None,
        init: Literal["zero", "xavier", "kaiming"],
        mode: Literal["add", "gated"],
        fallback_channel_dim: int | None = None,
    ) -> dict | None:
        latent = (
            first_frame_channel_condition_latent
            if first_frame_channel_condition_latent is not None
            else stage2_first_frame_channel_condition_latent
        )
        if latent is None:
            if fallback_channel_dim is None:
                return None
            return {
                "video_channel_condition_in_channels": int(fallback_channel_dim),
                "video_channel_condition_init": init,
                "video_channel_condition_mode": mode,
            }
        if latent.ndim not in (4, 5):
            raise ValueError(
                "first-frame channel condition latent must have shape [C, F, H, W] or [B, C, F, H, W], "
                f"got {tuple(latent.shape)}."
            )
        channel_dim = latent.shape[0] if latent.ndim == 4 else latent.shape[1]
        return {
            "video_channel_condition_in_channels": int(channel_dim),
            "video_channel_condition_init": init,
            "video_channel_condition_mode": mode,
        }

    @staticmethod
    def _prepend_decode_temporal_prefix_latent(
        *,
        latent: torch.Tensor,
        prefix_latent: torch.Tensor,
    ) -> torch.Tensor:
        if prefix_latent.ndim == 4:
            prefix_latent = prefix_latent.unsqueeze(0)
        if prefix_latent.ndim != 5:
            raise ValueError(
                "decode temporal prefix latent must have shape [C, F, H, W] or [B, C, F, H, W], "
                f"got {tuple(prefix_latent.shape)}."
            )
        if latent.ndim != 5:
            raise ValueError(f"Expected video latent with shape [B, C, F, H, W], got {tuple(latent.shape)}.")

        prefix_latent = prefix_latent.to(device=latent.device, dtype=latent.dtype)
        if prefix_latent.shape[0] != latent.shape[0]:
            if prefix_latent.shape[0] == 1:
                prefix_latent = prefix_latent.expand(latent.shape[0], -1, -1, -1, -1)
            else:
                raise ValueError(
                    "decode temporal prefix batch size must be 1 or match the generated latent batch. "
                    f"Got prefix batch {prefix_latent.shape[0]} and latent batch {latent.shape[0]}."
                )
        if (prefix_latent.shape[1], prefix_latent.shape[3], prefix_latent.shape[4]) != (
            latent.shape[1],
            latent.shape[3],
            latent.shape[4],
        ):
            raise ValueError(
                "decode temporal prefix latent must match generated latent channels and spatial shape. "
                f"Got prefix {tuple(prefix_latent.shape)} and latent {tuple(latent.shape)}."
            )
        return torch.cat([prefix_latent.contiguous(), latent], dim=2)

    @staticmethod
    def _replace_audio_latent_prefix(
        *,
        encoded_audio_latent: torch.Tensor,
        prefix_latent: torch.Tensor | None,
        prefix_frame_count: int | None,
        prepend_prefix: bool = False,
        target_frame_count: int | None = None,
    ) -> torch.Tensor:
        if target_frame_count is None:
            target_frame_count = encoded_audio_latent.shape[2]
        if prefix_latent is None:
            return A2VidDistilledPipeline._fit_audio_latent_to_frame_count(
                encoded_audio_latent,
                target_frame_count=target_frame_count,
            )
        if prefix_latent.ndim != 4:
            raise ValueError(
                "audio prefix latent must have shape [B, C, F, M], "
                f"got {tuple(prefix_latent.shape)}."
            )
        if encoded_audio_latent.ndim != 4:
            raise ValueError(
                "encoded audio latent must have shape [B, C, F, M], "
                f"got {tuple(encoded_audio_latent.shape)}."
            )

        requested_frames = prefix_latent.shape[2] if prefix_frame_count is None else prefix_frame_count
        if requested_frames <= 0:
            return A2VidDistilledPipeline._fit_audio_latent_to_frame_count(
                encoded_audio_latent,
                target_frame_count=target_frame_count,
            )

        copied_frames = min(requested_frames, prefix_latent.shape[2], target_frame_count)
        if copied_frames <= 0:
            return A2VidDistilledPipeline._fit_audio_latent_to_frame_count(
                encoded_audio_latent,
                target_frame_count=target_frame_count,
            )

        prefix = prefix_latent[:, :, -copied_frames:].to(
            device=encoded_audio_latent.device,
            dtype=encoded_audio_latent.dtype,
        )
        if prefix.shape[0] != encoded_audio_latent.shape[0]:
            if prefix.shape[0] == 1:
                prefix = prefix.expand(encoded_audio_latent.shape[0], -1, -1, -1)
            else:
                raise ValueError(
                    "audio prefix batch size must be 1 or match encoded audio latent batch. "
                    f"Got prefix batch {prefix.shape[0]} and audio batch {encoded_audio_latent.shape[0]}."
                )
        if (prefix.shape[1], prefix.shape[3]) != (
            encoded_audio_latent.shape[1],
            encoded_audio_latent.shape[3],
        ):
            raise ValueError(
                "audio prefix latent must match encoded audio channels and mel bins. "
                f"Got prefix {tuple(prefix.shape)} and audio {tuple(encoded_audio_latent.shape)}."
            )

        if prepend_prefix:
            return A2VidDistilledPipeline._fit_audio_latent_to_frame_count(
                torch.cat([prefix, encoded_audio_latent], dim=2),
                target_frame_count=target_frame_count,
            )

        audio_latent = A2VidDistilledPipeline._fit_audio_latent_to_frame_count(
            encoded_audio_latent,
            target_frame_count=target_frame_count,
        )
        audio_latent[:, :, :copied_frames] = prefix
        return audio_latent

    @staticmethod
    def _fit_audio_latent_to_frame_count(
        audio_latent: torch.Tensor,
        *,
        target_frame_count: int,
    ) -> torch.Tensor:
        if target_frame_count <= 0:
            raise ValueError(f"target audio latent frame count must be positive, got {target_frame_count}.")
        if audio_latent.shape[2] > target_frame_count:
            return audio_latent[:, :, :target_frame_count].contiguous()
        if audio_latent.shape[2] < target_frame_count:
            padding = audio_latent.new_zeros(
                audio_latent.shape[0],
                audio_latent.shape[1],
                target_frame_count - audio_latent.shape[2],
                audio_latent.shape[3],
            )
            return torch.cat([audio_latent, padding], dim=2).contiguous()
        return audio_latent.contiguous()

    @staticmethod
    def _slice_decoded_video(
        video: Iterator[torch.Tensor],
        *,
        start_frame: int,
        max_frames: int,
    ) -> Iterator[torch.Tensor]:
        if start_frame < 0:
            raise ValueError(f"start_frame must be non-negative, got {start_frame}.")
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}.")

        remaining_skip = start_frame
        remaining_take = max_frames
        for chunk in video:
            if remaining_skip:
                if chunk.shape[0] <= remaining_skip:
                    remaining_skip -= chunk.shape[0]
                    continue
                chunk = chunk[remaining_skip:]
                remaining_skip = 0

            if remaining_take <= 0:
                break
            if chunk.shape[0] > remaining_take:
                chunk = chunk[:remaining_take]
            remaining_take -= chunk.shape[0]
            if chunk.shape[0] > 0:
                yield chunk
            if remaining_take <= 0:
                break

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        negative_prompt: str = "",
        guidance_scale: float = 1.0,
        stg_scale: float = 0.0,
        stg_blocks: list[int] | None = None,
        stg_mode: STGMode = "stg_av",
        stage_mode: DistilledStageMode = "two-stage",
        stage1_sigmas: SigmaSchedule = None,
        stage2_sigmas: SigmaSchedule = None,
        video_frame_conditionings: list[VideoFrameConditioningInput] | None = None,
        stage2_video_frame_conditionings: list[VideoFrameConditioningInput] | None = None,
        latent_conditionings: list[LatentConditioningInput] | None = None,
        stage2_latent_conditionings: list[LatentConditioningInput] | None = None,
        global_condition_images: list[ImageConditioningInput] | None = None,
        global_latent_conditionings: list[GlobalLatentConditioningInput] | None = None,
        stage2_global_latent_conditionings: list[GlobalLatentConditioningInput] | None = None,
        global_condition_position: Literal["prepend", "append"] = "prepend",
        global_condition_repeat_to_target_frames: bool = False,
        global_condition_position_mode: Literal["zero", "match_target"] = "zero",
        use_global_condition_in_stage2: bool = False,
        fast_infer: bool = False,
        inference_behavior: A2VidInferenceBehavior = "default",
        ar_video_chunk_size: int | None = None,
        ar_history_chunk_count: int | None = None,
        ar_sink_first_chunk: bool = False,
        ar_relative_positions: bool = False,
        ar_history_feature_cache: bool = False,
        ar_first_frame_prefix_condition: bool = False,
        ar_first_frame_condition_position: Literal["prepend", "append"] = "prepend",
        ar_rope_max_temporal_index: int | None = None,
        ar_audio_alignment: AudioAlignment = "time",
        first_frame_channel_condition_latent: torch.Tensor | None = None,
        stage2_first_frame_channel_condition_latent: torch.Tensor | None = None,
        first_frame_channel_condition_init: Literal["zero", "xavier", "kaiming"] = "zero",
        first_frame_channel_condition_mode: Literal["add", "gated"] = "add",
        ar_first_frame_channel_condition_from_first_chunk: bool = False,
        ar_channel_condition_current_chunk_only: bool = False,
        audio_prefix_latent: torch.Tensor | None = None,
        audio_prefix_latent_frame_count: int | None = None,
        prepend_audio_prefix_latent: bool = False,
        decode_temporal_prefix_latent: torch.Tensor | None = None,
        stage2_decode_temporal_prefix_latent: torch.Tensor | None = None,
        decode_temporal_prefix_drop_frames: int = 1,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        if inference_behavior not in ("default", "ar"):
            raise ValueError(
                f"inference_behavior must be 'default' or 'ar', got {inference_behavior!r}."
            )
        if stage_mode not in ("one-stage", "two-stage"):
            raise ValueError(f"stage_mode must be 'one-stage' or 'two-stage', got {stage_mode!r}.")
        if guidance_scale < 1.0:
            raise ValueError(f"guidance_scale must be >= 1.0, got {guidance_scale}.")
        if stg_scale < 0.0:
            raise ValueError(f"stg_scale must be >= 0.0, got {stg_scale}.")
        if stg_mode not in ("stg_av", "stg_v"):
            raise ValueError(f"stg_mode must be 'stg_av' or 'stg_v', got {stg_mode!r}.")
        if stage_mode == "two-stage" and self.model_ledger.spatial_upsampler_path is None:
            raise ValueError("spatial_upsampler_path is required when stage_mode='two-stage'.")
        if global_condition_position not in ("prepend", "append"):
            raise ValueError(
                f"global_condition_position must be 'prepend' or 'append', got {global_condition_position!r}."
            )
        if global_condition_position_mode not in ("zero", "match_target"):
            raise ValueError(
                "global_condition_position_mode must be 'zero' or 'match_target', "
                f"got {global_condition_position_mode!r}."
            )
        if ar_first_frame_condition_position not in ("prepend", "append"):
            raise ValueError(
                "ar_first_frame_condition_position must be 'prepend' or 'append', "
                f"got {ar_first_frame_condition_position!r}."
            )
        if inference_behavior == "ar":
            if ar_video_chunk_size is None or ar_video_chunk_size <= 0:
                raise ValueError("ar_video_chunk_size must be a positive integer when inference_behavior='ar'.")
            if ar_history_chunk_count is not None and ar_history_chunk_count < 0:
                raise ValueError("ar_history_chunk_count must be non-negative.")
        if decode_temporal_prefix_drop_frames < 0:
            raise ValueError("decode_temporal_prefix_drop_frames must be non-negative.")

        assert_resolution(height=height, width=width, is_two_stage=(stage_mode == "two-stage"))

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = self.dtype
        images = list(images)
        video_frame_conditionings = list(video_frame_conditionings or ())
        stage2_video_frame_conditionings = list(stage2_video_frame_conditionings or ())
        latent_conditionings = list(latent_conditionings or ())
        stage2_latent_conditionings = list(stage2_latent_conditionings or ())
        global_condition_images = list(global_condition_images or ())
        global_latent_conditionings = list(global_latent_conditionings or ())
        stage2_global_latent_conditionings = list(stage2_global_latent_conditionings or ())
        transformer_extra_config = self._channel_condition_extra_config(
            first_frame_channel_condition_latent=first_frame_channel_condition_latent,
            stage2_first_frame_channel_condition_latent=stage2_first_frame_channel_condition_latent,
            init=first_frame_channel_condition_init,
            mode=first_frame_channel_condition_mode,
            fallback_channel_dim=(
                self.pipeline_components.video_latent_channels
                if ar_first_frame_channel_condition_from_first_chunk
                else None
            ),
        )
        self.last_audio_latent = None
        self.last_stage1_video_latent = None
        self.last_final_video_latent = None

        fast_modules = (
            self._get_fast_modules(stage_mode=stage_mode, transformer_extra_config=transformer_extra_config)
            if fast_infer
            else None
        )

        v_context_p, a_context_p, v_context_n, a_context_n = self._encode_guidance_contexts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            images=images,
            enhance_prompt=enhance_prompt,
            guidance_scale=guidance_scale,
            text_encoder=fast_modules.text_encoder if fast_modules is not None else None,
            embeddings_processor=fast_modules.embeddings_processor if fast_modules is not None else None,
        )

        decoded_audio = decode_audio_from_file(audio_path, self.device, audio_start_time, audio_max_duration)
        decoded_audio = ensure_stereo_audio(decoded_audio)
        if decoded_audio is None:
            raise ValueError(f"No audio stream found in {audio_path!r}.")
        audio_encoder = fast_modules.audio_encoder if fast_modules is not None else self.model_ledger.audio_encoder()
        encoded_audio_latent = vae_encode_audio(decoded_audio, audio_encoder)
        audio_shape = AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16)
        encoded_audio_latent = encoded_audio_latent[:, :, : audio_shape.frames]
        encoded_audio_latent = self._replace_audio_latent_prefix(
            encoded_audio_latent=encoded_audio_latent,
            prefix_latent=audio_prefix_latent,
            prefix_frame_count=audio_prefix_latent_frame_count,
            prepend_prefix=prepend_audio_prefix_latent,
            target_frame_count=audio_shape.frames,
        )
        self.last_audio_latent = encoded_audio_latent[:1].detach().to("cpu").contiguous()
        if fast_modules is None:
            del audio_encoder
            cleanup_memory()

        video_encoder = fast_modules.video_encoder if fast_modules is not None else self.model_ledger.video_encoder()
        transformer = (
            fast_modules.transformer
            if fast_modules is not None
            else self.model_ledger.transformer(extra_config=transformer_extra_config)
        )
        if transformer_extra_config is not None:
            _warn_if_channel_condition_is_inert(transformer)
        stage_1_sigmas = _resolve_sigmas(
            stage1_sigmas,
            DISTILLED_SIGMA_VALUES,
            device=self.device,
            name="stage1_sigmas",
        )

        def make_denoising_func():
            if guidance_scale == 1.0 and stg_scale == 0.0:
                return simple_denoising_func(
                    video_context=v_context_p,
                    audio_context=a_context_p,
                    transformer=transformer,
                )
            return guided_denoising_func(
                video_context=v_context_p,
                audio_context=a_context_p,
                negative_video_context=v_context_n,
                negative_audio_context=a_context_n,
                transformer=transformer,
                guidance_scale=guidance_scale,
                stg_scale=stg_scale,
                stg_blocks=stg_blocks,
                stg_mode=stg_mode,
            )

        def denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=make_denoising_func(),
            )

        output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2 if stage_mode == "two-stage" else width,
            height=height // 2 if stage_mode == "two-stage" else height,
            fps=frame_rate,
        )
        conditionings = self._build_video_conditionings(
            images=images,
            video_frame_conditionings=video_frame_conditionings,
            latent_conditionings=latent_conditionings,
            global_condition_images=global_condition_images,
            global_latent_conditionings=global_latent_conditionings,
            global_condition_position=global_condition_position,
            global_condition_repeat_to_target_frames=global_condition_repeat_to_target_frames,
            global_condition_position_mode=global_condition_position_mode,
            height=output_shape.height,
            width=output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
        )
        stage1_video_channel_condition = self._repeat_first_frame_channel_condition_latent(
            latent=first_frame_channel_condition_latent,
            output_shape=output_shape,
        )

        if inference_behavior == "ar":
            total_video_latent_frames = self._total_video_latent_frames(output_shape)

            video_state = self._denoise_video_only_ar(
                output_shape=output_shape,
                conditionings=conditionings,
                noiser=noiser,
                sigmas=stage_1_sigmas,
                stepper=stepper,
                denoise_fn_builder=make_denoising_func,
                total_video_latent_frames=total_video_latent_frames,
                ar_video_chunk_size=ar_video_chunk_size,
                ar_history_chunk_count=ar_history_chunk_count,
                ar_sink_first_chunk=ar_sink_first_chunk,
                ar_relative_positions=ar_relative_positions,
                ar_history_feature_cache=ar_history_feature_cache,
                ar_first_frame_prefix_condition=ar_first_frame_prefix_condition,
                ar_first_frame_condition_position=ar_first_frame_condition_position,
                ar_rope_max_temporal_index=ar_rope_max_temporal_index,
                ar_audio_alignment=ar_audio_alignment,
                dtype=dtype,
                initial_audio_latent=encoded_audio_latent,
                video_channel_condition=stage1_video_channel_condition,
                derive_channel_condition_from_first_chunk=ar_first_frame_channel_condition_from_first_chunk
                and first_frame_channel_condition_latent is None,
                channel_condition_current_chunk_only=ar_channel_condition_current_chunk_only,
                noise_scale=stage_1_sigmas[0],
            )
        else:
            video_state = denoise_video_only(
                output_shape=output_shape,
                conditionings=conditionings,
                noiser=noiser,
                sigmas=stage_1_sigmas,
                stepper=stepper,
                denoising_loop_fn=denoising_loop,
                components=self.pipeline_components,
                dtype=dtype,
                device=self.device,
                initial_audio_latent=encoded_audio_latent,
                video_channel_condition=stage1_video_channel_condition,
            )

        self.last_stage1_video_latent = video_state.latent[:1].detach().to("cpu").contiguous()

        if stage_mode == "two-stage":
            upscaled_video_latent = upsample_video(
                latent=video_state.latent[:1],
                video_encoder=video_encoder,
                upsampler=(
                    fast_modules.spatial_upsampler
                    if fast_modules is not None and fast_modules.spatial_upsampler is not None
                    else self.model_ledger.spatial_upsampler()
                ),
            )

            torch.cuda.synchronize()
            cleanup_memory()

            stage_2_sigmas = _resolve_sigmas(
                stage2_sigmas,
                STAGE_2_DISTILLED_SIGMA_VALUES,
                device=self.device,
                name="stage2_sigmas",
            )
            stage_2_output_shape = VideoPixelShape(
                batch=1,
                frames=num_frames,
                width=width,
                height=height,
                fps=frame_rate,
            )
            stage_2_conditionings = self._build_video_conditionings(
                images=images,
                video_frame_conditionings=stage2_video_frame_conditionings,
                latent_conditionings=stage2_latent_conditionings,
                global_condition_images=global_condition_images if use_global_condition_in_stage2 else [],
                global_latent_conditionings=(
                    stage2_global_latent_conditionings if use_global_condition_in_stage2 else []
                ),
                global_condition_position=global_condition_position,
                global_condition_repeat_to_target_frames=global_condition_repeat_to_target_frames,
                global_condition_position_mode=global_condition_position_mode,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
            )
            stage2_video_channel_condition = self._repeat_first_frame_channel_condition_latent(
                latent=stage2_first_frame_channel_condition_latent,
                output_shape=stage_2_output_shape,
            )
            if inference_behavior == "ar":
                stage_2_total_video_latent_frames = self._total_video_latent_frames(stage_2_output_shape)

                video_state = self._denoise_video_only_ar(
                    output_shape=stage_2_output_shape,
                    conditionings=stage_2_conditionings,
                    noiser=noiser,
                    sigmas=stage_2_sigmas,
                    stepper=stepper,
                    denoise_fn_builder=make_denoising_func,
                    total_video_latent_frames=stage_2_total_video_latent_frames,
                    ar_video_chunk_size=ar_video_chunk_size,
                    ar_history_chunk_count=ar_history_chunk_count,
                    ar_sink_first_chunk=ar_sink_first_chunk,
                    ar_relative_positions=ar_relative_positions,
                    ar_history_feature_cache=ar_history_feature_cache,
                    ar_first_frame_prefix_condition=ar_first_frame_prefix_condition,
                    ar_first_frame_condition_position=ar_first_frame_condition_position,
                    ar_rope_max_temporal_index=ar_rope_max_temporal_index,
                    ar_audio_alignment=ar_audio_alignment,
                    dtype=dtype,
                    initial_video_latent=upscaled_video_latent,
                    initial_audio_latent=encoded_audio_latent,
                    video_channel_condition=stage2_video_channel_condition,
                    derive_channel_condition_from_first_chunk=ar_first_frame_channel_condition_from_first_chunk
                    and stage2_first_frame_channel_condition_latent is None,
                    channel_condition_current_chunk_only=ar_channel_condition_current_chunk_only,
                    noise_scale=stage_2_sigmas[0],
                )
            else:
                video_state = denoise_video_only(
                    output_shape=stage_2_output_shape,
                    conditionings=stage_2_conditionings,
                    noiser=noiser,
                    sigmas=stage_2_sigmas,
                    stepper=stepper,
                    denoising_loop_fn=denoising_loop,
                    components=self.pipeline_components,
                    dtype=dtype,
                    device=self.device,
                    noise_scale=stage_2_sigmas[0],
                    initial_video_latent=upscaled_video_latent,
                    initial_audio_latent=encoded_audio_latent,
                    video_channel_condition=stage2_video_channel_condition,
                )

        self.last_final_video_latent = video_state.latent[:1].detach().to("cpu").contiguous()
        if self.last_stage1_video_latent is None:
            self.last_stage1_video_latent = self.last_final_video_latent.clone()

        if fast_modules is None:
            torch.cuda.synchronize()
            del transformer
            del video_encoder
            cleanup_memory()

        decode_prefix_latent = (
            stage2_decode_temporal_prefix_latent
            if stage_mode == "two-stage" and stage2_decode_temporal_prefix_latent is not None
            else decode_temporal_prefix_latent
        )
        decode_latent = video_state.latent
        if decode_prefix_latent is not None:
            decode_latent = self._prepend_decode_temporal_prefix_latent(
                latent=decode_latent,
                prefix_latent=decode_prefix_latent,
            )

        decoded_video = vae_decode_video(
            decode_latent,
            fast_modules.video_decoder if fast_modules is not None else self.model_ledger.video_decoder(),
            tiling_config,
            generator,
        )
        if decode_prefix_latent is not None:
            decoded_video = self._slice_decoded_video(
                decoded_video,
                start_frame=decode_temporal_prefix_drop_frames,
                max_frames=num_frames,
            )

        # Keep the source audio in the final file instead of VAE-decoding it.
        original_audio = Audio(waveform=decoded_audio.waveform.squeeze(0), sampling_rate=decoded_audio.sampling_rate)
        if fast_modules is None:
            cleanup_memory()
        return decoded_video, original_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    parser = default_2_stage_distilled_arg_parser(params=params)
    for action in parser._actions:
        if "--spatial-upsampler-path" in action.option_strings:
            action.required = False
            action.help += " Required only when --stage-mode=two-stage."
            break
    parser.add_argument(
        "--stage-mode",
        choices=("one-stage", "two-stage"),
        default="two-stage",
        help="Run only the one-stage distilled a2vid sampler or the default two-stage distilled a2vid pipeline.",
    )
    parser.add_argument(
        "--stage1-sigmas",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional stage-1 sigma schedule. Provide space-separated values and include the terminal 0.0 endpoint. "
            "Defaults to DISTILLED_SIGMA_VALUES."
        ),
    )
    parser.add_argument(
        "--stage2-sigmas",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional stage-2 sigma schedule. Provide space-separated values and include the terminal 0.0 endpoint. "
            "Defaults to STAGE_2_DISTILLED_SIGMA_VALUES. Ignored when --stage-mode=one-stage."
        ),
    )
    parser.add_argument(
        "--inference-behavior",
        choices=("default", "ar"),
        default="default",
        help="Use the default full-context sampler or chunk-wise autoregressive video generation.",
    )
    parser.add_argument(
        "--ar-video-chunk-size",
        type=int,
        default=None,
        help="Video latent chunk size for AR generation. Required when --inference-behavior=ar.",
    )
    parser.add_argument(
        "--ar-history-chunks",
        type=int,
        default=None,
        help="Number of previous AR chunks to keep as context. Default keeps the full history.",
    )
    parser.add_argument(
        "--ar-sink-first-chunk",
        action="store_true",
        help="Always keep the first AR chunk as an additional context anchor.",
    )
    parser.add_argument(
        "--ar-relative-positions",
        action="store_true",
        help="Rebuild AR window positions from local relative coordinates instead of original absolute positions.",
    )
    parser.add_argument(
        "--ar-history-feature-cache",
        action="store_true",
        help=(
            "Experimental: cache history token features from the first denoising step of each AR chunk and reuse them "
            "for later steps. Faster but approximate because full bidirectional history-current attention is skipped."
        ),
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to the audio file to condition the video generation.",
    )
    parser.add_argument(
        "--audio-start-time",
        type=float,
        default=0.0,
        help="Start time in seconds to read audio from (default: 0.0).",
    )
    parser.add_argument(
        "--audio-max-duration",
        type=float,
        default=None,
        help="Maximum audio duration in seconds. Defaults to video duration (num_frames / frame_rate).",
    )
    args = parser.parse_args()
    if args.inference_behavior == "ar" and (args.ar_video_chunk_size is None or args.ar_video_chunk_size <= 0):
        parser.error("--ar-video-chunk-size must be a positive integer when --inference-behavior=ar.")
    pipeline = A2VidDistilledPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
        stage_mode=args.stage_mode,
        stage1_sigmas=args.stage1_sigmas,
        stage2_sigmas=args.stage2_sigmas,
        audio_path=args.audio_path,
        audio_start_time=args.audio_start_time,
        audio_max_duration=args.audio_max_duration if args.audio_max_duration is not None else args.num_frames / args.frame_rate,
        inference_behavior=args.inference_behavior,
        ar_video_chunk_size=args.ar_video_chunk_size,
        ar_history_chunk_count=args.ar_history_chunks,
        ar_sink_first_chunk=args.ar_sink_first_chunk,
        ar_relative_positions=args.ar_relative_positions,
        ar_history_feature_cache=args.ar_history_feature_cache,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
