from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable, Literal

import torch
from tqdm import tqdm

from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.model.transformer.ar_feature_cache import ARFeatureCache
from ltx_core.types import LatentState
from ltx_pipelines.utils.helpers import post_process_latent
from ltx_pipelines.utils.types import DenoisingFunc


def build_progressive_chunk_ranges(total_steps: int, chunk_size: int) -> list[tuple[int, int]]:
    """Split a sequence into non-overlapping progressive chunks."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    chunk_ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_steps:
        end = min(start + chunk_size, total_steps)
        chunk_ranges.append((start, end))
        start = end
    return chunk_ranges


AudioAlignment = Literal["time", "proportional"]
AUDIO_ALIGNMENTS: tuple[str, ...] = ("time", "proportional")


def _pixel_frames_through(latent_frames: int, temporal_scale: int) -> int:
    """Pixel frames covered by the first ``latent_frames`` video latent frames.

    The video VAE is causal in time: the first latent frame stands for a single pixel frame and
    every later one for ``temporal_scale`` frames.
    """
    return 0 if latent_frames <= 0 else (latent_frames - 1) * temporal_scale + 1


def audio_steps_through(
    latent_frames: int,
    *,
    total_video_frames: int,
    total_audio_steps: int,
    alignment: AudioAlignment = "time",
    temporal_scale: int = 8,
) -> int:
    """Audio latent steps that belong to the first ``latent_frames`` video latent frames.

    ``"time"`` puts the boundary where those video frames end on the clock.

    ``"proportional"`` is the original rule: an equal share of the audio per latent frame. It
    ignores that the first latent frame is one pixel frame and the rest are ``temporal_scale``,
    so a boundary sits up to ``temporal_scale - 1`` pixel frames late at the start of a run and
    drifts to zero at the end. Because each chunk is denoised inside a window whose positions
    are rebuilt from zero, the video then lines up with audio that is progressively earlier
    than its own, by about 0.23s at the end at 25 fps, and which audio a chunk receives depends
    on the total length of the run.
    """
    if alignment not in AUDIO_ALIGNMENTS:
        raise ValueError(f"alignment must be one of {AUDIO_ALIGNMENTS}, got {alignment!r}.")
    if alignment == "proportional":
        return math.ceil(latent_frames * total_audio_steps / total_video_frames)
    total_pixel_frames = _pixel_frames_through(total_video_frames, temporal_scale)
    covered = _pixel_frames_through(latent_frames, temporal_scale)
    return math.floor(covered * total_audio_steps / total_pixel_frames + 0.5)


def build_aligned_audio_chunk_ranges(
    video_chunk_ranges: list[tuple[int, int]],
    *,
    total_video_frames: int,
    total_audio_steps: int,
    alignment: AudioAlignment = "time",
    temporal_scale: int = 8,
) -> list[tuple[int, int]]:
    """Build contiguous audio chunk ranges aligned with video chunk progress.

    See :func:`audio_steps_through` for the two alignment rules.
    """
    if total_video_frames <= 0:
        raise ValueError(f"total_video_frames must be positive, got {total_video_frames}")
    if alignment not in AUDIO_ALIGNMENTS:
        raise ValueError(f"alignment must be one of {AUDIO_ALIGNMENTS}, got {alignment!r}.")
    if total_audio_steps <= 0:
        return [(0, 0) for _ in video_chunk_ranges]

    audio_ranges: list[tuple[int, int]] = []
    audio_start = 0
    for chunk_idx, (_frame_start, frame_end) in enumerate(video_chunk_ranges):
        if chunk_idx == len(video_chunk_ranges) - 1:
            audio_end = total_audio_steps
        else:
            audio_end = audio_steps_through(
                frame_end,
                total_video_frames=total_video_frames,
                total_audio_steps=total_audio_steps,
                alignment=alignment,
                temporal_scale=temporal_scale,
            )
            if audio_end <= audio_start:
                audio_end = min(total_audio_steps, audio_start + 1)
        audio_ranges.append((audio_start, min(total_audio_steps, audio_end)))
        audio_start = min(total_audio_steps, audio_end)

    return audio_ranges


def select_inference_chunk_indices(
    *,
    current_chunk_idx: int,
    num_chunks: int,
    history_chunk_count: int | None,
    sink_first_chunk: bool,
) -> list[int]:
    """Select which chunks are visible when denoising the current AR chunk."""
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if current_chunk_idx < 0 or current_chunk_idx >= num_chunks:
        raise ValueError(f"current_chunk_idx must be in [0, {num_chunks}), got {current_chunk_idx}")
    if history_chunk_count is not None and history_chunk_count < 0:
        raise ValueError(f"history_chunk_count must be non-negative, got {history_chunk_count}")

    if history_chunk_count is None:
        start_idx = 0
    else:
        start_idx = max(0, current_chunk_idx - history_chunk_count)

    selected = list(range(start_idx, current_chunk_idx + 1))
    if sink_first_chunk and current_chunk_idx > 0 and 0 not in selected:
        selected = [0, *selected]
    return selected


def _slice_attention_mask(
    attention_mask: torch.Tensor | None,
    token_indices: torch.Tensor,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    return attention_mask.index_select(1, token_indices).index_select(2, token_indices)


def _prepend_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    num_prefix_tokens: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if attention_mask is None or num_prefix_tokens <= 0:
        return attention_mask
    total_tokens = num_prefix_tokens + attention_mask.shape[1]
    result = torch.ones((batch_size, total_tokens, total_tokens), device=device, dtype=dtype)
    result[:, num_prefix_tokens:, num_prefix_tokens:] = attention_mask
    return result


def _append_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    num_suffix_tokens: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if attention_mask is None or num_suffix_tokens <= 0:
        return attention_mask
    existing_tokens = attention_mask.shape[1]
    total_tokens = existing_tokens + num_suffix_tokens
    result = torch.ones((batch_size, total_tokens, total_tokens), device=device, dtype=dtype)
    result[:, :existing_tokens, :existing_tokens] = attention_mask
    return result


def _select_ar_chunk_state(
    state: LatentState,
    *,
    chunk_token_ranges: list[tuple[int, int]],
    selected_chunk_indices: list[int],
    current_chunk_idx: int,
    prefix_token_count: int = 0,
    relative_positions: bool = False,
    channel_condition_current_chunk_only: bool = False,
    video_tokens_per_frame: int | None = None,
    current_temporal_index: int | None = None,
    rope_max_temporal_index: int | None = None,
    dynamic_prefix_latent: torch.Tensor | None = None,
    dynamic_prefix_positions: torch.Tensor | None = None,
    dynamic_condition_position: Literal["prepend", "append"] = "prepend",
) -> tuple[LatentState, slice]:
    if dynamic_condition_position not in ("prepend", "append"):
        raise ValueError(
            f"dynamic_condition_position must be 'prepend' or 'append', got {dynamic_condition_position!r}."
        )
    latent_parts: list[torch.Tensor] = []
    denoise_mask_parts: list[torch.Tensor] = []
    token_indices_parts: list[torch.Tensor] = []
    target_token_indices_parts: list[torch.Tensor] = []
    channel_condition_parts: list[torch.Tensor] = []
    position_parts: list[torch.Tensor] = []

    current_slice: slice | None = None
    current_offset = 0
    dynamic_condition_token_count = 0
    dynamic_prefix_token_count = 0
    dynamic_suffix_token_count = 0
    use_relative_positions = relative_positions and rope_max_temporal_index is None

    if dynamic_prefix_latent is not None:
        if dynamic_prefix_positions is None:
            raise ValueError("dynamic_prefix_positions is required when dynamic_prefix_latent is provided.")
        if dynamic_prefix_latent.shape[1] != dynamic_prefix_positions.shape[2]:
            raise ValueError(
                "Dynamic AR prefix latent/position token counts must match. "
                f"latent={tuple(dynamic_prefix_latent.shape)}, positions={tuple(dynamic_prefix_positions.shape)}."
            )
        dynamic_condition_token_count = dynamic_prefix_latent.shape[1]
        if dynamic_condition_position == "prepend":
            dynamic_prefix_token_count = dynamic_condition_token_count
            latent_parts.append(dynamic_prefix_latent)
            denoise_mask_parts.append(
                state.denoise_mask.new_zeros(
                    state.denoise_mask.shape[0],
                    dynamic_prefix_token_count,
                    *state.denoise_mask.shape[2:],
                )
            )
            position_parts.append(dynamic_prefix_positions)
            if state.channel_condition is not None:
                channel_condition_parts.append(state.latent.new_zeros(*dynamic_prefix_latent.shape))
            current_offset = dynamic_prefix_token_count
        else:
            dynamic_suffix_token_count = dynamic_condition_token_count

    if prefix_token_count:
        latent_parts.append(state.latent[:, :prefix_token_count])
        denoise_mask_parts.append(state.denoise_mask[:, :prefix_token_count])
        position_parts.append(state.positions[:, :, :prefix_token_count])
        if state.channel_condition is not None:
            prefix_channel_condition = state.channel_condition[:, :prefix_token_count]
            if channel_condition_current_chunk_only:
                prefix_channel_condition = torch.zeros_like(prefix_channel_condition)
            channel_condition_parts.append(prefix_channel_condition)
        token_indices_parts.append(
            torch.arange(0, prefix_token_count, device=state.latent.device, dtype=torch.long)
        )
        current_offset += prefix_token_count

    for chunk_idx in selected_chunk_indices:
        token_start, token_end = chunk_token_ranges[chunk_idx]
        chunk_width = token_end - token_start
        latent_parts.append(state.latent[:, token_start:token_end])
        if state.channel_condition is not None:
            chunk_channel_condition = state.channel_condition[:, token_start:token_end]
            if channel_condition_current_chunk_only and chunk_idx != current_chunk_idx:
                chunk_channel_condition = torch.zeros_like(chunk_channel_condition)
            channel_condition_parts.append(chunk_channel_condition)
        token_indices_parts.append(
            torch.arange(token_start, token_end, device=state.latent.device, dtype=torch.long)
        )
        target_token_indices_parts.append(
            torch.arange(token_start, token_end, device=state.latent.device, dtype=torch.long)
        )
        if chunk_idx == current_chunk_idx:
            denoise_mask_parts.append(state.denoise_mask[:, token_start:token_end])
            current_slice = slice(current_offset, current_offset + chunk_width)
        else:
            denoise_mask_parts.append(torch.zeros_like(state.denoise_mask[:, token_start:token_end]))
        current_offset += chunk_width
        if not use_relative_positions:
            position_parts.append(state.positions[:, :, token_start:token_end])

    if current_slice is None:
        raise ValueError(
            f"Current chunk {current_chunk_idx} is missing from selected_chunk_indices={selected_chunk_indices}"
        )

    latent = torch.cat(latent_parts, dim=1)
    channel_condition = torch.cat(channel_condition_parts, dim=1) if channel_condition_parts else None
    if use_relative_positions:
        relative_token_count = latent.shape[1] - dynamic_prefix_token_count
        positions = state.positions[:, :, :relative_token_count]
        if dynamic_prefix_token_count:
            positions = torch.cat([dynamic_prefix_positions, positions], dim=2)
    else:
        positions = torch.cat(position_parts, dim=2)

    if dynamic_suffix_token_count:
        latent_parts.append(dynamic_prefix_latent)
        denoise_mask_parts.append(
            state.denoise_mask.new_zeros(
                state.denoise_mask.shape[0],
                dynamic_suffix_token_count,
                *state.denoise_mask.shape[2:],
            )
        )
        positions = torch.cat([positions, dynamic_prefix_positions], dim=2)
        if state.channel_condition is not None:
            channel_condition_parts.append(state.latent.new_zeros(*dynamic_prefix_latent.shape))
        latent = torch.cat(latent_parts, dim=1)
        channel_condition = torch.cat(channel_condition_parts, dim=1) if channel_condition_parts else None

    target_token_start = prefix_token_count + dynamic_prefix_token_count
    positions = _apply_rope_max_temporal_index(
        positions=positions,
        reference_positions=state.positions,
        target_token_start=target_token_start,
        reference_prefix_token_count=prefix_token_count,
        video_tokens_per_frame=video_tokens_per_frame,
        current_temporal_index=current_temporal_index,
        target_token_indices=torch.cat(target_token_indices_parts, dim=0) if target_token_indices_parts else None,
        rope_max_temporal_index=rope_max_temporal_index,
    )
    attention_mask = _slice_attention_mask(state.attention_mask, torch.cat(token_indices_parts, dim=0))
    attention_mask = _prepend_attention_mask(
        attention_mask,
        num_prefix_tokens=dynamic_prefix_token_count,
        batch_size=state.latent.shape[0],
        device=state.latent.device,
        dtype=state.denoise_mask.dtype,
    )
    attention_mask = _append_attention_mask(
        attention_mask,
        num_suffix_tokens=dynamic_suffix_token_count,
        batch_size=state.latent.shape[0],
        device=state.latent.device,
        dtype=state.denoise_mask.dtype,
    )
    return (
        replace(
            state,
            latent=latent,
            denoise_mask=torch.cat(denoise_mask_parts, dim=1),
            positions=positions,
            clean_latent=latent,
            attention_mask=attention_mask,
            conditioning_prefix_token_count=prefix_token_count + dynamic_prefix_token_count,
            channel_condition=channel_condition,
        ),
        current_slice,
    )


def _apply_rope_max_temporal_index(
    *,
    positions: torch.Tensor,
    reference_positions: torch.Tensor,
    target_token_start: int,
    reference_prefix_token_count: int,
    video_tokens_per_frame: int | None,
    current_temporal_index: int | None,
    target_token_indices: torch.Tensor | None,
    rope_max_temporal_index: int | None,
) -> torch.Tensor:
    if rope_max_temporal_index is None:
        return positions
    if rope_max_temporal_index < 0:
        raise ValueError(f"rope_max_temporal_index must be non-negative, got {rope_max_temporal_index}.")
    if video_tokens_per_frame is None or video_tokens_per_frame <= 0:
        raise ValueError("video_tokens_per_frame must be positive when rope_max_temporal_index is set.")
    if current_temporal_index is None:
        raise ValueError("current_temporal_index is required when rope_max_temporal_index is set.")
    if current_temporal_index <= rope_max_temporal_index:
        return positions

    if target_token_indices is None:
        raise ValueError("target_token_indices is required when rope_max_temporal_index is set.")
    target_token_count = target_token_indices.numel()
    if target_token_count <= 0:
        return positions
    if target_token_count % video_tokens_per_frame != 0:
        raise ValueError(
            "AR RoPE temporal rebasing expects a whole number of video latent frames. "
            f"target_tokens={target_token_count}, tokens_per_frame={video_tokens_per_frame}."
        )
    if target_token_indices.numel() != target_token_count:
        raise ValueError(
            "AR RoPE temporal rebasing target-token index count must match selected target tokens. "
            f"indices={target_token_indices.numel()}, target_tokens={target_token_count}."
        )
    if target_token_start + target_token_count > positions.shape[2]:
        raise ValueError(
            "AR RoPE temporal rebasing target span exceeds the selected position tensor. "
            f"start={target_token_start}, target_tokens={target_token_count}, positions={positions.shape[2]}."
        )

    overflow = current_temporal_index - rope_max_temporal_index
    target_token_indices = target_token_indices.to(device=reference_positions.device, dtype=torch.long)
    target_token_offsets = target_token_indices - reference_prefix_token_count
    target_temporal_indices = torch.div(target_token_offsets, video_tokens_per_frame, rounding_mode="floor")
    target_frame_token_offsets = target_token_offsets.remainder(video_tokens_per_frame)
    remapped_temporal_indices = (target_temporal_indices - overflow).clamp(min=0)
    remapped_token_indices = (
        reference_prefix_token_count + remapped_temporal_indices * video_tokens_per_frame + target_frame_token_offsets
    )
    if remapped_token_indices.max() >= reference_positions.shape[2]:
        return positions

    rebased = positions.clone()
    rebased[:, :, target_token_start : target_token_start + target_token_count] = reference_positions.index_select(
        2, remapped_token_indices
    ).to(
        device=positions.device,
        dtype=positions.dtype,
    )
    return rebased


def _repeated_first_latent_frame_channel_condition(
    *,
    state: LatentState,
    prefix_token_count: int,
    total_video_latent_frames: int,
    video_tokens_per_frame: int,
) -> torch.Tensor:
    target_token_count = total_video_latent_frames * video_tokens_per_frame
    first_frame_start = prefix_token_count
    first_frame_end = first_frame_start + video_tokens_per_frame
    first_frame_tokens = state.latent[:, first_frame_start:first_frame_end].detach().contiguous()
    target_condition = first_frame_tokens.repeat(1, total_video_latent_frames, 1)
    channel_condition = state.latent.new_zeros(state.latent.shape)
    channel_condition[:, prefix_token_count : prefix_token_count + target_token_count] = target_condition
    return channel_condition


def _first_latent_frame_prefix(
    *,
    state: LatentState,
    prefix_token_count: int,
    token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = prefix_token_count
    end = start + token_count
    latent = state.latent[:, start:end].detach().contiguous()
    positions = state.positions[:, :, start:end].detach().clone()
    return latent, positions


def _first_frame_audio_token_count(*, total_video_latent_frames: int, total_audio_steps: int) -> int:
    if total_audio_steps <= 0:
        return 0
    if total_video_latent_frames <= 1:
        return total_audio_steps
    return max(1, min(total_audio_steps, math.ceil(total_audio_steps / total_video_latent_frames)))


def autoregressive_euler_denoising_loop(
    *,
    sigmas: torch.Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn_builder: Callable[[], DenoisingFunc],
    total_video_latent_frames: int,
    video_chunk_size: int,
    history_chunk_count: int | None = None,
    sink_first_chunk: bool = False,
    relative_positions: bool = False,
    history_feature_cache: bool = False,
    derive_channel_condition_from_first_chunk: bool = False,
    channel_condition_current_chunk_only: bool = False,
    first_frame_prefix_condition: bool = False,
    first_frame_condition_position: Literal["prepend", "append"] = "prepend",
    rope_max_temporal_index: int | None = None,
    denoise_fn_builder_for_chunk: Callable[[int, int, int], DenoisingFunc] | None = None,
    max_generate_video_latent_frames: int | None = None,
    start_chunk_idx: int = 0,
    end_chunk_idx: int | None = None,
    audio_alignment: AudioAlignment = "time",
    video_temporal_scale: int = 8,
) -> tuple[LatentState, LatentState]:
    """Run chunk-wise prefix-growing denoising for joint video-audio generation."""
    if audio_alignment not in AUDIO_ALIGNMENTS:
        raise ValueError(f"audio_alignment must be one of {AUDIO_ALIGNMENTS}, got {audio_alignment!r}.")
    if first_frame_condition_position not in ("prepend", "append"):
        raise ValueError(
            "first_frame_condition_position must be 'prepend' or 'append', "
            f"got {first_frame_condition_position!r}."
        )
    if total_video_latent_frames <= 0:
        raise ValueError(f"total_video_latent_frames must be positive, got {total_video_latent_frames}")
    if video_chunk_size <= 0:
        raise ValueError(f"video_chunk_size must be positive, got {video_chunk_size}")
    video_prefix_token_count = video_state.conditioning_prefix_token_count
    video_target_token_count = video_state.latent.shape[1] - video_prefix_token_count
    if video_target_token_count <= 0:
        raise ValueError(
            "Video AR state has no target tokens after prefix conditioning. "
            f"prefix={video_prefix_token_count}, total={video_state.latent.shape[1]}"
        )
    if video_target_token_count % total_video_latent_frames != 0:
        raise ValueError(
            "Video token count is not divisible by the latent-frame count. "
            "Chunk-wise AR supports standard text/image-to-video targets plus optional prefix conditioning tokens."
        )

    generate_video_latent_frames = (
        total_video_latent_frames if max_generate_video_latent_frames is None else max_generate_video_latent_frames
    )
    if generate_video_latent_frames <= 0 or generate_video_latent_frames > total_video_latent_frames:
        raise ValueError(
            "max_generate_video_latent_frames must be in [1, total_video_latent_frames], "
            f"got {generate_video_latent_frames} for total={total_video_latent_frames}."
        )

    video_tokens_per_frame = video_target_token_count // total_video_latent_frames
    video_chunk_ranges = build_progressive_chunk_ranges(generate_video_latent_frames, video_chunk_size)
    audio_steps_for_generation = audio_state.latent.shape[1]
    if generate_video_latent_frames < total_video_latent_frames:
        audio_steps_for_generation = max(
            1,
            audio_steps_through(
                generate_video_latent_frames,
                total_video_frames=total_video_latent_frames,
                total_audio_steps=audio_state.latent.shape[1],
                alignment=audio_alignment,
                temporal_scale=video_temporal_scale,
            ),
        )
    audio_chunk_ranges = build_aligned_audio_chunk_ranges(
        video_chunk_ranges,
        total_video_frames=generate_video_latent_frames,
        total_audio_steps=audio_steps_for_generation,
        alignment=audio_alignment,
        temporal_scale=video_temporal_scale,
    )
    num_chunks = len(video_chunk_ranges)
    if start_chunk_idx < 0 or start_chunk_idx >= num_chunks:
        raise ValueError(f"start_chunk_idx must be in [0, {num_chunks}), got {start_chunk_idx}.")
    if end_chunk_idx is None:
        end_chunk_idx = num_chunks
    if end_chunk_idx <= start_chunk_idx or end_chunk_idx > num_chunks:
        raise ValueError(
            f"end_chunk_idx must be in ({start_chunk_idx}, {num_chunks}], got {end_chunk_idx}."
        )

    total_steps = (end_chunk_idx - start_chunk_idx) * max(0, len(sigmas) - 1)
    progress = tqdm(total=total_steps, desc="Generating", unit="step")
    try:
        first_frame_prefix_latent: torch.Tensor | None = None
        first_frame_prefix_positions: torch.Tensor | None = None
        first_frame_audio_prefix_latent: torch.Tensor | None = None
        first_frame_audio_prefix_positions: torch.Tensor | None = None
        first_frame_audio_prefix_token_count = _first_frame_audio_token_count(
            total_video_latent_frames=total_video_latent_frames,
            total_audio_steps=audio_state.latent.shape[1],
        )
        video_token_ranges = [
            (
                video_prefix_token_count + frame_start * video_tokens_per_frame,
                video_prefix_token_count + frame_end * video_tokens_per_frame,
            )
            for frame_start, frame_end in video_chunk_ranges
        ]
        if derive_channel_condition_from_first_chunk and start_chunk_idx > 0 and video_state.channel_condition is None:
            video_state = replace(
                video_state,
                channel_condition=_repeated_first_latent_frame_channel_condition(
                    state=video_state,
                    prefix_token_count=video_prefix_token_count,
                    total_video_latent_frames=total_video_latent_frames,
                    video_tokens_per_frame=video_tokens_per_frame,
                ),
            )
        if first_frame_prefix_condition and start_chunk_idx > 0:
            first_frame_prefix_latent, first_frame_prefix_positions = _first_latent_frame_prefix(
                state=video_state,
                prefix_token_count=video_prefix_token_count,
                token_count=video_tokens_per_frame,
            )
            first_frame_audio_prefix_latent, first_frame_audio_prefix_positions = _first_latent_frame_prefix(
                state=audio_state,
                prefix_token_count=0,
                token_count=first_frame_audio_prefix_token_count,
            )

        for chunk_idx, ((frame_start, frame_end), (audio_start, audio_end)) in enumerate(
            zip(video_chunk_ranges, audio_chunk_ranges, strict=True)
        ):
            if chunk_idx < start_chunk_idx:
                continue
            if chunk_idx >= end_chunk_idx:
                break
            if audio_end <= audio_start:
                raise ValueError(
                    "AR audio chunk mapping produced an empty chunk. "
                    "Reduce ar_video_chunk_size or disable audio generation for this configuration."
                )

            video_start = video_prefix_token_count + frame_start * video_tokens_per_frame
            video_end = video_prefix_token_count + frame_end * video_tokens_per_frame
            selected_chunk_indices = select_inference_chunk_indices(
                current_chunk_idx=chunk_idx,
                num_chunks=num_chunks,
                history_chunk_count=history_chunk_count,
                sink_first_chunk=sink_first_chunk,
            )
            use_first_frame_prefix = (
                first_frame_prefix_condition
                and first_frame_prefix_latent is not None
                and first_frame_audio_prefix_latent is not None
                and chunk_idx > 0
                and 0 not in selected_chunk_indices
            )
            chunk_denoise_fn = (
                denoise_fn_builder_for_chunk(chunk_idx, frame_start, frame_end)
                if denoise_fn_builder_for_chunk is not None
                else denoise_fn_builder()
            )
            chunk_feature_cache = ARFeatureCache() if history_feature_cache else None

            for step_idx, _sigma in enumerate(sigmas[:-1]):
                current_video_state, current_video_slice = _select_ar_chunk_state(
                    video_state,
                    chunk_token_ranges=video_token_ranges,
                    selected_chunk_indices=selected_chunk_indices,
                    current_chunk_idx=chunk_idx,
                    prefix_token_count=video_prefix_token_count,
                    relative_positions=relative_positions,
                    channel_condition_current_chunk_only=channel_condition_current_chunk_only,
                    video_tokens_per_frame=video_tokens_per_frame,
                    current_temporal_index=frame_end - 1,
                    rope_max_temporal_index=rope_max_temporal_index,
                    dynamic_prefix_latent=first_frame_prefix_latent if use_first_frame_prefix else None,
                    dynamic_prefix_positions=first_frame_prefix_positions if use_first_frame_prefix else None,
                    dynamic_condition_position=first_frame_condition_position,
                )
                current_audio_state, current_audio_slice = _select_ar_chunk_state(
                    audio_state,
                    chunk_token_ranges=audio_chunk_ranges,
                    selected_chunk_indices=selected_chunk_indices,
                    current_chunk_idx=chunk_idx,
                    relative_positions=relative_positions,
                    dynamic_prefix_latent=first_frame_audio_prefix_latent if use_first_frame_prefix else None,
                    dynamic_prefix_positions=first_frame_audio_prefix_positions if use_first_frame_prefix else None,
                    dynamic_condition_position=first_frame_condition_position,
                )

                if chunk_feature_cache is None:
                    denoised_video, denoised_audio = chunk_denoise_fn(
                        current_video_state,
                        current_audio_state,
                        sigmas,
                        step_idx,
                    )
                else:
                    denoised_video, denoised_audio = chunk_denoise_fn(
                        current_video_state,
                        current_audio_state,
                        sigmas,
                        step_idx,
                        ar_feature_cache=chunk_feature_cache,
                        video_current_slice=current_video_slice,
                        audio_current_slice=current_audio_slice,
                    )
                denoised_video = post_process_latent(
                    denoised_video,
                    current_video_state.denoise_mask,
                    current_video_state.clean_latent,
                )
                denoised_audio = post_process_latent(
                    denoised_audio,
                    current_audio_state.denoise_mask,
                    current_audio_state.clean_latent,
                )

                video_step = stepper.step(current_video_state.latent, denoised_video, sigmas, step_idx)
                video_step = post_process_latent(
                    video_step,
                    current_video_state.denoise_mask,
                    current_video_state.clean_latent,
                )
                new_video_latent = video_state.latent.clone()
                new_video_latent[:, video_start:video_end] = video_step[:, current_video_slice]
                video_state = replace(video_state, latent=new_video_latent)

                audio_step = stepper.step(current_audio_state.latent, denoised_audio, sigmas, step_idx)
                audio_step = post_process_latent(
                    audio_step,
                    current_audio_state.denoise_mask,
                    current_audio_state.clean_latent,
                )
                new_audio_latent = audio_state.latent.clone()
                new_audio_latent[:, audio_start:audio_end] = audio_step[:, current_audio_slice]
                audio_state = replace(audio_state, latent=new_audio_latent)

                progress.update(1)

            if derive_channel_condition_from_first_chunk and chunk_idx == 0 and video_state.channel_condition is None:
                video_state = replace(
                    video_state,
                    channel_condition=_repeated_first_latent_frame_channel_condition(
                        state=video_state,
                        prefix_token_count=video_prefix_token_count,
                        total_video_latent_frames=total_video_latent_frames,
                        video_tokens_per_frame=video_tokens_per_frame,
                    ),
                )
            if first_frame_prefix_condition and chunk_idx == 0:
                first_frame_prefix_latent, first_frame_prefix_positions = _first_latent_frame_prefix(
                    state=video_state,
                    prefix_token_count=video_prefix_token_count,
                    token_count=video_tokens_per_frame,
                )
                first_frame_audio_prefix_latent, first_frame_audio_prefix_positions = _first_latent_frame_prefix(
                    state=audio_state,
                    prefix_token_count=0,
                    token_count=first_frame_audio_prefix_token_count,
                )
    finally:
        progress.close()

    return video_state, audio_state
