from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Literal

import torch

from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.a2vid_distilled import (
    A2VidDistilledPipeline,
    DistilledStageMode,
    LatentConditioningInput,
)
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
from ltx_pipelines.utils.media_io import encode_video

SigmaSchedule = list[float] | tuple[float, ...] | torch.Tensor | None

SIGMA_SCHEDULE_ALIASES = {
    "distilled": DISTILLED_SIGMA_VALUES,
    "distilled-stage1": DISTILLED_SIGMA_VALUES,
    "stage1-distilled": DISTILLED_SIGMA_VALUES,
    "stage2-distilled": STAGE_2_DISTILLED_SIGMA_VALUES,
    "distilled-stage2": STAGE_2_DISTILLED_SIGMA_VALUES,
}


class ARA2VidDistilledPipeline(A2VidDistilledPipeline):
    """Chunk-wise autoregressive distilled audio-to-video pipeline with one-stage and two-stage support."""

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
        ar_video_chunk_size: int,
        ar_history_chunk_count: int | None = None,
        ar_sink_first_chunk: bool = False,
        ar_relative_positions: bool = False,
        ar_history_feature_cache: bool = False,
        ar_first_frame_prefix_condition: bool = False,
        ar_first_frame_condition_position: Literal["prepend", "append"] = "prepend",
        ar_rope_max_temporal_index: int | None = None,
        ar_audio_alignment: Literal["time", "proportional"] = "time",
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        stage_mode: DistilledStageMode = "two-stage",
        stage1_sigmas: SigmaSchedule = None,
        stage2_sigmas: SigmaSchedule = None,
        latent_conditionings: list[LatentConditioningInput] | None = None,
        stage2_latent_conditionings: list[LatentConditioningInput] | None = None,
        global_condition_images: list[ImageConditioningInput] | None = None,
        global_condition_position: Literal["prepend", "append"] = "prepend",
        use_global_condition_in_stage2: bool = False,
        fast_infer: bool = False,
        first_frame_channel_condition_latent: torch.Tensor | None = None,
        stage2_first_frame_channel_condition_latent: torch.Tensor | None = None,
        first_frame_channel_condition_init: Literal["zero", "xavier", "kaiming"] = "zero",
        first_frame_channel_condition_mode: Literal["add", "gated"] = "add",
        ar_first_frame_channel_condition_from_first_chunk: bool = False,
        ar_channel_condition_current_chunk_only: bool = False,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        return super().__call__(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
            stage_mode=stage_mode,
            stage1_sigmas=stage1_sigmas,
            stage2_sigmas=stage2_sigmas,
            latent_conditionings=latent_conditionings,
            stage2_latent_conditionings=stage2_latent_conditionings,
            global_condition_images=global_condition_images,
            global_condition_position=global_condition_position,
            use_global_condition_in_stage2=use_global_condition_in_stage2,
            fast_infer=fast_infer,
            first_frame_channel_condition_latent=first_frame_channel_condition_latent,
            stage2_first_frame_channel_condition_latent=stage2_first_frame_channel_condition_latent,
            first_frame_channel_condition_init=first_frame_channel_condition_init,
            first_frame_channel_condition_mode=first_frame_channel_condition_mode,
            ar_first_frame_channel_condition_from_first_chunk=ar_first_frame_channel_condition_from_first_chunk,
            ar_channel_condition_current_chunk_only=ar_channel_condition_current_chunk_only,
            inference_behavior="ar",
            ar_video_chunk_size=ar_video_chunk_size,
            ar_history_chunk_count=ar_history_chunk_count,
            ar_sink_first_chunk=ar_sink_first_chunk,
            ar_relative_positions=ar_relative_positions,
            ar_history_feature_cache=ar_history_feature_cache,
            ar_first_frame_prefix_condition=ar_first_frame_prefix_condition,
            ar_first_frame_condition_position=ar_first_frame_condition_position,
            ar_rope_max_temporal_index=ar_rope_max_temporal_index,
            ar_audio_alignment=ar_audio_alignment,
        )


def _parse_sigmas_arg(values: list[str] | None) -> list[float] | None:
    if values is None:
        return None

    if len(values) == 1:
        alias = values[0].lower().replace("_", "-")
        if alias in SIGMA_SCHEDULE_ALIASES:
            return list(SIGMA_SCHEDULE_ALIASES[alias])

    tokens: list[str] = []
    for value in values:
        tokens.extend(part for part in value.split(",") if part)
    try:
        return [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(
            "Could not parse sigma schedule. Pass numeric values, comma-separated numeric values, "
            f"or one of: {', '.join(sorted(SIGMA_SCHEDULE_ALIASES))}."
        ) from exc


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
        help="Run the distilled AR sampler directly at target resolution or in the default two-stage flow.",
    )
    parser.add_argument(
        "--stage1-sigmas",
        nargs="+",
        default=None,
        help=(
            "Optional stage-1 sigma schedule. Pass values directly, comma-separated values, or an alias. "
            f"Aliases: {', '.join(sorted(SIGMA_SCHEDULE_ALIASES))}."
        ),
    )
    parser.add_argument(
        "--stage2-sigmas",
        nargs="+",
        default=None,
        help=(
            "Optional stage-2 sigma schedule. Pass values directly, comma-separated values, or an alias. "
            f"Aliases: {', '.join(sorted(SIGMA_SCHEDULE_ALIASES))}."
        ),
    )
    parser.add_argument(
        "--ar-video-chunk-size",
        type=int,
        required=True,
        help="Video latent chunk size for chunk-wise autoregressive generation.",
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

    if args.stage_mode == "two-stage" and args.spatial_upsampler_path is None:
        parser.error("--spatial-upsampler-path is required when --stage-mode=two-stage")

    try:
        stage1_sigmas = _parse_sigmas_arg(args.stage1_sigmas)
        stage2_sigmas = _parse_sigmas_arg(args.stage2_sigmas)
    except ValueError as exc:
        parser.error(str(exc))

    pipeline = ARA2VidDistilledPipeline(
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
        audio_path=args.audio_path,
        audio_start_time=args.audio_start_time,
        audio_max_duration=args.audio_max_duration if args.audio_max_duration is not None else args.num_frames / args.frame_rate,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
        stage_mode=args.stage_mode,
        stage1_sigmas=stage1_sigmas,
        stage2_sigmas=stage2_sigmas,
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


__all__ = ["ARA2VidDistilledPipeline"]
