from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from ltx_core.model.video_vae import get_video_chunks_number
from ltx_pipelines import ARA2VidDistilledPipeline
from ltx_pipelines.utils.media_io import encode_video
from util import (
    build_first_frame_images,
    build_tiling_config,
    encode_first_frame_channel_condition,
    optional_int,
    resolve_num_frames,
    resolve_output_path,
    valid_num_frames,
)

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

# Previous scene-specific default prompt: A friendly young professional sits in a softly lit home office in a
# medium close-up shot, facing the camera with natural eye contact. Realistic conversational video-call framing,
# subtle head motion, natural blinking, soft facial expressions, clear lip sync, gentle breathing, and small
# relaxed hand gestures while speaking.


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AvatarForever one-stage autoregressive A2V inference.")

    model = parser.add_argument_group("model")
    model.add_argument("--distilled-checkpoint-path", type=Path, required=True)
    model.add_argument("--gemma-root", type=Path, required=True)

    generation = parser.add_argument_group("generation")
    generation.add_argument("--audio-path", type=Path, required=True)
    generation.add_argument("--prompt", default=DEFAULT_PROMPT)
    generation.add_argument("--seed", type=int, default=42)
    generation.add_argument("--height", type=int, default=512)
    generation.add_argument("--width", type=int, default=768)
    generation.add_argument(
        "--num-frames",
        type=valid_num_frames,
        default=None,
        help="Output frame count. Must follow 8n+1. Default: as many frames as the audio covers.",
    )
    generation.add_argument("--frame-rate", type=float, default=25.0)
    generation.add_argument(
        "--stage1-sigmas",
        type=float,
        nargs="+",
        default=(1.0, 0.98125, 0.909375, 0.421875, 0.0),
    )
    generation.add_argument("--fast-infer", action=argparse.BooleanOptionalAction, default=False)

    autoregressive = parser.add_argument_group("autoregressive")
    autoregressive.add_argument("--ar-video-chunk-size", type=int, default=4)
    autoregressive.add_argument("--ar-history-chunk-count", type=optional_int, default=1)
    autoregressive.add_argument("--ar-sink-first-chunk", action=argparse.BooleanOptionalAction, default=True)
    autoregressive.add_argument("--ar-relative-positions", action=argparse.BooleanOptionalAction, default=True)
    autoregressive.add_argument("--ar-history-feature-cache", action=argparse.BooleanOptionalAction, default=False)
    autoregressive.add_argument(
        "--ar-audio-alignment",
        choices=("time", "proportional"),
        default="time",
        help=(
            "How the audio is split across AR chunks. 'time' cuts where each chunk's video frames end on "
            "the clock. 'proportional' is the original equal-share rule, which lets lip sync slip by about "
            "0.23s over a run; kept for comparison."
        ),
    )
    autoregressive.add_argument(
        "--first-frame-prefix-condition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse chunk-0 video latent and aligned audio tokens as a dynamic AR condition.",
    )
    autoregressive.add_argument(
        "--first-frame-condition-position",
        choices=("prepend", "append"),
        default="prepend",
    )

    first_frame = parser.add_argument_group("first-frame conditioning")
    first_frame.add_argument("--first-frame-condition-image-path", type=Path, default=None)
    first_frame.add_argument("--first-frame-image-strength", type=float, default=1.0)
    first_frame.add_argument("--first-frame-image-crf", type=int, default=0)
    first_frame.add_argument(
        "--first-frame-channel-condition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the supplied first frame, or derive the channel condition from generated chunk 0.",
    )
    first_frame.add_argument(
        "--first-frame-channel-condition-mode",
        choices=("add", "gated"),
        default="gated",
    )
    first_frame.add_argument(
        "--first-frame-channel-condition-init",
        choices=("zero", "xavier", "kaiming"),
        default="zero",
    )

    tiling = parser.add_argument_group("VAE tiling")
    tiling.add_argument("--spatial-tile-size", type=int, default=512)
    tiling.add_argument("--spatial-tile-overlap", type=int, default=64)
    tiling.add_argument("--temporal-tile-size", type=int, default=256)
    tiling.add_argument("--temporal-tile-overlap", type=int, default=8)

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "videos")
    output.add_argument("--output-path", type=Path, default=None)
    output.add_argument("--video-crf", type=int, default=12, help="x264 CRF (0-51; lower is higher quality).")
    output.add_argument(
        "--video-preset",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"),
        default="fast",
        help="Optional x264 encoding preset. Slower presets improve compression efficiency.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args.num_frames = resolve_num_frames(args.num_frames, args.audio_path, args.frame_rate)

    pipeline = ARA2VidDistilledPipeline(
        distilled_checkpoint_path=str(args.distilled_checkpoint_path),
        spatial_upsampler_path=None,
        gemma_root=str(args.gemma_root),
        loras=(),
    )
    images = build_first_frame_images(
        args.first_frame_condition_image_path,
        strength=args.first_frame_image_strength,
        crf=args.first_frame_image_crf,
    )
    first_frame_channel_condition_latent = encode_first_frame_channel_condition(
        pipeline,
        image_path=args.first_frame_condition_image_path,
        enabled=args.first_frame_channel_condition,
        height=args.height,
        width=args.width,
        crf=args.first_frame_image_crf,
    )
    derive_channel_condition_from_first_chunk = (
        args.first_frame_channel_condition and first_frame_channel_condition_latent is None
    )
    tiling_config = build_tiling_config(args)
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    output_path = resolve_output_path(args)

    start_time = time.time()
    with torch.inference_mode():
        video, audio = pipeline(
            prompt=args.prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            images=images,
            audio_path=str(args.audio_path),
            audio_start_time=0.0,
            audio_max_duration=args.num_frames / args.frame_rate,
            tiling_config=tiling_config,
            enhance_prompt=False,
            stage_mode="one-stage",
            stage1_sigmas=args.stage1_sigmas,
            ar_video_chunk_size=args.ar_video_chunk_size,
            ar_history_chunk_count=args.ar_history_chunk_count,
            ar_sink_first_chunk=args.ar_sink_first_chunk,
            ar_relative_positions=args.ar_relative_positions,
            ar_history_feature_cache=args.ar_history_feature_cache,
            ar_audio_alignment=args.ar_audio_alignment,
            ar_first_frame_prefix_condition=args.first_frame_prefix_condition,
            ar_first_frame_condition_position=args.first_frame_condition_position,
            first_frame_channel_condition_latent=first_frame_channel_condition_latent,
            first_frame_channel_condition_init=args.first_frame_channel_condition_init,
            first_frame_channel_condition_mode=args.first_frame_channel_condition_mode,
            ar_first_frame_channel_condition_from_first_chunk=derive_channel_condition_from_first_chunk,
            ar_channel_condition_current_chunk_only=args.first_frame_channel_condition,
            fast_infer=args.fast_infer,
        )

        encode_video(
            video=video,
            fps=args.frame_rate,
            audio=audio,
            output_path=str(output_path),
            video_chunks_number=video_chunks_number,
            crf=args.video_crf,
            preset=args.video_preset,
        )

    generation_seconds = time.time() - start_time
    logging.info("Generation finished in %.2fs (%d frames).", generation_seconds, args.num_frames)
    logging.info("Saved video to %s", output_path)


if __name__ == "__main__":
    main()
