"""Run the streaming engine on an audio file and report what each chunk cost.

    uv run --no-sync python -m server.selftest --audio inputs/a.wav --image inputs/a.jpg --chunks 8
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import torch
import torchaudio

from ltx_core.model.audio_vae import encode_audio
from ltx_core.types import Audio
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from server.engine import DEFAULT_SIGMAS, FPS, THREE_STEP_SIGMAS, Engine, Session, chunk_seconds


def mono16k(path: str, device: torch.device) -> torch.Tensor:
    audio = decode_audio_from_file(path, device, 0.0, None)
    wave = audio.waveform.float().mean(dim=1).squeeze(0)
    if audio.sampling_rate != 16000:
        wave = torchaudio.functional.resample(wave, audio.sampling_rate, 16000)
    return wave.cpu()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/avatarforever-ltx-2.3-22b.safetensors")
    p.add_argument("--gemma-root", default="checkpoints/gemma-3-12b-it-qat-q4_0-unquantized")
    p.add_argument("--audio", required=True)
    p.add_argument("--image", default=None)
    p.add_argument("--chunks", type=int, default=8)
    p.add_argument("--steps", type=int, choices=(3, 4), default=4)
    p.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--decode-context", type=int, default=1)
    p.add_argument("--decode-delay", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/selftest.mp4")
    p.add_argument("--check-audio", action="store_true", help="compare chunked audio latents with a whole-file encode")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    t0 = time.time()
    engine = Engine(args.checkpoint, args.gemma_root)
    load_s = time.time() - t0
    wave = mono16k(args.audio, engine.device)
    session = Session(
        engine, image_path=args.image, seed=args.seed, height=args.height, width=args.width,
        sigmas=DEFAULT_SIGMAS if args.steps == 4 else THREE_STEP_SIGMAS,
        feature_cache=args.cache, decode_context=args.decode_context, decode_delay=args.decode_delay,
    )

    frames, rows, chunked_audio = [], [], []
    offset = 0
    for index in range(args.chunks):
        n = int(round(chunk_seconds(index) * 16000))
        piece = wave[offset : offset + n]
        if piece.numel() < n:
            piece = torch.cat([piece, torch.zeros(n - piece.numel())])
        offset += n
        chunk = session.next(piece)
        chunked_audio.append(session.previous_audio.float().cpu())
        frames.append(chunk.frames)
        t = chunk.timing
        rows.append(t)
        print(
            f"chunk {index:2d}  {chunk.frames.shape[0]} frames  audio {t.audio_ms:6.0f} ms  "
            f"denoise {t.denoise_ms:6.0f} ms ({' '.join(f'{s:.0f}' for s in t.steps_ms)})  "
            f"decode {t.decode_ms:6.0f} ms  total {t.total_ms:6.0f} ms  for {chunk.seconds * 1000:.0f} ms of video",
            flush=True,
        )

    steady = rows[2:] if len(rows) > 3 else rows
    med = lambda xs: statistics.median(xs)  # noqa: E731
    total = med([t.total_ms for t in steady])
    summary = {
        "load_s": round(load_s, 1),
        "steps": args.steps, "cache": args.cache, "decode_context": args.decode_context, "decode_delay": args.decode_delay,
        "size": f"{args.width}x{args.height}",
        "steady_chunks": len(steady),
        "median_audio_ms": round(med([t.audio_ms for t in steady])),
        "median_denoise_ms": round(med([t.denoise_ms for t in steady])),
        "median_decode_ms": round(med([t.decode_ms for t in steady])),
        "median_total_ms": round(total),
        "chunk_ms": 1280,
        "realtime_factor": round(1280 / total, 2),
        "fps": round(32 / (total / 1000), 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)

    if args.check_audio:
        with torch.inference_mode():
            whole = wave[:offset].to(engine.device).view(1, 1, -1).repeat(1, 2, 1)
            full = encode_audio(Audio(waveform=whole, sampling_rate=16000), engine.audio_encoder, engine.audio_processor)
        full = full.float().cpu()
        cat = torch.cat(chunked_audio, dim=2)
        n = min(full.shape[2], cat.shape[2])
        scale = full[:, :, :n].abs().mean().item()
        for shift in (-1, 0, 1):
            a = cat[:, :, max(0, shift) : n + min(0, shift)]
            b = full[:, :, max(0, -shift) : n + min(0, -shift)]
            m = min(a.shape[2], b.shape[2])
            print(f"AUDIO-CHECK shift {shift:+d}: mean abs diff {(a[:, :, :m] - b[:, :, :m]).abs().mean().item():.4f}  (latent scale {scale:.4f})")
        per_chunk, at = [], 0
        for c in chunked_audio:
            k = c.shape[2]
            if at + k <= n:
                per_chunk.append(round((c - full[:, :, at : at + k]).abs().mean().item(), 4))
            at += k
        print("AUDIO-CHECK per chunk:", per_chunk)

    video = torch.cat(frames, dim=0)
    audio = wave[: int(video.shape[0] / FPS * 16000)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        video=video, fps=FPS, audio=Audio(waveform=audio.view(1, -1).repeat(2, 1), sampling_rate=16000),
        output_path=args.out, video_chunks_number=1, preset="veryfast",
    )
    print(f"saved {args.out}: {video.shape[0]} frames", flush=True)


if __name__ == "__main__":
    main()
