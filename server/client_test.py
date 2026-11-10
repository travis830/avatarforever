"""Talk to the streaming interface the way a page would, and check what comes back.

    uv run --no-sync python -m server.client_test --audio inputs/a.wav --image inputs/a.jpg --seconds 14
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import time
import wave as wavelib
from pathlib import Path

import av
import numpy as np
import websockets


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8811/ws/stream")
    p.add_argument("--audio", required=True, help="16-bit mono wav")
    p.add_argument("--image", default=None)
    p.add_argument("--seconds", type=float, default=14.0)
    p.add_argument("--say-at", type=float, default=3.0, help="send the speech this long after the stream starts")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--act", default=None, help="an action sentence to send")
    p.add_argument("--motion", default="calm", choices=("calm", "original"))
    p.add_argument("--crf", type=int, default=21)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--act-at", type=float, default=5.0)
    p.add_argument("--act-seconds", type=float, default=6.0)
    p.add_argument("--save", default=None, help="write what came back to this mp4 (video only)")
    args = p.parse_args()

    with wavelib.open(args.audio, "rb") as w:
        assert w.getsampwidth() == 2, "need 16-bit audio"
        sr, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)

    decoder = av.CodecContext.create("h264", "r")
    start = {"type": "start", "sr": sr, "steps": args.steps, "motion": args.motion, "crf": args.crf,
             "width": args.width, "height": args.height}
    if args.image:
        start["image_b64"] = base64.b64encode(Path(args.image).read_bytes()).decode()

    async with websockets.connect(args.url, max_size=None) as ws:
        t0 = time.time()
        await ws.send(json.dumps(start))
        print("ready:", json.loads(await ws.recv()))
        said = False
        acted = args.act is None
        pictures = []
        decoded = keys = 0
        stream_s = 0.0
        speech_first = None
        while time.time() - t0 < args.seconds:
            if not said and time.time() - t0 >= args.say_at:
                await ws.send(pcm.tobytes())
                said, said_at = True, time.time()
                print(f"  sent {pcm.size / sr:.2f}s of speech at wall {said_at - t0:.2f}s")
            if not acted and time.time() - t0 >= args.act_at:
                await ws.send(json.dumps({"type": "action", "text": args.act, "seconds": args.act_seconds}))
                acted = True
                print(f"  sent the action at wall {time.time() - t0:.2f}s")
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            if isinstance(message, str):
                print("text:", message[:200])
                continue
            (n,) = struct.unpack(">I", message[:4])
            header = json.loads(message[4 : 4 + n])
            at = 4 + n
            for size, key in header["packets"]:
                packet = av.Packet(message[at : at + size])
                at += size
                keys += key
                got = decoder.decode(packet)
                decoded += len(got)
                if args.save:
                    pictures.extend(f.to_ndarray(format="rgb24") for f in got)
            audio = np.frombuffer(message[at:], dtype=np.int16)
            assert audio.size == header["samples"], (audio.size, header["samples"])
            wall = time.time() - t0
            if header["speech"] and speech_first is None:
                speech_first = wall
                print(f"  first chunk carrying speech arrived {wall - (said_at - t0):.2f}s after the speech was sent")
            stream_s = header["t"] + header["frames"] / header["fps"]
            print(
                f"chunk {header['index']:2d} t={header['t']:6.2f}s frames={header['frames']} speech={int(header['speech'])} acting={int(header.get('acting', False))} "
                f"bytes video={sum(s for s, _ in header['packets'])} audio_rms={int(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))} "
                f"timing={header['timing']} ahead={header['ahead']} | wall {wall:.2f}s, stream ahead of wall {stream_s - wall:+.2f}s"
            )
        await ws.send(json.dumps({"type": "end"}))
    print(f"decoded {decoded} frames, {keys} keyframes, {stream_s:.2f}s of stream in {time.time() - t0:.2f}s of wall time")
    if args.save and pictures:
        out = av.open(args.save, mode="w")
        stream = out.add_stream("libx264", rate=25, options={"crf": "18", "preset": "veryfast"})
        stream.width, stream.height, stream.pix_fmt = pictures[0].shape[1], pictures[0].shape[0], "yuv420p"
        for array in pictures:
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
        out.close()
        print(f"saved {args.save}: {len(pictures)} frames")


if __name__ == "__main__":
    asyncio.run(main())
