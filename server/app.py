"""The streaming interface: audio in, video out, over one WebSocket.

    uv run --no-sync uvicorn server.app:app --host 127.0.0.1 --port 8811

It listens on 127.0.0.1 only. Reach it through an SSH tunnel; there is no login of its own.

GET  /health            what is loaded and whether a stream is running
WS   /ws/stream         one stream at a time; a new connection takes over from the old one

Client to server
    {"type": "start", "image_b64": "...", "sr": 24000, "steps": 4, "cache": true, "seed": 42,
     "width": 768, "height": 512, "prompt": null, "lead": 1.5, "motion": "calm", "crf": 21}
                               motion "calm": a steady-head prompt while she speaks and a listening
                               prompt in silence. "original": the released prompt for everything.
                               crf: H.264 quality, lower is sharper and bigger.
                               zoom (default 1): crop the picture closer around the face first.
                               The model works on a grid 32 times coarser than the picture, so
                               a face 135 pixels tall is four cells; closer in, it gets more.
                               focus_x / focus_y: where the face is, as shares of the picture.
                               anchor "still": every window looks back at the given picture itself
                               rather than at the first second as it was generated.
                               subject: a description of who and where, put in front of the prompts.
    binary                     speech, 16-bit mono PCM at ``sr``; send a whole utterance at once
    {"type": "clock", "played": 12.3}    seconds the page has played, a few times a second
    {"type": "flush"}          drop speech that has not been drawn yet
    {"type": "action", "text": "She picks up the camera ...", "seconds": 5}
                               draw the next ``seconds`` with this sentence in front of the prompt,
                               then go back. The sentence is used as it is: write it in English.
    {"type": "end"}

Server to client
    {"type": "ready", "fps": 25, "width": 768, "height": 512, "codec": "avc1.64001f", "sr": 24000}
    binary chunk               4-byte big-endian header length, a JSON header, then the H.264
                               packets (Annex B) and the 16-bit PCM that goes with those frames
    {"type": "action", "state": "ready", "text": "...", "chunks": 4, "encode_ms": 180}
    {"type": "error", "message": "..."}

Between utterances the stream keeps going on silence, a little ahead of what the page has played,
so she is always on screen. Speech is drawn as soon as it arrives.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import struct
import tempfile
import time
from contextlib import suppress
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from server.engine import (
    CALM_SPEAKING_PROMPT,
    DEFAULT_PROMPT,
    DEFAULT_SIGMAS,
    FPS,
    IDLE_PROMPT,
    THREE_STEP_SIGMAS,
    Engine,
    Session,
    chunk_seconds,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("avatar.server")

CHECKPOINT = os.environ.get("AVATAR_CHECKPOINT", "checkpoints/avatarforever-ltx-2.3-22b.safetensors")
GEMMA_ROOT = os.environ.get("AVATAR_GEMMA_ROOT", "checkpoints/gemma-3-12b-it-qat-q4_0-unquantized")
CODEC = "avc1.64001f"
MAX_AHEAD_S = 6.0          # never draw more than this beyond what the page has played
RESAMPLE_CONTEXT = 2400    # input samples kept to the left of a chunk when resampling for the model

app = FastAPI()
state: dict = {"engine": None, "loading": False, "stream": None, "last": None, "loaded_at": None}
gpu = asyncio.Lock()


@app.on_event("startup")
async def load() -> None:
    state["loading"] = True

    def build() -> Engine:
        return Engine(CHECKPOINT, GEMMA_ROOT, live_prompts=os.environ.get("AVATAR_LIVE_PROMPTS", "1") != "0")

    state["engine"] = await asyncio.to_thread(build)
    state["loading"] = False
    state["loaded_at"] = time.time()
    logger.info("ready")


@app.get("/health")
async def health() -> dict:
    engine = state["engine"]
    return {
        "ok": engine is not None,
        "loading": state["loading"],
        "streaming": state["stream"] is not None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "memory_gb": round(torch.cuda.memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None,
        "live_prompts": engine is not None and engine.text_encoder is not None,
        "last": state["last"],
    }


def framed(data: bytes, width: int, height: int, zoom: float, focus_x: float, focus_y: float) -> Image.Image:
    """The part of the picture to start from: the output's shape, ``zoom`` times closer, kept on the face."""
    picture = Image.open(io.BytesIO(data)).convert("RGB")
    full_w, full_h = picture.size
    aspect = width / height
    box_w = min(full_w, full_h * aspect) / max(1.0, zoom)
    box_h = box_w / aspect
    left = min(max(0.0, focus_x * full_w - box_w / 2), full_w - box_w)
    # the face a little above the middle of what is kept, as in the uncropped scenes
    top = min(max(0.0, focus_y * full_h - 0.38 * box_h), full_h - box_h)
    return picture.crop((round(left), round(top), round(left + box_w), round(top + box_h)))


class Encoder:
    """H.264, one packet per frame, a keyframe at the start of every chunk."""

    def __init__(self, width: int, height: int, crf: int = 21) -> None:
        ctx = av.CodecContext.create("libx264", "w")
        ctx.width, ctx.height = width, height
        ctx.pix_fmt = "yuv420p"
        ctx.time_base = Fraction(1, int(FPS))
        ctx.framerate = Fraction(int(FPS), 1)
        ctx.options = {
            "preset": "veryfast", "tune": "zerolatency", "crf": str(crf), "profile": "high", "forced-idr": "1",
            "x264-params": "keyint=250:min-keyint=1:scenecut=0:bframes=0:repeat-headers=1",
        }
        ctx.open()
        self.ctx = ctx
        self.pts = 0

    def encode(self, frames: np.ndarray) -> list[tuple[bytes, bool]]:
        out: list[tuple[bytes, bool]] = []
        for i, array in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = self.pts
            self.pts += 1
            if i == 0:
                frame.pict_type = av.video.frame.PictureType.I
            for packet in self.ctx.encode(frame):
                out.append((bytes(packet), bool(packet.is_keyframe)))
        return out


class Stream:
    def __init__(self, ws: WebSocket, engine: Engine, start: dict) -> None:
        self.ws, self.engine = ws, engine
        self.sr = int(start.get("sr") or 24000)
        self.width = int(start.get("width") or 768)
        self.height = int(start.get("height") or 512)
        if self.width * self.height > 900 * 620 or self.width % 32 or self.height % 32:
            raise ValueError("width and height must be multiples of 32 and no larger than 896x608 on this card")
        self.lead = float(start.get("lead") or 1.5)
        steps = int(start.get("steps") or 4)
        motion = "original" if start.get("motion") == "original" else "calm"
        cache = start.get("cache", True)
        cache = cache if cache in ("all", "last_full", "off") else ("all" if cache else "off")
        crf = max(14, min(32, int(start.get("crf") or 21)))
        self.options = {
            "steps": steps, "cache": cache, "seed": int(start.get("seed") or 42),
            "size": f"{self.width}x{self.height}", "motion": motion, "crf": crf,
        }
        # Which prompt a chunk is drawn with depends on whether it carries speech.
        anchor = "still" if start.get("anchor") == "still" else "generated"
        subject = " ".join(str(start.get("subject") or "").split())[:900]
        self.options["anchor"] = anchor
        self.options["subject"] = bool(subject)
        if start.get("prompt"):
            self.speak_prompt = self.idle_prompt = str(start["prompt"])
        elif motion == "original":
            self.speak_prompt = self.idle_prompt = DEFAULT_PROMPT
        else:
            self.speak_prompt, self.idle_prompt = CALM_SPEAKING_PROMPT, IDLE_PROMPT
        if subject:
            self.speak_prompt = f"{subject} {self.speak_prompt}"
            self.idle_prompt = f"{subject} {self.idle_prompt}"
        engine.context(self.speak_prompt)
        self.current_prompt = self.idle_prompt
        self.image_path: str | None = None
        if start.get("image_b64"):
            zoom = max(1.0, min(3.0, float(start.get("zoom") or 1.0)))
            self.options["zoom"] = zoom
            handle = tempfile.NamedTemporaryFile(prefix="avatar-", suffix=".png", delete=False)
            handle.close()
            framed(
                base64.b64decode(start["image_b64"]), self.width, self.height, zoom,
                float(start.get("focus_x") or 0.5), float(start.get("focus_y") or 0.33),
            ).save(handle.name)
            self.image_path = handle.name
        self.session = Session(
            engine, image_path=self.image_path, prompt=self.idle_prompt,
            seed=self.options["seed"], height=self.height, width=self.width,
            sigmas=DEFAULT_SIGMAS if steps >= 4 else THREE_STEP_SIGMAS,
            feature_cache=self.options["cache"], decode_context=2, decode_delay=True, anchor=anchor,
        )
        self.encoder = Encoder(self.width, self.height, crf)
        self.speech = np.zeros(0, dtype=np.int16)   # speech waiting to be drawn
        self.speech_at = 0.0
        self.left = np.zeros(0, dtype=np.int16)     # tail of what was already drawn, for the resampler
        self.playback = np.zeros(0, dtype=np.int16) # audio drawn but not yet sent (the decode delay)
        self.emitted = 0.0      # seconds of video sent
        self.played = 0.0       # seconds the page says it has played
        self.played_at = time.time()
        self.closed = False
        self.wake = asyncio.Event()
        self.chunks = 0
        self.busy_ms: list[float] = []
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.out = asyncio.Lock()            # chunks and notes share the connection: one at a time
        self.action: dict | None = None      # {"text", "chunks", "on"}
        self.sender: asyncio.Task | None = None

    # what the page has played by now, assuming it kept playing since its last report
    def playhead(self) -> float:
        return min(self.emitted, self.played + (time.time() - self.played_at))

    def hear(self, pcm: bytes) -> None:
        self.speech = np.concatenate([self.speech, np.frombuffer(pcm, dtype=np.int16)])
        self.speech_at = time.time()
        self.wake.set()

    def flush(self) -> None:
        self.speech = np.zeros(0, dtype=np.int16)

    async def act(self, text: str, seconds: float) -> None:
        """Put an action sentence in front of the prompt for the next ``seconds`` of video."""
        text = " ".join(text.split())[:400]
        if not text:
            return
        started = time.perf_counter()
        # Encode now, with either base it may be joined to, so the switch itself costs nothing.
        for base in {self.speak_prompt, self.idle_prompt}:
            await asyncio.to_thread(self.engine.context, f"{text} {base}")
        chunks = max(1, round(max(1.0, min(30.0, seconds)) / 1.28))
        self.action = {"text": text, "chunks": chunks, "on": False}
        self.wake.set()
        logger.info("action for %d chunks: %s", chunks, text)
        async with self.out:
            with suppress(Exception):
                await self.ws.send_json({
                    "type": "action", "state": "ready", "text": text, "chunks": chunks,
                    "encode_ms": round((time.perf_counter() - started) * 1000),
                })

    def _prompt_for_next(self, spoke: bool) -> bool:
        """Give the session the prompt the next chunk needs. True while an action is being drawn."""
        want = self.speak_prompt if spoke else self.idle_prompt
        acting = False
        action = self.action
        if action is not None:
            if action["chunks"] > 0:
                action["on"] = True
                action["chunks"] -= 1
                acting = True
                want = f"{action['text']} {want}"
            else:
                self.action = None
        if want != self.current_prompt:
            self.session.set_prompt(want)
            self.current_prompt = want
        return acting

    def _take(self, seconds: float) -> tuple[np.ndarray, bool]:
        """The next ``seconds`` of audio: waiting speech first, silence for whatever is missing."""
        need = int(round(seconds * self.sr))
        got = self.speech[:need]
        self.speech = self.speech[need:]
        spoke = got.size > 0
        if got.size < need:
            got = np.concatenate([got, np.zeros(need - got.size, dtype=np.int16)])
        return got, spoke

    def _for_model(self, piece: np.ndarray) -> torch.Tensor:
        """16 kHz float for the model. Resampled together with a little of what came before it."""
        if self.sr == 16000:
            out = torch.from_numpy(piece.astype(np.float32) / 32768.0)
        else:
            joined = np.concatenate([self.left, piece]).astype(np.float32) / 32768.0
            resampled = torchaudio.functional.resample(torch.from_numpy(joined), self.sr, 16000)
            out = resampled[-int(round(piece.size * 16000 / self.sr)) :]
        self.left = piece[-RESAMPLE_CONTEXT:]
        return out

    async def run(self) -> None:
        await self.ws.send_json({
            "type": "ready", "fps": FPS, "width": self.width, "height": self.height, "codec": CODEC,
            "sr": self.sr, "options": self.options,
        })
        self.sender = asyncio.create_task(self._send())
        try:
            await self._loop()
        except asyncio.CancelledError:
            self.sender.cancel()
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("generation failed")
            self.closed = True
            with suppress(Exception):
                await self.ws.send_json({"type": "error", "message": str(exc)[:300]})
                await self.ws.close()

    async def _loop(self) -> None:
        while not self.closed:
            ahead = self.emitted - self.playhead()
            # An action that has not started yet is drawn at once, like speech.
            waiting = self.speech.size > 0 or (self.action is not None and not self.action["on"])
            # Speech is drawn at once; silence only to stay ``lead`` seconds ahead of the page.
            if ahead > MAX_AHEAD_S or (not waiting and ahead > self.lead):
                self.wake.clear()
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.wake.wait(), timeout=0.05)
                continue
            seconds = chunk_seconds(self.session.index)
            piece, spoke = self._take(seconds)
            started = time.perf_counter()
            async with gpu:
                acting = await asyncio.to_thread(self._prompt_for_next, spoke)
                chunk = await asyncio.to_thread(self.session.next, self._for_model(piece))
            drawn_ms = (time.perf_counter() - started) * 1000.0
            at = self.emitted
            self.emitted += chunk.frames.shape[0] / FPS
            self.chunks += 1
            self.busy_ms.append(drawn_ms)
            # Encoding and sending happen beside the next chunk's drawing: the card is free again
            # the moment the frames exist. The queue keeps chunks in order.
            await self.outbox.put((chunk, piece, spoke, at, ahead, drawn_ms, acting))

    async def _send(self) -> None:
        while True:
            chunk, piece, spoke, at, ahead, drawn_ms, acting = await self.outbox.get()
            started = time.perf_counter()
            packets = await asyncio.to_thread(self.encoder.encode, chunk.frames.numpy())
            encode_ms = (time.perf_counter() - started) * 1000.0
            self.playback = np.concatenate([self.playback, piece])
            samples = int(round(chunk.frames.shape[0] / FPS * self.sr))
            audio, self.playback = self.playback[:samples], self.playback[samples:]
            header = {
                "type": "chunk", "index": chunk.index, "t": round(at, 4),
                "frames": int(chunk.frames.shape[0]), "fps": FPS,
                "packets": [[len(data), int(key)] for data, key in packets],
                "sr": self.sr, "samples": int(audio.size), "speech": spoke, "acting": acting,
                "timing": {
                    "audio": round(chunk.timing.audio_ms), "denoise": round(chunk.timing.denoise_ms),
                    "decode": round(chunk.timing.decode_ms), "encode": round(encode_ms), "total": round(drawn_ms),
                    "steps": [round(s) for s in chunk.timing.steps_ms],
                },
                "ahead": round(ahead, 3),
            }
            raw = json.dumps(header).encode()
            body = struct.pack(">I", len(raw)) + raw + b"".join(data for data, _ in packets) + audio.tobytes()
            try:
                async with self.out:
                    await self.ws.send_bytes(body)
            except Exception:  # noqa: BLE001  the page has gone
                self.closed = True
                return

    def close(self) -> None:
        self.closed = True
        self.wake.set()
        if self.sender is not None:
            self.sender.cancel()
        # Give back what this stream's sizes left in the allocator: a larger picture leaves the card nearly
        # full, and with the text encoder loaded there are only a few GB to spare.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.image_path:
            with suppress(OSError):
                Path(self.image_path).unlink()
        if self.busy_ms:
            steady = sorted(self.busy_ms[2:] or self.busy_ms)
            state["last"] = {
                "chunks": self.chunks, "seconds": round(self.emitted, 1),
                "median_chunk_ms": round(steady[len(steady) // 2]), "options": self.options,
            }


@app.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    engine: Engine | None = state["engine"]
    if engine is None:
        await ws.send_json({"type": "error", "message": "the model is still loading"})
        await ws.close()
        return
    current: Stream | None = None
    runner: asyncio.Task | None = None
    try:
        first = json.loads(await ws.receive_text())
        if first.get("type") != "start":
            await ws.send_json({"type": "error", "message": "the first message must be start"})
            return
        previous: Stream | None = state["stream"]
        if previous is not None:      # one stream at a time: the newcomer takes over
            previous.close()
            with suppress(Exception):
                await previous.ws.close()
        current = await asyncio.to_thread(Stream, ws, engine, first)
        state["stream"] = current
        runner = asyncio.create_task(current.run())
        logger.info("stream started %s", current.options)
        while not current.closed:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                current.hear(message["bytes"])
            elif message.get("text") is not None:
                data = json.loads(message["text"])
                kind = data.get("type")
                if kind == "clock":
                    current.played, current.played_at = float(data.get("played") or 0.0), time.time()
                elif kind == "flush":
                    current.flush()
                elif kind == "action":
                    asyncio.create_task(current.act(str(data.get("text") or ""), float(data.get("seconds") or 5.0)))
                elif kind == "end":
                    break
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("stream failed")
        with suppress(Exception):
            await ws.send_json({"type": "error", "message": str(exc)[:300]})
    finally:
        if current is not None:
            current.close()
            if state["stream"] is current:
                state["stream"] = None
            logger.info("stream ended %s", state["last"])
        if runner is not None:
            runner.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await runner
        with suppress(Exception):
            await ws.close()
