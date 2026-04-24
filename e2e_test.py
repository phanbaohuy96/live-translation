"""End-to-end test for the streaming translator.

Replaces mic capture with a WAV file reader that streams into the same
AudioRing at realtime pace, replaces the Tk overlay with stdout sinks,
and exercises StreamingASR + SentenceBuffer. If ANTHROPIC_API_KEY is set,
also exercises the Claude translation step; otherwise stubs it.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
# Monkeypatch sounddevice before importing translator so the import doesn't
# probe audio devices.
import types
sd_stub = types.ModuleType("sounddevice")
sd_stub.query_devices = lambda: []
sd_stub.default = types.SimpleNamespace(device=(0, 0))
sd_stub.RawInputStream = object
sys.modules["sounddevice"] = sd_stub

from translator import (  # noqa: E402
    AudioRing, SentenceBuffer, StreamingASR, FRAME_SAMPLES, SAMPLE_RATE,
    TICK_INTERVAL, TRANSLATION_STOP, build_translator,
)
from faster_whisper import WhisperModel  # noqa: E402


class FakeOverlay:
    def __init__(self) -> None:
        self.last_live = ""
        self.pairs: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def set_live(self, text: str) -> None:
        if text != self.last_live:
            self.last_live = text
            sys.stdout.write(f"\r\x1b[K… {text[:100]}"); sys.stdout.flush()

    def add_pair(self, english: str, vietnamese: str) -> None:
        with self.lock:
            self.pairs.append((english, vietnamese))
        sys.stdout.write(f"\r\x1b[K[EN] {english}\n[VI] {vietnamese}\n\n"); sys.stdout.flush()


def load_wav_float32(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz WAV, got {wf.getframerate()} Hz")
        if wf.getnchannels() != 1:
            raise ValueError(f"expected mono WAV, got {wf.getnchannels()} channels")
        if wf.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit WAV, got {wf.getsampwidth() * 8}-bit")
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16)


def file_capture_loop(
    ring: AudioRing,
    audio_int16: np.ndarray,
    stop_evt: threading.Event,
    realtime: bool = True,
) -> None:
    """Push WAV data into the ring at ~realtime pace (30 ms frames)."""
    pos = 0
    next_tick = time.monotonic()
    frame_sec = FRAME_SAMPLES / SAMPLE_RATE
    while pos < len(audio_int16) and not stop_evt.is_set():
        chunk = audio_int16[pos : pos + FRAME_SAMPLES]
        if len(chunk) < FRAME_SAMPLES:
            chunk = np.concatenate([chunk, np.zeros(FRAME_SAMPLES - len(chunk), dtype=np.int16)])
        ring.append_int16(chunk)
        pos += FRAME_SAMPLES
        if realtime:
            next_tick += frame_sec
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
    time.sleep(2.5)  # let the tail drain through idle-flush


def stt_loop(
    asr: StreamingASR,
    sentences: SentenceBuffer,
    overlay: FakeOverlay,
    stop_evt: threading.Event,
) -> None:
    next_tick = time.monotonic()
    while not stop_evt.is_set():
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        next_tick = time.monotonic() + TICK_INTERVAL
        t0 = time.time()
        live, committed = asr.tick()
        took_ms = int((time.time() - t0) * 1000)
        if committed:
            sys.stderr.write(f"\n[stt {took_ms}ms] +{' '.join(committed)}\n")
            sentences.add(committed)
        sentences.check_idle()
        overlay.set_live(live)


def translate_loop(
    in_q: "queue.Queue[object]",
    overlay: FakeOverlay,
    translate,
) -> None:
    while True:
        item = in_q.get()
        try:
            if item is TRANSLATION_STOP:
                return
            english = str(item)
            t0 = time.time()
            if translate is None:
                vi = f"[stub VI of: {english}]"
            else:
                try:
                    vi = translate(english)
                except Exception as exc:
                    vi = f"[dịch lỗi: {exc}]"
            sys.stderr.write(f"[tr {int((time.time()-t0)*1000)}ms] flush\n")
            overlay.add_pair(english, vi)
        finally:
            in_q.task_done()


def main() -> int:
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test.wav"
    model_name = os.environ.get("WHISPER_MODEL", "base.en")

    print(f"[test] wav={wav_path}  model={model_name}", file=sys.stderr)
    audio = load_wav_float32(wav_path)
    print(f"[test] loaded {len(audio)/SAMPLE_RATE:.2f}s of audio ({len(audio)} samples)", file=sys.stderr)

    print(f"[test] loading faster-whisper: {model_name}", file=sys.stderr)
    model = WhisperModel(model_name, device="auto", compute_type="int8")
    print("[test] model ready", file=sys.stderr)

    ring = AudioRing()
    asr = StreamingASR(model, ring)
    trans_q: queue.Queue[object] = queue.Queue()
    sentences = SentenceBuffer(trans_q)
    overlay = FakeOverlay()
    stop_evt = threading.Event()

    translate = None
    try:
        translate = build_translator()
    except Exception as exc:
        print(f"[test] translator STUBBED: {exc}", file=sys.stderr)

    t_capture = threading.Thread(target=file_capture_loop, args=(ring, audio, stop_evt), daemon=True)
    t_stt = threading.Thread(target=stt_loop, args=(asr, sentences, overlay, stop_evt), daemon=True)
    t_tr = threading.Thread(target=translate_loop, args=(trans_q, overlay, translate), daemon=False)
    t_capture.start(); t_stt.start(); t_tr.start()

    t_capture.join()
    time.sleep(2.0)  # let LocalAgreement-2 catch up before we force-finalize
    stop_evt.set()
    t_stt.join()

    tail = asr.finalize()
    if tail:
        print(f"\n[stt final] +{' '.join(tail)}", file=sys.stderr)
        sentences.add(tail)
    sentences.force_flush()

    trans_q.put(TRANSLATION_STOP)
    trans_q.join()
    t_tr.join()

    print(f"\n[test] captured {len(overlay.pairs)} translated pairs", file=sys.stderr)
    return 0 if overlay.pairs else 1


if __name__ == "__main__":
    sys.exit(main())
