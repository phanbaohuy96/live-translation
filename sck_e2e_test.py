"""End-to-end test for ScreenCaptureKit system-audio capture.

Starts the ScreenCaptureKit helper, plays the synthetic test script through the
macOS default output with `say`, then runs the same StreamingASR,
SentenceBuffer, and translation worker used by the live app. The test passes if
at least one translated pair reaches the fake overlay.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))

from faster_whisper import WhisperModel  # noqa: E402
from translator import (  # noqa: E402
    AudioRing,
    SentenceBuffer,
    StreamingASR,
    TRANSLATION_STOP,
    WHISPER_COMPUTE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
    build_translator,
    screencapturekit_loop,
    stt_loop,
    translate_loop,
    validate_screencapturekit_helper,
)


class FakeOverlay:
    def __init__(self) -> None:
        self.last_live = ""
        self.pairs: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def set_live(self, text: str) -> None:
        if text and text != self.last_live:
            self.last_live = text
            sys.stdout.write(f"\r\x1b[K... {text[:100]}")
            sys.stdout.flush()

    def add_pair(self, english: str, vietnamese: str) -> None:
        with self.lock:
            self.pairs.append((english, vietnamese))
        sys.stdout.write(f"\r\x1b[K[EN] {english}\n[VI] {vietnamese}\n\n")
        sys.stdout.flush()


def play_test_script(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    subprocess.run(["say", "-v", "Samantha", "-r", "175", "-f", path], check=True)


def main() -> int:
    script_path = sys.argv[1] if len(sys.argv) > 1 else "test_script.txt"
    validate_screencapturekit_helper()

    print(f"[test] script={script_path}", file=sys.stderr)
    print(f"[test] loading faster-whisper: {WHISPER_MODEL}", file=sys.stderr)
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    print("[test] model ready", file=sys.stderr)

    translate = None
    try:
        translate = build_translator()
    except Exception as exc:
        print(f"[test] translator STUBBED: {exc}", file=sys.stderr)

    ring = AudioRing()
    asr = StreamingASR(model, ring)
    trans_q: queue.Queue[object] = queue.Queue()
    sentences = SentenceBuffer(trans_q)
    overlay = FakeOverlay()
    stop_evt = threading.Event()
    capture_errors: list[BaseException] = []

    def capture_target() -> None:
        try:
            screencapturekit_loop(ring, stop_evt)
        except BaseException as exc:
            capture_errors.append(exc)
            stop_evt.set()

    t_capture = threading.Thread(target=capture_target, daemon=True)
    t_stt = threading.Thread(target=stt_loop, args=(asr, sentences, overlay, stop_evt), daemon=True)
    t_tr = threading.Thread(target=translate_loop, args=(trans_q, overlay, translate), daemon=False)

    t_capture.start()
    time.sleep(1.0)
    if capture_errors:
        print(f"[test] capture failed before playback: {capture_errors[0]}", file=sys.stderr)
        return 1

    t_stt.start()
    t_tr.start()

    print("[test] playing synthetic speech through system output", file=sys.stderr)
    try:
        play_test_script(script_path)
    except Exception as exc:
        print(f"[test] playback failed: {exc}", file=sys.stderr)
        stop_evt.set()
        trans_q.put(TRANSLATION_STOP)
        trans_q.join()
        t_tr.join()
        return 1

    print("[test] waiting for capture and ASR drain", file=sys.stderr)
    deadline = time.time() + 35.0
    while time.time() < deadline and not overlay.pairs and not capture_errors:
        time.sleep(0.25)

    stop_evt.set()
    t_capture.join(timeout=2.0)
    t_stt.join()

    try:
        tail = asr.finalize()
        if tail:
            print(f"\n[stt final] +{' '.join(tail)}", file=sys.stderr)
            sentences.add(tail)
        sentences.force_flush()
    except Exception as exc:
        print(f"[test] final ASR drain failed: {exc}", file=sys.stderr)

    trans_q.put(TRANSLATION_STOP)
    trans_q.join()
    t_tr.join()

    if capture_errors:
        print(f"[test] capture failed: {capture_errors[0]}", file=sys.stderr)
        return 1

    print(f"\n[test] captured {len(overlay.pairs)} translated pairs", file=sys.stderr)
    return 0 if overlay.pairs else 1


if __name__ == "__main__":
    sys.exit(main())
