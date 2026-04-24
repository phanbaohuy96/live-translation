"""Live English -> Vietnamese translator with streaming STT (LocalAgreement-2).

Pipeline:
  system audio (BlackHole) -> sounddevice capture -> rolling audio ring
  -> periodic faster-whisper ticks with word timestamps
  -> LocalAgreement-2: confirm words that appear in two consecutive hypotheses
  -> sentence buffer (flush on .!? or idle timeout)
  -> Anthropic Claude (VI translation) -> always-on-top Tk overlay.

The overlay shows a history of confirmed EN/VI pairs plus the unconfirmed live
English hypothesis as a trailing italic line, so it feels realtime instead of
"one line per pause".
"""
from __future__ import annotations

import os
import queue
import re
import select
import sys
import threading
import time
import tkinter as tk
import subprocess
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
FRAME_SAMPLES = 480
TICK_INTERVAL = 0.4
MIN_TICK_AUDIO_SEC = 1.0
MAX_BUFFER_SEC = 25.0            # ring cap so runaway audio can't OOM
SENTENCE_END = re.compile(r'[.!?]["\')\]]?$')
IDLE_FLUSH_SEC = 2.0
MAX_SENTENCE_WORDS = 50          # force-flush if speaker never uses punctuation
TRIM_PAD_SAMPLES = 0             # trim exactly to avoid recommitting overlap words
SILENCE_RMS_THRESHOLD = 0.005    # skip Whisper below this; it hallucinates "..." on silence
TRANSLATION_STOP = object()

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")
AUDIO_SOURCE = os.environ.get("AUDIO_SOURCE", "sounddevice").lower()  # sounddevice | screencapturekit
INPUT_DEVICE = os.environ.get("AUDIO_INPUT")  # substring match against device name
SCK_AUDIO_HELPER = os.environ.get("SCK_AUDIO_HELPER", "./.build/screencapture_audio")

TRANSLATION_BACKEND = os.environ.get("TRANSLATION_BACKEND", "claude").lower()  # claude | ollama
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

SYSTEM_PROMPT = (
    "You are a live interpreter. Translate the user's English caption into "
    "natural, conversational Vietnamese. Output ONLY the Vietnamese "
    "translation with no preamble, quotes, or notes."
)

_NORM_RE = re.compile(r"[^\w']")


def _norm(tok: str) -> str:
    return _NORM_RE.sub("", tok).lower()


@dataclass
class Word:
    text: str
    start: float
    end: float


class AudioRing:
    """Thread-safe preallocated rolling float32 buffer at 16 kHz.

    Preallocated so per-frame append is O(frame), not O(buffer). Overflow does
    a one-shot memmove, which only happens if confirmation isn't keeping up.
    """

    def __init__(self, max_seconds: float = MAX_BUFFER_SEC) -> None:
        self.max_samples = int(max_seconds * SAMPLE_RATE)
        self.buf = np.zeros(self.max_samples, dtype=np.float32)
        self.size = 0
        self.lock = threading.Lock()

    def append_int16(self, frame: np.ndarray) -> None:
        n = len(frame)
        with self.lock:
            overflow = self.size + n - self.max_samples
            if overflow > 0:
                self.buf[: self.size - overflow] = self.buf[overflow : self.size]
                self.size -= overflow
            self.buf[self.size : self.size + n] = frame.astype(np.float32) / 32768.0
            self.size += n

    def snapshot(self) -> np.ndarray:
        with self.lock:
            return self.buf[: self.size].copy()

    def drop_front(self, samples: int) -> None:
        with self.lock:
            samples = min(samples, self.size)
            if samples <= 0:
                return
            self.buf[: self.size - samples] = self.buf[samples : self.size]
            self.size -= samples


class StreamingASR:
    """LocalAgreement-2 on top of faster-whisper.

    Every tick, transcribe the whole ring with word timestamps. A word is
    "confirmed" when it matches the same position in the previous hypothesis.
    Confirmed words trim the ring and become part of the committed stream.
    """

    def __init__(self, model: WhisperModel, ring: AudioRing) -> None:
        self.model = model
        self.ring = ring
        self.last_hypothesis: list[Word] = []           # timestamps rebased to ring start
        self.committed_tail: deque[str] = deque(maxlen=40)  # sliding window for initial_prompt

    def tick(self) -> tuple[str, list[str]]:
        """Run one inference pass. Returns (live_unconfirmed_text, newly_committed_words)."""
        buf = self.ring.snapshot()
        if len(buf) < int(MIN_TICK_AUDIO_SEC * SAMPLE_RATE):
            return "", []

        # Silence skip: Whisper reliably hallucinates "..." on near-silent audio.
        # Better to return nothing than to feed garbage into the translator.
        rms = float(np.sqrt(np.mean(buf * buf)))
        if rms < SILENCE_RMS_THRESHOLD:
            self.last_hypothesis = []
            return "", []

        prompt = " ".join(self.committed_tail) or None
        segments, _ = self.model.transcribe(
            buf,
            language="en",
            beam_size=1,
            word_timestamps=True,
            initial_prompt=prompt,
            condition_on_previous_text=False,
            vad_filter=False,
        )

        new_words: list[Word] = []
        for seg in segments:
            for w in (seg.words or []):
                tok = w.word.strip()
                if tok:
                    new_words.append(Word(tok, w.start, w.end))

        # LocalAgreement-2: longest common prefix with previous hypothesis becomes committed.
        # Skip tokens that normalize to empty (e.g. "...", "—") — otherwise two
        # punctuation-only hallucinations would trivially "agree" and get confirmed.
        common: list[Word] = []
        for a, b in zip(new_words, self.last_hypothesis):
            na, nb = _norm(a.text), _norm(b.text)
            if na and na == nb:
                common.append(a)
            else:
                break

        if not common:
            self.last_hypothesis = new_words
            return " ".join(w.text for w in new_words), []

        trim_end = common[-1].end
        trim_samples = max(0, int(trim_end * SAMPLE_RATE) - TRIM_PAD_SAMPLES)
        self.ring.drop_front(trim_samples)

        committed_text = [w.text for w in common]
        for t in committed_text:
            self.committed_tail.append(t)

        remaining = new_words[len(common):]
        self.last_hypothesis = [
            Word(w.text, w.start - trim_end, w.end - trim_end) for w in remaining
        ]
        return " ".join(w.text for w in remaining), committed_text

    def finalize(self) -> list[str]:
        """Best-effort final pass: commit whatever is left without waiting for
        LocalAgreement-2 to confirm. Called on shutdown so the last few words
        aren't dropped when capture ends."""
        buf = self.ring.snapshot()
        if len(buf) < int(0.2 * SAMPLE_RATE):
            return []
        prompt = " ".join(self.committed_tail) or None
        segments, _ = self.model.transcribe(
            buf,
            language="en",
            beam_size=1,
            word_timestamps=False,
            initial_prompt=prompt,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        words: list[str] = []
        for seg in segments:
            words.extend(tok for tok in seg.text.split() if tok)
        return words


class SentenceBuffer:
    """Accumulates confirmed words; flushes on sentence boundary, word cap, or idle."""

    def __init__(self, out_q: "queue.Queue[str]", max_idle: float = IDLE_FLUSH_SEC) -> None:
        self.out_q = out_q
        self.words: list[str] = []
        self.last_update = time.time()
        self.max_idle = max_idle

    def add(self, words: list[str]) -> None:
        if not words:
            return
        self.words.extend(words)
        self.last_update = time.time()
        if SENTENCE_END.search(self.words[-1]) or len(self.words) >= MAX_SENTENCE_WORDS:
            self._flush()

    def check_idle(self) -> None:
        if self.words and (time.time() - self.last_update) > self.max_idle:
            self._flush()

    def force_flush(self) -> None:
        self._flush()

    def _flush(self) -> None:
        text = " ".join(self.words).strip()
        self.words = []
        self.last_update = time.time()
        if text:
            self.out_q.put(text)


class Overlay:
    """Always-on-top caption window. Single Text widget with tagged lines.

    Worker threads only enqueue UI events. The Tk main thread polls and applies
    them, so workers never touch widget state or `pairs` directly.
    """

    HISTORY = 3

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Live EN -> VI")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#111111")
        self.root.geometry("880x230+80+80")
        self.pairs: deque[tuple[str, str]] = deque(maxlen=self.HISTORY)
        self.live_text = ""
        self.events: "queue.Queue[tuple[Any, ...]]" = queue.Queue()

        self.text = tk.Text(
            self.root,
            bg="#111111", fg="#e6e6e6",
            font=("Helvetica", 14), wrap="word",
            bd=0, padx=12, pady=10, highlightthickness=0,
        )
        self.text.pack(fill="both", expand=True)
        self.text.tag_config("en", foreground="#7aa7ff")
        self.text.tag_config("vi", foreground="#f2f2f2", font=("Helvetica", 15, "bold"))
        self.text.tag_config("live", foreground="#888888", font=("Helvetica", 13, "italic"))
        self.text.configure(state="disabled")
        self.root.after(50, self._poll_events)

    def set_live(self, text: str) -> None:
        self.events.put(("live", text))

    def add_pair(self, english: str, vietnamese: str) -> None:
        self.events.put(("pair", english, vietnamese))

    def _poll_events(self) -> None:
        dirty = False
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "live":
                text = event[1]
                if text != self.live_text:
                    self.live_text = text
                    dirty = True
            elif kind == "pair":
                self.pairs.append((event[1], event[2]))
                dirty = True
        if dirty:
            self._render()
        self.root.after(50, self._poll_events)

    def _render(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        for en, vi in self.pairs:
            self.text.insert("end", f"{en}\n", "en")
            self.text.insert("end", f"{vi}\n\n", "vi")
        if self.live_text:
            self.text.insert("end", f"… {self.live_text}\n", "live")
        self.text.see("end")
        self.text.configure(state="disabled")

    def run(self) -> None:
        self.root.mainloop()


def pick_input_device() -> int:
    devices = sd.query_devices()
    if INPUT_DEVICE:
        for idx, dev in enumerate(devices):
            if INPUT_DEVICE.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
                print(f"[audio] using device {idx}: {dev['name']}", file=sys.stderr)
                return idx
        print(f"[audio] no device matching {INPUT_DEVICE!r}; using default", file=sys.stderr)
    default = sd.default.device[0]
    if default is None or default < 0 or default >= len(devices) or devices[default]["max_input_channels"] <= 0:
        print("[audio] available input devices:", file=sys.stderr)
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                print(f"  [{idx}] {dev['name']}", file=sys.stderr)
        raise RuntimeError("No valid default input device. Set AUDIO_INPUT to an input device name.")
    print(f"[audio] using default input device: {devices[default]['name']}", file=sys.stderr)
    return default


def validate_input_stream(device: int) -> None:
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=FRAME_SAMPLES,
        dtype="int16", channels=1, device=device,
    ):
        pass
    print("[audio] input stream validated", file=sys.stderr)


def validate_screencapturekit_helper() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("ScreenCaptureKit audio is only available on macOS.")
    if not os.path.exists(SCK_AUDIO_HELPER):
        raise RuntimeError(
            f"ScreenCaptureKit helper not found at {SCK_AUDIO_HELPER!r}. "
            "Run 'make screencapture-helper' first."
        )
    if not os.access(SCK_AUDIO_HELPER, os.X_OK):
        raise RuntimeError(f"ScreenCaptureKit helper is not executable: {SCK_AUDIO_HELPER!r}")
    print(f"[audio] using ScreenCaptureKit helper: {SCK_AUDIO_HELPER}", file=sys.stderr)


def build_translator():
    """Returns a (english_text) -> vietnamese_text callable for the configured backend."""
    backend = TRANSLATION_BACKEND
    if backend == "claude":
        from anthropic import Anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "TRANSLATION_BACKEND=claude but ANTHROPIC_API_KEY is unset. "
                "Set the key, or pick TRANSLATION_BACKEND=ollama."
            )
        client = Anthropic()

        def translate(english: str) -> str:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": english}],
            )
            return resp.content[0].text.strip()
        print(f"[tr] backend=claude model={CLAUDE_MODEL}", file=sys.stderr)
        return translate

    if backend == "ollama":
        chat_url = f"{OLLAMA_URL.rstrip('/')}/api/chat"

        def translate(english: str) -> str:
            r = requests.post(
                chat_url,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": english},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        print(f"[tr] backend=ollama model={OLLAMA_MODEL} url={OLLAMA_URL}", file=sys.stderr)
        return translate

    raise ValueError(
        f"Unknown TRANSLATION_BACKEND={backend!r}. Expected 'claude' or 'ollama'."
    )


def capture_loop(ring: AudioRing, stop_evt: threading.Event, device: int) -> None:
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=FRAME_SAMPLES,
        dtype="int16", channels=1, device=device,
    ) as stream:
        print("[audio] capture started", file=sys.stderr)
        while not stop_evt.is_set():
            data, _ = stream.read(FRAME_SAMPLES)
            ring.append_int16(np.frombuffer(data, dtype=np.int16))


def screencapturekit_loop(ring: AudioRing, stop_evt: threading.Event) -> None:
    frame_bytes = FRAME_SAMPLES * 2
    pending = bytearray()
    proc = subprocess.Popen(
        [SCK_AUDIO_HELPER],
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
    )
    try:
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        print("[audio] ScreenCaptureKit capture started", file=sys.stderr)
        while not stop_evt.is_set():
            readable, _, _ = select.select([fd], [], [], 0.2)
            if not readable:
                if proc.poll() is not None:
                    raise RuntimeError(f"ScreenCaptureKit helper exited with code {proc.returncode}")
                continue

            chunk = os.read(fd, frame_bytes * 4)
            if not chunk:
                raise RuntimeError(f"ScreenCaptureKit helper exited with code {proc.poll()}")
            pending.extend(chunk)

            full_bytes = (len(pending) // frame_bytes) * frame_bytes
            if full_bytes:
                data = bytes(pending[:full_bytes])
                del pending[:full_bytes]
                ring.append_int16(np.frombuffer(data, dtype=np.int16))
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def stt_loop(
    asr: StreamingASR,
    sentences: SentenceBuffer,
    overlay: Overlay,
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
        took = int((time.time() - t0) * 1000)
        if committed:
            print(f"[stt {took}ms] +{' '.join(committed)}", file=sys.stderr)
            sentences.add(committed)
        sentences.check_idle()
        overlay.set_live(live)


def translate_loop(
    in_q: "queue.Queue[object]",
    overlay: Overlay,
    translate,
) -> None:
    while True:
        item = in_q.get()
        try:
            if item is TRANSLATION_STOP:
                return
            english = str(item)
            t0 = time.time()
            try:
                vi = translate(english)
            except Exception as exc:
                vi = f"[dịch lỗi: {exc}]"
            print(f"[tr {int((time.time()-t0)*1000)}ms] {english}  ->  {vi}", file=sys.stderr)
            overlay.add_pair(english, vi)
        finally:
            in_q.task_done()


def main() -> None:
    try:
        translate = build_translator()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    input_device: int | None = None
    try:
        if AUDIO_SOURCE == "sounddevice":
            input_device = pick_input_device()
            validate_input_stream(input_device)
        elif AUDIO_SOURCE == "screencapturekit":
            validate_screencapturekit_helper()
        else:
            raise ValueError("AUDIO_SOURCE must be 'sounddevice' or 'screencapturekit'.")
    except Exception as exc:
        print(f"error: audio input unavailable: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[stt] loading faster-whisper: {WHISPER_MODEL}", file=sys.stderr)
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    print("[stt] model ready", file=sys.stderr)

    ring = AudioRing()
    asr = StreamingASR(model, ring)
    trans_q: queue.Queue[object] = queue.Queue()
    sentences = SentenceBuffer(trans_q)
    overlay = Overlay()
    stop_evt = threading.Event()

    if AUDIO_SOURCE == "sounddevice":
        assert input_device is not None
        capture_thread = threading.Thread(target=capture_loop, args=(ring, stop_evt, input_device), daemon=True)
    else:
        capture_thread = threading.Thread(target=screencapturekit_loop, args=(ring, stop_evt), daemon=True)
    stt_thread = threading.Thread(target=stt_loop, args=(asr, sentences, overlay, stop_evt), daemon=True)
    tr_thread = threading.Thread(target=translate_loop, args=(trans_q, overlay, translate), daemon=False)
    capture_thread.start()
    stt_thread.start()
    tr_thread.start()

    try:
        overlay.run()
    finally:
        stop_evt.set()
        capture_thread.join(timeout=2.0)
        stt_thread.join()
        # Drain: capture any tail that LocalAgreement-2 hadn't confirmed yet,
        # then let the translator worker empty its queue before exit.
        try:
            tail = asr.finalize()
            if tail:
                print(f"[stt final] +{' '.join(tail)}", file=sys.stderr)
                sentences.add(tail)
            sentences.force_flush()
        except Exception as exc:
            print(f"[drain] finalize failed: {exc}", file=sys.stderr)
        trans_q.put(TRANSLATION_STOP)
        trans_q.join()
        tr_thread.join()


if __name__ == "__main__":
    main()
