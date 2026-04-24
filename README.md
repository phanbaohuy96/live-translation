# Live EN → VI Translator

Captures system audio during a Teams / Google Meet / any-app call, transcribes
English with faster-whisper locally, translates to Vietnamese with either
Claude (cloud) or Ollama (local), and shows captions in an always-on-top
overlay window.

## Quick start (via Makefile)

```bash
cd /Users/huy.phan/personal/projects/ai/live-translate
make doctor                 # check prerequisites
make setup                  # create venv + install deps

# Cloud translation (Claude):
export ANTHROPIC_API_KEY=sk-ant-...
make run                   # no-admin ScreenCaptureKit audio

# -- or --

# Local translation (Ollama):
make ollama-pull            # downloads qwen2.5:7b (~4.7 GB)
make run-local              # no-admin ScreenCaptureKit audio
```

macOS may prompt for Screen Recording / System Audio Recording permission the
first time the helper starts. Grant it in **System Settings → Privacy & Security
→ Screen & System Audio Recording**, then run the command again.

If you prefer the old virtual-driver path and have admin access, install
BlackHole and create a Multi-Output Device:

```bash
make install-blackhole
make run-blackhole
```

In **Audio MIDI Setup** → `+` → **Create Multi-Output Device**, tick your
speakers/headphones and `BlackHole 2ch`, then select that Multi-Output Device in
System Settings → Sound → Output.

## Makefile targets

| target | purpose |
|---|---|
| `make help` | list all targets |
| `make setup` | create `.venv`, install `requirements.txt` |
| `make doctor` | check Python / Homebrew / BlackHole / Ollama / API key |
| `make install-blackhole` | `brew install blackhole-2ch` |
| `make screencapture-helper` | build the no-admin macOS ScreenCaptureKit audio helper |
| `make model` | pre-download the Whisper model (default `base.en`) |
| `make ollama-pull` | pre-pull the Ollama model (default `qwen2.5:7b`) |
| `make run` | live translator, Claude backend, ScreenCaptureKit audio |
| `make run-local` | live translator, Ollama backend, ScreenCaptureKit audio |
| `make run-sck` | alias for `make run` |
| `make run-sck-local` | alias for `make run-local` |
| `make run-blackhole` | live translator, Claude backend, BlackHole audio |
| `make run-blackhole-local` | live translator, Ollama backend, BlackHole audio |
| `make test` | generate synthetic audio (`say`) and run the e2e pipeline |
| `make test-sck` | play synthetic speech and test ScreenCaptureKit capture end to end |
| `make test-sck-local` | same as `test-sck`, but with Ollama translation |
| `make test-wav WAV=…` | run the e2e pipeline on a specific WAV file |
| `make clean` | remove `.venv` |

Overrides: `PYTHON`, `WHISPER_MODEL`, `OLLAMA_MODEL`, `OLLAMA_URL`, `WAV`.
Example:

```bash
make run-local OLLAMA_MODEL=qwen2.5:3b WHISPER_MODEL=tiny.en
```

## Translation backends

| backend | select with | pros | cons |
|---|---|---|---|
| **Claude API** (default) | `TRANSLATION_BACKEND=claude` | best VI quality, sub-second per short turn | needs `ANTHROPIC_API_KEY`, audio text leaves machine |
| **Ollama** (local LLM) | `TRANSLATION_BACKEND=ollama` | fully offline, no per-call cost | needs Ollama running + model pulled; quality depends on model |

There is no "local Claude" — Claude's weights aren't public. For fully-local
operation use the Ollama backend. Recommended Ollama models for Vietnamese:

- `qwen2.5:7b` *(default — strong VI)*
- `qwen2.5:3b` *(half the RAM, a bit worse)*
- `aya:8b` *(Cohere, multilingual tuned)*

## Tunables (env vars)

| var | default | notes |
|---|---|---|
| `AUDIO_SOURCE` | `sounddevice` | `sounddevice` for mic/BlackHole, `screencapturekit` for no-admin system audio on macOS 13+ |
| `AUDIO_INPUT` | default mic | substring match against device name; `BlackHole` for system audio |
| `SCK_AUDIO_HELPER` | `./.build/screencapture_audio` | helper binary used when `AUDIO_SOURCE=screencapturekit` |
| `WHISPER_MODEL` | `base.en` | streaming needs a model that finishes a tick in < ~400 ms. `tiny.en` is snappier on weak CPUs; `small.en` / `medium.en` only if you have a GPU. |
| `WHISPER_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `WHISPER_COMPUTE` | `int8` | `int8` (CPU), `float16` (GPU) |
| `TRANSLATION_BACKEND` | `claude` | `claude` or `ollama` |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Haiku is fast + cheap; swap for Sonnet/Opus for domain lingo |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server base URL |
| `OLLAMA_MODEL` | `qwen2.5:7b` | any chat model pulled in Ollama |

## Python version

faster-whisper 1.1.0 ships wheels for Python 3.9–3.12. The Makefile auto-picks
`python3.12` → `3.11` → `3.10` → `python3` if available. Python 3.13/3.14 may
require source builds of `ctranslate2` / `onnxruntime`.

## How it works

- Audio capture feeds 30 ms PCM frames at 16 kHz into a preallocated ring buffer
  (~25 s cap). The default path uses `sounddevice`; the no-admin macOS path uses
  a ScreenCaptureKit helper that writes raw PCM to `translator.py`.
- Every `TICK_INTERVAL` (~400 ms) a worker runs `faster-whisper` over the whole
  current ring with word-level timestamps.
- **LocalAgreement-2**: a word is "confirmed" when it appears at the same
  position in two consecutive hypotheses. Confirmed words are trimmed from the
  front of the ring and pushed into a sentence buffer.
- The sentence buffer flushes to the translator on `.`, `!`, `?`, after 2 s of
  no new confirmed words, or at 50 words hard cap — so the translator only
  sees whole thoughts, not partials.
- On shutdown, a `finalize()` pass commits whatever's still unconfirmed in the
  ring so the last few words aren't dropped.
- The overlay shows confirmed EN/VI pairs on top and the unconfirmed live
  English hypothesis on the bottom in grey italic.

## Latency budget

| stage | typical |
|---|---|
| confirm a word (LocalAgreement-2) | ~400–800 ms after it's spoken |
| sentence flush (`.!?` or idle) | +0–2000 ms depending on pause |
| Claude Haiku translate | 300–700 ms |
| Ollama `qwen2.5:7b` translate (M-series) | 500–1200 ms |
| **total EN → VI** | **~1.0–3.0 s after speaker finishes a thought** |

The live English bar updates much faster than that (~2.5 Hz), so you can
follow along before the VI line arrives.

## Known trade-offs

- **CPU use**: Whisper runs on every tick, not just at pauses. On an M-series
  Mac with `base.en`, expect 20–40% of one core. Ollama adds more load on top.
- **Accuracy** on heavy accents / noisy calls is the usual Whisper story. Bump
  to `small.en` (GPU) if you need it; `tiny.en` drops accuracy noticeably.
- **Privacy**: STT is always local. With the Ollama backend the whole pipeline
  is offline. With the Claude backend, only the confirmed English text leaves
  your machine.
- **Speaker diarization** is not done — all voices in the mix transcribe as one
  stream.
