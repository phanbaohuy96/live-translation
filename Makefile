# Pick the newest Python that faster-whisper 1.1.0 has wheels for (3.10–3.12
# are the happy path; 3.9 works with a relaxed numpy pin; 3.13+ may need source builds).
PYTHON       ?= $(shell command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3.10 2>/dev/null || command -v python3)
VENV         ?= .venv
PY           := $(VENV)/bin/python
PIP          := $(VENV)/bin/pip
WHISPER_MODEL ?= base.en
OLLAMA_MODEL ?= qwen2.5:7b
SCK_HELPER   ?= .build/screencapture_audio
WAV          ?= /tmp/test.wav

.DEFAULT_GOAL := help

.PHONY: help setup install-blackhole screencapture-helper install-ollama model ollama-pull ollama-warmup run run-local run-sck run-sck-local run-blackhole run-blackhole-local test test-local test-wav mic-test clean doctor

help:
	@echo "Live EN → VI translator"
	@echo ""
	@echo "  make setup              create venv and install deps"
	@echo "  make install-blackhole  install BlackHole 2ch (requires Homebrew)"
	@echo "  make screencapture-helper build no-admin macOS system-audio helper"
	@echo "  make install-ollama     install Ollama + start the server (requires Homebrew)"
	@echo "  make model              pre-download the Whisper model ($(WHISPER_MODEL))"
	@echo "  make ollama-pull        pre-pull and warm up the Ollama model ($(OLLAMA_MODEL))"
	@echo "  make ollama-warmup      pin the Ollama model into RAM (speeds up first translation)"
	@echo ""
	@echo "  make run                live translator — ScreenCaptureKit audio + Claude"
	@echo "  make run-local          live translator — ScreenCaptureKit audio + Ollama"
	@echo "  make run-blackhole      live translator — BlackHole audio + Claude"
	@echo "  make run-blackhole-local live translator — BlackHole audio + Ollama"
	@echo ""
	@echo "  make test               e2e on synthetic WAV — Claude backend (needs ANTHROPIC_API_KEY)"
	@echo "  make test-local         e2e on synthetic WAV — Ollama backend"
	@echo "  make test-wav WAV=path  e2e on a specific WAV file (honours TRANSLATION_BACKEND)"
	@echo "  make mic-test           live meter — diagnose whether BlackHole is receiving audio"
	@echo "  make doctor             check that prerequisites are installed"
	@echo ""
	@echo "  make clean              remove the venv"
	@echo ""
	@echo "Overrides:  PYTHON, VENV, WHISPER_MODEL, OLLAMA_MODEL, WAV"

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip >/dev/null

$(VENV)/.deps-installed: $(VENV)/bin/python requirements.txt
	$(PIP) install -r requirements.txt
	@touch $(VENV)/.deps-installed

setup: $(VENV)/.deps-installed
	@echo ""
	@echo "Setup complete. Next:"
	@echo "  1) make install-blackhole     (one-time)"
	@echo "  2) export ANTHROPIC_API_KEY=sk-ant-...  (or 'make ollama-pull' for local)"
	@echo "  3) make run                   (or 'make run-local')"

install-blackhole:
	@command -v brew >/dev/null || { echo "Homebrew not found. Install from https://brew.sh first."; exit 1; }
	@brew list blackhole-2ch >/dev/null 2>&1 && echo "BlackHole already installed." || brew install blackhole-2ch
	@echo ""
	@echo "Next: Audio MIDI Setup → '+' → Create Multi-Output Device,"
	@echo "tick your speakers/headphones AND 'BlackHole 2ch', then"
	@echo "System Settings → Sound → Output → select that Multi-Output Device."

screencapture-helper: macos_screencapture_audio.swift
	@mkdir -p .build
	xcrun swiftc -O -parse-as-library -module-cache-path .build/swift-module-cache \
		-framework ScreenCaptureKit -framework CoreMedia -framework AudioToolbox \
		macos_screencapture_audio.swift -o $(SCK_HELPER)
	@echo "Built $(SCK_HELPER)"

model: setup
	$(PY) -c "from faster_whisper import WhisperModel; WhisperModel('$(WHISPER_MODEL)')"
	@echo "Whisper model '$(WHISPER_MODEL)' cached."

install-ollama:
	@command -v brew >/dev/null || { echo "Homebrew not found. Install from https://brew.sh first, or download Ollama from https://ollama.com/download."; exit 1; }
	@command -v ollama >/dev/null && echo "Ollama already installed ($$(ollama --version | head -1))." || brew install ollama
	@echo ""
	@echo "Starting Ollama server (if it's not already running)…"
	@curl -s -o /dev/null --max-time 2 http://localhost:11434/api/tags && echo "Ollama server is already running." || { \
		echo "Run 'ollama serve' in another terminal, or 'brew services start ollama' to run it in the background."; \
		echo "Then: make ollama-pull"; \
	}

ollama-pull:
	@command -v ollama >/dev/null || { echo "Ollama not found. Run 'make install-ollama' first (or install from https://ollama.com/download)."; exit 1; }
	@curl -s -o /dev/null --max-time 2 http://localhost:11434/api/tags || { \
		echo "Ollama server not reachable at http://localhost:11434."; \
		echo "Start it with 'ollama serve' (foreground) or 'brew services start ollama' (background)."; \
		exit 1; }
	ollama pull $(OLLAMA_MODEL)
	@$(MAKE) --no-print-directory ollama-warmup

ollama-warmup:
	@echo "Warming up '$(OLLAMA_MODEL)' (loads weights into RAM so the first real translation isn't slow)..."
	@curl -s -X POST http://localhost:11434/api/generate \
		-d '{"model":"$(OLLAMA_MODEL)","prompt":"hi","stream":false,"keep_alive":"30m"}' \
		-o /dev/null --max-time 120 && echo "Model warmed up and pinned in RAM for 30 min." || echo "Warmup failed — first translation may be slow."

run: setup screencapture-helper
	@test -n "$$ANTHROPIC_API_KEY" || { echo "error: ANTHROPIC_API_KEY is unset. Export it or use 'make run-local'."; exit 1; }
	AUDIO_SOURCE=screencapturekit SCK_AUDIO_HELPER=$(SCK_HELPER) WHISPER_MODEL=$(WHISPER_MODEL) TRANSLATION_BACKEND=claude $(PY) translator.py

run-local: setup screencapture-helper
	@curl -s -o /dev/null --max-time 2 $(or $(OLLAMA_URL),http://localhost:11434)/api/tags || { \
		echo "error: Ollama doesn't appear to be running at $(or $(OLLAMA_URL),http://localhost:11434)."; \
		echo "Start it with 'ollama serve' (or open the Ollama app), then 'make ollama-pull'."; \
		exit 1; }
	AUDIO_SOURCE=screencapturekit SCK_AUDIO_HELPER=$(SCK_HELPER) WHISPER_MODEL=$(WHISPER_MODEL) TRANSLATION_BACKEND=ollama OLLAMA_MODEL=$(OLLAMA_MODEL) $(PY) translator.py

run-sck: run

run-sck-local: run-local

run-blackhole: setup
	@test -n "$$ANTHROPIC_API_KEY" || { echo "error: ANTHROPIC_API_KEY is unset. Export it or use 'make run-blackhole-local'."; exit 1; }
	AUDIO_INPUT=BlackHole WHISPER_MODEL=$(WHISPER_MODEL) TRANSLATION_BACKEND=claude $(PY) translator.py

run-blackhole-local: setup
	@curl -s -o /dev/null --max-time 2 $(or $(OLLAMA_URL),http://localhost:11434)/api/tags || { \
		echo "error: Ollama doesn't appear to be running at $(or $(OLLAMA_URL),http://localhost:11434)."; \
		echo "Start it with 'ollama serve' (or open the Ollama app), then 'make ollama-pull'."; \
		exit 1; }
	AUDIO_INPUT=BlackHole WHISPER_MODEL=$(WHISPER_MODEL) TRANSLATION_BACKEND=ollama OLLAMA_MODEL=$(OLLAMA_MODEL) $(PY) translator.py

/tmp/test.wav: test_script.txt
	@command -v say >/dev/null || { echo "error: 'say' not found (macOS-only)."; exit 1; }
	say -v Samantha -r 175 -f test_script.txt -o /tmp/test.aiff
	afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/test.aiff /tmp/test.wav
	@rm -f /tmp/test.aiff

test: setup /tmp/test.wav
	@test -n "$$ANTHROPIC_API_KEY" || { echo "error: ANTHROPIC_API_KEY is unset. Export it, or run 'make test-local' to use Ollama."; exit 1; }
	WHISPER_MODEL=$(WHISPER_MODEL) TRANSLATION_BACKEND=claude $(PY) e2e_test.py /tmp/test.wav

test-local: setup /tmp/test.wav
	@curl -s -o /dev/null --max-time 2 $(or $(OLLAMA_URL),http://localhost:11434)/api/tags || { \
		echo "error: Ollama server not reachable at $(or $(OLLAMA_URL),http://localhost:11434)."; \
		echo "Start it with 'ollama serve' or 'brew services start ollama'."; \
		exit 1; }
	WHISPER_MODEL=$(WHISPER_MODEL) TRANSLATION_BACKEND=ollama OLLAMA_MODEL=$(OLLAMA_MODEL) $(PY) e2e_test.py /tmp/test.wav

test-wav: setup
	@test -f "$(WAV)" || { echo "error: WAV file not found: $(WAV)"; echo "Usage: make test-wav WAV=/path/to/file.wav"; exit 1; }
	WHISPER_MODEL=$(WHISPER_MODEL) $(PY) e2e_test.py "$(WAV)"

mic-test: setup
	$(PY) mic_check.py $(or $(AUDIO_INPUT),BlackHole)

doctor:
	@echo "Python:           $$($(PYTHON) --version 2>&1)"
	@test -d $(VENV) && echo "Venv:             $(VENV) present" || echo "Venv:             MISSING (run 'make setup')"
	@command -v brew >/dev/null && echo "Homebrew:         $$(brew --version | head -1)" || echo "Homebrew:         not installed"
	@brew list blackhole-2ch >/dev/null 2>&1 && echo "BlackHole 2ch:    installed" || echo "BlackHole 2ch:    not installed (run 'make install-blackhole')"
	@command -v ollama >/dev/null && echo "Ollama:           $$(ollama --version 2>&1 | head -1)" || echo "Ollama:           not installed (run 'make install-ollama' for local backend)"
	@curl -s -o /dev/null --max-time 2 http://localhost:11434/api/tags && echo "Ollama server:    reachable" || echo "Ollama server:    not running"
	@test -n "$$ANTHROPIC_API_KEY" && echo "ANTHROPIC_API_KEY: set" || echo "ANTHROPIC_API_KEY: unset"

clean:
	rm -rf $(VENV)
