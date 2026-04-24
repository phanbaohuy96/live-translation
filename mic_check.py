"""Audio level meter for diagnosing BlackHole / input-device setup.

Usage:  python mic_check.py [device-name-substring]

Opens the chosen input device at 16 kHz mono and prints RMS + peak every
50 ms for 10 s. Verdict at the end tells you whether the signal level is
useful for Whisper.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import sounddevice as sd

DURATION_SEC = 10.0
SAMPLE_RATE = 16000
BLOCK_SAMPLES = 480

USEFUL_RMS = 0.02   # clearly in-range for speech
MIN_RMS = 0.005     # our SILENCE_RMS_THRESHOLD; below this Whisper hallucinates


def pick_device(name_hint: str | None) -> int:
    devices = sd.query_devices()
    if name_hint:
        for idx, dev in enumerate(devices):
            if name_hint.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
                return idx
        print(f"error: no input device matching {name_hint!r}. Available inputs:", file=sys.stderr)
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                print(f"  [{idx}] {dev['name']}", file=sys.stderr)
        sys.exit(2)
    return sd.default.device[0]


def main() -> int:
    hint = sys.argv[1] if len(sys.argv) > 1 else None
    idx = pick_device(hint)
    dev = sd.query_devices()[idx]
    print(f"Listening on device [{idx}] {dev['name']!r}  (channels={dev['max_input_channels']}, default_sr={int(dev['default_samplerate'])})")
    print(f"Play your meeting audio / YouTube now. Monitoring for {int(DURATION_SEC)}s...\n")

    peak_rms = 0.0
    latest_rms = 0.0

    def cb(indata, frames, time_info, status):
        nonlocal peak_rms, latest_rms
        audio = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio * audio)))
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        latest_rms = rms
        if rms > peak_rms:
            peak_rms = rms
        bar_len = int(min(rms * 60, 60))
        sys.stdout.write(f"\rRMS={rms:.4f}  peak={peak:.4f}  {'█' * bar_len:<60}")
        sys.stdout.flush()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCK_SAMPLES,
        dtype="int16", channels=1, device=idx, callback=cb,
    ):
        time.sleep(DURATION_SEC)

    print("\n")
    print(f"Peak RMS over the window: {peak_rms:.4f}")

    if peak_rms < MIN_RMS:
        print("\n⚠️  No usable signal. BlackHole (or your chosen device) isn't receiving audio.")
        print("Likely causes:")
        print("  1. System output isn't set to your Multi-Output Device")
        print("     → menu-bar speaker icon or System Settings → Sound → Output")
        print("  2. Multi-Output Device doesn't include 'BlackHole 2ch'")
        print("     → Audio MIDI Setup → right-click the device → Show Info")
        print("  3. BlackHole channel levels are at 0 in Audio MIDI Setup")
        print("  4. The source app is using a per-app output (some browsers, Teams) —")
        print("     check the app's own audio settings")
        return 1
    if peak_rms < USEFUL_RMS:
        print("\n⚠️  Signal is detectable but quiet. Whisper may still work but accuracy drops.")
        print("Raise source volume or increase BlackHole channel levels in Audio MIDI Setup.")
        return 0
    print("\n✓ Good signal. BlackHole is receiving audio and Whisper will get usable input.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
