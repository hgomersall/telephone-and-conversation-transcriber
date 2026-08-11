#!/usr/bin/env python3
"""Tests for SpeechDetector — the Silero VAD wrapper in caption_app.py.

The property that matters is gain invariance: the fixed `energy < 0.005`
threshold this replaces could be defeated in both directions by the microphone
gain, which is exactly how it failed in practice — too high and room noise was
transcribed as speech, too low and speech was discarded silently.

Needs a real speech sample. Synthetic tones do not work: Silero correctly
rejects them, so a buzz proves nothing about detection. Pass a path to any
speech audio, or let it fetch one:

    ./vad-detector-test.py [path-to-speech-audio]

Without a sample the speech-dependent cases are SKIPPED, not passed.
"""

import os
import sys
import types
import urllib.request

import numpy as np

SAMPLE_URL = 'https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac'
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.vad-test-sample')
SR = 16000
FAILURES = []
SKIPPED = []


def stub_pyqt():
    """caption_app imports PyQt6 at module scope; we only want the detector."""
    class Any:
        def __init__(self, *a, **k): pass
        def __getattr__(self, n): return Any()
        def __call__(self, *a, **k): return Any()
    mods = {
        'PyQt6.QtWidgets': ['QApplication', 'QMainWindow', 'QTextEdit', 'QLabel',
                            'QVBoxLayout', 'QHBoxLayout', 'QWidget', 'QScroller',
                            'QStackedWidget'],
        'PyQt6.QtCore': ['Qt', 'pyqtSignal', 'QObject', 'QTimer'],
        'PyQt6.QtGui': ['QFont', 'QFontDatabase', 'QTextCursor', 'QTextCharFormat',
                        'QPainter', 'QColor', 'QLinearGradient', 'QPen'],
    }
    sys.modules['PyQt6'] = types.ModuleType('PyQt6')
    for mod, names in mods.items():
        sys.modules[mod] = types.ModuleType(mod)
        for n in names:
            setattr(sys.modules[mod], n, Any)


def load_speech(path=None):
    """Return a float32 16 kHz speech array, or None if unavailable."""
    from faster_whisper.audio import decode_audio
    if path:
        return decode_audio(path, sampling_rate=SR)
    if not os.path.exists(CACHE):
        try:
            urllib.request.urlretrieve(SAMPLE_URL, CACHE)
        except Exception as e:
            print(f'  could not fetch a speech sample ({e})')
            return None
    try:
        return decode_audio(CACHE, sampling_rate=SR)
    except Exception as e:
        print(f'  could not decode {CACHE} ({e})')
        return None


def to_pcm(a):
    return (np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes()


def transitions(detector_cls, pcm, sr=SR):
    """Feed audio in 100 ms reads. Returns (detector, [(audio_t, active, speaking)]).

    feed() reports changes in `active` — the raw state driving the indicator —
    so `speaking` is sampled at those moments rather than at its own.
    """
    d = detector_cls(sr)
    out = []
    for i in range(0, len(pcm), 3200):
        if d.feed(pcm[i:i + 3200]):
            out.append((round(d._t, 2), d.active, d.speaking))
    return d, out


def check(name, cond, detail=''):
    if cond:
        print(f'  PASS  {name}')
    else:
        print(f'  FAIL  {name}' + (f'\n        {detail}' if detail else ''))
        FAILURES.append(name)


def skip(name, why):
    print(f'  SKIP  {name} — {why}')
    SKIPPED.append(name)


def main():
    stub_pyqt()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import caption_app
    SpeechDetector = caption_app.SpeechDetector

    if not SpeechDetector(SR).enabled:
        print('SpeechDetector could not load Silero — is faster-whisper installed?')
        return 1

    rng = np.random.default_rng(0)
    quiet = (rng.standard_normal(2 * SR) * 0.001).astype(np.float32)
    loud = (rng.standard_normal(2 * SR) * 0.02).astype(np.float32)  # loud, not speech

    print('\nnon-speech cases (no sample needed)')
    _, tr = transitions(SpeechDetector, to_pcm(quiet))
    check('quiet room stays silent', not tr, str(tr))

    _, tr = transitions(SpeechDetector, to_pcm(loud))
    rms = float(np.sqrt(np.mean(loud ** 2)))
    check('loud non-speech rejected', not tr,
          f'rms={rms:.4f} — the old energy gate (0.005) would have passed this')

    print('\nspeech cases')
    speech = load_speech(sys.argv[1] if len(sys.argv) > 1 else None)
    if speech is None:
        for n in ['real speech detected', 'speech bracketed by silence',
                  'indicator leads the debounced state', 'gain invariance']:
            skip(n, 'no speech sample available')
    else:
        d, tr = transitions(SpeechDetector, to_pcm(speech))
        check('real speech detected', any(a for _, a, _ in tr), str(tr))

        seq = np.concatenate([quiet, speech, quiet])
        d, tr = transitions(SpeechDetector, to_pcm(seq))
        check('speech bracketed by silence',
              any(s for _, _, s in tr) and not d.speaking,
              f'speaking seen={any(s for _, _, s in tr)} at_end={d.speaking} {tr}')

        # The indicator must respond to the voice, not to the hangover. The
        # first raw detection necessarily precedes the debounced one, which
        # still has min_speech_duration to satisfy.
        first_active = next((t for t, a, _ in tr if a), None)
        first_speaking = next((t for t, _, s in tr if s), None)
        check('indicator leads the debounced state',
              first_active is not None and
              (first_speaking is None or first_active <= first_speaking),
              f'active at {first_active}s, speaking at {first_speaking}s')

        # The one the old threshold could not do: same signal, 8x quieter.
        _, tr = transitions(SpeechDetector, to_pcm(speech * 0.125))
        check('gain invariance (speech at 1/8 level)', any(a for _, a, _ in tr), str(tr))

    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES))
        return 1
    if SKIPPED:
        print(f'passed, but {len(SKIPPED)} skipped for want of a speech sample')
        return 0
    print('ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
