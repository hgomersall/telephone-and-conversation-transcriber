#!/usr/bin/env python3
"""Gramps Captions - Online (Deepgram) / Offline (Vosk) hybrid - BULLETPROOF VERSION"""
import sys
import os
import subprocess
import threading
import re
import time
import queue
import json
import collections
import numpy as np
from datetime import datetime

USAGE = """Gramps Captions — live transcription for landline and in-room speech

  caption_app.py [--log] [--log-interims]

  --log            print recognised speech to this terminal
  --log-interims   as --log, plus every partial result (very noisy)
  --log-raw        as --log, plus raw provider messages (for debugging a
                   provider's response format)

Speech is not logged by default. These are command-line flags rather than
config keys on purpose: a flag lasts exactly as long as the process you typed
it into, whereas a config setting is easy to enable while debugging and then
forget, after which every conversation in the house accumulates in the journal
indefinitely. The systemd unit does not pass them.
"""

# Answered before the GUI toolkit is imported, so --help works on a machine
# with no display libraries.
if '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
    print(USAGE)
    sys.exit(0)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLabel,
    QVBoxLayout, QHBoxLayout, QWidget, QScroller, QStackedWidget)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QFont, QFontDatabase, QTextCursor, QTextCharFormat, QPainter, QColor, QLinearGradient, QPen

# Load config from setup wizard, or use defaults
from gramps_config import find_config_file, load_config

CONFIG_PATH = find_config_file()
# Strict when a person is at the terminal — a typo'd key is silent otherwise,
# and you lose an hour wondering why a setting has no effect. Not strict when
# running as a service: refusing to start leaves someone with no captions at
# all, which is a far worse outcome than one setting not applying. The problem
# is still printed either way.
CONFIG = load_config(strict=sys.stdout.isatty())

# Paths
VOSK_MODEL = os.path.expanduser('~/vosk-uk')
FONT_PATH = os.path.expanduser('~/gramps-transcriber/fonts/DSEG14Classic-Bold.ttf')
PHONE_MUTED_FILE = '/tmp/phone_muted'
SILENCE_TIMEOUT = 90
PHONE_SILENCE_TIMEOUT = 10
# Speaker turn colours. Cycled on speaker CHANGE, not bound to identity.
# Amber/blue is the blue-yellow axis, preserved under protanopia and deuteranopia;
# both are kept light because an ageing lens absorbs short wavelengths.
SPEAKER_PALETTE_DARK = ['#ffffff', '#ffb000', '#56b4e9']
SPEAKER_PALETTE_LIGHT = ['#1a1a1a', '#8a4b00', '#00457a']
SPEAKER_MARKER = '▸ '
MODE_FILE = '/tmp/gramps_mode'

_ARGV = sys.argv[1:]

# Recognised speech is NOT logged unless explicitly asked for. It would
# otherwise be a verbatim record of every conversation and phone call in the
# house — including callers who never agreed to any of it.
_LOG_FLAGS = [a for a in _ARGV if a in ('--log', '--log-interims', '--log-raw')]
LOG_TRANSCRIPTS = bool(CONFIG.get('log_transcripts')) or bool(_LOG_FLAGS)
LOG_TRANSCRIPTS_VIA = (f'{_LOG_FLAGS[0]} flag' if _LOG_FLAGS
                       else 'log_transcripts in config')

# Interim results are extremely noisy — one line per partial — but they are
# what the latency diagnostic needs. Content either way, so they follow the
# same rule.
LOG_INTERIMS = bool(CONFIG.get('log_interims')) or '--log-interims' in _ARGV

# Raw provider messages, for working out a new provider's response shape.
# Contains transcript text, so it follows the same rule as everything else.
LOG_RAW = '--log-raw' in _ARGV

# Colour and mark caption text on speaker change. Off means no explicit
# character format is applied, which is what lets the colour-scheme buttons
# control the text colour again.
SPEAKER_COLOURS = bool(CONFIG.get('speaker_colours', True))


def log_transcript(text, prefix='>>>'):
    """Print recognised speech, if and only if that has been asked for."""
    if LOG_TRANSCRIPTS:
        print(f'{prefix} {text}', flush=True)

# Try to load secrets — check credentials.py first, then config.json
try:
    from credentials import DEEPGRAM_KEY
except ImportError:
    DEEPGRAM_KEY = CONFIG.get('deepgram_key')

# Thread-safe state management
class TranscriptionState:
    def __init__(self):
        self._lock = threading.RLock()
        self._mode = 'offline'
        self._use_phone_audio = False
        self._last_phone_speech = 0
        self._stop_event = threading.Event()
        self._thread_alive = False
        self._last_text_time = 0
        self._restart_count = 0
        self._restarting = False
        self._max_restarts = 5
        self._current_proc = None
        self._generation = 0
        self._success_time = 0
        self._retry_online_at = 0  # Timestamp to retry online mode (0=disabled)
        self._gave_up_at = 0  # Timestamp when max restarts exceeded (0=not given up)
        self._thread_loop_time = 0  # Updated by thread on each loop iteration
        self._provider_ready = False  # Set when provider signals ready (model loaded, arecord started)
        self._retry_backoff = 600  # Seconds before retrying online (exponential: 600->1200->2400->3600)

    @property
    def mode(self):
        with self._lock:
            return self._mode

    @mode.setter
    def mode(self, value):
        with self._lock:
            self._mode = value

    @property
    def use_phone_audio(self):
        with self._lock:
            return self._use_phone_audio

    @use_phone_audio.setter
    def use_phone_audio(self, value):
        with self._lock:
            self._use_phone_audio = value

    @property
    def last_phone_speech(self):
        with self._lock:
            return self._last_phone_speech

    @last_phone_speech.setter
    def last_phone_speech(self, value):
        with self._lock:
            self._last_phone_speech = value

    @property
    def thread_alive(self):
        with self._lock:
            return self._thread_alive

    @thread_alive.setter
    def thread_alive(self, value):
        with self._lock:
            self._thread_alive = value

    @property
    def last_text_time(self):
        with self._lock:
            return self._last_text_time

    @last_text_time.setter
    def last_text_time(self, value):
        with self._lock:
            self._last_text_time = value

    @property
    def generation(self):
        with self._lock:
            return self._generation

    def next_generation(self):
        with self._lock:
            self._generation += 1
            return self._generation

    def stop(self):
        self._stop_event.set()

    def clear_stop(self):
        self._stop_event.clear()

    def is_stopped(self):
        return self._stop_event.is_set()

    def can_restart(self):
        with self._lock:
            return self._restart_count < self._max_restarts

    def increment_restart(self):
        with self._lock:
            self._restart_count += 1
            return self._restart_count

    def reset_restart_count(self):
        with self._lock:
            self._restart_count = 0
            self._restarting = False

    def mark_success(self):
        """Mark working transcription — resets restart count after 60s sustained"""
        with self._lock:
            now = time.time()
            self._last_text_time = now
            if self._success_time == 0:
                self._success_time = now
            elif now - self._success_time > 60 and self._restart_count > 0:
                print(f"Sustained success for 60s, resetting restart count (was {self._restart_count})", flush=True)
                self._restart_count = 0
                self._retry_online_at = 0  # Online working, stop retry
                self._gave_up_at = 0  # Clear any gave-up state
                self._retry_backoff = 600  # Reset backoff to 10 min

    def reset_success_timer(self):
        with self._lock:
            self._success_time = 0

    @property
    def thread_loop_time(self):
        with self._lock:
            return self._thread_loop_time

    @thread_loop_time.setter
    def thread_loop_time(self, value):
        with self._lock:
            self._thread_loop_time = value

    @property
    def provider_ready(self):
        with self._lock:
            return self._provider_ready

    @provider_ready.setter
    def provider_ready(self, value):
        with self._lock:
            self._provider_ready = value

    def set_proc(self, proc):
        with self._lock:
            if self._current_proc:
                try:
                    self._current_proc.kill()
                except:
                    pass
                try:
                    self._current_proc.wait(timeout=1)
                except:
                    pass
            self._current_proc = proc

    def is_restarting(self):
        with self._lock:
            return self._restarting

    def set_restarting(self, value):
        with self._lock:
            self._restarting = value

    def kill_proc(self):
        with self._lock:
            if self._current_proc:
                try:
                    self._current_proc.kill()
                except:
                    pass
                try:
                    self._current_proc.wait(timeout=2)
                except:
                    pass
                self._current_proc = None

    def proc_alive(self):
        with self._lock:
            if self._current_proc is None:
                return False
            return self._current_proc.poll() is None


state = TranscriptionState()


class Emitter(QObject):
    new_text = pyqtSignal(str)
    new_segment = pyqtSignal(dict)  # {'text','is_final','speech_final','speaker'} - streaming path
    speakers_reset = pyqtSignal()   # a new STT session began; speaker labels restart
    vad_state = pyqtSignal(bool)    # speech present / absent, for the status indicator
    status_changed = pyqtSignal(str)
    mode_changed = pyqtSignal(str)
    mode_ready = pyqtSignal(str)
    thread_died = pyqtSignal(str)  # NEW: signal when thread dies

emitter = Emitter()



def status_display(status, speaking):
    """(text, stylesheet) for a status indicator.

    Shared by the caption and clock views so the two can never disagree.
    Faults outrank speech activity — an error must never be masked by the
    listening indicator.
    """
    fault = {
        'no-key': ('⚠️ NO KEY', '#ff0000', '#330000'),
        'error': ('⚠️ ERROR', '#ff0000', '#330000'),
        'offline-fallback': ('⚠️ OFFLINE', '#ffaa00', '#332200'),
    }.get(status)
    if fault:
        text, fg, bg = fault
        return text, (f'color: {fg}; background: {bg}; padding: 8px 15px; '
                      'border-radius: 10px; font-size: 18px; font-weight: bold;')

    if status == 'connecting':
        return '🔌', 'font-size: 30px; background: transparent;'
    if status == 'switching':
        return '⏳', 'font-size: 30px; background: transparent;'
    if status == 'restarting':
        return '🔄', 'font-size: 30px; background: transparent;'

    if status in ('vosk', 'deepgram', 'speechmatics', 'assemblyai', 'azure',
                  'google', 'openai', 'groq', 'interfaze', 'whisper',
                  'faster-whisper'):
        if not CONFIG.get('vad_indicator', True):
            return '🎤', 'font-size: 30px; background: transparent;'
        # The speech state is carried by the glyph, not by colour or opacity.
        # Qt Style Sheets have no `opacity` property, and `color` does not tint
        # a colour emoji — so styling 🎤 renders identically either way, which
        # reads as "permanently active". A filled vs hollow circle cannot be
        # ignored by any font, and being plain text it takes the colour too.
        if speaking:
            return '🎤 ●', 'font-size: 30px; background: transparent; color: #00ff66;'
        return '🎤 ○', 'font-size: 30px; background: transparent; color: #666666;'

    return '', 'background: transparent;'


class SpeechDetector:
    """Silero VAD over a raw PCM stream — reports whether speech is present.

    Observational only: it never withholds audio. Feed it the same bytes that
    go to the recogniser.

    Two states, deliberately:

      active   — the raw per-window verdict, for the on-screen indicator. It
                 should track the voice as closely as it can, because its job
                 is to answer "is this thing hearing me?" the moment you speak.
      speaking — the same verdict with a minimum duration and a hangover
                 applied, for decisions about the audio itself. Cutting a chunk
                 or closing a gate on the raw state would land in the pause
                 between two words.

    Unlike the fixed `energy < 0.005` threshold it replaces, this is not a
    function of amplitude, so changing the mic gain cannot break it.

    The model zeroes its recurrent state on every call, so frames are batched
    into a longer window to give it some context rather than asking it about
    each 32 ms in isolation.
    """

    BATCH_FRAMES = 8  # ~256 ms at 16 kHz

    def __init__(self, sample_rate, label=''):
        self.sample_rate = sample_rate
        self.label = label
        self.enabled = False
        self.active = False     # raw — drives the indicator
        self.speaking = False   # debounced — drives decisions about audio
        self._model = None
        self._buf = np.empty(0, dtype=np.float32)
        # Durations are measured in AUDIO time, not wall-clock. The offline loop
        # blocks inside model.transcribe() for seconds at a time, so wall-clock
        # would run on while no audio was being consumed and the hangover would
        # expire against silence that never happened.
        self._t = 0.0
        self._last_speech = 0.0
        self._speech_since = 0.0
        self._failures = 0
        # Silero wants 512-sample frames at 16 kHz, 256 at 8 kHz.
        self._frame = 512 if sample_rate >= 16000 else 256
        self._threshold = float(CONFIG.get('vad_threshold', 0.5))
        self._min_speech = float(CONFIG.get('vad_min_speech_ms', 250)) / 1000.0
        self._min_silence = float(CONFIG.get('vad_min_silence_ms', 500)) / 1000.0
        self._hangover = float(CONFIG.get('vad_hangover_s', 1.0))

        try:
            import os as _os
            from faster_whisper.vad import SileroVADModel, get_assets_path
            self._model = SileroVADModel(
                _os.path.join(get_assets_path(), 'silero_vad_v6.onnx')
            )
            self.enabled = True
            print(f'VAD ready{self.label} @ {sample_rate}Hz', flush=True)
        except Exception as e:
            # Never fatal — the recogniser works without us.
            print(f'VAD unavailable{self.label}: {e}', flush=True)

    def feed(self, pcm_bytes):
        """Push raw S16_LE mono audio. Returns True if `active` changed.

        The return value drives the indicator, so it reports the raw state.
        Read `speaking` for the debounced one.
        """
        if not self.enabled:
            return False
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            self._buf = np.concatenate((self._buf, samples))
            window = self._frame * self.BATCH_FRAMES
            changed = False
            while len(self._buf) >= window:
                probs = self._model(self._buf[:window], num_samples=self._frame)
                self._buf = self._buf[window:]
                self._t += window / float(self.sample_rate)
                was_active = self.active
                self._update(float(np.max(probs)))
                if self.active != was_active:
                    changed = True
            return changed
        except Exception as e:
            self._failures += 1
            if self._failures >= 5:
                self.enabled = False
                print(f'VAD disabled after repeated errors{self.label}: {e}', flush=True)
            return False

    def _update(self, prob):
        """Update both states from one window's verdict.

        `now` is audio time — seconds of audio fed in, not seconds elapsed.
        """
        now = self._t
        loud = prob >= self._threshold

        # Raw state: no smoothing beyond the window itself, which already takes
        # the maximum across its frames. The indicator follows the voice.
        self.active = loud

        if loud:
            self._last_speech = now
            if not self.speaking:
                if self._speech_since == 0.0:
                    self._speech_since = now
                # Require sustained speech so one noisy window cannot latch it
                # on. Measured over BATCH_FRAMES, so a min_speech shorter than
                # that window resolves on the first one.
                if now - self._speech_since >= self._min_speech:
                    self.speaking = True
                    self._log(True)
            return

        self._speech_since = 0.0
        if self.speaking and now - self._last_speech >= max(self._min_silence, self._hangover):
            self.speaking = False
            self._log(False)

    def _log(self, speaking):
        if CONFIG.get('log_vad'):
            # Machine-parseable: epoch plus state, so a run can be reduced to a
            # speech duty cycle without re-deriving anything.
            print(f'VAD {time.time():.3f} {"speech" if speaking else "silence"}'
                  f'{self.label}', flush=True)


class AudioGate:
    """Holds audio back while nobody is speaking.

    Deepgram bills on audio sent rather than on connection time, so withholding
    silence is the whole saving. The risk is clipping the start of an utterance:
    a detector only knows speech began after hearing some of it, and the onset
    of a word is its quietest part. So recent audio is kept in a ring buffer and
    flushed the instant the gate opens, which recovers the leading word.

    The buffer is sized in time rather than chunks — a 3200-byte read is 100 ms
    at 16 kHz but 200 ms at 8 kHz, so a fixed count would silently give the
    phone tap twice the pre-roll.
    """

    def __init__(self, sample_rate, preroll_s, bytes_per_chunk=3200):
        self.bytes_per_sec = sample_rate * 2
        # Round UP to a whole chunk. A read is 200ms at 8kHz, so the requested
        # duration usually is not representable, and erring short is the
        # direction that clips words.
        need = int(preroll_s * self.bytes_per_sec)
        self.chunks = max(1, (need + bytes_per_chunk - 1) // bytes_per_chunk)
        self._buf = collections.deque(maxlen=self.chunks)
        self.bytes_per_chunk = bytes_per_chunk
        self.open = False
        self.sent_bytes = 0

    @property
    def preroll_s(self):
        return self.chunks * self.bytes_per_chunk / float(self.bytes_per_sec)

    def feed(self, chunk, want_open):
        """Return the chunks to transmit for this read — possibly none."""
        if want_open:
            out = []
            if not self.open:
                out.extend(self._buf)
                self._buf.clear()
                self.open = True
            out.append(chunk)
            self.sent_bytes += sum(len(c) for c in out)
            return out
        self.open = False
        self._buf.append(chunk)
        return []

    def billed_seconds(self):
        return self.sent_bytes / float(self.bytes_per_sec)


def emit_utterance_end():
    """Signal end-of-utterance from our own VAD rather than the provider's.

    Both services derive end-of-utterance from silence — Deepgram through
    endpointing, Speechmatics through end_of_utterance_silence_trigger. The
    cost gate withholds exactly that silence, so neither ever sees an utterance
    finish and the captions run together into one unbroken paragraph.

    The gate closing IS the end of an utterance: our VAD heard the speech stop,
    which is why it closed. So the signal is generated here instead.

    Timing does not matter. add_segment records this and applies the break to
    the NEXT text, so emitting it any time before the next utterance produces
    identical output — which is why it can safely wait for the gate's hangover
    and be sure every pending final has arrived first.
    """
    emitter.new_segment.emit({
        'text': '', 'is_final': True, 'speech_final': True, 'speaker': None,
    })


def write_phone_status(active):
    """Write phone status file"""
    try:
        with open(PHONE_MUTED_FILE, 'w') as f:
            f.write('1' if active else '0')
    except:
        pass

def find_audio_device(name_pattern):
    """Find ALSA device by name pattern, returns hw:X,0 or None"""
    try:
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if name_pattern.lower() in line.lower():
                match = re.search(r'card (\d+):', line)
                if match:
                    return f"hw:{match.group(1)},0"
    except Exception as e:
        print(f"Device detection error: {e}", flush=True)
    return None


def to_plughw(dev):
    """Wrap a raw hw: device in ALSA's plug layer so it resamples/downmixes.
    Many USB mics only support 48kHz stereo natively; arecord on the raw hw:
    device at 16kHz mono fails instantly. plughw: converts transparently and
    is a no-op when the device already supports the requested format."""
    if dev and dev.startswith('hw:'):
        return 'plug' + dev
    return dev


def get_audio_device():
    """Get appropriate audio device based on current state"""
    if state.use_phone_audio:
        configured = CONFIG.get('phone_device')
        if configured:
            return to_plughw(configured)
        dev = find_audio_device('0x4d9') or find_audio_device('2832') or find_audio_device('phone') or 'hw:0,0'
    else:
        configured = CONFIG.get('room_device')
        if configured:
            return to_plughw(configured)
        dev = find_audio_device('tonor') or find_audio_device('usb') or 'hw:1,0'
    return to_plughw(dev)


def ensure_mic_volume():
    """Ensure microphone volume is set correctly"""
    try:
        # Try multiple cards
        for card in range(3):
            subprocess.run(['amixer', '-c', str(card), 'set', 'Mic', '100%'],
                         capture_output=True, timeout=5)
    except:
        pass


def cleanup_audio_processes():
    """Kill any stale audio processes"""
    try:
        subprocess.run(['pkill', '-9', '-f', 'arecord'], capture_output=True, timeout=5)
        time.sleep(0.5)
    except:
        pass



def faster_whisper_thread():
    """Run faster-whisper for high quality offline transcription"""
    print('Starting faster-whisper...', flush=True)
    emitter.status_changed.emit('whisper')
    # thread_alive already set by start_transcription
    arecord = None

    try:
        from faster_whisper import WhisperModel
        import numpy as np

        # Load model
        print('Loading Whisper model (small.en)...', flush=True)
        model = WhisperModel(
            "small.en",
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            num_workers=1
        )
        print('Whisper model loaded', flush=True)

        # Audio settings
        SAMPLE_RATE = 16000
        CHUNK_SECONDS = float(CONFIG.get('offline_chunk_s', 3))
        MAX_CHUNK_SECONDS = float(CONFIG.get('offline_max_chunk_s', 6))
        CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_SECONDS) * 2
        MAX_CHUNK_BYTES = int(SAMPLE_RATE * MAX_CHUNK_SECONDS) * 2

        detector = SpeechDetector(SAMPLE_RATE)

        # Get audio device
        audio_device = get_audio_device()
        print(f"Using audio device: {audio_device}", flush=True)

        # Start arecord with retry
        for attempt in range(4):
            arecord = subprocess.Popen(
                ['arecord', '-D', audio_device, '-f', 'S16_LE', '-r', '16000', '-c', '1', '-t', 'raw', '-q'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            state.set_proc(arecord)
            time.sleep(0.3)
            if arecord.poll() is None:
                print(f'arecord ready (attempt {attempt+1})', flush=True)
                break
            else:
                arecord = None
                if attempt < 3:
                    time.sleep(1)

        if not arecord:
            raise RuntimeError('Could not start arecord after 4 attempts')


        print('faster-whisper ready', flush=True)
        emitter.mode_ready.emit('offline')


        buffer = b''

        while not state.is_stopped():
            state.thread_loop_time = time.time()
            # Read audio chunk
            data = arecord.stdout.read(3200)  # 100ms at 16kHz, 16-bit
            if not data:
                print('faster-whisper: No audio data', flush=True)
                break

            buffer += data
            if detector.feed(data):
                emitter.vad_state.emit(detector.active)

            # Cut at a silence rather than on a fixed tick, so a word straddling
            # the boundary is never split in half. Unbroken speech still has to
            # be cut eventually or nothing would ever be transcribed.
            if len(buffer) >= CHUNK_BYTES:
                at_max = len(buffer) >= MAX_CHUNK_BYTES
                if detector.enabled and detector.speaking and not at_max:
                    continue  # mid-utterance — keep accumulating

                audio = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                buffer = b''

                # Transcribe
                segments, info = model.transcribe(
                    audio,
                    language="en",
                    beam_size=1,
                    best_of=1,
                    temperature=0,
                    vad_filter=bool(CONFIG.get('offline_vad', True)),
                    vad_parameters={
                        "threshold": float(CONFIG.get('vad_threshold', 0.5)),
                        "min_speech_duration_ms": int(CONFIG.get('vad_min_speech_ms', 250)),
                        "min_silence_duration_ms": int(CONFIG.get('vad_min_silence_ms', 500))
                    },
                )

                # Emit text
                segments_list = list(segments)

                for segment in segments_list:
                    text = segment.text.strip()
                    if text:
                        log_transcript(text)
                        state.mark_success()
                        emitter.new_text.emit(text)

    except ImportError as e:
        print(f'faster-whisper not available: {e}', flush=True)
        print('Falling back to Vosk...', flush=True)
        state.thread_alive = False
        vosk_thread()
        return
    except Exception as e:
        print(f'faster-whisper error: {e}', flush=True)
        emitter.status_changed.emit('error')
        import traceback
        traceback.print_exc()
    finally:
        state.thread_alive = False
        if arecord:
            try:
                arecord.terminate()
                arecord.wait(timeout=2)
            except:
                try:
                    arecord.kill()
                    arecord.wait(timeout=1)  # Reap zombie
                except:
                    pass
        state.kill_proc()
        print('faster-whisper stopped', flush=True)

        if not state.is_stopped():
            emitter.thread_died.emit('offline')


def vosk_thread():
    """Run Vosk streaming with robust error handling"""
    print('Starting Vosk...', flush=True)
    emitter.status_changed.emit('vosk')
    # thread_alive already set by start_transcription
    arecord = None

    try:
        import vosk
        import json

        vosk.SetLogLevel(-1)

        # Verify model exists
        if not os.path.exists(VOSK_MODEL):
            raise FileNotFoundError(f"Vosk model not found: {VOSK_MODEL}")

        model = vosk.Model(VOSK_MODEL)
        rec = vosk.KaldiRecognizer(model, 16000)
        rec.SetWords(False)

        # Get audio device with retry
        audio_device = None
        for attempt in range(3):
            audio_device = get_audio_device()
            print(f"Using audio device: {audio_device} (attempt {attempt+1})", flush=True)

            arecord = subprocess.Popen(
                ['arecord', '-D', audio_device, '-f', 'S16_LE', '-r', '16000', '-c', '1', '-t', 'raw', '-q'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            state.set_proc(arecord)

            # Test if we get data
            time.sleep(0.3)
            if arecord.poll() is None:  # Still running
                break
            else:
                stderr = arecord.stderr.read().decode() if arecord.stderr else ""
                print(f"arecord failed: {stderr}", flush=True)
                arecord = None
                time.sleep(1)

        if not arecord:
            raise RuntimeError("Could not start arecord after 3 attempts")

        print('Vosk ready', flush=True)
        emitter.mode_ready.emit('offline')


        consecutive_empty = 0
        while not state.is_stopped():
            state.thread_loop_time = time.time()
            try:
                data = arecord.stdout.read(4000)
                if not data:
                    consecutive_empty += 1
                    if consecutive_empty > 10:
                        print("Vosk: No audio data, restarting...", flush=True)
                        break
                    time.sleep(0.1)
                    continue

                consecutive_empty = 0
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get('text', '').strip()
                    if text:
                        log_transcript(text)
                        state.mark_success()
                        emitter.new_text.emit(text)
            except Exception as e:
                print(f"Vosk read error: {e}", flush=True)
                break

    except FileNotFoundError as e:
        print(f'Vosk model error: {e}', flush=True)
        emitter.status_changed.emit('error')
    except Exception as e:
        print(f'Vosk error: {e}', flush=True)
        emitter.status_changed.emit('error')
        import traceback
        traceback.print_exc()
    finally:
        state.thread_alive = False
        if arecord:
            try:
                arecord.terminate()
                arecord.wait(timeout=2)
            except:
                try:
                    arecord.kill()
                    arecord.wait(timeout=1)  # Reap zombie
                except:
                    pass
        state.kill_proc()
        print('Vosk stopped', flush=True)

        # Signal that thread died (for restart logic)
        if not state.is_stopped():
            emitter.thread_died.emit('offline')



def whisper_thread():
    """Run whisper.cpp streaming for better quality offline transcription"""
    print('Starting Whisper.cpp stream...', flush=True)
    emitter.status_changed.emit('whisper')
    # thread_alive already set by start_transcription
    proc = None

    try:
        WHISPER_BIN = os.path.expanduser('~/whisper.cpp/build/bin/whisper-stream')
        WHISPER_MODEL = os.path.expanduser('~/whisper.cpp/models/ggml-base.en-q5_0.bin')

        if not os.path.exists(WHISPER_BIN):
            raise FileNotFoundError(f"whisper-stream not found: {WHISPER_BIN}")
        if not os.path.exists(WHISPER_MODEL):
            raise FileNotFoundError(f"Whisper model not found: {WHISPER_MODEL}")

        # Determine audio device index for ALSA
        audio_device = "0" if state.use_phone_audio else "1"
        print(f"Using audio device index: {audio_device}", flush=True)

        cmd = [
            WHISPER_BIN,
            '-m', WHISPER_MODEL,
            '-c', audio_device,
            '--step', '3000',
            '--length', '5000',
            '-l', 'en',
        ]

        env = os.environ.copy()
        env['TERM'] = 'dumb'

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        state.set_proc(proc)

        print('Whisper ready', flush=True)
        emitter.mode_ready.emit('offline')


        for line in iter(proc.stdout.readline, ''):
            if state.is_stopped():
                break

            # Clean ANSI codes
            line = re.sub(r'\x1b\[[0-9;]*[mK]', '', line)
            line = re.sub(r'\[2K', '', line)
            line = line.strip()

            if not line:
                continue
            if line.startswith('[') or line.startswith('init:') or line.startswith('whisper'):
                continue
            if line.startswith('main:'):
                continue
            if 'BLANK_AUDIO' in line or 'INAUDIBLE' in line:
                continue

            log_transcript(line)
            state.mark_success()
            emitter.new_text.emit(line)

    except FileNotFoundError as e:
        print(f'Whisper not available: {e}', flush=True)
        print('Falling back to Vosk...', flush=True)
        state.thread_alive = False
        vosk_thread()
        return
    except Exception as e:
        print(f'Whisper error: {e}', flush=True)
        emitter.status_changed.emit('error')
        import traceback
        traceback.print_exc()
    finally:
        state.thread_alive = False
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except:
                try:
                    proc.kill()
                    proc.wait(timeout=1)  # Reap zombie
                except:
                    pass
        state.kill_proc()
        print('Whisper stopped', flush=True)

        if not state.is_stopped():
            emitter.thread_died.emit('offline')


def deepgram_thread():
    """Run Deepgram streaming with robust error handling"""
    if not DEEPGRAM_KEY:
        print('No Deepgram API key', flush=True)
        emitter.status_changed.emit('no-key')
        state.thread_alive = False
        emitter.thread_died.emit('online')
        return

    print('Starting Deepgram...', flush=True)
    emitter.status_changed.emit('connecting')
    # thread_alive already set by start_transcription
    arecord = None

    try:
        import websocket
        import json

        # Get audio device with retry
        audio_device = get_audio_device()
        # Phone recorder is 8kHz, room mic is 16kHz
        sample_rate = 8000 if state.use_phone_audio else 16000
        print(f"Using audio device: {audio_device} @ {sample_rate}Hz", flush=True)

        detector = SpeechDetector(sample_rate, label=' (deepgram)')

        test_data = b''
        for attempt in range(4):
            arecord = subprocess.Popen(
                ['arecord', '-D', audio_device, '-f', 'S16_LE', '-r', str(sample_rate), '-c', '1', '-t', 'raw'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            state.set_proc(arecord)
            time.sleep(0.3)
            test_data = arecord.stdout.read(3200)
            if len(test_data) > 0:
                print(f'arecord ready (attempt {attempt+1})', flush=True)
                break
            else:
                arecord.terminate()
                arecord = None
                if attempt < 3:
                    time.sleep(1)

        if not arecord:
            raise RuntimeError('Could not start arecord after 4 attempts')

        params = [
            'model=nova-2',
            'language=en-GB',
            'smart_format=true',
            'interim_results=true',
            'endpointing=400',
            'encoding=linear16',
            f'sample_rate={sample_rate}',
        ]
        # Diarization is billed separately and is pointless on the phone tap,
        # which already has exactly one remote talker.
        diarize = bool(CONFIG.get('speaker_colours', True)) and not state.use_phone_audio

        # Gating has been observed to make Deepgram's speaker labels collapse
        # to 0, intermittently. That was never pinned down, and the same claim
        # about Speechmatics turned out to be a measurement error — short clips
        # rather than the gaps — so treat it as reported rather than
        # established. Both run together here and you can judge; if the colours
        # stop tracking, turn one of them off.
        gating_wanted = bool(CONFIG.get('vad_gate', True))
        gate_hangover = float(CONFIG.get('gate_hangover_s', 4.0))
        if diarize and gating_wanted:
            print('Note: gating has previously disturbed Deepgram speaker labels. '
                  'If colours stop tracking the speaker, set vad_gate=false '
                  '(or speaker_colours=false to keep the saving).', flush=True)

        if diarize:
            params.append('diarize_model=latest')
        url = 'wss://api.deepgram.com/v1/listen?' + '&'.join(params)

        # This is a fresh session, so Deepgram's speaker label space restarts.
        # Covers reconnects, phone/room switches and the online retry after an
        # offline fallback — all of them arrive here.
        emitter.speakers_reset.emit()

        ws_connected = threading.Event()
        ws_error = threading.Event()

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'channel' not in data:
                    return
                alt = data['channel']['alternatives'][0]
                t = alt.get('transcript', '')
                is_final = bool(data.get('is_final', False))
                speech_final = bool(data.get('speech_final', False))
                if not t or not t.strip():
                    # Empty finals still carry endpointing information, so they
                    # are forwarded rather than dropped.
                    if not is_final:
                        return
                else:
                    state.mark_success()
                speaker = None
                words = alt.get('words') or []
                labels = [w['speaker'] for w in words if w.get('speaker') is not None]
                if labels:
                    speaker = max(set(labels), key=labels.count)
                if t.strip():
                    if is_final:
                        log_transcript(t.strip(), f'>>> [spk {speaker}]')
                    elif LOG_INTERIMS:
                        log_transcript(t.strip(), '...')
                emitter.new_segment.emit({
                    'text': t,
                    'is_final': is_final,
                    'speech_final': speech_final,
                    'speaker': speaker,
                })
            except Exception as e:
                print(f'Parse error: {e}', flush=True)

        def on_error(ws, error):
            print(f'WS error: {error}', flush=True)
            ws_error.set()

        def on_open(ws):
            print('Deepgram connected', flush=True)
            ws_connected.set()
            # Only now are we actually listening. Claiming it earlier showed the
            # microphone icon during the whole handshake, so a slow link looked
            # identical to a working one.
            emitter.status_changed.emit('deepgram')
            emitter.mode_ready.emit('online')

            ws.send(test_data, opcode=2)

            def send_audio():
                # Deepgram bills on audio sent, not on connection time, and
                # KeepAlive is not charged — so the socket is held open for the
                # life of the thread and only the audio is gated. Closing and
                # reopening would cost a reconnect on every utterance.
                gating = gating_wanted and detector.enabled
                gate = AudioGate(sample_rate, float(CONFIG.get('preroll_s', 0.5)))
                keepalive_every = float(CONFIG.get('keepalive_s', 4.0))
                keepalive_msg = json.dumps({'type': 'KeepAlive'})

                last_keepalive = time.time()
                started = time.time()
                last_report = started
                # Start open, so the stream begins exactly as an ungated one
                # does and closes only once the hangover expires with nothing
                # said. Starting closed meant Deepgram received nothing but
                # KeepAlive until the first word — possibly minutes — and then
                # a pre-roll burst arriving faster than real time. It also left
                # the very first utterance of a session resting entirely on the
                # pre-roll, which is the one place there is no margin.
                last_speech = time.time()

                if gating:
                    print(f'Gate on: {gate.preroll_s:.2f}s pre-roll, '
                          f'{gate_hangover:.1f}s hangover', flush=True)
                else:
                    print('Gate off: streaming continuously', flush=True)

                while not state.is_stopped() and not ws_error.is_set():
                    state.thread_loop_time = time.time()
                    try:
                        chunk = arecord.stdout.read(gate.bytes_per_chunk)
                        if not chunk:
                            if arecord.poll() is not None:
                                print('Deepgram: arecord process died', flush=True)
                                break
                            print('send_audio: arecord returned empty data, stopping', flush=True); break

                        if detector.feed(chunk):
                            emitter.vad_state.emit(detector.active)

                        now = time.time()
                        if detector.active:
                            last_speech = now

                        # Opens on the raw state, so it reacts as fast as the
                        # indicator. Stays open for gate_hangover_s afterwards —
                        # its own, much longer than the detector's, so the
                        # stream survives the pauses between turns and Deepgram
                        # does not restart its speaker numbering mid-conversation.
                        want_open = (not gating) or detector.active \
                            or (now - last_speech) < gate_hangover

                        was_open = gate.open
                        for outgoing in gate.feed(chunk, want_open):
                            ws.send(outgoing, opcode=2)
                        if gating and gate.open != was_open:
                            if CONFIG.get('log_vad'):
                                # Distinct from the VAD lines: the detector
                                # reports whether anyone is speaking, this
                                # reports whether audio is being transmitted.
                                # They differ by gate_hangover_s, which is the
                                # point.
                                print(f'GATE {time.time():.3f} '
                                      f'{"open" if gate.open else "closed"}', flush=True)
                            if not gate.open:
                                emit_utterance_end()
                        if not gate.open:
                            if now - last_keepalive >= keepalive_every:
                                # Text frame. Sent as binary it would be treated
                                # as audio and would not prevent the 10s
                                # NET-0001 timeout.
                                ws.send(keepalive_msg)
                                last_keepalive = now

                        if CONFIG.get('log_vad') and now - last_report >= 300:
                            elapsed = now - started
                            billed = gate.billed_seconds()
                            print(f'GATE billed {billed:.0f}s of {elapsed:.0f}s '
                                  f'({100.0 * billed / elapsed:.1f}%)', flush=True)
                            last_report = now
                    except Exception as e:
                        print(f"Send error: {e}", flush=True)
                        break

                if gating and gate.sent_bytes:
                    elapsed = max(1e-6, time.time() - started)
                    billed = gate.billed_seconds()
                    print(f'Gate closed: billed {billed:.0f}s of {elapsed:.0f}s '
                          f'({100.0 * billed / elapsed:.1f}%)', flush=True)
                try:
                    ws.close()
                except:
                    pass

            threading.Thread(target=send_audio, daemon=True).start()

        def on_close(ws, code, msg):
            print(f'Deepgram closed: {code} {msg}', flush=True)

        print('Connecting to Deepgram...', flush=True)
        ws = websocket.WebSocketApp(
            url,
            header={'Authorization': f'Token {DEEPGRAM_KEY}'},
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        # Run with timeout
        ws.run_forever(ping_interval=30, ping_timeout=10)

    except Exception as e:
        print(f'Deepgram error: {e}', flush=True)
        emitter.status_changed.emit('error')
        import traceback
        traceback.print_exc()
    finally:
        state.thread_alive = False
        if arecord:
            try:
                arecord.terminate()
                arecord.wait(timeout=2)
            except:
                try:
                    arecord.kill()
                    arecord.wait(timeout=1)  # Reap zombie
                except:
                    pass
        state.kill_proc()
        print('Deepgram stopped', flush=True)

        if not state.is_stopped():
            emitter.thread_died.emit('online')


def parse_speechmatics_transcript(data):
    """(transcript, speaker) from an AddTranscript/AddPartialTranscript message.

    Deliberately defensive about the response shape: the transcript is taken
    from metadata where present and rebuilt from the word results otherwise.
    Guessing wrong would present as a silent recogniser rather than a parse
    bug, and a silent recogniser is the one thing this device must never be.
    """
    text = ''
    meta = data.get('metadata') or {}
    if isinstance(meta, dict):
        text = meta.get('transcript') or ''

    results = data.get('results') or []
    if not text:
        parts = []
        for r in results:
            alts = r.get('alternatives') or []
            if alts:
                parts.append(alts[0].get('content', ''))
        text = ' '.join(p for p in parts if p)

    labels = []
    for r in results:
        alts = r.get('alternatives') or []
        if alts:
            spk = alts[0].get('speaker')
            # UU is Speechmatics for "unknown". Treating it as a speaker would
            # read as a turn change and recolour the captions for no reason.
            if spk and spk != 'UU':
                labels.append(spk)
    speaker = max(set(labels), key=labels.count) if labels else None
    return text, speaker


def speechmatics_thread():
    """Run Speechmatics realtime streaming.

    Protocol differs from Deepgram: JSON control messages plus binary audio
    frames, with the session opened by a StartRecognition message rather than
    by query parameters. Partials and finals arrive as separate message types,
    which maps onto the same provisional-region display as Deepgram's interims.

    Audio IS gated here, and unlike Deepgram that costs nothing: measured with
    clip length controlled, the same two voices give S1/S2/S1 whether streamed
    continuously or with 12s withheld between them. So speaker colours and the
    cost saving coexist on this provider.

    There is no KeepAlive equivalent, but none is needed — a session survived
    40s of receiving nothing and transcribed normally on resume. If a very long
    silence does drop it, the supervisor reconnects during silence, when nobody
    is talking and no captions are lost.
    """
    api_key = CONFIG.get('speechmatics_key')
    if not api_key:
        print('No Speechmatics API key', flush=True)
        emitter.status_changed.emit('no-key')
        state.thread_alive = False
        emitter.thread_died.emit('online')
        return

    print('Starting Speechmatics...', flush=True)
    emitter.status_changed.emit('connecting')
    arecord = None

    try:
        import websocket
        import json

        audio_device = get_audio_device()
        sample_rate = 8000 if state.use_phone_audio else 16000
        print(f"Using audio device: {audio_device} @ {sample_rate}Hz", flush=True)

        detector = SpeechDetector(sample_rate, label=' (speechmatics)')

        for attempt in range(4):
            arecord = subprocess.Popen(
                ['arecord', '-D', audio_device, '-f', 'S16_LE', '-r', str(sample_rate),
                 '-c', '1', '-t', 'raw', '-q'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            state.set_proc(arecord)
            time.sleep(0.3)
            if arecord.poll() is None:
                print(f'arecord ready (attempt {attempt+1})', flush=True)
                break
            arecord = None
            if attempt < 3:
                time.sleep(1)

        if not arecord:
            raise RuntimeError('Could not start arecord after 4 attempts')

        # Same rule as Deepgram: pointless on the phone tap, which has exactly
        # one remote talker, and billed separately.
        diarize = bool(CONFIG.get('speaker_colours', True)) and not state.use_phone_audio

        # `model` is the live field; `operating_point` is deprecated but still
        # accepted. An invalid value is a protocol_error that kills the session
        # outright, so it is checked here rather than left to fail on a device
        # nobody is watching. melia-1 exists but is batch-only.
        model = str(CONFIG.get('speechmatics_model', 'enhanced'))
        if model not in ('standard', 'enhanced'):
            print(f'Speechmatics: model {model!r} is not valid for realtime '
                  f'(use standard or enhanced) — falling back to enhanced', flush=True)
            model = 'enhanced'

        transcription_config = {
            'language': CONFIG.get('speechmatics_language', 'en'),
            'model': model,
            'enable_partials': True,
            'max_delay': float(CONFIG.get('speechmatics_max_delay', 1.0)),
        }
        # Proper end-of-turn signal. Without it there is nothing to hang a
        # paragraph break on, because a Speechmatics final is not the end of an
        # utterance — see below.
        eou = float(CONFIG.get('speechmatics_end_of_utterance_s', 0.8))
        if eou > 0:
            transcription_config['conversation_config'] = {
                'end_of_utterance_silence_trigger': eou,
            }
        if diarize:
            transcription_config['diarization'] = 'speaker'
            # Knobs Deepgram does not offer, aimed at unstable labelling.
            speaker_config = {}
            if CONFIG.get('speechmatics_max_speakers'):
                speaker_config['max_speakers'] = int(CONFIG['speechmatics_max_speakers'])
            if CONFIG.get('speechmatics_prefer_current_speaker'):
                speaker_config['prefer_current_speaker'] = True
            if CONFIG.get('speechmatics_speaker_sensitivity') is not None:
                speaker_config['speaker_sensitivity'] = float(
                    CONFIG['speechmatics_speaker_sensitivity'])
            if speaker_config:
                transcription_config['speaker_diarization_config'] = speaker_config

        start_msg = {
            'message': 'StartRecognition',
            'audio_format': {
                'type': 'raw',
                'encoding': 'pcm_s16le',
                'sample_rate': sample_rate,
            },
            'transcription_config': transcription_config,
        }

        url = CONFIG.get('speechmatics_url') or 'wss://eu2.rt.speechmatics.com/v2'
        emitter.speakers_reset.emit()

        ws_error = threading.Event()
        seq = [0]  # AddAudio messages sent, for EndOfStream

        def extract(data):
            return parse_speechmatics_transcript(data)

        def on_message(ws, message):
            try:
                data = json.loads(message)
                kind = data.get('message')

                if LOG_RAW:
                    print(f'SM<< {message[:800]}', flush=True)

                if kind == 'RecognitionStarted':
                    print('Speechmatics recognition started', flush=True)
                    emitter.status_changed.emit('speechmatics')
                    emitter.mode_ready.emit('online')
                    return
                if kind == 'EndOfUtterance':
                    # An empty final carrying speech_final: add_segment commits
                    # whatever is on screen and starts the next segment in a
                    # new paragraph.
                    emitter.new_segment.emit({
                        'text': '', 'is_final': True,
                        'speech_final': True, 'speaker': None,
                    })
                    return
                if kind == 'Info':
                    # Surfaces the concurrent-session quota, which is small
                    # (2 on a trial account) and is the thing a reconnect loop
                    # will exhaust.
                    if data.get('type') == 'concurrent_session_usage':
                        print(f"Speechmatics sessions: {data.get('usage')} of "
                              f"{data.get('quota')}", flush=True)
                    return

                if kind in ('Error', 'Warning'):
                    print(f"Speechmatics {kind}: {data.get('type')} "
                          f"{data.get('reason', '')}", flush=True)
                    if kind == 'Error':
                        if data.get('type') == 'quota_exceeded':
                            # The previous session has not finished closing.
                            # Reconnecting straight away is guaranteed to fail
                            # again, so wait for it to drain rather than
                            # burning restarts against a wall.
                            print('Speechmatics: waiting for the previous '
                                  'session to close', flush=True)
                            time.sleep(10)
                        ws_error.set()
                        emitter.status_changed.emit('error')
                    return
                if kind not in ('AddTranscript', 'AddPartialTranscript'):
                    return

                is_final = kind == 'AddTranscript'
                text, speaker = extract(data)
                if not text or not text.strip():
                    if is_final:
                        emitter.new_segment.emit({
                            'text': '', 'is_final': True,
                            'speech_final': False, 'speaker': speaker,
                        })
                    return

                state.mark_success()
                if is_final:
                    log_transcript(text.strip(), f'>>> [spk {speaker}]')
                elif LOG_INTERIMS:
                    log_transcript(text.strip(), '...')

                emitter.new_segment.emit({
                    'text': text,
                    'is_final': is_final,
                    # NOT is_final. Verified against the live service: a
                    # Speechmatics final carries only the newly-finalised
                    # words, so a 10s clip produced 22 of them — "He ",
                    # "hoped ", "there would ". Treating each as the end of an
                    # utterance would break the line after every word or two.
                    # EndOfUtterance is what actually ends one.
                    'speech_final': False,
                    'speaker': speaker,
                })
            except Exception as e:
                print(f'Speechmatics parse error: {e}', flush=True)

        def on_error(ws, error):
            print(f'Speechmatics WS error: {error}', flush=True)
            ws_error.set()

        def on_close(ws, code, msg):
            print(f'Speechmatics closed: {code} {msg}', flush=True)

        def on_open(ws):
            print('Speechmatics connected', flush=True)
            ws.send(json.dumps(start_msg))

            def send_audio():
                gating = bool(CONFIG.get('vad_gate', True)) and detector.enabled
                gate = AudioGate(sample_rate, float(CONFIG.get('preroll_s', 0.5)))
                gate_hangover = float(CONFIG.get('gate_hangover_s', 4.0))
                last_speech = time.time()   # start open, as the Deepgram path does
                started = time.time()
                print(f'Gate {"on" if gating else "off"}'
                      + (f': {gate.preroll_s:.2f}s pre-roll, {gate_hangover:.1f}s hangover'
                         if gating else ': streaming continuously'), flush=True)

                while not state.is_stopped() and not ws_error.is_set():
                    state.thread_loop_time = time.time()
                    try:
                        chunk = arecord.stdout.read(gate.bytes_per_chunk)
                        if not chunk:
                            if arecord.poll() is not None:
                                print('Speechmatics: arecord process died', flush=True)
                                break
                            print('send_audio: arecord returned empty data, stopping', flush=True)
                            break
                        if detector.feed(chunk):
                            emitter.vad_state.emit(detector.active)

                        now = time.time()
                        if detector.active:
                            last_speech = now
                        want_open = (not gating) or detector.active \
                            or (now - last_speech) < gate_hangover

                        was_open = gate.open
                        for outgoing in gate.feed(chunk, want_open):
                            ws.send(outgoing, opcode=2)
                            seq[0] += 1
                        if gating and gate.open != was_open:
                            if CONFIG.get('log_vad'):
                                print(f'GATE {now:.3f} '
                                      f'{"open" if gate.open else "closed"}', flush=True)
                            if not gate.open:
                                emit_utterance_end()
                    except Exception as e:
                        print(f'Speechmatics send error: {e}', flush=True)
                        break
                if gating and gate.sent_bytes:
                    elapsed = max(1e-6, time.time() - started)
                    billed = gate.billed_seconds()
                    print(f'Gate closed: sent {billed:.0f}s of {elapsed:.0f}s '
                          f'({100.0 * billed / elapsed:.1f}%)', flush=True)
                try:
                    ws.send(json.dumps({'message': 'EndOfStream',
                                        'last_seq_no': seq[0]}))
                except Exception:
                    pass
                try:
                    ws.close()
                except Exception:
                    pass

            threading.Thread(target=send_audio, daemon=True).start()

        print('Connecting to Speechmatics...', flush=True)
        ws = websocket.WebSocketApp(
            url,
            header={'Authorization': f'Bearer {api_key}'},
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.run_forever(ping_interval=30, ping_timeout=10)

    except Exception as e:
        print(f'Speechmatics error: {e}', flush=True)
        emitter.status_changed.emit('error')
        import traceback
        traceback.print_exc()
    finally:
        state.thread_alive = False
        if arecord:
            try:
                arecord.terminate()
                arecord.wait(timeout=2)
            except Exception:
                try:
                    arecord.kill()
                    arecord.wait(timeout=1)
                except Exception:
                    pass
        state.kill_proc()
        print('Speechmatics stopped', flush=True)

        if not state.is_stopped():
            emitter.thread_died.emit('online')


def assemblyai_thread():
    """Run AssemblyAI real-time streaming via WebSocket"""
    api_key = CONFIG.get('assemblyai_key')
    if not api_key:
        print('No AssemblyAI API key', flush=True)
        emitter.status_changed.emit('no-key')
        state.thread_alive = False
        emitter.thread_died.emit('online')
        return

    print('Starting AssemblyAI...', flush=True)
    emitter.status_changed.emit('assemblyai')
    # thread_alive already set by start_transcription
    arecord = None

    try:
        import websocket
        import json
        import base64

        audio_device = get_audio_device()
        sample_rate = 8000 if state.use_phone_audio else 16000
        print(f"Using audio device: {audio_device} @ {sample_rate}Hz", flush=True)

        # Start arecord with retry
        for attempt in range(4):
            arecord = subprocess.Popen(
                ['arecord', '-D', audio_device, '-f', 'S16_LE', '-r', str(sample_rate), '-c', '1', '-t', 'raw'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            state.set_proc(arecord)
            time.sleep(0.3)
            test_data = arecord.stdout.read(3200)
            if len(test_data) > 0:
                print(f'arecord ready (attempt {attempt+1})', flush=True)
                break
            else:
                arecord.terminate()
                arecord = None
                if attempt < 3:
                    time.sleep(1)

        if not arecord:
            raise RuntimeError('Could not start arecord after 4 attempts')

        url = f'wss://api.assemblyai.com/v2/realtime/ws?sample_rate={sample_rate}'
        ws_error = threading.Event()

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data.get('message_type') == 'FinalTranscript':
                    text = data.get('text', '').strip()
                    if text:
                        log_transcript(text)
                        state.mark_success()
                        emitter.new_text.emit(text + '\n')
                elif data.get('message_type') == 'PartialTranscript':
                    text = data.get('text', '').strip()
                    if text:
                        state.mark_success()
            except Exception as e:
                print(f'AssemblyAI parse error: {e}', flush=True)

        def on_error(ws, error):
            print(f'AssemblyAI WS error: {error}', flush=True)
            ws_error.set()

        def on_open(ws):
            print('AssemblyAI connected', flush=True)
            emitter.mode_ready.emit('online')


            # Send initial audio
            ws.send(json.dumps({'audio_data': base64.b64encode(test_data).decode()}))

            def send_audio():
                while not state.is_stopped() and not ws_error.is_set():
                    state.thread_loop_time = time.time()
                    try:
                        chunk = arecord.stdout.read(3200)
                        if chunk:
                            ws.send(json.dumps({'audio_data': base64.b64encode(chunk).decode()}))
                        else:
                            if arecord.poll() is not None:
                                print('AssemblyAI: arecord process died', flush=True)
                                break
                            print('send_audio: arecord returned empty data, stopping', flush=True); break
                    except Exception as e:
                        print(f"AssemblyAI send error: {e}", flush=True)
                        break
                try:
                    ws.send(json.dumps({'terminate_session': True}))
                    ws.close()
                except:
                    pass

            threading.Thread(target=send_audio, daemon=True).start()

        def on_close(ws, code, msg):
            print(f'AssemblyAI closed: {code} {msg}', flush=True)

        print('Connecting to AssemblyAI...', flush=True)
        ws = websocket.WebSocketApp(
            url,
            header={'Authorization': api_key},
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.run_forever(ping_interval=30, ping_timeout=10)

    except Exception as e:
        print(f'AssemblyAI error: {e}', flush=True)
        emitter.status_changed.emit('error')
        import traceback
        traceback.print_exc()
    finally:
        state.thread_alive = False
        if arecord:
            try:
                arecord.terminate()
                arecord.wait(timeout=2)
            except:
                try:
                    arecord.kill()
                    arecord.wait(timeout=1)  # Reap zombie
                except:
                    pass
        state.kill_proc()
        print('AssemblyAI stopped', flush=True)
        if not state.is_stopped():
            emitter.thread_died.emit('online')


def azure_thread():
    """Run Azure Speech Services with SDK streaming"""
    api_key = CONFIG.get('azure_key')
    region = CONFIG.get('azure_region', 'uksouth')
    if not api_key:
        print('No Azure API key', flush=True)
        emitter.status_changed.emit('no-key')
        state.thread_alive = False
        emitter.thread_died.emit('online')
        return

    print('Starting Azure Speech...', flush=True)
    emitter.status_changed.emit('azure')
    # thread_alive already set by start_transcription

    try:
        import azure.cognitiveservices.speech as speechsdk

        speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
        speech_config.speech_recognition_language = 'en-GB'

        audio_device = get_audio_device()
        # Azure SDK can use ALSA device directly
        audio_config = speechsdk.audio.AudioConfig(device_name=audio_device)

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        def on_recognized(evt):
            text = evt.result.text.strip()
            if text:
                log_transcript(text)
                state.mark_success()
                emitter.new_text.emit(text + '\n')

        def on_recognizing(evt):
            if evt.result.text.strip():
                state.mark_success()

        def on_canceled(evt):
            print(f'Azure canceled: {evt.result.cancellation_details.reason}', flush=True)
            if evt.result.cancellation_details.error_details:
                print(f'Azure error: {evt.result.cancellation_details.error_details}', flush=True)

        def on_session_started(evt):
            print('Azure session started', flush=True)
            emitter.mode_ready.emit('online')


        recognizer.recognized.connect(on_recognized)
        recognizer.recognizing.connect(on_recognizing)
        recognizer.canceled.connect(on_canceled)
        recognizer.session_started.connect(on_session_started)

        print('Starting continuous recognition...', flush=True)
        recognizer.start_continuous_recognition()

        # Wait until stopped
        while not state.is_stopped():
            state.thread_loop_time = time.time()
            time.sleep(0.5)

        recognizer.stop_continuous_recognition()

    except ImportError:
        print('Azure Speech SDK not installed. Install with: pip install azure-cognitiveservices-speech', flush=True)
        emitter.status_changed.emit('error')
    except Exception as e:
        print(f'Azure error: {e}', flush=True)
        emitter.status_changed.emit('error')
        import traceback
        traceback.print_exc()
    finally:
        state.thread_alive = False
        state.kill_proc()
        print('Azure stopped', flush=True)
        if not state.is_stopped():
            emitter.thread_died.emit('online')


def _chunked_api_thread(provider_name, transcribe_fn):
    """Shared logic for providers that use chunked batch transcription (Google, OpenAI, Groq).

    Records audio in chunks and sends each chunk to transcribe_fn(audio_bytes, sample_rate)
    which should return the transcribed text or empty string.
    """
    print(f'Starting {provider_name}...', flush=True)
    emitter.status_changed.emit(provider_name.lower())
    # thread_alive already set by start_transcription
    arecord = None

    try:
        audio_device = get_audio_device()
        sample_rate = 8000 if state.use_phone_audio else 16000
        detector = SpeechDetector(sample_rate, label=f' ({provider_name.lower()})')
        chunk_seconds = 4
        chunk_bytes = sample_rate * 2 * chunk_seconds  # 16-bit mono
        print(f"Using audio device: {audio_device} @ {sample_rate}Hz", flush=True)

        # Start arecord with retry
        for attempt in range(4):
            arecord = subprocess.Popen(
                ['arecord', '-D', audio_device, '-f', 'S16_LE', '-r', str(sample_rate), '-c', '1', '-t', 'raw'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            state.set_proc(arecord)
            time.sleep(0.3)
            if arecord.poll() is None:
                print(f'arecord ready (attempt {attempt+1})', flush=True)
                break
            else:
                arecord = None
                if attempt < 3:
                    time.sleep(1)

        if not arecord:
            raise RuntimeError('Could not start arecord after 4 attempts')

        emitter.mode_ready.emit('online')


        buffer = b''
        speech_in_chunk = False
        while not state.is_stopped():
            state.thread_loop_time = time.time()
            data = arecord.stdout.read(3200)
            if not data:
                if arecord.poll() is not None:
                    print(f'{provider_name}: arecord process died', flush=True)
                    break
                print(f'{provider_name}: arecord returned empty data, stopping', flush=True); break
                continue

            buffer += data
            if detector.feed(data):
                emitter.vad_state.emit(detector.active)
            if detector.active:
                # Over-inclusive on purpose: sending a chunk that turns out to
                # be quiet costs a fraction of a penny, dropping one that had a
                # short word in it costs the word.
                speech_in_chunk = True

            if len(buffer) >= chunk_bytes:
                audio_chunk = buffer[:chunk_bytes]
                buffer = buffer[chunk_bytes:]

                # Skip chunks containing no speech, to avoid paying for silence.
                # Was a fixed amplitude threshold, which meant turning the mic
                # gain up defeated it entirely.
                had_speech = speech_in_chunk or detector.active
                speech_in_chunk = False
                if detector.enabled and not had_speech:
                    continue

                try:
                    text = transcribe_fn(audio_chunk, sample_rate)
                    if text and text.strip():
                        text = text.strip()
                        log_transcript(text)
                        state.mark_success()
                        emitter.new_text.emit(text + '\n')
                except Exception as e:
                    print(f'{provider_name} API error: {e}', flush=True)

    except Exception as e:
        print(f'{provider_name} error: {e}', flush=True)
        emitter.status_changed.emit('error')
        import traceback
        traceback.print_exc()
    finally:
        state.thread_alive = False
        if arecord:
            try:
                arecord.terminate()
                arecord.wait(timeout=2)
            except:
                try:
                    arecord.kill()
                    arecord.wait(timeout=1)  # Reap zombie
                except:
                    pass
        state.kill_proc()
        print(f'{provider_name} stopped', flush=True)
        if not state.is_stopped():
            emitter.thread_died.emit('online')


def _make_wav(raw_audio, sample_rate):
    """Wrap raw PCM bytes in a minimal WAV header."""
    import struct as st
    num_samples = len(raw_audio) // 2
    wav = bytearray()
    wav += b'RIFF'
    wav += st.pack('<I', 36 + len(raw_audio))
    wav += b'WAVE'
    wav += b'fmt '
    wav += st.pack('<I', 16)           # chunk size
    wav += st.pack('<H', 1)            # PCM format
    wav += st.pack('<H', 1)            # mono
    wav += st.pack('<I', sample_rate)
    wav += st.pack('<I', sample_rate * 2)  # byte rate
    wav += st.pack('<H', 2)            # block align
    wav += st.pack('<H', 16)           # bits per sample
    wav += b'data'
    wav += st.pack('<I', len(raw_audio))
    wav += raw_audio
    return bytes(wav)


def google_thread():
    """Google Cloud Speech-to-Text via REST API (chunked)"""
    api_key = CONFIG.get('google_key')
    if not api_key:
        print('No Google Cloud API key', flush=True)
        emitter.status_changed.emit('no-key')
        state.thread_alive = False
        emitter.thread_died.emit('online')
        return

    import requests
    import base64

    def transcribe(audio_bytes, sample_rate):
        audio_b64 = base64.b64encode(audio_bytes).decode()
        resp = requests.post(
            f'https://speech.googleapis.com/v1/speech:recognize?key={api_key}',
            json={
                'config': {
                    'encoding': 'LINEAR16',
                    'sampleRateHertz': sample_rate,
                    'languageCode': 'en-GB',
                    'enableAutomaticPunctuation': True,
                },
                'audio': {'content': audio_b64},
            },
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get('results', [])
        return ' '.join(
            r['alternatives'][0]['transcript']
            for r in results
            if r.get('alternatives')
        )

    _chunked_api_thread('Google', transcribe)


def openai_thread():
    """OpenAI Whisper API (chunked)"""
    api_key = CONFIG.get('openai_key')
    if not api_key:
        print('No OpenAI API key', flush=True)
        emitter.status_changed.emit('no-key')
        state.thread_alive = False
        emitter.thread_died.emit('online')
        return

    import requests

    def transcribe(audio_bytes, sample_rate):
        wav_data = _make_wav(audio_bytes, sample_rate)
        resp = requests.post(
            'https://api.openai.com/v1/audio/transcriptions',
            headers={'Authorization': f'Bearer {api_key}'},
            files={'file': ('chunk.wav', wav_data, 'audio/wav')},
            data={'model': 'whisper-1', 'language': 'en'},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get('text', '')

    _chunked_api_thread('OpenAI', transcribe)


def groq_thread():
    """Groq Whisper API (chunked) — free tier, very fast"""
    api_key = CONFIG.get('groq_key')
    if not api_key:
        print('No Groq API key', flush=True)
        emitter.status_changed.emit('no-key')
        state.thread_alive = False
        emitter.thread_died.emit('online')
        return

    import requests

    def transcribe(audio_bytes, sample_rate):
        wav_data = _make_wav(audio_bytes, sample_rate)
        resp = requests.post(
            'https://api.groq.com/openai/v1/audio/transcriptions',
            headers={'Authorization': f'Bearer {api_key}'},
            files={'file': ('chunk.wav', wav_data, 'audio/wav')},
            data={'model': 'whisper-large-v3', 'language': 'en'},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get('text', '')

    _chunked_api_thread('Groq', transcribe)


def interfaze_thread():
    """Interfaze STT API (chunked, OpenAI-compatible)"""
    api_key = CONFIG.get('interfaze_key')
    if not api_key:
        print('No Interfaze API key', flush=True)
        emitter.status_changed.emit('no-key')
        state.thread_alive = False
        emitter.thread_died.emit('online')
        return

    import requests

    def transcribe(audio_bytes, sample_rate):
        wav_data = _make_wav(audio_bytes, sample_rate)
        resp = requests.post(
            'https://api.interfaze.ai/v1/audio/transcriptions',
            headers={'Authorization': f'Bearer {api_key}'},
            files={'file': ('chunk.wav', wav_data, 'audio/wav')},
            data={'model': 'interfaze-beta', 'language': 'en'},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get('text', '')

    _chunked_api_thread('Interfaze', transcribe)


def start_transcription(mode):
    """Start transcription with cleanup"""
    state.kill_proc()  # Targeted kill only — no blanket pkill
    time.sleep(0.3)
    ensure_mic_volume()
    state.clear_stop()
    state.last_text_time = 0
    state.mode = mode
    state.reset_success_timer()
    state.thread_alive = True
    state.thread_loop_time = time.time()
    state.provider_ready = False
    gen = state.next_generation()
    print(f"Starting transcription gen={gen} mode={mode}", flush=True)

    if mode == 'online':
        provider = CONFIG.get('stt_provider', 'deepgram')
        provider_threads = {
            'deepgram': deepgram_thread,
            'speechmatics': speechmatics_thread,
            'assemblyai': assemblyai_thread,
            'azure': azure_thread,
            'groq': groq_thread,
            'interfaze': interfaze_thread,
            'openai': openai_thread,
            'google': google_thread,
        }
        target = provider_threads.get(provider, deepgram_thread)
        print(f'Starting online transcription with {provider}', flush=True)
        threading.Thread(target=target, daemon=True).start()
    else:
        offline_model = CONFIG.get('offline_model', 'faster-whisper')
        offline_threads = {
            'faster-whisper': faster_whisper_thread,
            'vosk': vosk_thread,
            'whisper-cpp': whisper_thread,
        }
        target = offline_threads.get(offline_model, faster_whisper_thread)
        print(f'Starting offline transcription with {offline_model}', flush=True)
        threading.Thread(target=target, daemon=True).start()



def stop_transcription():
    """Stop transcription cleanly"""
    state.stop()
    state.kill_proc()  # Targeted kill only — no blanket pkill
    time.sleep(0.5)



def switch_mode(new_mode):
    """Switch modes with proper cleanup"""
    if new_mode != state.mode:
        print(f'Switching from {state.mode} to {new_mode}', flush=True)
        emitter.status_changed.emit('switching')
        stop_transcription()
        time.sleep(0.5)
        start_transcription(new_mode)
        emitter.mode_changed.emit(new_mode)


class FlipFlap(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(420, 450)
        self._text = '00'
        self._dimmed = False
        font_id = QFontDatabase.addApplicationFont(FONT_PATH)
        if font_id >= 0:
            self._font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        else:
            self._font_family = 'Arial Narrow'

    def set_text(self, text):
        if text != self._text:
            self._text = text
            self.update()

    def set_dimmed(self, dimmed):
        self._dimmed = dimmed
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        gap = 8
        flap_h = (h - gap) // 2
        r = 15
        text_col = QColor('#555') if self._dimmed else QColor('#f0f0f0')
        painter.setPen(QPen(QColor('#444'), 2))
        top_g = QLinearGradient(0, 0, 0, flap_h)
        top_g.setColorAt(0, QColor('#3d3d3d'))
        top_g.setColorAt(0.9, QColor('#2a2a2a'))
        top_g.setColorAt(1, QColor('#222'))
        painter.setBrush(top_g)
        painter.drawRoundedRect(0, 0, w, flap_h, r, r)
        bot_g = QLinearGradient(0, flap_h + gap, 0, h)
        bot_g.setColorAt(0, QColor('#1a1a1a'))
        bot_g.setColorAt(0.1, QColor('#252525'))
        bot_g.setColorAt(1, QColor('#333'))
        painter.setBrush(bot_g)
        painter.drawRoundedRect(0, flap_h + gap, w, flap_h, r, r)
        font = QFont(self._font_family, 280, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(text_col)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(self._text)
        tx = (w - tw) // 2
        cap_h = fm.capHeight()
        ty = (h // 2) + (cap_h // 2)
        painter.setClipRect(0, 0, w, flap_h)
        painter.drawText(tx, ty, self._text)
        painter.setClipRect(0, flap_h + gap, w, flap_h)
        painter.drawText(tx, ty, self._text)
        painter.setClipping(False)
        painter.setPen(QPen(QColor(255, 255, 255, 40), 2))
        painter.drawLine(r, 2, w - r, 2)


class ClockView(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet('background: black;')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 15, 25, 15)

        # Status belongs here as much as on the caption view. A fault stops text
        # arriving, no text for 90s brings up this clock, and without an
        # indicator here the clock would hide the very warning that explains it.
        status_row = QHBoxLayout()
        status_row.addStretch()
        self.status_label = QLabel('')
        self.status_label.setStyleSheet('background: transparent;')
        status_row.addWidget(self.status_label)
        layout.addLayout(status_row)

        layout.addStretch()
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setSpacing(30)
        self.hours = FlipFlap()
        self.mins = FlipFlap()
        row_layout.addWidget(self.hours)
        row_layout.addWidget(self.mins)
        layout.addWidget(row, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()

        self.dimmed = False
        self._status = ''
        self._speaking = False

    def set_status(self, status):
        self._status = status
        self._render_status()

    def set_speaking(self, speaking):
        self._speaking = speaking
        self._render_status()

    def _render_status(self):
        text, style = status_display(self._status, self._speaking)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(style)

    def update_time(self):
        now = datetime.now()
        self.hours.set_text(now.strftime('%H'))
        self.mins.set_text(now.strftime('%M'))
        h = now.hour
        night = h >= 22 or h < 7
        if night != self.dimmed:
            self.dimmed = night
            self.hours.set_dimmed(night)
            self.mins.set_dimmed(night)


class CaptionView(QWidget):
    def __init__(self):
        super().__init__()
        self.font_sizes = {'S': 28, 'M': 36, 'L': 48}
        self.current_size = 'M'
        self.color_schemes = [
            ('W/B', '#ffffff', '#000000'),
            ('B/W', '#000000', '#ffffff'),
            ('Y/B', '#ffff00', '#000000'),
            ('G/B', '#00ff00', '#000000'),
        ]
        self.current_scheme = 0
        self.current_mode = CONFIG.get('speech_mode', 'offline')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 15, 25, 15)
        top_bar = QHBoxLayout()

        self.size_buttons = {}
        for size in ['S', 'M', 'L']:
            btn = QLabel(size)
            btn.setFixedSize(50, 50)
            btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
            btn.mousePressEvent = lambda e, s=size: self.set_size(s)
            self.size_buttons[size] = btn
            top_bar.addWidget(btn)

        spacer = QLabel('  ')
        spacer.setFixedWidth(30)
        top_bar.addWidget(spacer)

        self.color_buttons = []
        for i, (name, text_col, bg_col) in enumerate(self.color_schemes):
            btn = QLabel('A')
            btn.setFixedSize(50, 50)
            btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
            btn.setStyleSheet(f'background: {bg_col}; color: {text_col}; border-radius: 25px; font-size: 24px; font-weight: bold; border: 2px solid #444;')
            btn.mousePressEvent = lambda e, idx=i: self.set_color(idx)
            self.color_buttons.append(btn)
            top_bar.addWidget(btn)

        top_bar.addStretch()

        self.phone_icon = QLabel('📞')
        self.phone_icon.setStyleSheet('color: #00ff00; background: transparent; font-size: 40px;')
        self.phone_icon.hide()
        top_bar.addWidget(self.phone_icon)

        self.mode_btn = QLabel('OFFLINE')
        self.mode_btn.setFixedSize(140, 50)
        self.mode_btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_btn.mousePressEvent = self.toggle_mode
        self.update_mode_button()
        top_bar.addWidget(self.mode_btn)

        self.status_label = QLabel('')
        top_bar.addWidget(self.status_label)

        layout.addLayout(top_bar)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.text.setStyleSheet('background: black; color: white; border: none;')
        self.text.verticalScrollBar().setStyleSheet(
            'QScrollBar:vertical { background: #222; width: 30px; border-radius: 15px; }'
            'QScrollBar::handle:vertical { background: #666; min-height: 60px; border-radius: 15px; }'
            'QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }'
        )
        self.text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        QScroller.grabGesture(self.text.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture)
        layout.addWidget(self.text)

        self.update_size_buttons()
        self.update_color_buttons()
        self.set_size('M')
        self.set_color(0)
        self._waiting_for_ready = False
        self._last_text_time = 0
        self._status = ''
        self._speaking = False
        self._prov_start = None      # doc position where uncommitted (interim) text begins
        self._last_speaker = None
        self._colour_idx = 0
        self._last_speech_final = False  # did the last commit end an utterance?

    def toggle_mode(self, event):
        new_mode = 'online' if self.current_mode == 'offline' else 'offline'
        self.mode_btn.setText('⏳ WAIT...')
        self.mode_btn.setStyleSheet(
            'background: #333300; color: #ffff00; border-radius: 10px; '
            'font-size: 20px; font-weight: bold; border: 3px solid #ffff00;'
        )
        self.mode_btn.repaint()
        QApplication.processEvents()
        self.current_mode = new_mode
        self._waiting_for_ready = True
        switch_mode(new_mode)

    def update_mode_button(self):
        if self.current_mode == 'online':
            self.mode_btn.setText('🌐 ONLINE')
            self.mode_btn.setStyleSheet(
                'background: #004400; color: #00ff00; border-radius: 10px; '
                'font-size: 20px; font-weight: bold; border: 3px solid #00ff00;'
            )
        else:
            self.mode_btn.setText('💾 OFFLINE')
            self.mode_btn.setStyleSheet(
                'background: #442200; color: #ffaa00; border-radius: 10px; '
                'font-size: 20px; font-weight: bold; border: 3px solid #ffaa00;'
            )

    def set_mode(self, mode):
        self.current_mode = mode
        self.update_mode_button()

    def set_size(self, size):
        self.current_size = size
        self.text.setFont(QFont('Helvetica', self.font_sizes[size], QFont.Weight.Bold))
        self.update_size_buttons()

    def set_color(self, idx):
        self.current_scheme = idx
        name, text_col, bg_col = self.color_schemes[idx]
        self.text.setStyleSheet(f'background: {bg_col}; color: {text_col}; border: none;')
        self.setStyleSheet(f'background: {bg_col};')
        self.update_color_buttons()

    def update_size_buttons(self):
        for size, btn in self.size_buttons.items():
            if size == self.current_size:
                btn.setStyleSheet('background: #444; color: white; border-radius: 25px; font-size: 24px; font-weight: bold;')
            else:
                btn.setStyleSheet('background: #222; color: #888; border-radius: 25px; font-size: 24px;')

    def update_color_buttons(self):
        for i, btn in enumerate(self.color_buttons):
            name, text_col, bg_col = self.color_schemes[i]
            if i == self.current_scheme:
                btn.setStyleSheet(f'background: {bg_col}; color: {text_col}; border-radius: 25px; font-size: 24px; font-weight: bold; border: 3px solid #0af;')
            else:
                btn.setStyleSheet(f'background: {bg_col}; color: {text_col}; border-radius: 25px; font-size: 24px; font-weight: bold; border: 2px solid #444;')

    def set_status(self, status):
        self._status = status
        self._render_status()

    def set_speaking(self, speaking):
        self._speaking = speaking
        self._render_status()

    def _render_status(self):
        text, style = status_display(self._status, self._speaking)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(style)

    def _trim_if_needed(self):
        """Trim old text to prevent unbounded memory growth.

        Only safe when nothing is provisional — removing blocks invalidates
        self._prov_start.
        """
        doc = self.text.document()
        if doc.blockCount() > 250:
            trim_cursor = QTextCursor(doc)
            trim_cursor.movePosition(QTextCursor.MoveOperation.Start)
            # Select first 50 blocks for removal
            for _ in range(50):
                trim_cursor.movePosition(QTextCursor.MoveOperation.NextBlock, QTextCursor.MoveMode.KeepAnchor)
            trim_cursor.removeSelectedText()
            trim_cursor.deleteChar()  # Remove the leftover empty block

    def _speaker_palette(self):
        _, _, bg_col = self.color_schemes[self.current_scheme]
        if bg_col.lower() in ('#ffffff', '#fff'):
            return SPEAKER_PALETTE_LIGHT
        return SPEAKER_PALETTE_DARK

    def reset_speakers(self):
        """A new STT session began — Deepgram's speaker label space restarts.

        Resuming the colour cycle across that boundary would be false
        precision, so the cycle restarts at turn A. Any text left provisional
        by the previous session stays on screen and is committed as-is.
        """
        self._prov_start = None
        self._last_speaker = None
        self._colour_idx = 0
        self._last_speech_final = True  # next segment starts a fresh paragraph

    def add_segment(self, seg):
        """Streaming path: interim results overwrite, finals commit.

        The provisional region runs from self._prov_start to end of document and
        INCLUDES its leading separator, so a speaker change detected at commit
        time can retroactively turn a space into a paragraph break.
        """
        text = (seg.get('text') or '').strip()
        is_final = bool(seg.get('is_final'))
        speech_final = bool(seg.get('speech_final'))
        speaker = seg.get('speaker')

        if self._waiting_for_ready:
            self._waiting_for_ready = False
            self.update_mode_button()

        if not text:
            # Deepgram sends empty finals for segments containing no speech.
            # Commit whatever interim text is already on screen rather than
            # deleting it — those were real words.
            if is_final:
                self._prov_start = None
                self._last_speech_final = speech_final
            return

        c = self.text.textCursor()
        if self._prov_start is None:
            self._trim_if_needed()
            c.movePosition(QTextCursor.MoveOperation.End)
            self._prov_start = c.position()
        else:
            c.setPosition(self._prov_start)
            c.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
            if c.hasSelection():
                c.removeSelectedText()

        palette = self._speaker_palette()
        colour_idx = self._colour_idx
        turn_change = False
        # Speaker labels firm up as evidence arrives, so only act on finals.
        if is_final and speaker is not None and speaker != self._last_speaker:
            if self._last_speaker is not None:
                colour_idx = (self._colour_idx + 1) % len(palette)
                turn_change = True

        now = time.time()
        at_start = (self._prov_start == 0)
        if not at_start:
            if turn_change:
                sep = '\n\n'      # speaker change — the strongest break
            elif self._last_speech_final:
                sep = '\n'        # end of an utterance
            elif self._last_text_time > 0 and (now - self._last_text_time) > 2:
                sep = '\n'        # long gap and no speech_final arrived
            else:
                sep = ' '
            c.insertText(sep)

        if SPEAKER_COLOURS:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(palette[colour_idx]))
            marker = SPEAKER_MARKER if (turn_change or at_start) else ''
            c.insertText(marker + text, fmt)
        else:
            # No explicit character format. An explicit one overrides the
            # widget stylesheet, which is why the colour-scheme buttons stopped
            # changing the text colour once speaker colouring arrived.
            c.insertText(text)

        if is_final:
            self._prov_start = None
            self._colour_idx = colour_idx
            self._last_speech_final = speech_final
            if speaker is not None:
                self._last_speaker = speaker

        self._last_text_time = now
        self.text.setTextCursor(c)
        self.text.ensureCursorVisible()

    def add_text(self, t):
        if self._waiting_for_ready:
            self._waiting_for_ready = False
            self.update_mode_button()
        # Any provisional text from a streaming session is committed where it
        # stands. Leaving _prov_start set would let a later interim delete
        # everything appended here — e.g. a whole offline fallback session.
        self._prov_start = None
        self._trim_if_needed()
        c = self.text.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        now = time.time()
        # A trailing newline means the provider signalled end-of-utterance
        # (Deepgram's speech_final). How much space that is worth is decided
        # here rather than by the provider.
        ends_utterance = t.endswith('\n')
        t = t.rstrip('\n')
        existing = self.text.toPlainText()
        if existing:
            if existing.endswith('\n'):
                pass  # previous utterance already left the break — adding a
                      # separator here is what put a stray space on every line
            elif self._last_text_time > 0 and (now - self._last_text_time) > 2:
                c.insertText('\n\n')
            else:
                c.insertText(' ')
        c.insertText(t)
        if ends_utterance:
            c.insertText('\n\n')  # blank line between utterances
        self._last_text_time = now
        self.text.setTextCursor(c)
        self.text.ensureCursorVisible()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Gramps')
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.showFullScreen()

        self.stack = QStackedWidget()
        self.stack.setStyleSheet('background: black; border: none;')
        self.stack.setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet('background: black; border: none;')
        self.setCentralWidget(self.stack)

        self.clock_view = ClockView()
        self.caption_view = CaptionView()

        self.stack.addWidget(self.clock_view)
        self.stack.addWidget(self.caption_view)

        self.last_activity = 0
        self.phone_was_active = False
        self._pending_restart = None

        # Main tick timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)

        # Phone check timer
        self.mute_timer = QTimer()
        self.mute_timer.timeout.connect(self.check_muted)
        self.mute_timer.start(500)

        # Health check timer - monitors transcription thread
        self.health_timer = QTimer()
        self.health_timer.timeout.connect(self.health_check)
        self.health_timer.start(5000)

        # Connect signals
        emitter.new_text.connect(self.on_text)
        emitter.new_segment.connect(self.on_segment)
        emitter.speakers_reset.connect(self.caption_view.reset_speakers)
        emitter.vad_state.connect(self.on_vad_state)
        emitter.status_changed.connect(self.on_status_changed)
        emitter.mode_changed.connect(self.on_mode_changed)
        emitter.mode_ready.connect(self.on_mode_ready)
        emitter.thread_died.connect(self.on_thread_died)

        self.stack.setCurrentIndex(0)

    def signal_activity(self):
        self.last_activity = time.time()
        if self.stack.currentIndex() != 1:
            self.stack.setCurrentIndex(1)

    def tick(self):
        self.clock_view.update_time()
        if self.last_activity > 0:
            age = time.time() - self.last_activity
            if age > SILENCE_TIMEOUT and self.stack.currentIndex() == 1:
                self.stack.setCurrentIndex(0)

    def health_check(self):
        """Monitor transcription health and restart if needed"""
        if state.is_restarting() or state.is_stopped():
            return

        problem = None

        if not state.thread_alive:
            problem = "thread dead"
        elif state.provider_ready and not state.proc_alive():
            problem = "arecord subprocess dead"
        elif state.provider_ready and state.thread_loop_time > 0 and (time.time() - state.thread_loop_time) > 120:
            problem = f"thread stuck (no loop for {time.time() - state.thread_loop_time:.0f}s)"
        elif state.last_text_time > 0 and state.mode == 'online':
            stale_time = time.time() - state.last_text_time
            if stale_time > 600:
                problem = f"no transcription for {stale_time:.0f}s"

        if not problem:
            # Retry online mode if we previously fell back to offline
            if (state._retry_online_at > 0
                    and state.mode == 'offline'
                    and time.time() > state._retry_online_at
                    and not state.is_restarting()):
                state._retry_online_at = 0
                state._retry_backoff = 600  # Reset backoff
                state.reset_restart_count()
                print("Retrying online mode...", flush=True)
                stop_transcription()
                QTimer.singleShot(2000, lambda: start_transcription('online'))
                return

            # Write heartbeat file (primary health signal for monitor)
            try:
                elapsed = f"{time.time() - state.last_text_time:.0f}s" if state.last_text_time > 0 else "never"
                loop_age = f"{time.time() - state.thread_loop_time:.0f}s" if state.thread_loop_time > 0 else "never"
                with open('/tmp/caption_heartbeat', 'w') as f:
                    f.write(f"{time.time()} gen={state.generation} mode={state.mode} thread={state.thread_alive} proc={state.proc_alive()} last_text={elapsed} loop={loop_age}")
            except Exception:
                pass

            # Print to log every 5 min (every 10th check) for diagnostics
            self._hb_count = getattr(self, '_hb_count', 0) + 1
            if self._hb_count % 10 == 1:
                print(f"heartbeat: gen={state.generation} mode={state.mode}", flush=True)
            return

        if problem:
            if state.can_restart():
                state.set_restarting(True)
                count = state.increment_restart()
                gen = state.generation
                print(f"Health check: {problem}, restarting (attempt {count}, gen={gen})...", flush=True)
                self.caption_view.set_status('restarting')
                mode = state.mode
                if state.thread_alive:
                    stop_transcription()
                def do_health_restart(expected_gen=gen):
                    if state.generation != expected_gen:
                        print(f"Health restart skipped: gen changed {expected_gen}->{state.generation}", flush=True)
                        state.set_restarting(False)
                        return
                    start_transcription(mode)
                    state.set_restarting(False)
                QTimer.singleShot(2000, do_health_restart)
            else:
                now = time.time()
                # First time giving up? Record when
                if state._gave_up_at == 0:
                    state._gave_up_at = now
                    print(f"Health check: {problem}, max restarts exceeded, waiting 30 min to retry", flush=True)
                    self.caption_view.set_status('error')
                elif now - state._gave_up_at > 1800:
                    # 30 minutes passed — reset and try again
                    print(f"Health check: 30 min cooldown expired, resetting and retrying", flush=True)
                    state._gave_up_at = 0
                    state.reset_restart_count()
                    state.set_restarting(True)
                    self.caption_view.set_status('restarting')
                    mode = state.mode or 'offline'
                    stop_transcription()
                    gen = state.generation
                    def do_cooldown_restart(expected_gen=gen):
                        if state.generation != expected_gen:
                            print(f"Cooldown restart skipped: gen changed {expected_gen}->{state.generation}", flush=True)
                            state.set_restarting(False)
                            return
                        start_transcription(mode)
                        state.set_restarting(False)
                    QTimer.singleShot(3000, do_cooldown_restart)
                # Always write heartbeat even when in gave-up state
                try:
                    elapsed = f"{time.time() - state.last_text_time:.0f}s" if state.last_text_time > 0 else "never"
                    with open('/tmp/caption_heartbeat', 'w') as f:
                        f.write(f"{time.time()} gen={state.generation} mode={state.mode} thread={state.thread_alive} proc={state.proc_alive()} last_text={elapsed} status=gave_up")
                except Exception:
                    pass


    def on_thread_died(self, mode):
        """Handle thread death signal with automatic fallback"""
        if state.is_restarting():
            print("Restart already in progress, skipping thread_died signal", flush=True)
            return
        if state.is_stopped():
            return
        if state.can_restart():
            state.set_restarting(True)
            count = state.increment_restart()
            gen = state.generation

            if mode == 'online' and count >= 3 and not state.use_phone_audio:
                print(f"Online mode failed {count} times, falling back to offline (gen={gen})", flush=True)
                self.caption_view.set_status('fallback')
                def do_fallback(expected_gen=gen):
                    if state.generation != expected_gen:
                        state.set_restarting(False)
                        return
                    state.reset_restart_count()
                    state._retry_online_at = time.time() + state._retry_backoff
                    print(f"Will retry online mode in {state._retry_backoff}s", flush=True)
                    state._retry_backoff = min(state._retry_backoff * 2, 3600)  # Double up to 1 hour max
                    start_transcription('offline')
                    self.caption_view.set_mode('offline')
                    state.set_restarting(False)
                QTimer.singleShot(2000, do_fallback)
            else:
                print(f"Thread died (gen={gen}), scheduling restart (attempt {count})...", flush=True)
                self.caption_view.set_status('restarting')
                def do_restart(expected_gen=gen):
                    if state.generation != expected_gen:
                        state.set_restarting(False)
                        return
                    start_transcription(mode)
                    state.set_restarting(False)
                QTimer.singleShot(3000, do_restart)
        else:
            now = time.time()
            if state._gave_up_at == 0:
                state._gave_up_at = now
                print(f"Thread died, max restarts exceeded, waiting 30 min to retry", flush=True)
                self.caption_view.set_status('error')


    def check_muted(self):
        """Check phone status and handle audio device switching"""
        try:
            phone_active = False
            if os.path.exists(PHONE_MUTED_FILE):
                with open(PHONE_MUTED_FILE, 'r') as f:
                    c = f.read().strip()
                    phone_active = c == '1'

            # If using phone audio, check for silence to end call
            if state.use_phone_audio and state.last_phone_speech > 0:
                silence_duration = time.time() - state.last_phone_speech
                if silence_duration > PHONE_SILENCE_TIMEOUT:
                    print(f'Phone silent for {silence_duration:.0f}s - ending call', flush=True)
                    write_phone_status(False)
                    phone_active = False

            if phone_active:
                self.caption_view.phone_icon.show()
                if not self.phone_was_active:
                    state.use_phone_audio = True
                    state.last_phone_speech = time.time()
                    state.last_text_time = time.time()
                    if not state.is_restarting():
                        print('Phone active - restarting with phone recorder', flush=True)
                        state.set_restarting(True)
                        stop_transcription()
                        def do_phone_start():
                            start_transcription(state.mode)
                            state.set_restarting(False)
                        QTimer.singleShot(500, do_phone_start)
                    else:
                        print('Phone active - restart in progress, waiting', flush=True)
                self.phone_was_active = True
            else:
                self.caption_view.phone_icon.hide()
                if self.phone_was_active:
                    state.use_phone_audio = False
                    if not state.is_restarting():
                        print('Phone ended - restarting with room mic', flush=True)
                        state.set_restarting(True)
                        stop_transcription()
                        def do_room_start():
                            start_transcription(state.mode)
                            state.set_restarting(False)
                        QTimer.singleShot(500, do_room_start)
                    else:
                        print('Phone ended - restart in progress, waiting', flush=True)
                self.phone_was_active = False

        except Exception as e:
            print(f'check_muted error: {e}', flush=True)


    def on_text(self, t):
        self.signal_activity()
        self.caption_view.add_text(t)
        if state.use_phone_audio:
            state.last_phone_speech = time.time()

    def on_segment(self, seg):
        has_text = bool((seg.get('text') or '').strip())
        if has_text:
            self.signal_activity()
        self.caption_view.add_segment(seg)
        if state.use_phone_audio and has_text:
            state.last_phone_speech = time.time()

    def on_status_changed(self, status):
        # Both views, so a fault stays visible after the clock takes over.
        self.caption_view.set_status(status)
        self.clock_view.set_status(status)

    def on_vad_state(self, speaking):
        self.caption_view.set_speaking(speaking)
        self.clock_view.set_speaking(speaking)

    def on_mode_changed(self, mode):
        self.caption_view.set_mode(mode)

    def on_mode_ready(self, mode):
        state.provider_ready = True
        self.caption_view.update_mode_button()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            QApplication.quit()


def clear_stale_state():
    """Clear stale state files on startup"""
    print("Clearing stale state...", flush=True)
    try:
        # Clear phone muted state
        if os.path.exists(PHONE_MUTED_FILE):
            with open(PHONE_MUTED_FILE, 'w') as f:
                f.write('0')
            print(f"  Cleared {PHONE_MUTED_FILE}", flush=True)
    except Exception as e:
        print(f"  Error clearing state: {e}", flush=True)


def main():
    print('='*50, flush=True)
    print('Starting Gramps Captions (BULLETPROOF VERSION)', flush=True)
    print('='*50, flush=True)

    if LOG_TRANSCRIPTS:
        # Say so, every time. Speech logging that runs unannounced is how a
        # device ends up holding months of private conversation that nobody
        # remembers switching on.
        print(f'*** SPEECH IS BEING LOGGED ({LOG_TRANSCRIPTS_VIA}) ***', flush=True)
        print('*** Everything said in this room will appear below. ***', flush=True)

    clear_stale_state()
    cleanup_audio_processes()
    ensure_mic_volume()

    # Create QApplication FIRST — Qt signals require this
    # Our flags are not Qt's; hand it only what it understands.
    app = QApplication([sys.argv[0]] + [
        a for a in sys.argv[1:] if a not in ('--log', '--log-interims', '--log-raw')
    ])

    try:
        import systemd.daemon
        systemd.daemon.notify('READY=1')
    except:
        pass

    start_transcription(CONFIG.get('speech_mode', 'offline'))

    def watchdog():
        try:
            import systemd.daemon
            while True:
                time.sleep(10)
                systemd.daemon.notify('WATCHDOG=1')
        except:
            pass
    threading.Thread(target=watchdog, daemon=True).start()

    win = MainWindow()
    win.show()
    sys.exit(app.exec())



if __name__ == '__main__':
    main()
