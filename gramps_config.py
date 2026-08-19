#!/usr/bin/env python3
"""Config loading for the Gramps transcriber.

Resolves a single config.json from an ordered list of locations and merges it
over a table of defaults. Existing installs are unaffected: the historical
~/gramps-transcriber/config.json is still searched, and every default here
matches the value the calling code used to pass to CONFIG.get() inline.

Search order (first file that exists wins):

  1. $GRAMPS_CONFIG                              — explicit override, a file path
  2. <repo>/config.json                          — alongside this checkout
  3. $XDG_CONFIG_HOME/gramps-transcriber/config.json
  4. ~/gramps-transcriber/config.json            — where the installer puts it

On a Pi the checkout *is* ~/gramps-transcriber, so 2 and 4 are the same file and
the behaviour is identical to before. On a dev machine with the repo elsewhere,
a config.json next to the code takes precedence, so you can test without
touching the installed one.
"""

import difflib
import json
import os

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_CONFIG_PATH = os.path.expanduser('~/gramps-transcriber/config.json')

# Every key the app reads, with the default it used to hard-code at the call
# site. Adding a key here is enough to make it configurable.
DEFAULTS = {
    # Audio devices — None means "autodetect" (see get_audio_device())
    'room_device': None,
    'phone_device': None,

    # Speech engine selection
    'speech_mode': 'offline',
    'stt_provider': 'deepgram',
    'offline_model': 'faster-whisper',

    # Cloud provider credentials
    'deepgram_key': None,
    'assemblyai_key': None,
    'azure_key': None,
    'azure_region': 'uksouth',
    'groq_key': None,
    'interfaze_key': None,
    'openai_key': None,
    'google_key': None,
    'speechmatics_key': None,

    # Speechmatics realtime. Regional endpoints exist for data residency;
    # global.rt routes to the nearest.
    'speechmatics_url': 'wss://eu2.rt.speechmatics.com/v2',
    'speechmatics_language': 'en',
    # standard or enhanced. Enhanced is the more accurate; standard favours
    # throughput. Defaulting to enhanced because a misread word costs the
    # reader more here than a little latency does. Speechmatics defaults to
    # standard if unset. (melia-1 exists but is batch-only, so it cannot be
    # used for live captions.)
    'speechmatics_model': 'enhanced',
    # Delay between the end of a word and its final transcript. Their range is
    # 0.7-4s; lower is more responsive, which is what this device needs.
    'speechmatics_max_delay': 1.0,
    # Diarization stability controls, which Deepgram does not offer. Deepgram's
    # labels proved unusable here once audio was gated, so these are worth
    # having: prefer_current_speaker reduces switching between similar voices.
    'speechmatics_max_speakers': None,
    'speechmatics_prefer_current_speaker': True,
    'speechmatics_speaker_sensitivity': None,
    # Silence that ends an utterance, producing an EndOfUtterance message. This
    # is what paragraph breaks hang on: a Speechmatics "final" carries only the
    # newly-finalised words, not the end of anything. 0 disables it.
    'speechmatics_end_of_utterance_s': 0.8,

    # Used by the network watchdog
    'gateway_ip': None,

    # Write recognised speech to the log. OFF, and it should stay off outside
    # of debugging: the log is a verbatim, permanent, unencrypted record of
    # every conversation and phone call in the house, on the SD card of a
    # device sitting in someone's home. Callers have not agreed to it either.
    'log_transcripts': False,

    # Log every interim transcript, not just finals. Needed to measure
    # time-to-first-word. Also transcript content, so it needs log_transcripts
    # as well as this.
    'log_interims': False,

    # Voice activity detection. Silero ships inside faster-whisper and needs no
    # download. It replaces a fixed amplitude threshold, so unlike the old
    # `energy < 0.005` check it is not thrown off by changing the mic gain.
    'offline_vad': True,          # let Silero, not amplitude, decide what is speech
    'vad_threshold': 0.5,         # speech probability above which a frame counts
    'vad_min_speech_ms': 250,     # ignore speech bursts shorter than this
    'vad_min_silence_ms': 500,    # ignore gaps shorter than this
    'vad_hangover_s': 1.0,        # keep reporting speech this long after it stops
    'vad_indicator': True,        # show speech activity in the on-screen status
    # Log speech/silence transitions and the periodic billed-vs-elapsed report.
    # On by default: without it there is no way to tell what the gate is
    # actually saving, or to reduce a run to a speech duty cycle. Lines are
    # emitted per transition, so a silent room produces none.
    'log_vad': True,

    # Colour and mark caption text when the speaker changes (Deepgram, room mic).
    #
    # Works alongside vad_gate. On Speechmatics that is measured: the same two
    # voices give S1/S2/S1 whether streamed continuously or with 12s of audio
    # withheld between them.
    #
    # On Deepgram, gating has been observed to make speaker labels collapse to
    # 0, intermittently. That was never pinned down, and the equivalent claim
    # about Speechmatics turned out to be a measurement error — too little
    # audio per speaker rather than the gaps — so it is reported rather than
    # established. If colours stop tracking the speaker on Deepgram, turn off
    # vad_gate (or this, to keep the saving).
    #
    # Turning this off stops diarization being requested at all, which also
    # removes the blank line between speakers: with no speaker labels there is
    # no speaker change to break on. Utterance breaks are unaffected, so the
    # text still separates, just less strongly. That is the trade — diarization
    # is billed separately, so requesting it purely for paragraph breaks would
    # be paying for a layout hint.
    #
    # Set this false if the cost saving matters more than knowing who is
    # talking. Phone calls are unaffected — they never had diarization.
    'speaker_colours': True,

    # Gating the audio stream. Both providers bill on audio sent rather than on
    # connection time, so holding a socket open costs nothing and only the
    # audio is withheld. Deepgram documents this and does not charge for
    # KeepAlive. Speechmatics was measured: two 60s connections four minutes
    # apart, one submitting 60s of audio and one submitting 5s, billed a minute
    # and about four seconds respectively.
    #
    # Set vad_gate false to stream continuously as before.
    #
    'vad_gate': True,

    # How long the gate stays open after speech stops. Much longer than the
    # detector's hangover on purpose: across a gap in the audio Deepgram
    # restarts its speaker numbering, so a gate that shuts between turns makes
    # every utterance come back as speaker 0 and speaker colours stop working.
    # Several seconds bridges conversational pauses and keeps a whole
    # conversation as one unbroken stream. It costs very little, because nearly
    # all the saving is silent hours rather than the gaps between turns.
    'gate_hangover_s': 4.0,
    'preroll_s': 0.5,             # audio held back and flushed when speech starts
    'keepalive_s': 4.0,           # KeepAlive interval; Deepgram drops at 10s of nothing

    # Seconds of unbroken press on the status indicator before the on-screen
    # exit is offered, which then needs a second tap to confirm. There is no
    # keyboard on the appliance, so some route out has to exist — but it must
    # be hard to hit by accident, because the person using this cannot hear
    # that it has stopped and would be left with a blank screen. Set 0 to
    # remove the touch exit entirely.
    'exit_hold_s': 5.0,

    # Offline chunking. Audio is cut at a silence rather than on a fixed tick so
    # words are never split, but unbroken speech has to be cut eventually or
    # nothing would ever be transcribed.
    'offline_chunk_s': 3,         # start looking for a cut point after this long
    'offline_max_chunk_s': 6,     # cut regardless after this long
}


def config_search_paths():
    """The ordered candidate locations, highest precedence first."""
    paths = []

    override = os.environ.get('GRAMPS_CONFIG')
    if override:
        paths.append(os.path.expanduser(override))

    paths.append(os.path.join(REPO_DIR, 'config.json'))

    xdg = os.environ.get('XDG_CONFIG_HOME') or os.path.expanduser('~/.config')
    paths.append(os.path.join(xdg, 'gramps-transcriber', 'config.json'))

    paths.append(LEGACY_CONFIG_PATH)

    # De-duplicate while preserving order — on a Pi several of these collapse
    # to the same file.
    seen = set()
    unique = []
    for path in paths:
        real = os.path.normpath(path)
        if real not in seen:
            seen.add(real)
            unique.append(real)
    return unique


def find_config_file():
    """Return the first config path that exists, or None if there is no config.

    A $GRAMPS_CONFIG that points at a missing file is reported but not fatal —
    this runs on a device that has to come up unattended, so a typo in an env
    var must not stop it starting.
    """
    override = os.environ.get('GRAMPS_CONFIG')
    for path in config_search_paths():
        if os.path.isfile(path):
            return path
        if override and path == os.path.normpath(os.path.expanduser(override)):
            print(f'Config: $GRAMPS_CONFIG points at {path}, which does not '
                  f'exist — falling back to the normal search order', flush=True)
    return None


def read_raw_config(verbose=False):
    """Load the config file exactly as written, with no defaults merged.

    Use this when writing config back out — merging DEFAULTS in first would
    bake today's defaults into the file and freeze them there.
    """
    return _read_file(find_config_file(), verbose)


def _read_file(path, verbose):
    """Read and validate a config file. Returns {} for anything unusable."""
    if not path:
        if verbose:
            print('Config: no config file found — using defaults', flush=True)
        return {}

    try:
        with open(path) as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f'Config: could not read {path} ({e}) — using defaults', flush=True)
        return {}

    if not isinstance(loaded, dict):
        print(f'Config: {path} is not a JSON object — ignoring it', flush=True)
        return {}

    if verbose:
        print(f'Config: loaded {path}', flush=True)
    return loaded


def unknown_keys(loaded):
    """Config keys that mean nothing to the app.

    Keys starting with _ are ignored on purpose — JSON has no comments, so that
    is how you leave notes in the file.
    """
    return sorted(k for k in loaded
                  if k not in DEFAULTS and not k.startswith('_'))


def describe_unknown_keys(unknown, path):
    """A message naming each unknown key, with a suggestion where one is close."""
    lines = [f'Config error in {path}:']
    for key in unknown:
        near = difflib.get_close_matches(key, list(DEFAULTS), n=1)
        lines.append(f'  unknown key {key!r}' +
                     (f' — did you mean {near[0]!r}?' if near else ''))
    lines.append('A typo here is silent: the setting you meant is simply never '
                 'applied. Keys starting with _ are ignored, so use those for notes.')
    return '\n'.join(lines)


def load_config(verbose=True, strict=False):
    """Load config, merged over DEFAULTS.

    With strict, an unrecognised key raises SystemExit. Callers running
    unattended should leave it off: a device that will not start is worse for
    the person relying on it than one setting quietly not applying.
    """
    path = find_config_file()
    raw = _read_file(path, verbose)
    unknown = unknown_keys(raw)
    if unknown:
        message = describe_unknown_keys(unknown, path)
        if strict:
            raise SystemExit(message)
        print(message, flush=True)

    config = dict(DEFAULTS)
    for key, value in raw.items():
        # The setup wizard writes "" for every field left blank, which would
        # otherwise shadow a real default (e.g. azure_region).
        if value == '':
            continue
        config[key] = value
    return config


def config_write_path():
    """Where the setup wizard should save.

    Writes back to whichever file is currently in effect so the wizard edits
    the config the app actually reads. With no config anywhere yet, falls back
    to the installer's location so a fresh Pi behaves exactly as before.
    """
    return find_config_file() or LEGACY_CONFIG_PATH


def save_config(data, path=None):
    """Write config as JSON, creating the parent directory if needed."""
    path = path or config_write_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    return path
