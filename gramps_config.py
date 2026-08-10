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

    # Used by the network watchdog
    'gateway_ip': None,
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


def load_config(verbose=True):
    """Load config, merged over DEFAULTS. Never raises."""
    config = dict(DEFAULTS)
    for key, value in _read_file(find_config_file(), verbose).items():
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
