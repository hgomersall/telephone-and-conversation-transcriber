#!/usr/bin/env python3
"""Gramps Transcriber — Setup Wizard (runs on port 8080)"""

import os
import re
import subprocess
import sys
import time
import struct

from flask import Flask, render_template, request, jsonify

# The wizard lives in setup/ — put the repo root on the path so it shares the
# same config resolution as caption_app.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gramps_config import (  # noqa: E402
    DEFAULTS, REPO_DIR, config_write_path, load_config, read_raw_config,
    save_config, unknown_keys,
)

app = Flask(__name__)

# Every setting this wizard writes. Checked against DEFAULTS at start-up rather
# than discovered when a save produces a config the app then rejects.
WIZARD_FIELDS = {
    'room_device': '',
    'phone_device': '',
    'speech_mode': 'online',
    'stt_provider': 'deepgram',
    'offline_model': 'faster-whisper',
    'deepgram_key': '',
    'speechmatics_key': '',
    'assemblyai_key': '',
    'azure_key': '',
    'azure_region': 'uksouth',
    'groq_key': '',
    'interfaze_key': '',
    'openai_key': '',
    'google_key': '',
    'gateway_ip': '',
    # Booleans are written even when False — False is not "empty", and an
    # explicit false is exactly how you turn off something that defaults on.
    'speaker_colours': True,
    'vad_gate': True,
    'speechmatics_model': 'enhanced',
}

_unknown = [k for k in WIZARD_FIELDS if k not in DEFAULTS]
if _unknown:
    print(f'Setup wizard: these fields are not config keys and will be '
          f'rejected by the app: {_unknown}', flush=True)

# credentials.py is imported by caption_app.py, so it has to sit next to it in
# the repo root regardless of where config.json ends up.
CREDENTIALS_PATH = os.path.join(REPO_DIR, 'credentials.py')


def save_credentials(deepgram_key=None, azure_key=None, azure_region=None):
    """Write credentials.py so caption_app.py can import it."""
    lines = []
    if deepgram_key:
        lines.append(f'DEEPGRAM_KEY = "{deepgram_key}"')
    if azure_key:
        lines.append(f'AZURE_KEY = "{azure_key}"')
        lines.append(f'AZURE_REGION = "{azure_region or "uksouth"}"')
    if lines:
        os.makedirs(os.path.dirname(CREDENTIALS_PATH), exist_ok=True)
        with open(CREDENTIALS_PATH, 'w') as f:
            f.write('\n'.join(lines) + '\n')


def detect_audio_devices():
    """Parse arecord -l into a friendly list of microphones."""
    devices = []
    try:
        result = subprocess.run(
            ['arecord', '-l'], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split('\n'):
            match = re.search(r'card (\d+):.*\[(.+?)\].*device (\d+):.*\[(.+?)\]', line)
            if match:
                card, card_name, device, device_name = match.groups()
                hw_id = f'hw:{card},{device}'
                # Build a friendly label
                label = device_name.strip()
                if card_name.strip().lower() != device_name.strip().lower():
                    label = f'{card_name.strip()} — {device_name.strip()}'
                devices.append({
                    'hw_id': hw_id,
                    'card': int(card),
                    'label': label,
                    'raw': line.strip(),
                })
    except Exception:
        pass
    return devices


def test_audio_device(hw_id, duration=3, sample_rate=16000):
    """Record a short clip from a device and return the audio energy level (0-100)."""
    # Use plughw: so ALSA resamples/downmixes for mics that don't natively
    # support 16kHz mono (most USB mics only do 48kHz stereo). Recording from
    # the raw hw: device fails instantly on those, which reads as silence.
    if hw_id and hw_id.startswith('hw:'):
        hw_id = 'plug' + hw_id
    try:
        proc = subprocess.run(
            ['arecord', '-D', hw_id, '-f', 'S16_LE', '-r', str(sample_rate),
             '-c', '1', '-t', 'raw', '-d', str(duration), '-q'],
            capture_output=True, timeout=duration + 5
        )
        raw = proc.stdout
        if not raw:
            return 0
        # Calculate RMS energy
        samples = struct.unpack(f'<{len(raw)//2}h', raw)
        if not samples:
            return 0
        rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
        # Normalise to 0-100 (32768 is max for 16-bit)
        level = min(100, int(rms / 327.68 * 10))
        return level
    except Exception:
        return -1


def get_service_status(service_name):
    """Check if a systemd user service is running."""
    try:
        result = subprocess.run(
            ['systemctl', '--user', 'is-active', service_name],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    config = load_config(verbose=False)
    return render_template('index.html', config=config)


@app.route('/api/devices')
def api_devices():
    devices = detect_audio_devices()
    return jsonify(devices)


@app.route('/api/test-audio', methods=['POST'])
def api_test_audio():
    data = request.get_json() or {}
    hw_id = data.get('hw_id', 'hw:0,0')
    level = test_audio_device(hw_id)
    return jsonify({'level': level, 'hw_id': hw_id})


@app.route('/api/save', methods=['POST'])
def api_save():
    data = request.get_json() or {}

    # Start from what is on disk, so anything hand-added — including the
    # underscore-prefixed keys used as comments — survives a save.
    config = read_raw_config()

    for key, fallback in WIZARD_FIELDS.items():
        value = data.get(key, fallback)
        if value in ('', None):
            # A cleared field REMOVES the key rather than writing "". Empty
            # values are ignored by the loader anyway, so writing them only
            # accumulated dead entries: choosing one provider used to leave
            # seven empty key fields in the file for good.
            config.pop(key, None)
        else:
            config[key] = value

    path = save_config(config)
    print(f'Setup wizard: saved {path}', flush=True)

    leftover = unknown_keys(config)
    if leftover:
        # Not fatal — they may be deliberate. But the app reports them too, so
        # say it here where whoever typed them is looking.
        print(f'Setup wizard: config contains keys the app does not know: '
              f'{leftover}', flush=True)

    # Also write credentials.py for backwards compatibility
    if config.get('deepgram_key'):
        save_credentials(deepgram_key=config['deepgram_key'])
    elif config.get('azure_key'):
        save_credentials(azure_key=config['azure_key'], azure_region=config['azure_region'])

    # Restart the caption service so it picks up new config
    try:
        subprocess.run(
            ['systemctl', '--user', 'restart', 'caption'],
            capture_output=True, timeout=10
        )
    except Exception:
        pass

    return jsonify({'ok': True, 'config_path': path,
                    'unknown_keys': leftover})


@app.route('/api/status')
def api_status():
    caption = get_service_status('caption')
    mute = get_service_status('gramps-mute')
    raw = read_raw_config()
    config = load_config(verbose=False)
    provider_key = f"{config.get('stt_provider') or ''}_key"
    configured = bool(config.get('room_device') or config.get(provider_key)
                      or config.get('deepgram_key')
                      or config.get('speechmatics_key'))
    return jsonify({
        'caption': caption,
        'mute': mute,
        'configured': configured,
        # Which file is actually in effect. There are four places it can live
        # now, so "I edited the config and nothing changed" is a real way to
        # lose an afternoon.
        'config_path': config_write_path(),
        'unknown_keys': unknown_keys(raw),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
