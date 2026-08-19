#!/bin/bash
# =============================================================================
# Gramps Transcriber - Easy Installer
# Run with: curl -sSL https://raw.githubusercontent.com/hgomersall/telephone-and-conversation-transcriber/main/install.sh | bash
# =============================================================================

# Wrap everything in main() so bash must receive the complete script before
# executing anything — protects against truncated downloads via curl | bash.
main() {

set -e

# Colours for friendly output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

INSTALL_DIR="$HOME/gramps-transcriber"
VENV_DIR="$HOME/gramps-env"
VOSK_DIR="$HOME/vosk-uk"
SYSTEMD_DIR="$HOME/.config/systemd/user"
# Override both to install from a fork:
#   curl -sSL <your-raw-install.sh-url> | \
#     GRAMPS_REPO=https://github.com/you/telephone-and-conversation-transcriber.git \
#     GRAMPS_BRANCH=your-branch bash
REPO_URL="${GRAMPS_REPO:-https://github.com/hgomersall/telephone-and-conversation-transcriber.git}"
BRANCH="${GRAMPS_BRANCH:-main}"
VOSK_MODEL_URL="https://alphacephei.com/vosk/models/vosk-model-small-en-gb-0.15.zip"
TOTAL_STEPS=8

step() {
    echo ""
    echo -e "${BLUE}${BOLD}[$1/$TOTAL_STEPS]${NC} ${BOLD}$2${NC}"
}

ok() {
    echo -e "  ${GREEN}Done!${NC} $1"
}

warn() {
    echo -e "  ${YELLOW}Note:${NC} $1"
}

fail() {
    echo ""
    echo -e "  ${RED}Something went wrong:${NC} $1"
    echo -e "  ${YELLOW}Need help? Open an issue at:${NC}"
    echo -e "  https://github.com/hgomersall/telephone-and-conversation-transcriber/issues"
    exit 1
}

echo ""
echo -e "${BOLD}================================================${NC}"
echo -e "${BOLD}   Gramps Transcriber — Easy Installer${NC}"
echo -e "${BOLD}================================================${NC}"
echo ""
echo "This will set up everything you need."
echo "It usually takes about 5-10 minutes."
echo ""

# ─── Step 1: Check we're on a Raspberry Pi ───────────────────────────────────

step 1 "Checking your Raspberry Pi..."

if [ ! -f /etc/os-release ]; then
    fail "Can't detect your operating system. This installer is for Raspberry Pi OS."
fi

# Read OS info (in a subshell to avoid polluting the namespace)
OS_ID=$(. /etc/os-release && echo "$ID")
OS_PRETTY_NAME=$(. /etc/os-release && echo "$PRETTY_NAME")

if [[ "$OS_ID" != "debian" && "$OS_ID" != "raspbian" ]]; then
    fail "This installer is designed for Raspberry Pi OS (Bookworm). You seem to be running $OS_PRETTY_NAME."
fi

# Check for 64-bit
if [[ "$(uname -m)" != "aarch64" ]]; then
    fail "You need the 64-bit version of Raspberry Pi OS. You're running 32-bit ($(uname -m))."
fi

# Check Python version
if ! command -v python3 &>/dev/null; then
    fail "Python 3 is not installed. This is unusual — please reinstall Raspberry Pi OS."
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 11 ]; then
    fail "Python 3.11 or newer is required. You have Python $PYTHON_VERSION."
fi

ok "Raspberry Pi OS (64-bit), Python $PYTHON_VERSION"

# ─── Step 2: Install system packages ─────────────────────────────────────────

step 2 "Installing system packages (this may take a couple of minutes)..."

sudo apt-get update -qq || fail "Couldn't update package list. Are you connected to the internet?"

sudo apt-get install -y -qq \
    python3-pyqt6 \
    qt6-qpa-plugins \
    python3-venv \
    python3-dev \
    libasound2-dev \
    portaudio19-dev \
    git \
    curl \
    unzip \
    avahi-daemon \
    2>/dev/null || fail "Couldn't install required packages."

ok "All system packages installed"

# ─── Step 3: Download the transcriber ─────────────────────────────────────────

step 3 "Downloading the transcriber..."

if [ -d "$INSTALL_DIR" ]; then
    warn "Already downloaded — updating to latest version"
    git -C "$INSTALL_DIR" pull --quiet || warn "Couldn't update. Using existing version."
else
    git clone --quiet --branch "$BRANCH" -- "$REPO_URL" "$INSTALL_DIR" || fail "Couldn't download the transcriber. Check your internet connection."
fi

ok "Transcriber downloaded to $INSTALL_DIR"

# ─── Step 4: Set up Python environment ────────────────────────────────────────

step 4 "Setting up Python environment..."

if [ -d "$VENV_DIR" ]; then
    if "$VENV_DIR/bin/python3" -c "import sys" 2>/dev/null; then
        warn "Python environment already exists — reusing it"
    else
        warn "Python environment is broken — recreating it"
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR" --system-site-packages || fail "Couldn't create Python environment."
    fi
else
    python3 -m venv "$VENV_DIR" --system-site-packages || fail "Couldn't create Python environment."
fi

ok "Python environment ready"

# ─── Step 5: Install Python packages ─────────────────────────────────────────

step 5 "Installing Python packages..."

REQUIREMENTS="$INSTALL_DIR/requirements.txt"

if [ ! -f "$REQUIREMENTS" ]; then
    fail "requirements.txt not found. The download may be incomplete — try running the installer again."
fi

# uv resolves and installs far faster than pip, which is worth real minutes on
# a Pi. It is only ever an accelerator here: every failure path below falls
# back to pip, so a fresh install can never be blocked by uv being missing,
# unreachable or broken.
UV_BIN=""
if command -v uv &>/dev/null; then
    UV_BIN="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
else
    if curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh >/dev/null 2>&1; then
        if [ -x "$HOME/.local/bin/uv" ]; then
            UV_BIN="$HOME/.local/bin/uv"
        fi
    fi
fi

PACKAGES_INSTALLED=0
if [ -n "$UV_BIN" ]; then
    # Target the venv explicitly rather than relying on VIRTUAL_ENV. PyQt6 is
    # deliberately absent from requirements.txt — it comes from apt, and the
    # venv was created with --system-site-packages so it can see it.
    if "$UV_BIN" pip install --quiet --python "$VENV_DIR/bin/python" -r "$REQUIREMENTS" 2>/dev/null; then
        PACKAGES_INSTALLED=1
        ok "All Python packages installed (uv)"
    else
        warn "uv couldn't install the packages — falling back to pip"
    fi
fi

if [ "$PACKAGES_INSTALLED" -eq 0 ]; then
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip 2>/dev/null || warn "Couldn't upgrade pip (continuing with existing version)"
    "$VENV_DIR/bin/pip" install --quiet -r "$REQUIREMENTS" || fail "Couldn't install Python packages."
    ok "All Python packages installed (pip)"
fi

# ─── Step 6: Download offline speech model ────────────────────────────────────

step 6 "Downloading offline speech model (~40 MB)..."

if [ -d "$VOSK_DIR" ] && [ -f "$VOSK_DIR/conf/model.conf" ]; then
    warn "Speech model already downloaded — skipping"
else
    rm -rf "$VOSK_DIR"  # Clean up any partial download
    VOSK_ZIP="/tmp/vosk-model.zip"
    curl -sSL "$VOSK_MODEL_URL" -o "$VOSK_ZIP" || fail "Couldn't download the speech model."
    unzip -o -q "$VOSK_ZIP" -d /tmp || fail "Couldn't unpack the speech model."
    mv /tmp/vosk-model-small-en-gb-0.15 "$VOSK_DIR" || fail "Couldn't move the speech model into place."
    rm -f "$VOSK_ZIP"
fi

ok "Offline speech model ready"

# ─── Step 7: Install services ─────────────────────────────────────────────────

step 7 "Setting up auto-start services..."

mkdir -p "$SYSTEMD_DIR"

# Copy all user services
for service_file in "$INSTALL_DIR"/systemd/caption.service \
                    "$INSTALL_DIR"/systemd/gramps-mute.service; do
    if [ -f "$service_file" ]; then
        cp "$service_file" "$SYSTEMD_DIR/"
    fi
done

# Install the setup wizard service
if [ -f "$INSTALL_DIR/setup/gramps-setup.service" ]; then
    cp "$INSTALL_DIR/setup/gramps-setup.service" "$SYSTEMD_DIR/"
else
    fail "Setup wizard service file not found. The download may be incomplete — try running the installer again."
fi

systemctl --user daemon-reload || fail "Couldn't reload systemd. Make sure you're running this from an interactive SSH login or desktop session."

ok "Services installed"

# ─── Step 8: Start the setup wizard ──────────────────────────────────────────

step 8 "Starting the setup wizard..."

# ─── Desktop shortcuts ───────────────────────────────────────────────────────

# Both the Desktop and the application menu, mode 755 because some file
# managers refuse to launch a .desktop file without the executable bit.
# (An earlier version also ran `gio set metadata::trusted` — that is a GNOME
# Files mechanism, is not supported by gio for this attribute, and is not read
# by the file manager Raspberry Pi OS uses. It did nothing.)
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
mkdir -p "$DESKTOP_DIR" "$HOME/.local/share/applications"
for launcher in "$INSTALL_DIR"/desktop/*.desktop; do
    [ -f "$launcher" ] || continue
    install -m 755 "$launcher" "$DESKTOP_DIR/" 2>/dev/null || \
        warn "Couldn't put a shortcut on the desktop"
    install -m 755 "$launcher" "$HOME/.local/share/applications/" 2>/dev/null || \
        warn "Couldn't add a shortcut to the menu"
done
# User services only start at boot if the account has lingering enabled.
# Without it a headless Pi comes back from a power cut with no setup wizard
# and no transcriber, which looks exactly like a broken install.
loginctl enable-linger "$USER" 2>/dev/null || warn "Couldn't enable start-on-boot for services (they'll still start when you log in)"

systemctl --user enable --now gramps-setup 2>/dev/null || warn "Couldn't auto-start the wizard (you can start it manually)"

# Give it a moment to start
sleep 2

# Get the Pi's hostname
PI_HOSTNAME="$(hostname).local"

ok "Setup wizard is running!"

# ─── Done! ────────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}${BOLD}================================================${NC}"
echo -e "${GREEN}${BOLD}   All done! Just one more step...${NC}"
echo -e "${GREEN}${BOLD}================================================${NC}"
echo ""
echo -e "Open this address on your phone or computer:"
echo ""
echo -e "  ${BOLD}${BLUE}http://${PI_HOSTNAME}:8080${NC}"
echo ""
echo -e "The setup page will walk you through the rest."
echo ""
echo -e "If that address doesn't work, try:"
echo ""
IP_ADDR=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.' | head -1)
if [ -n "$IP_ADDR" ]; then
    echo -e "  ${BOLD}http://${IP_ADDR}:8080${NC}"
    echo ""
fi
echo -e "${YELLOW}Tip:${NC} You can come back to this setup page any time"
echo -e "     to change settings or check on things."
echo ""

}

# Run the main function — this ensures the entire script is downloaded before execution
main "$@"
