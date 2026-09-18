#!/usr/bin/env bash
# Install or update LLM Scanner on this Linux machine. No sudo needed.
#
#   From GitHub:     curl -fsSL https://raw.githubusercontent.com/Kadn0/llm-scanner/main/install.sh | bash
#   From a clone:    ./install.sh
#   Move your data:  ./install.sh --data llm-scanner-data-DATE.tar.gz   (made with export-data.sh)
#
# Installs, all inside your home folder:
#   ~/.local/bin/uv        Python/package manager
#   ~/.local/ollama        Ollama (chat model engine)
#   ~/.local/sd-cpp        stable-diffusion.cpp (image engine)
#   ~/llm-scanner          the app, its Python environment, and your data
# and sets up user services that start at boot and open the app when you log in.
#
# Safe to run again: it updates the app and keeps your existing data.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

  --data FILE       restore chats, images, probe groups and reports from an export-data.sh file
  --pull-models     with --data, also re-download the Ollama chat models listed in the export
  --no-autostart    don't open the app automatically when you log in
  --no-start        install everything but don't start the services or open the browser
  -h, --help        show this help
EOF
}

REPO="${LLM_SCANNER_REPO:-Kadn0/llm-scanner}"
PULL_MODELS=0
AUTOSTART=1
START=1
DATA_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA_FILE="$(realpath "$2")"; shift ;;
    --pull-models) PULL_MODELS=1 ;;
    --no-autostart) AUTOSTART=0 ;;
    --no-start) START=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done
# LLM_SCANNER_TEST=1 skips systemd, linger and desktop steps (used to test the installer in a throwaway HOME).
TEST_MODE="${LLM_SCANNER_TEST:-0}"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)"
APP_DIR="$HOME/llm-scanner"
LOCAL_BIN="$HOME/.local/bin"
OLLAMA_DIR="$HOME/.local/ollama"
SD_DIR="$HOME/.local/sd-cpp"
UV="$LOCAL_BIN/uv"
PORT=7861

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    \033[33mWarning:\033[0m %s\n' "$*"; }
fail() { printf '\n\033[31mError:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- checks
step "Checking this machine"
[ "$(uname -s)" = "Linux" ] || fail "This installer supports Linux only."
[ "$(uname -m)" = "x86_64" ] || fail "This installer supports x86_64 (64-bit Intel/AMD) only."
for cmd in curl tar; do
  command -v "$cmd" >/dev/null || fail "'$cmd' is required. Install it with your package manager, then run this again."
done
if [ ! -f "$SRC_DIR/app.py" ]; then
  # Run through curl | bash: fetch the latest release from GitHub.
  info "Downloading the latest LLM Scanner release from github.com/$REPO"
  FETCH_DIR="$(mktemp -d)"
  tag="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)"
  [ -n "$tag" ] || fail "Couldn't find a release of $REPO on GitHub."
  curl -fsSL "https://codeload.github.com/$REPO/tar.gz/refs/tags/$tag" | tar -xz -C "$FETCH_DIR"
  SRC_DIR="$(find "$FETCH_DIR" -mindepth 1 -maxdepth 1 -type d | head -1)"
  [ -f "$SRC_DIR/app.py" ] || fail "The downloaded release doesn't contain app.py."
fi
info "Installing LLM Scanner $(cat "$SRC_DIR/VERSION" 2>/dev/null || echo "(unknown version)")"
if [ -n "$DATA_FILE" ]; then
  [ -f "$DATA_FILE" ] || fail "Data file not found: $DATA_FILE"
  DATA_DIR_SRC="$(mktemp -d)"
  tar -xzf "$DATA_FILE" -C "$DATA_DIR_SRC"
  DATA_DIR_SRC="$DATA_DIR_SRC/llm-scanner-data"
  [ -d "$DATA_DIR_SRC" ] || fail "$DATA_FILE isn't an LLM Scanner data export."
fi
. /etc/os-release 2>/dev/null || true
if [ "${NAME:-}" = "Pop!_OS" ]; then
  info "System: ${PRETTY_NAME:-Pop!_OS}"
else
  warn "This installer is made for Pop!_OS 24.04. You're on ${PRETTY_NAME:-an unknown system}; it may still work."
fi
# NVIDIA card present but no working driver (e.g. Pop!_OS installed from the non-NVIDIA download)?
if grep -qs 0x10de /sys/bus/pci/devices/*/vendor && ! { command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; }; then
  warn "An NVIDIA graphics card was found, but its driver isn't working, so models would run on the CPU (slowly)."
  warn "Install the driver, reboot, then run this installer again:"
  warn "    sudo apt install -y system76-driver-nvidia && sudo reboot"
  if [ -t 0 ]; then
    read -r -p "    Continue without the NVIDIA driver anyway? [y/N] " answer
    [[ "$answer" =~ ^[Yy]$ ]] || exit 1
  else
    fail "Install the NVIDIA driver first (command above), reboot, then run the installer again."
  fi
fi
GPU="none"
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  GPU="nvidia"; info "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
elif ls /dev/dri/renderD* >/dev/null 2>&1; then
  GPU="other"; info "GPU: non-NVIDIA graphics found (image generation will use Vulkan if available)"
else
  info "GPU: none found. Everything still works on the CPU, just more slowly."
fi
info "Disk space free in your home folder: $(df -h "$HOME" | awk 'NR==2 {print $4}')"
mkdir -p "$LOCAL_BIN"

if [ "$TEST_MODE" != "1" ] && systemctl --user is-active --quiet llm-scanner 2>/dev/null; then
  info "Stopping the running LLM Scanner so it can be updated"
  systemctl --user stop llm-scanner
fi

# ---------------------------------------------------------------- uv
step "Installing uv (Python package manager)"
if [ -x "$UV" ]; then
  info "Already installed: $("$UV" --version)"
else
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$LOCAL_BIN" INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null
  info "Installed $("$UV" --version)"
fi

# ---------------------------------------------------------------- Ollama
step "Installing Ollama (chat model engine)"
if [ -x "$OLLAMA_DIR/bin/ollama" ]; then
  info "Already installed. Update it later from the app's update notice."
else
  tmp="$(mktemp -d)"
  if command -v zstd >/dev/null; then
    curl -fL --progress-bar -o "$tmp/ollama.tar.zst" https://ollama.com/download/ollama-linux-amd64.tar.zst
    mkdir -p "$OLLAMA_DIR" && tar --zstd -xf "$tmp/ollama.tar.zst" -C "$OLLAMA_DIR"
  else
    curl -fL --progress-bar -o "$tmp/ollama.tgz" https://ollama.com/download/ollama-linux-amd64.tgz
    mkdir -p "$OLLAMA_DIR" && tar -xzf "$tmp/ollama.tgz" -C "$OLLAMA_DIR"
  fi
  rm -rf "$tmp"
  info "Installed to $OLLAMA_DIR"
fi
ln -sf "$OLLAMA_DIR/bin/ollama" "$LOCAL_BIN/ollama"

# ---------------------------------------------------------------- app + Python environment
step "Installing the LLM Scanner app"
mkdir -p "$APP_DIR/static" "$APP_DIR/data"
if [ "$SRC_DIR" != "$APP_DIR" ]; then
  for f in app.py analyst_report.py garak_runner.py llm-scanner.sh install.sh uninstall.sh export-data.sh \
           README.md VERSION requirements.txt; do
    if [ -f "$SRC_DIR/$f" ]; then cp "$SRC_DIR/$f" "$APP_DIR/"; fi
  done
  cp "$SRC_DIR"/static/* "$APP_DIR/static/"
fi
chmod +x "$APP_DIR"/*.sh
info "App files copied to $APP_DIR"

step "Building the Python environment (garak, Gradio, PyTorch). This is the slow part, several GB."
[ -x "$APP_DIR/.venv/bin/python" ] || "$UV" venv --quiet --python 3.12 "$APP_DIR/.venv"
if ! "$UV" pip install --quiet --python "$APP_DIR/.venv/bin/python" -r "$APP_DIR/requirements.txt"; then
  warn "Exact package versions failed to install; installing the latest compatible versions instead."
  "$UV" pip install --quiet --python "$APP_DIR/.venv/bin/python" garak gradio requests pypdf python-docx
fi
info "garak $("$APP_DIR/.venv/bin/python" -c 'import importlib.metadata as m; print(m.version("garak"))'), Gradio $("$APP_DIR/.venv/bin/python" -c 'import importlib.metadata as m; print(m.version("gradio"))')"

# ---------------------------------------------------------------- image engine
step "Installing stable-diffusion.cpp (image engine)"
if [ -x "$SD_DIR/sd-cli" ]; then
  info "Already installed"
else
  # Vulkan build for NVIDIA and other GPUs with Vulkan drivers, CPU build otherwise.
  variant="cpu"
  if [ "$GPU" != "none" ] && { compgen -G "/usr/share/vulkan/icd.d/*.json" >/dev/null || compgen -G "/etc/vulkan/icd.d/*.json" >/dev/null; }; then
    variant="vulkan"
  fi
  url="$(curl -fsSL https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest | "$APP_DIR/.venv/bin/python" -c "
import json, sys
assets = json.load(sys.stdin)['assets']
want = '$variant'
for a in assets:
    n = a['name']
    if 'Linux' in n and 'x86_64' in n and n.endswith('.zip') and 'rocm' not in n:
        if (want == 'vulkan') == ('vulkan' in n):
            print(a['browser_download_url']); break
")"
  if [ -z "$url" ]; then
    warn "Couldn't find an image engine download. Image generation won't work until this step succeeds (re-run install.sh)."
  else
    tmp="$(mktemp -d)"
    curl -fL --progress-bar -o "$tmp/sd.zip" "$url"
    mkdir -p "$SD_DIR"
    "$APP_DIR/.venv/bin/python" -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$tmp/sd.zip" "$SD_DIR"
    chmod +x "$SD_DIR/sd-cli" "$SD_DIR/sd-server" 2>/dev/null || true
    rm -rf "$tmp"
    info "Installed the $variant build to $SD_DIR"
  fi
fi

# ---------------------------------------------------------------- your data
if [ -n "$DATA_FILE" ]; then
  step "Restoring your data (chats, images, probe groups, reports)"
  for item in chats attachments images probe_groups.json image_models.json; do
    if [ -e "$DATA_DIR_SRC/$item" ]; then
      cp -rn "$DATA_DIR_SRC/$item" "$APP_DIR/data/" 2>/dev/null || true
    fi
  done
  if [ -d "$DATA_DIR_SRC/garak_runs" ]; then
    mkdir -p "$HOME/.local/share/garak/garak_runs"
    cp -rn "$DATA_DIR_SRC/garak_runs/." "$HOME/.local/share/garak/garak_runs/" 2>/dev/null || true
  fi
  info "Existing files on this machine were kept; only missing ones were added."
fi

# ---------------------------------------------------------------- services, startup, menu entry
if [ "$TEST_MODE" = "1" ]; then
  step "Test mode: skipping services and desktop entries"
else
  step "Setting up services (start at boot, open at login)"
  units="$HOME/.config/systemd/user"
  mkdir -p "$units"
  cat > "$units/ollama.service" <<'EOF'
[Unit]
Description=Ollama (user-level, no sudo)
After=network-online.target

[Service]
ExecStart=%h/.local/ollama/bin/ollama serve
Environment=OLLAMA_HOST=127.0.0.1:11434
Environment=OLLAMA_MODELS=%h/.ollama/models
# Models that don't fit on the GPU are split across GPU and system memory automatically.
Environment=OLLAMA_FLASH_ATTENTION=1
Environment=OLLAMA_KV_CACHE_TYPE=q8_0
Environment=OLLAMA_CONTEXT_LENGTH=8192
Environment=OLLAMA_MAX_LOADED_MODELS=1
Environment=OLLAMA_NUM_PARALLEL=1
# Unload models right away; the app keeps them loaded only while chatting or scanning.
Environment=OLLAMA_KEEP_ALIVE=0
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF
  cat > "$units/llm-scanner.service" <<EOF
[Unit]
Description=LLM Scanner GUI (http://127.0.0.1:$PORT)
After=ollama.service
Wants=ollama.service

[Service]
WorkingDirectory=%h/llm-scanner
ExecStart=%h/llm-scanner/.venv/bin/python %h/llm-scanner/app.py --no-browser
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable ollama llm-scanner >/dev/null 2>&1
  info "Services enabled"
  if loginctl enable-linger "$USER" 2>/dev/null; then
    info "Services will start at boot, before you log in"
  else
    warn "Couldn't enable start-at-boot (the system didn't allow it without sudo). Services will start when you log in instead."
  fi

  icons="$HOME/.local/share/icons/hicolor"
  mkdir -p "$icons/scalable/apps" "$icons/256x256/apps"
  cp "$APP_DIR/static/icon.svg" "$icons/scalable/apps/llm-scanner.svg"
  cp "$APP_DIR/static/icon.png" "$icons/256x256/apps/llm-scanner.png"

  desktop="$HOME/.local/share/applications"
  mkdir -p "$desktop"
  cat > "$desktop/llm-scanner.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=LLM Scanner
Comment=Download models, chat, generate images, and scan AI models with garak
Exec=$APP_DIR/llm-scanner.sh
Icon=llm-scanner
Terminal=false
Categories=Development;Security;
StartupNotify=false
EOF
  command -v update-desktop-database >/dev/null && update-desktop-database "$desktop" >/dev/null 2>&1 || true
  info "Added LLM Scanner to your app menu"

  # Desktop shortcut: the same launcher, shown on the desktop.
  desk_dir="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
  mkdir -p "$desk_dir"
  cp "$desktop/llm-scanner.desktop" "$desk_dir/llm-scanner.desktop"
  chmod +x "$desk_dir/llm-scanner.desktop"
  if command -v gio >/dev/null; then gio set "$desk_dir/llm-scanner.desktop" metadata::trusted true 2>/dev/null || true; fi
  info "Added an LLM Scanner shortcut to your desktop"
  if [ "$AUTOSTART" = "1" ]; then
    mkdir -p "$HOME/.config/autostart"
    cat > "$HOME/.config/autostart/llm-scanner.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=LLM Scanner
Comment=Open LLM Scanner when you log in
Exec=$APP_DIR/llm-scanner.sh
Icon=llm-scanner
Terminal=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=5
EOF
    info "LLM Scanner will open in your browser when you log in"
  else
    rm -f "$HOME/.config/autostart/llm-scanner.desktop"
  fi
fi

# ---------------------------------------------------------------- start
if [ "$START" = "1" ] && [ "$TEST_MODE" != "1" ]; then
  step "Starting LLM Scanner"
  systemctl --user restart ollama llm-scanner
  for _ in $(seq 1 120); do curl -s -o /dev/null "http://127.0.0.1:$PORT" && break; sleep 1; done
  if curl -s -o /dev/null "http://127.0.0.1:$PORT"; then
    info "Running at http://127.0.0.1:$PORT"
  else
    warn "The app didn't respond yet. Check: journalctl --user -u llm-scanner -n 50"
  fi
  if [ "$PULL_MODELS" = "1" ] && [ -n "$DATA_FILE" ] && [ -s "$DATA_DIR_SRC/ollama-models.txt" ]; then
    step "Re-downloading your Ollama chat models"
    while read -r model; do
      [ -n "$model" ] || continue
      info "Pulling $model"
      "$LOCAL_BIN/ollama" pull "$model" || warn "Couldn't download $model"
    done < "$DATA_DIR_SRC/ollama-models.txt"
  fi
  command -v xdg-open >/dev/null && xdg-open "http://127.0.0.1:$PORT" >/dev/null 2>&1 &
fi

step "Done"
info "Open LLM Scanner from your app menu, or go to http://127.0.0.1:$PORT"
if [ -n "$DATA_FILE" ] && [ -s "$DATA_DIR_SRC/ollama-models.txt" ] && [ "$PULL_MODELS" != "1" ]; then
  info "Your chat models weren't copied. Re-download them in the app, or re-run with: --data FILE --pull-models"
fi
info "LLM Scanner checks GitHub for new versions at startup and every 6 hours, and shows an Update button."
if [ -n "$DATA_FILE" ]; then
  info "Image models: download them again from the Images tab (the ones you used are listed with a Download button)."
else
  info "Download chat models from the Models tab and image models from the Images tab."
fi
