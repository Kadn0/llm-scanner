#!/usr/bin/env bash
# Remove LLM Scanner from this machine. Asks before deleting anything.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./uninstall.sh [options]

  --keep-data        keep ~/llm-scanner/data (chats, images, probe groups) and the garak scan reports
  --remove-models    also delete downloaded Ollama chat models (~/.ollama/models)
  -y, --yes          don't ask for confirmation
  -h, --help         show this help

Image models live in ~/llm-scanner/data and are removed with the app unless --keep-data is used.
EOF
}

KEEP_DATA=0
REMOVE_MODELS=0
YES=0
for arg in "$@"; do
  case "$arg" in
    --keep-data) KEEP_DATA=1 ;;
    --remove-models) REMOVE_MODELS=1 ;;
    -y|--yes) YES=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg"; usage; exit 1 ;;
  esac
done

APP_DIR="$HOME/llm-scanner"
echo "This will permanently remove:"
echo "  - the LLM Scanner and Ollama services, app menu entry and login startup"
echo "  - $HOME/.local/ollama (Ollama) and $HOME/.local/sd-cpp (image engine)"
if [ "$KEEP_DATA" = "1" ]; then
  echo "  - $APP_DIR, except its data folder"
else
  echo "  - $APP_DIR, including your chats, images, image models and probe groups"
  echo "  - garak scan reports in $HOME/.local/share/garak"
fi
[ "$REMOVE_MODELS" = "1" ] && echo "  - all Ollama chat models in $HOME/.ollama/models"
if [ "$YES" != "1" ]; then
  read -r -p "Continue? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }
fi

systemctl --user disable --now llm-scanner ollama >/dev/null 2>&1 || true
rm -f "$HOME/.config/systemd/user/llm-scanner.service" "$HOME/.config/systemd/user/ollama.service"
systemctl --user daemon-reload >/dev/null 2>&1 || true
rm -f "$HOME/.local/share/applications/llm-scanner.desktop" "$HOME/.config/autostart/llm-scanner.desktop"
rm -rf "$HOME/.local/ollama" "$HOME/.local/sd-cpp"
rm -f "$HOME/.local/bin/ollama"

if [ "$KEEP_DATA" = "1" ]; then
  find "$APP_DIR" -mindepth 1 -maxdepth 1 ! -name data -exec rm -rf {} +
  echo "Kept your data in $APP_DIR/data and $HOME/.local/share/garak"
else
  rm -rf "$APP_DIR" "$HOME/.local/share/garak"
fi
[ "$REMOVE_MODELS" = "1" ] && rm -rf "$HOME/.ollama/models"
echo "LLM Scanner has been removed. (uv in ~/.local/bin was left in place; delete it if you don't use it.)"
