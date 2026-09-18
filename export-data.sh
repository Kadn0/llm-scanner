#!/usr/bin/env bash
# Export your LLM Scanner data to one file, to restore on another machine with: ./install.sh --data FILE
# Includes chats (and their attachments), generated images, probe groups, the list of image models you added,
# scan reports, and the names of your Ollama chat models. Model files themselves are NOT included.
set -euo pipefail

OUT="$HOME/llm-scanner-data-$(date +%Y%m%d-%H%M).tar.gz"
case "${1:-}" in
  -o) OUT="$2" ;;
  -h|--help) echo "Usage: ./export-data.sh [-o FILE]"; exit 0 ;;
  "") ;;
  *) echo "Unknown option: $1"; exit 1 ;;
esac

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
D="$STAGE/llm-scanner-data"
mkdir -p "$D"

for item in chats attachments images probe_groups.json image_models.json; do
  if [ -e "$APP_DIR/data/$item" ]; then cp -r "$APP_DIR/data/$item" "$D/"; fi
done
rm -rf "$D/attachments/staging"
if [ -d "$HOME/.local/share/garak/garak_runs" ]; then cp -r "$HOME/.local/share/garak/garak_runs" "$D/garak_runs"; fi
if "$HOME/.local/bin/ollama" list >/dev/null 2>&1; then
  "$HOME/.local/bin/ollama" list | awk 'NR > 1 {print $1}' > "$D/ollama-models.txt"
fi

count() { find "$D/$1" -name "$2" 2>/dev/null | wc -l; }
echo "Exporting: $(count chats '*.json') chats, $(count images '*.png') images, $(count garak_runs '*.report.jsonl') scan reports," \
     "$(wc -l < "$D/ollama-models.txt" 2>/dev/null || echo 0) chat model names"
tar -czf "$OUT" -C "$STAGE" llm-scanner-data
echo "Saved $OUT ($(du -h "$OUT" | cut -f1))"
echo "On the new machine: ./install.sh --data $(basename "$OUT")   (add --pull-models to re-download chat models)"
