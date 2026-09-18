#!/usr/bin/env bash
# Start Ollama and LLM Scanner (user services) and open the app in the default browser.
systemctl --user start ollama llm-scanner
for _ in $(seq 1 120); do
  curl -s -o /dev/null http://127.0.0.1:7861 && break
  sleep 0.5
done
xdg-open http://127.0.0.1:7861
