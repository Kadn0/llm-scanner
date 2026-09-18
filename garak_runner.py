"""Run garak with LLM Scanner's Ollama tweaks.

garak's Ollama generator has no way to control "thinking" on reasoning models (Qwen3.x, DeepSeek-R1, ...).
With thinking on, the model can spend the whole token budget reasoning and return an empty answer, which
garak would score as harmless. LLM_SCANNER_THINK=0/1 sets `think` on every chat call, but only for models
that report the thinking capability (other models reject the parameter).
"""

import os
import sys

import ollama
import requests

think = os.environ.get("LLM_SCANNER_THINK")
model = os.environ.get("LLM_SCANNER_MODEL", "")

_chat_for_scan = ollama.Client.chat


def _scan_chat(self, *args, **kwargs):
    kwargs.setdefault("keep_alive", "15m")  # Ollama's default is to unload immediately; keep it for the scan
    return _chat_for_scan(self, *args, **kwargs)


ollama.Client.chat = _scan_chat

if think in ("0", "1") and model:
    try:
        caps = requests.post("http://127.0.0.1:11434/api/show", json={"model": model}, timeout=30).json().get("capabilities", [])
    except Exception:
        caps = []
    if "thinking" in caps:
        _chat = ollama.Client.chat

        def chat(self, *args, **kwargs):
            kwargs.setdefault("think", think == "1")
            return _chat(self, *args, **kwargs)

        ollama.Client.chat = chat
        print(f"LLM Scanner: thinking {'on' if think == '1' else 'off'} for {model}", flush=True)

from garak import cli  # noqa: E402

cli.main(sys.argv[1:])
