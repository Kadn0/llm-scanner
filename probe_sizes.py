"""Count how many prompts each garak probe sends, so the Probes tab can say which probes take long.

Prints JSON: {"garak": version, "probes": {name: prompt count}, "reply_tokens": {name: tokens}, "prompt_chars":
{name: average characters per prompt}, "turns": {name: average turns per prompt}, "judged": [names],
"scoring_models": [names], "uncapped": {name: prompts it has}, "prompt_cap": n}. A count of 0 means the probe writes
its prompts during the scan (an attacker model or a search) or inspects files instead. reply_tokens lists probes that let each reply run longer than garak's usual limit
(divergence asks for a word repeated forever, so every reply runs to that limit). judged lists probes whose replies an
AI model judges (one more model reply per reply); scoring_models lists probes scored by a Hugging Face model that is
downloaded the first time. Probes that can't be loaded here (missing API keys or optional packages) are left out.
garak's prompt cap (soft_probe_prompt_cap, 256 by default) is applied to the counts, as in a scan; uncapped lists the
probes the cap cuts down, with how many prompts they have.
"""

import contextlib
import importlib
import importlib.metadata
import io
import json
import logging
import signal

logging.disable(logging.CRITICAL)

from garak import _config, _plugins  # noqa: E402


class _TooSlow(Exception):
    pass


def _alarm(*_):
    raise _TooSlow()


def _prompt_text(prompt):
    """(text, user turns) of one prompt: a string, or a conversation of several turns."""
    if isinstance(prompt, str):
        return prompt, 1
    turns = getattr(prompt, "turns", None) or []
    texts = [str(getattr(getattr(t, "content", None), "text", "") or "") for t in turns]
    users = sum(1 for t in turns if getattr(t, "role", "user") == "user")
    return " ".join(texts) if turns else str(prompt), max(1, users)


def _detector_kinds():
    """(judge detectors, Hugging Face model detectors), from their classes; nothing is loaded or downloaded."""
    from garak.detectors.base import HFDetector

    judges, models = set(), set()
    for key in _plugins.enumerate_plugins("detectors", skip_base_classes=True):
        name = (key[0] if isinstance(key, tuple) else key).removeprefix("detectors.")
        module, _, cls = name.rpartition(".")
        try:
            klass = getattr(importlib.import_module(f"garak.detectors.{module}"), cls)
        except Exception:
            continue
        if module == "judge":
            judges.add(name)
        elif isinstance(klass, type) and issubclass(klass, HFDetector):
            models.add(name)
    return judges, models


def _uncapped_counts(names):
    """How many prompts the capped probes have without garak's cap."""
    cap = _config.run.soft_probe_prompt_cap
    _config.run.soft_probe_prompt_cap = 10**9
    out = {}
    try:
        for name in names:
            signal.alarm(30)
            try:
                module, _, cls = name.rpartition(".")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    probe = getattr(importlib.import_module(f"garak.probes.{module}"), cls)()  # load_plugin reuses
                n = len(getattr(probe, "prompts", None) or [])
                if n > cap:
                    out[name] = n
            except (_TooSlow, Exception):
                pass
            finally:
                signal.alarm(0)
    finally:
        _config.run.soft_probe_prompt_cap = cap
    return out


def main():
    _config.load_base_config()
    signal.signal(signal.SIGALRM, _alarm)
    counts, reply_tokens, prompt_chars, turns, judged, scoring_models = {}, {}, {}, {}, [], []
    judges, models = _detector_kinds()
    for key in _plugins.enumerate_plugins("probes", skip_base_classes=True):
        name = key[0] if isinstance(key, tuple) else key
        signal.alarm(30)  # a probe that fetches a large dataset is skipped rather than holding up the rest
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                probe = _plugins.load_plugin(name)
            short = name.removeprefix("probes.")
            prompts = list(getattr(probe, "prompts", None) or [])
            counts[short] = len(prompts)
            if getattr(probe, "override_maxlen", False) and getattr(probe, "new_max_tokens", None):
                reply_tokens[short] = int(probe.new_max_tokens)
            if prompts:
                texts = [_prompt_text(x) for x in prompts]
                prompt_chars[short] = round(sum(len(t) for t, _ in texts) / len(texts))
                turns[short] = round(sum(n for _, n in texts) / len(texts), 2)
            detectors = [getattr(probe, "primary_detector", None)] + list(getattr(probe, "extended_detectors", None) or [])
            detectors += list(getattr(probe, "recommended_detector", None) or [])
            detectors = {d.removeprefix("detectors.") for d in detectors if d}
            if detectors & judges:
                judged.append(short)
            if detectors & models:
                scoring_models.append(short)
        except (_TooSlow, Exception):
            pass
        finally:
            signal.alarm(0)
    cap = _config.run.soft_probe_prompt_cap
    uncapped = _uncapped_counts([n for n, c in counts.items() if c == cap])
    print(json.dumps({"garak": importlib.metadata.version("garak"), "probes": counts, "reply_tokens": reply_tokens,
                      "prompt_chars": prompt_chars, "turns": turns, "judged": sorted(judged),
                      "scoring_models": sorted(scoring_models), "uncapped": uncapped, "prompt_cap": cap},
                     sort_keys=True))


if __name__ == "__main__":
    main()
