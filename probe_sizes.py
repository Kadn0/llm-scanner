"""Count how many prompts each garak probe sends, so the Probes tab can say which probes take long.

Prints {"garak": version, "probes": {name: prompt count}, "reply_tokens": {name: tokens}} as JSON. A count of 0 means
the probe writes its prompts during the scan (an attacker model or a search) or inspects files instead. reply_tokens
lists the probes that let each reply run longer than garak's usual limit (divergence asks for a word repeated forever,
so every reply runs to that limit). Probes that can't be loaded here (missing API keys or optional packages) are left
out. garak's prompt cap is already applied to the counts.
"""

import contextlib
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


def main():
    _config.load_base_config()
    signal.signal(signal.SIGALRM, _alarm)
    counts, reply_tokens = {}, {}
    for key in _plugins.enumerate_plugins("probes", skip_base_classes=True):
        name = key[0] if isinstance(key, tuple) else key
        signal.alarm(30)  # a probe that fetches a large dataset is skipped rather than holding up the rest
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                probe = _plugins.load_plugin(name)
            short = name.removeprefix("probes.")
            counts[short] = len(getattr(probe, "prompts", None) or [])
            if getattr(probe, "override_maxlen", False) and getattr(probe, "new_max_tokens", None):
                reply_tokens[short] = int(probe.new_max_tokens)
        except (_TooSlow, Exception):
            pass
        finally:
            signal.alarm(0)
    print(json.dumps({"garak": importlib.metadata.version("garak"), "probes": counts, "reply_tokens": reply_tokens},
                     sort_keys=True))


if __name__ == "__main__":
    main()
