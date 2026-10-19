"""Count how many prompts each garak probe sends, so the Probes tab can say which probes take long.

Prints {"garak": version, "probes": {name: prompt count}} as JSON. A count of 0 means the probe writes its prompts
during the scan (an attacker model or a search) or inspects files instead. Probes that can't be loaded here (missing
API keys or optional packages) are left out. garak's prompt cap is already applied to the counts.
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
    counts = {}
    for key in _plugins.enumerate_plugins("probes", skip_base_classes=True):
        name = key[0] if isinstance(key, tuple) else key
        signal.alarm(30)  # a probe that fetches a large dataset is skipped rather than holding up the rest
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                probe = _plugins.load_plugin(name)
            counts[name.removeprefix("probes.")] = len(getattr(probe, "prompts", None) or [])
        except (_TooSlow, Exception):
            pass
        finally:
            signal.alarm(0)
    print(json.dumps({"garak": importlib.metadata.version("garak"), "probes": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
