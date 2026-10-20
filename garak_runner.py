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


_failing = [False]


def _scan_chat(self, *args, **kwargs):
    # Keep the model loaded for the whole scan, however long garak spends scoring or the scan is paused (Ollama's
    # default here is to unload at once). LLM Scanner unloads it when the scan ends, halts or restarts a probe.
    kwargs.setdefault("keep_alive", -1)
    try:
        response = _chat_for_scan(self, *args, **kwargs)
        if _failing[0]:
            _failing[0] = False
            print("LLM Scanner: request answered after a retry", flush=True)
        return response
    except Exception as e:
        _failing[0] = True
        # garak retries timeouts and Ollama errors without limit and without printing anything, so a model that
        # never answers would look frozen. LLM Scanner counts these lines to tell a stuck scan from a slow one.
        kind = "timeout" if "timeout" in type(e).__name__.lower() else "error"
        print(f"LLM Scanner: request {kind}: {type(e).__name__}: {str(e)[:200]}; garak will retry", flush=True)
        raise


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


# Nothing the model has answered is lost when a scan is cut short (a crash, a restart, a stuck probe, the computer
# shutting down). Every answer is saved the moment it arrives to LLM_SCANNER_RESPONSES (one JSON line, flushed to
# disk), and when the scan continues as its next part, prompts answered before (LLM_SCANNER_REUSE: the earlier parts'
# files) get their saved answers back instead of being sent to the model again. garak then scores them as usual. So
# that a probe picks the same prompts again where it samples them at random, each probe draws its random numbers
# from LLM_SCANNER_SEED and its own name.
import functools  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
from collections import defaultdict, deque  # noqa: E402

import garak.attempt  # noqa: E402
import garak.probes.base  # noqa: E402


def _probe_name(probe):
    return probe.__class__.__module__.replace("garak.probes.", "") + "." + probe.__class__.__name__


def _prompt_key(prompt):
    turns = getattr(prompt, "turns", None) or []
    return json.dumps([[t.role, getattr(t.content, "text", None)] for t in turns], ensure_ascii=False)


def _load_saved(paths):
    """{(probe, prompt): queue of answer lists} from earlier parts. A line cut off by the interruption is skipped."""
    saved = defaultdict(deque)
    for path in filter(None, paths.split(os.pathsep)):
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        saved[(r["probe"], r["prompt"])].append(r["outputs"])
                    except (ValueError, KeyError, TypeError):
                        continue
        except OSError:
            continue
    return saved


_saved = _load_saved(os.environ.get("LLM_SCANNER_REUSE", ""))
_out_path = os.environ.get("LLM_SCANNER_RESPONSES")
_out = open(_out_path, "a", encoding="utf-8") if _out_path else None
_reused = set()


def _save_answer(probe, prompt, outputs):
    if not _out:
        return
    _out.write(json.dumps({"probe": probe, "prompt": prompt, "outputs": outputs}, ensure_ascii=False) + "\n")
    _out.flush()
    try:
        os.fsync(_out.fileno())  # answers take seconds each, so this costs nothing and survives a power cut
    except OSError:
        pass


_execute_attempt = garak.probes.base.Probe._execute_attempt


@functools.wraps(_execute_attempt)
def _execute_attempt_saved(self, this_attempt):
    probe, prompt = _probe_name(self), _prompt_key(this_attempt.prompt)
    gen = self.generator
    queue = _saved.get((probe, prompt))
    answers = queue.popleft() if queue and len(queue[0]) == self.generations else None
    real = gen.generate
    had_own = "generate" in gen.__dict__

    def generate(prompt_conv, generations_this_call=1, *args, **kwargs):
        if answers is not None:
            if probe not in _reused:
                _reused.add(probe)
                print(f"\nLLM Scanner: reusing answers saved before the interruption for {probe}", flush=True)
            outputs = [garak.attempt.Message(text=a["text"], lang=a.get("lang")) for a in answers]
        else:
            outputs = real(prompt_conv, generations_this_call, *args, **kwargs)
        if outputs and len(outputs) == generations_this_call and all(
                o is not None and getattr(o, "text", None) is not None for o in outputs):
            _save_answer(probe, prompt, [{"text": o.text, "lang": getattr(o, "lang", None)} for o in outputs])
        return outputs

    gen.generate = generate
    try:
        return _execute_attempt(self, this_attempt)
    finally:
        if had_own:
            gen.generate = real
        else:
            del gen.generate


garak.probes.base.Probe._execute_attempt = _execute_attempt_saved

_seed = os.environ.get("LLM_SCANNER_SEED")
if _seed:
    _probe_init, _probe_probe = garak.probes.base.Probe.__init__, garak.probes.base.Probe.probe

    @functools.wraps(_probe_init)  # garak checks the signature of a probe's __init__
    def _seeded_init(self, *args, **kwargs):
        random.seed(f"{_seed}:{_probe_name(self)}")
        _probe_init(self, *args, **kwargs)

    @functools.wraps(_probe_probe)
    def _seeded_probe(self, *args, **kwargs):
        random.seed(f"{_seed}:{_probe_name(self)}:probe")
        return _probe_probe(self, *args, **kwargs)

    garak.probes.base.Probe.__init__ = _seeded_init
    garak.probes.base.Probe.probe = _seeded_probe

from garak import cli  # noqa: E402

cli.main(sys.argv[1:])
