"""Scenario tests across every tab: the real app runs once on a spare port, against a fake Ollama / ollama.com /
Hugging Face and a scratch data folder inside tests/, and each test presses the same Gradio events the browser's
buttons do (through gradio_client). Exactly 100 tests: 63 written out, 37 generated from the tables at the bottom.
Run from the app folder:  .venv/bin/python -m unittest discover -s tests -p test_scenarios_e2e.py"""

import html
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import unittest
import urllib.parse
from datetime import datetime, timedelta
from http.server import ThreadingHTTPServer
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
APP_DIR = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))
from test_ui import FAKE_ENGINE, PY, FakeOllama, free_port, model_entry, write_run  # noqa: E402

CURRENT = "Current selection (Probes tab)"
GROUP = "Saved group: "
FULL, HITLOG = "Full report (report.jsonl)", "Hitlog (failing responses)"
CANNOT_LOAD = "Ollama can't load this format"
IDLE_PROGRESS = "No scan running"

# ollama.com's library as the fake serves it: model -> [(tag, size in GB)]
OLLAMA_LIB = {
    "qwen3": [("0.6b", "0.5"), ("1.7b", "0.9"), ("4b-mlx", "2.0"), ("235b", "900")],
    "qwen3-coder": [("30b", "900")],
    "gemma3n": [("e2b", "0.5")],
    "gemma3": [("1b", "0.8")],
    "kimi-cloud": [],
    "giant": [("900b", "900"), ("1t", "1200")],
    "slowpoke": [("1b", "0.1")],
}

# Hugging Face repositories as the fake serves them
HF_REPOS = {
    "acme/Tiny-Chat-GGUF": {
        "gated": True, "private": False,
        "files": [("Tiny-Chat-PQ2_0.gguf", 200_000_000), ("Tiny-Chat-Q4_K_M.gguf", 400_000_000),
                  ("Tiny-Chat-Q8_0.gguf", 700_000_000), ("mmproj-Tiny-Chat-F16.gguf", 100_000_000),
                  ("BF16/Tiny-Chat-BF16-00001-of-00002.gguf", 5_000_000_000), ("README.md", 1000)],
        "info": {"downloads": 1_234_567, "likes": 42, "gated": "auto", "private": False,
                 "cardData": {"base_model": ["acme/Tiny-Base"], "license": "apache-2.0"}}},
    "acme/Tiny-Chat-abliterated-GGUF": {
        "gated": False, "private": False, "files": [("tiny-chat-abliterated.Q4_K_M.gguf", 460_000_000)],
        "info": {"downloads": 10}},
    "bob/Tiny-Secret-GGUF": {
        "gated": False, "private": True, "files": [("secret-Q5_K_M.gguf", 520_000_000)],
        "info": {"downloads": 3, "private": True}},
}

HELLO = [{"message": {"content": "Hello "}}, {"message": {"content": "there"}},
         {"message": {"content": ""}, "done": True, "prompt_eval_count": 12, "eval_count": 2,
          "eval_duration": 1e8, "load_duration": 0}]

# the image engine from test_ui, which also fails like a real out-of-memory run when the prompt says FAILME
ENGINE = FAKE_ENGINE.replace("out = sys.argv", "if 'FAILME' in sys.argv[sys.argv.index('-p') + 1]:\n"
                             "    print('ggml: out of video memory', flush=True)\n    sys.exit(3)\nout = sys.argv", 1)

IMAGE_SIZES = {"Square 1024 x 1024": (1024, 1024), "Square 768 x 768 (faster)": (768, 768),
               "Portrait 768 x 1024": (768, 1024), "Landscape 1024 x 768": (1024, 768),
               "Wide 1280 x 720": (1280, 720), "Small 512 x 512 (fastest)": (512, 512)}

MODELS = [("broken-pq2:latest", 1_100_000_000), ("tiny:latest", 270_000_000), ("thinker:8b", 5_200_000_000),
          ("big:70b", 40_000_000_000)]

RUNS = {  # prefix -> (write_run options, seconds newer than the oldest)
    "hf.co_org_alpha_20260917-090000": ({}, 500),
    "beta_20260917-100000": ({"complete": False}, 400),
    "gamma_20260917-110000": ({"hits": False}, 300),
    "delta_20260917-120000": ({"complete": False, "evals": False, "hits": False}, 200),
    "victim1_20260917-130000": ({}, 100),
    "victim2_20260917-140000": ({"complete": False}, 50),
    "victim3_20260917-150000": ({}, 20),
}
ALPHA, BETA, GAMMA, DELTA = (p + ".report.html" for p in list(RUNS)[:4])
VICTIM1, VICTIM2, VICTIM3 = (p + ".report.html" for p in list(RUNS)[4:])


class E2EOllama(FakeOllama):
    """test_ui's fake Ollama, plus: an ollama.com library that filters by the query, a small Hugging Face API, a
    switch that makes Ollama look stopped, models loaded in memory, a model Ollama can't load, a reasoning model,
    and a model whose chat fails."""
    down = False
    loaded = []
    hf_seen = []   # (path, Authorization header) of every Hugging Face API request
    web_seen = []  # search terms sent to ollama.com

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        path, query = url.path, urllib.parse.parse_qs(url.query)
        if path == "/search":
            term = (query.get("q") or [""])[0].lower()
            E2EOllama.web_seen.append(term)
            if term == "explode":
                return self.reply({"error": "upstream exploded"}, 500)
            items = "".join(f'<li x-test><a href="/library/{n}"><p class="max-w-lg break-words">{n} test model</p>'
                            f'<span class="bg-indigo-50 x"> tools </span><span >1.2M</span><span>&nbsp;Pulls</span>'
                            f'</a></li>' for n in OLLAMA_LIB if term and term in n)
            return self.html(f"<ul>{items}</ul>")
        if path.startswith("/library/") and path.endswith("/tags"):
            name = path.split("/")[2]
            return self.html("".join(f'<a href="/library/{name}:{tag}" class="md:hidden flex"><span class="font-mono">'
                                     f' abc123 </span> • {size}GB</a>' for tag, size in OLLAMA_LIB.get(name, [])))
        if path == "/api/version":
            if E2EOllama.down:
                return self.reply({"error": "connection refused"}, 503)
            return self.reply({"version": "0.34.2"})
        if path == "/api/ps":
            return self.reply({"models": E2EOllama.loaded})
        auth = self.headers.get("Authorization") or ""
        if path.startswith("/api/models") or path == "/api/whoami-v2":
            E2EOllama.hf_seen.append((path, auth))
        if path == "/api/whoami-v2":
            if "hf_down" in auth:
                return self.reply({"error": "service unavailable"}, 500)
            if "hf_good" in auth:
                return self.reply({"name": "e2e-account"})
            return self.reply({"error": "Invalid credentials"}, 401)
        if path == "/api/models":
            term = (query.get("search") or [""])[0].lower()
            return self.reply([{"id": r, "gated": v["gated"], "private": v["private"]}
                               for r, v in HF_REPOS.items() if term in r.lower()])
        if path.startswith("/api/models/"):
            rest = path[len("/api/models/"):]
            tree = rest.endswith("/tree/main")
            repo = rest[:-len("/tree/main")] if tree else rest
            if repo not in HF_REPOS:
                return self.reply({"error": "Repository not found"}, 404)
            if tree:
                return self.reply([{"type": "file", "path": p, "size": s} for p, s in HF_REPOS[repo]["files"]])
            return self.reply({"id": repo, **HF_REPOS[repo]["info"]})
        return super().do_GET()

    def do_POST(self):
        if self.path not in ("/api/show", "/api/chat"):
            return super().do_POST()
        data = self.body()
        FakeOllama.calls.append((self.path, data))
        model = data.get("model", "")
        if self.path == "/api/show":
            if "broken" in model:
                return self.reply({"error": 'read GGUF metadata "/blobs/sha256-e2e": unsupported tensor type PQ2_0'},
                                  500)
            return self.reply({"capabilities": ["completion", "thinking"] if "thinker" in model else ["completion"]})
        if "big" in model:
            return self.stream([{"error": "model requires more system memory (64.0 GiB) than is available"}])
        return self.stream(HELLO)


def garak_catalog():
    """The probe list the app shows, read the same way the app reads it (garak's plugin cache)."""
    paths = [Path.home() / ".cache/garak/resources/plugin_cache.json"]
    try:
        spec = importlib.util.find_spec("garak")
        if spec and spec.origin:
            paths.append(Path(spec.origin).parent / "resources/plugin_cache.json")
    except (ImportError, ValueError):
        pass
    data = None
    for p in paths:
        if p.exists():
            try:
                data = json.loads(p.read_text())["probes"]
            except Exception:
                return {}
            break
    if data is None:
        return {}
    cat = {}
    for key, v in data.items():
        name = key.removeprefix("probes.")
        if "text" not in v.get("modality", {}).get("in", ["text"]):
            continue
        cat[name] = {"module": name.split(".")[0], "desc": (v.get("description") or "").strip().split("\n")[0],
                     "active": v.get("active", True)}
    return dict(sorted(cat.items()))


CAT = garak_catalog()


def filtered(query="", module="All categories", show_all=False):
    q = (query or "").lower().strip()
    return [n for n, v in CAT.items() if (show_all or v["active"])
            and (module in (None, "", "All categories") or v["module"] == module)
            and (not q or q in n.lower() or q in v["desc"].lower())]


def expand(specs):
    out = []
    for s in specs:
        if s in CAT:
            out.append(s)
        else:
            out += [n for n, v in CAT.items() if v["module"] == s and v["active"]]
    return list(dict.fromkeys(out))


def summary_of(selected):
    if not selected:
        return "No probes selected."
    shown = ", ".join(selected[:8]) + (f", and {len(selected) - 8} more" if len(selected) > 8 else "")
    return f"**{len(selected)} selected:** {shown}"


def text_of(message):
    """Plain text of a chatbot message, whichever shape Gradio sends its content in."""
    content = message.get("content") if isinstance(message, dict) else message
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or ""
    if isinstance(content, (list, tuple)):
        return "".join(text_of({"content": c}) if not isinstance(c, dict) else (c.get("text") or "") for c in content)
    return "" if content is None else str(content)


def noop(value):
    """A gr.update() that changes nothing."""
    return isinstance(value, dict) and "value" not in value


def values(update):
    return [c[1] if isinstance(c, (list, tuple)) else c for c in update.get("choices", [])]


def labels(update):
    return {(c[1] if isinstance(c, (list, tuple)) else c): (c[0] if isinstance(c, (list, tuple)) else c)
            for c in update.get("choices", [])}


class Scenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gradio_client import Client
        cls.proc = None
        cls.tmp = TESTS_DIR / f".e2e-scratch-{os.getpid()}"
        shutil.rmtree(cls.tmp, ignore_errors=True)
        cls.data, cls.runs, cls.sd = cls.tmp / "data", cls.tmp / "runs", cls.tmp / "sd"
        for d in (cls.data, cls.runs, cls.sd, cls.tmp / "gradio-tmp", cls.tmp / "tmp"):
            d.mkdir(parents=True)
        oldest = time.time() - 3600
        for prefix, (options, newer) in RUNS.items():
            write_run(cls.runs, prefix, **options)
            for f in cls.runs.glob(prefix + ".*"):
                os.utime(f, (oldest + newer, oldest + newer))
        assert "FAILME" in ENGINE
        (cls.sd / "sd-cli").write_text(ENGINE)
        (cls.sd / "sd-cli").chmod(0o755)
        for rel in ("image-models/z-image-turbo/z_image_turbo-Q4_K.gguf",
                    "image-models/shared/unsloth-qwen3-4b-instruct-2507-gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                    "image-models/shared/comfy-org-z-image-turbo/ae.safetensors",
                    "image-models/dreamy-sd15/dreamshaper_8.safetensors"):
            (cls.data / rel).parent.mkdir(parents=True, exist_ok=True)
            (cls.data / rel).write_bytes(b"x")
        (cls.data / "image_models.json").write_text(json.dumps({"dreamy-sd15": {
            "name": "Dreamy SD15", "family": "sd15", "desc": "E2E custom model.",
            "files": {"model": ["e2e/dreamshaper-8", "dreamshaper_8.safetensors", False]},
            "steps": 25, "size": "Small 512 x 512 (fastest)", "args": ["--cfg-scale", "7.0"]}}))
        FakeOllama.models[:] = [dict(model_entry(n), size=s) for n, s in MODELS]
        E2EOllama.down, E2EOllama.loaded = False, []
        cls.ollama = ThreadingHTTPServer(("127.0.0.1", free_port()), E2EOllama)
        threading.Thread(target=cls.ollama.serve_forever, daemon=True).start()
        cls.fake_url = f"http://127.0.0.1:{cls.ollama.server_port}"
        port = free_port()
        env = os.environ | {"LLM_SCANNER_CHECK": "1", "LLM_SCANNER_PORT": str(port), "LLM_SCANNER_DATA": str(cls.data),
                            "LLM_SCANNER_GARAK_RUNS": str(cls.runs), "LLM_SCANNER_SD_DIR": str(cls.sd),
                            "OLLAMA_URL": cls.fake_url, "LLM_SCANNER_HF_API": cls.fake_url + "/api",
                            "LLM_SCANNER_OLLAMA_WEB": cls.fake_url,
                            "HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": "", "HUGGINGFACE_TOKEN": "",
                            "GRADIO_TEMP_DIR": str(cls.tmp / "gradio-tmp"), "TMPDIR": str(cls.tmp / "tmp"),
                            "OLLAMA_MODELS": str(cls.tmp / "ollama-models"),  # never near the real model store
                            "GRADIO_ANALYTICS_ENABLED": "False"}
        cls.log = open(cls.tmp / "app.log", "wb")
        cls.proc = subprocess.Popen([PY, "app.py", "--no-browser"], cwd=APP_DIR, env=env, stdout=cls.log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + 120
        while True:
            try:
                cls.client = Client(f"http://127.0.0.1:{port}/", verbose=False, download_files=False,
                                    analytics_enabled=False)
                break
            except Exception:
                if time.time() > deadline or cls.proc.poll() is not None:
                    tail = (cls.tmp / "app.log").read_text(errors="replace")[-1500:]
                    cls.tearDownClass()
                    raise RuntimeError("the app did not start: " + tail)
                time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        if cls.proc is not None and cls.proc.poll() is None:
            try:
                os.killpg(cls.proc.pid, 15)
                cls.proc.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(cls.proc.pid, 9)
                except ProcessLookupError:
                    pass
        if getattr(cls, "ollama", None) is not None:
            cls.ollama.shutdown()
            cls.ollama.server_close()
        if getattr(cls, "log", None) is not None:
            cls.log.close()
        E2EOllama.down, E2EOllama.loaded = False, []
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.client.reset_session()  # every test starts like a freshly opened page

    # ------------------------------------------------------------ helpers
    def call(self, api, *args):
        return self.client.predict(*args, api_name=api)

    def config_choices(self, match):
        for c in self.client.config["components"]:
            props = c.get("props") or {}
            if c.get("type") == "dropdown" and props.get("choices") and match(props):
                return values(props)
        return []

    def need_catalog(self):
        if not CAT:
            self.skipTest("garak's probe catalog is not available to the test")
        if not getattr(type(self), "_catalog_checked", False):
            _, count = self.call("/filter_probes", "", "All categories", False)
            if f"of {len(CAT)} probes" not in count:
                self.skipTest("the app reads a different garak catalog than the test")
            type(self)._catalog_checked = True

    def need_module(self, *modules):
        self.need_catalog()
        offered = self.config_choices(lambda p: "All categories" in values(p))
        for m in modules:
            if m not in offered or not filtered(module=m):
                self.skipTest(f"garak has no active {m} probes here")

    def pick_module(self):
        self.need_catalog()
        mod = "goodside" if filtered(module="goodside") else next(v["module"] for n, v in CAT.items() if v["active"])
        return mod, filtered(module=mod)

    def chats(self):
        folder = self.data / "chats"
        return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(folder.glob("*.json"))] if folder.exists() \
            else []

    def saved_chat(self, first):
        found = [c for c in self.chats() if c["messages"] and c["messages"][0]["content"] == first]
        self.assertEqual(len(found), 1, f"one saved chat starting with {first!r}")
        return found[0]

    def chat_calls(self):
        return [data for path, data in FakeOllama.calls if path == "/api/chat"]

    def images(self):
        folder = self.data / "images"
        return set(folder.glob("*.png")) if folder.exists() else set()

    def generate(self, model, prompt, size, steps, seed):
        before = self.images()
        out = self.call("/generate_image", model, prompt, size, steps, seed)
        new = self.images() - before
        self.assertEqual(len(new), 1, out[0])
        png = new.pop()
        return out, png, json.loads(png.with_suffix(".json").read_text())

    def groups(self):
        f = self.data / "probe_groups.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def settings(self):
        f = self.data / "settings.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def pending(self):
        f = self.data / "downloads.json"
        entries = json.loads(f.read_text()) if f.exists() else []
        return [e if isinstance(e, str) else e["ref"] for e in entries]

    def search(self, source, query):
        return self.call("/search_models", source, query)

    def versions(self, source, query, repo):
        self.search(source, query)
        return self.call("/list_versions", source, repo)

    def report_labels(self):
        listing, _ = self.call("/report_list_state")
        return labels(listing)

    # ============================================================ status bar (4)
    def test_status_when_ollama_is_stopped(self):
        E2EOllama.down = True
        try:
            status, ollama, activity = self.call("/status_controls")
        finally:
            E2EOllama.down = False
        self.assertIn("Ollama stopped", ollama)
        self.assertIn("dot bad", ollama)
        self.assertNotIn("0.34.2", ollama)
        _, ollama, _ = self.call("/status_controls")
        self.assertIn("Ollama 0.34.2", ollama, "back up once Ollama answers again")

    def test_status_lists_loaded_models_with_gpu_share(self):
        expires = (datetime.now() + timedelta(seconds=620)).isoformat()
        E2EOllama.loaded = [{"name": "tiny:latest", "size": 270_000_000, "size_vram": 270_000_000, "expires_at": expires},
                            {"name": "big:70b", "size": 40_000_000_000, "size_vram": 20_000_000_000}]
        try:
            _, _, activity = self.call("/status_controls")
        finally:
            E2EOllama.loaded = []
        self.assertIn("Model loaded: <b>tiny:latest</b> (0.3 GB, GPU, unloads in 10 min)", activity)
        self.assertIn("Model loaded: <b>big:70b</b> (40.0 GB, 50% GPU / 50% CPU)", activity)
        self.assertNotIn("Idle", activity)

    def test_halted_scan_shows_in_activity_until_stopped(self):
        job = self.data / "scan_job.json"
        self.addCleanup(job.unlink, missing_ok=True)
        job.write_text(json.dumps({"model": "tiny:latest", "scan_type": "Quick check", "requested": [], "queue": [],
                                   "done": ["dan.DanInTheWild"], "failed": [], "skipped": [], "generations": 1,
                                   "timeout": 60, "thinking": False, "prefix": "tiny_latest_20260917-160000",
                                   "parts": 1, "stalls": 0, "halted": True, "stuck_on": "goodside.WhoIsRiley"}))
        self.assertIn("Scan halted, open Scan to continue", self.call("/status_controls")[2])
        self.assertEqual(self.call("/stop_scan"), "Scan ended. Results from the finished probes are in Reports.")
        self.assertFalse(job.exists(), "Stop forgets the halted scan")
        self.assertIn("Idle", self.call("/status_controls")[2])

    def test_refresh_picks_first_runnable_model_and_lists_choices(self):
        scan_model, status, scan_type, group_pick, chat_model = self.call("/refresh_all")
        self.assertEqual(scan_model["value"], "tiny:latest", "the model Ollama can't load is not preselected")
        self.assertIn(CANNOT_LOAD, labels(scan_model)["broken-pq2:latest"])
        self.assertNotIn(CANNOT_LOAD, labels(scan_model)["tiny:latest"])
        self.assertEqual(values(chat_model)[:len(MODELS)], [n for n, _ in MODELS])
        self.assertEqual(chat_model["value"], "broken-pq2:latest")
        self.assertTrue(chat_model.get("interactive"))
        self.assertEqual(values(scan_type)[0], "Quick check")
        self.assertEqual(values(scan_type)[-1], CURRENT)
        self.assertIn("Preset: Quick check", values(group_pick))
        self.assertNotIn("Preset: Full scan", values(group_pick), "Full scan has no probe list to load")
        self.assertIn("RAM", status)

    # ============================================================ Hugging Face token behind the gear (6)
    def test_gear_opens_and_closes_with_empty_box_and_shows_masked_token(self):
        self.addCleanup(self.call, "/remove_hf_token")
        for _ in range(2):  # open, then close
            box, note = self.call("/toggle_hf_panel")
            self.assertEqual(box["value"], "")
            self.assertIn("No token set", note)
        self.call("/save_hf_token", "hf_good_e2e_masked_token")
        box, note = self.call("/toggle_hf_panel")
        self.assertEqual(box["value"], "", "the token is never sent back into the box")
        self.assertIn("Saved token hf_\u2026oken is in use.", note)
        self.assertNotIn("hf_good_e2e_masked_token", note)

    def test_token_hugging_face_cannot_check_is_saved_with_warning(self):
        self.addCleanup(self.call, "/remove_hf_token")
        box, note = self.call("/save_hf_token", "hf_down_e2e_token_9876")
        self.assertEqual(box["value"], "")
        self.assertIn("Saved token hf_\u20269876.", note)
        self.assertIn("be reached to check it", note)
        self.assertEqual(self.settings()["hf_token"], "hf_down_e2e_token_9876")

    def test_short_token_is_fully_masked(self):
        self.addCleanup(self.call, "/remove_hf_token")
        _, note = self.call("/save_hf_token", "hf_good1")
        self.assertIn("Saved. Hugging Face recognizes this token as e2e-account.", note)
        status = self.call("/hf_token_status")
        self.assertIn("Saved token \u2026 is in use.", status)
        self.assertNotIn("hf_good1", status)

    def test_token_save_and_remove_keep_other_settings(self):
        f = self.data / "settings.json"
        original = f.read_text() if f.exists() else None
        self.addCleanup(lambda: f.write_text(original) if original is not None else f.unlink(missing_ok=True))
        f.write_text(json.dumps({"ui_hint": "keep-me"}))
        self.call("/save_hf_token", "hf_good_e2e_keep_1234")
        self.assertEqual(self.settings(), {"ui_hint": "keep-me", "hf_token": "hf_good_e2e_keep_1234"})
        _, note = self.call("/remove_hf_token")
        self.assertIn("No token set", note)
        self.assertEqual(self.settings(), {"ui_hint": "keep-me"})

    def test_saved_token_goes_with_searches_and_stops_after_removal(self):
        self.addCleanup(self.call, "/remove_hf_token")
        self.call("/save_hf_token", "hf_good_e2e_search_5678")
        E2EOllama.hf_seen.clear()
        self.search("Hugging Face", "tiny")
        sent = [auth for path, auth in E2EOllama.hf_seen if path == "/api/models"]
        self.assertTrue(sent and all("hf_good_e2e_search_5678" in a for a in sent), sent)
        self.call("/remove_hf_token")
        E2EOllama.hf_seen.clear()
        self.search("Hugging Face", "tiny")
        sent = [auth for path, auth in E2EOllama.hf_seen if path == "/api/models"]
        self.assertTrue(sent)
        self.assertFalse(any("hf_good_e2e_search_5678" in a for a in sent), "a removed token is not sent")

    def test_rejected_token_keeps_the_saved_one(self):
        self.addCleanup(self.call, "/remove_hf_token")
        self.call("/save_hf_token", "hf_good_e2e_first_1111")
        box, note = self.call("/save_hf_token", "hf_bad_e2e_second_2222")
        self.assertIn("rejected", note)
        self.assertTrue(noop(box), "the box keeps what was typed so it can be corrected")
        self.assertEqual(self.settings()["hf_token"], "hf_good_e2e_first_1111")
        self.assertIn("Saved token hf_\u20261111 is in use.", self.call("/hf_token_status"))

    # ============================================================ models: source, search, versions (12)
    def test_switching_source_resets_fields_both_ways(self):
        query, repo, version, note = self.call("/source_changed", "Ollama")
        self.assertEqual((query["placeholder"], query["value"]), ("qwen3, llama3.2, gemma3", ""))
        self.assertEqual((repo["label"], values(repo), repo["interactive"]), ("Model", [""], False))
        self.assertEqual((values(version), version["value"]), ([""], ""))
        self.assertEqual(note, "")
        query, repo, _, _ = self.call("/source_changed", "Hugging Face")
        self.assertEqual(query["placeholder"], "Qwen3.8-27B, or a Hugging Face link")
        self.assertEqual(repo["label"], "Repository")

    def test_ollama_search_lists_matches_with_pulls_and_capabilities(self):
        repo, version, note, loading = self.search("Ollama", "qwen")
        self.assertEqual(values(repo), ["", "qwen3", "qwen3-coder"])
        self.assertEqual(labels(repo)["qwen3"], "qwen3   1.2M pulls   tools")
        self.assertEqual((repo["value"], repo["interactive"]), ("", True))
        self.assertIn("Found 2 models in the Ollama library. Choose one to see its versions.", note)
        self.assertEqual((values(version), version.get("value", "")), ([""], ""))
        self.assertIs(loading["visible"], False, "the loading screen is hidden again")

    def test_ollama_search_accepts_a_library_link_and_puts_exact_match_first(self):
        repo, _, note, _ = self.search("Ollama", "https://ollama.com/library/gemma3/tags")
        self.assertEqual(E2EOllama.web_seen[-1], "gemma3", "only the model name is searched for")
        self.assertEqual(values(repo), ["", "gemma3", "gemma3n"])
        self.assertIn("Found 2 models", note)

    def test_ollama_search_for_nonsense_finds_nothing(self):
        repo, _, note, _ = self.search("Ollama", "zzqx-nothing")
        self.assertIn("No models named &quot;zzqx-nothing&quot; found in the Ollama library.", note)
        self.assertIn("error", note)
        self.assertEqual((values(repo), repo["interactive"]), ([""], False))

    def test_empty_search_asks_for_a_name_without_searching(self):
        for source, query in (("Ollama", "   "), ("Hugging Face", "")):
            seen_web, seen_hf = len(E2EOllama.web_seen), len(E2EOllama.hf_seen)
            repo, version, note, loading = self.search(source, query)
            self.assertIn("Enter a model name to search.", note)
            self.assertTrue(noop(repo) and noop(version), "the previous results stay")
            self.assertIs(loading["visible"], False)
            self.assertEqual((len(E2EOllama.web_seen), len(E2EOllama.hf_seen)), (seen_web, seen_hf), source)

    def test_hugging_face_search_shows_size_and_access_marks(self):
        repo, _, note, _ = self.search("Hugging Face", "tiny")
        found = labels(repo)
        self.assertEqual(found["acme/Tiny-Chat-GGUF"], "acme/Tiny-Chat-GGUF   0.7 GB   (gated)")
        self.assertEqual(found["acme/Tiny-Chat-abliterated-GGUF"],
                         "acme/Tiny-Chat-abliterated-GGUF   0.5 GB   (modified: abliterated / uncensored)")
        self.assertEqual(found["bob/Tiny-Secret-GGUF"], "bob/Tiny-Secret-GGUF   0.5 GB   (private)")
        self.assertIn("Found 3 GGUF repositories, sorted by downloads.", note)

    def test_hugging_face_link_to_a_gguf_repo_skips_the_search(self):
        E2EOllama.hf_seen.clear()
        repo, _, note, _ = self.search("Hugging Face", "https://huggingface.co/acme/Tiny-Chat-GGUF")
        self.assertEqual(values(repo), ["", "acme/Tiny-Chat-GGUF"])
        self.assertEqual(labels(repo)["acme/Tiny-Chat-GGUF"], "acme/Tiny-Chat-GGUF   0.7 GB")
        self.assertIn("Found 1 GGUF", note)
        self.assertNotIn("/api/models", [p for p, _ in E2EOllama.hf_seen], "no search request for a direct link")

    def test_ollama_versions_recommend_largest_that_fits_and_hide_mlx(self):
        version, note, _ = self.versions("Ollama", "qwen", "qwen3")
        self.assertEqual(values(version), ["", "qwen3:0.6b", "qwen3:1.7b", "qwen3:235b"])
        self.assertEqual(version["value"], "qwen3:1.7b")
        self.assertEqual(labels(version)["qwen3:1.7b"], "1.7b   0.9 GB VRAM   (Recommended)")
        self.assertIn("qwen3 test model Recommended: 1.7b (0.9 GB)", note)

    def test_ollama_cloud_only_model_has_no_versions(self):
        version, note, _ = self.versions("Ollama", "kimi", "kimi-cloud")
        self.assertIn("No downloadable versions found", note)
        self.assertEqual((values(version), version["interactive"]), ([""], False))

    def test_ollama_model_too_big_for_any_gpu_preselects_smallest(self):
        version, note, _ = self.versions("Ollama", "giant", "giant")
        self.assertEqual(version["value"], "giant:900b")
        self.assertIn("No version fits your", note)
        self.assertIn("(900b, 900.0 GB) is selected", note)

    def test_hugging_face_versions_mark_unloadable_formats_and_describe_repo(self):
        self.assertNotIn("hf_token", self.settings())
        version, note, _ = self.versions("Hugging Face", "tiny", "acme/Tiny-Chat-GGUF")
        self.assertEqual(values(version), ["", "PQ2_0", "Q4_K_M", "Q8_0"], "shards and projectors are not versions")
        self.assertEqual(version["value"], "Q8_0")
        self.assertEqual(labels(version)["PQ2_0"], f"PQ2_0   0.2 GB   {CANNOT_LOAD}")
        self.assertEqual(labels(version)["Q8_0"], "Q8_0   0.7 GB VRAM   (Recommended)")
        for part in ("Gated: accept this model", "then add a token with the gear above.", "Based on Tiny-Base.",
                     "apache-2.0 licence.", "1.2M downloads on Hugging Face.", "Sees images",
                     "Recommended: Q8_0 (0.7 GB)", "PQ2_0 is in a format Ollama cannot load"):
            self.assertIn(part, note)

    def test_gated_repo_with_a_token_drops_the_gear_hint(self):
        self.addCleanup(self.call, "/remove_hf_token")
        self.call("/save_hf_token", "hf_good_e2e_gated_4321")
        _, note, _ = self.versions("Hugging Face", "tiny", "acme/Tiny-Chat-GGUF")
        self.assertIn("Hugging Face page.", note)
        self.assertNotIn("add a token", note)

    # ============================================================ downloads (3)
    def test_download_needs_a_version_for_either_source(self):
        before = self.pending()
        self.versions("Ollama", "qwen", "qwen3")
        self.assertIn("Choose a version before downloading.", self.call("/pull_model", "Ollama", "qwen3", ""))
        self.call("/source_changed", "Hugging Face")
        self.versions("Hugging Face", "tiny", "acme/Tiny-Chat-GGUF")
        self.assertIn("Choose a version before downloading.",
                      self.call("/pull_model", "Hugging Face", "acme/Tiny-Chat-GGUF", ""))
        self.assertIn("Choose a repository before downloading.", self.call("/pull_model", "Hugging Face", "", "Q8_0"))
        self.assertEqual(self.pending(), before, "nothing was queued")

    def test_downloading_the_same_version_twice_is_refused_then_deleted(self):
        self.addCleanup(self.call, "/remove_download", "slowpoke:1b")
        self.versions("Ollama", "slowpoke", "slowpoke")
        self.assertIn("Downloading slowpoke:1b.", self.call("/pull_model", "Ollama", "slowpoke", "slowpoke:1b"))
        self.assertIn("slowpoke:1b is already downloading.",
                      self.call("/pull_model", "Ollama", "slowpoke", "slowpoke:1b"))
        self.assertIn('data-ref="slowpoke:1b"', self.call("/downloads_tick", None)[0])
        self.assertIn("slowpoke:1b", self.pending())
        note, ref_box = self.call("/remove_download", "slowpoke:1b")
        self.assertIn("Deleted slowpoke:1b from downloads.", note)
        self.assertEqual(ref_box, "")
        self.assertNotIn("slowpoke:1b", self.call("/downloads_tick", None)[0])
        self.assertNotIn("slowpoke:1b", self.pending())

    def test_idle_downloads_panel_ignores_unknown_actions(self):
        f = self.data / "downloads.json"
        before = f.read_text() if f.exists() else None
        view, pick = self.call("/downloads_tick", None)
        self.assertIn("No active downloads.", view)
        self.assertEqual((values(pick), pick.get("value")), ([], None))
        status, ref_box = self.call("/toggle_download", "pause:ghost:1b")
        self.assertTrue(noop(status))
        self.assertEqual(ref_box, "")
        self.assertTrue(all(noop(x) for x in self.call("/remove_download", "ghost:1b")))
        self.assertEqual(f.read_text() if f.exists() else None, before)

    # ============================================================ chat (7, plus 4 generated)
    def submit_and_reply(self, message, model="tiny:latest"):
        self.call("/chat_submit", message, model)
        return self.call("/chat_respond")

    def test_chat_blank_message_changes_nothing(self):
        self.call("/start_new_chat")
        before = len(self.chats())
        out = self.call("/chat_submit", "  \n\t ", "tiny:latest")
        self.assertEqual(len(out), 4)
        self.assertTrue(all(noop(x) for x in out), out)
        self.assertEqual(len(self.chats()), before, "nothing is saved for a blank message")

    def test_chat_without_a_model_asks_for_one(self):
        self.call("/start_new_chat")
        box, bot, _, _ = self.call("/chat_submit", "E2E no model chosen", None)
        self.assertTrue(noop(box), "the message stays in the box to send again")
        self.assertEqual(len(bot), 1)
        self.assertEqual(text_of(bot[-1]), "Choose a model above first.")
        self.assertFalse([c for c in self.chats() if c["messages"] and c["messages"][0]["content"] == "E2E no model chosen"])

    def test_chat_reply_without_a_new_message_sends_nothing(self):
        self.call("/start_new_chat")
        sent = len(self.chat_calls())
        bot, usage = self.call("/chat_respond")
        self.assertTrue(noop(bot) and noop(usage))
        self.assertEqual(len(self.chat_calls()), sent)

    def test_chat_two_turns_keep_history_and_totals_then_new_chat_unlocks(self):
        self.call("/start_new_chat")
        self.submit_and_reply("E2E turn one")
        bot, usage = self.submit_and_reply("E2E turn two")
        chat = self.saved_chat("E2E turn one")
        self.assertEqual([(m["role"], m["content"]) for m in chat["messages"]],
                         [("user", "E2E turn one"), ("assistant", "Hello there"), ("user", "E2E turn two"),
                          ("assistant", "Hello there")])
        self.assertEqual((chat["totals"]["replies"], chat["totals"]["prompt"], chat["totals"]["output"]), (2, 24, 4))
        self.assertEqual([(m["role"], m["content"]) for m in self.chat_calls()[-1]["messages"]],
                         [("user", "E2E turn one"), ("assistant", "Hello there"), ("user", "E2E turn two")])
        self.assertIn("2 replies", usage)
        self.assertIn("28 total", usage)
        self.assertEqual(len(bot), 4)
        bot, usage, model = self.call("/start_new_chat")
        self.assertEqual((bot, usage), ([], "No messages yet."))
        self.assertEqual((model["interactive"], model["label"]), (True, "Model"))

    def test_chat_model_is_locked_after_the_first_message(self):
        self.call("/start_new_chat")
        _, _, model, _ = self.call("/chat_submit", "E2E lock check", "tiny:latest")
        self.assertEqual((model["value"], model["interactive"], model["label"]),
                         ("tiny:latest", False, "Model (locked to this chat)"))
        self.call("/chat_respond")
        _, _, model, _ = self.call("/chat_submit", "E2E lock second", "thinker:8b")
        self.assertEqual(model["value"], "tiny:latest")
        self.call("/chat_respond")
        self.assertEqual(self.chat_calls()[-1]["model"], "tiny:latest", "the chat keeps talking to its own model")
        self.assertEqual(self.saved_chat("E2E lock check")["model"], "tiny:latest")

    def test_chat_reply_records_timing_and_token_stats(self):
        self.call("/start_new_chat")
        bot, usage = self.submit_and_reply("E2E stats check")
        reply = self.saved_chat("E2E stats check")["messages"][-1]
        for part in ("to first token", "12 prompt + 2 response tokens", "20.0 tokens/s", "tiny:latest"):
            self.assertIn(part, reply["stats"])
        self.assertIn("**This chat:** 1 reply", usage)
        self.assertIn("14 total", usage)
        self.assertTrue(text_of(bot[-1]).startswith("Hello there"))

    def test_chat_error_from_ollama_is_shown_in_the_reply(self):
        self.call("/start_new_chat")
        bot, _ = self.submit_and_reply("E2E memory error", "big:70b")
        reply = self.saved_chat("E2E memory error")["messages"][-1]
        self.assertIn("Error: model requires more system memory (64.0 GiB) than is available", reply["content"])
        self.assertNotIn("prompt +", reply["stats"], "no token counts without a finished reply")
        self.assertTrue(reply["stats"].endswith("big:70b"))
        self.assertIn("Error: model requires more system memory", text_of(bot[-1]))

    # ============================================================ images (7, plus 6 generated)
    def test_random_seed_is_recorded_with_the_image(self):
        out, png, meta = self.generate("z-image-turbo", "E2E random seed", "Small 512 x 512 (fastest)", 1, -1)
        self.assertTrue(0 <= meta["seed"] < 2 ** 31)
        self.assertTrue(png.name.endswith(f"-{meta['seed']}.png"))
        self.assertIn(f"seed {meta['seed']}", out[0])

    def test_custom_sd15_model_generates_and_sets_its_defaults(self):
        steps, size = self.call("/image_model_defaults", "dreamy-sd15")
        self.assertEqual((steps["value"], size["value"]), (25, "Small 512 x 512 (fastest)"))
        self.assertTrue(all(noop(x) for x in self.call("/image_model_defaults", None)))
        _, _, meta = self.generate("dreamy-sd15", "E2E dreamy castle", "Small 512 x 512 (fastest)", 2, 3301)
        self.assertEqual((meta["model"], meta["width"], meta["height"], meta["steps"]), ("Dreamy SD15", 512, 512, 2))

    def test_generate_without_an_image_model(self):
        before = self.images()
        progress, note, result, gallery = self.call("/generate_image", None, "E2E a cat", "Square 1024 x 1024", 8, 5)
        self.assertIn("Download an image model first", note)
        self.assertIn("Ready", progress)
        self.assertTrue(noop(result) and noop(gallery))
        self.assertEqual(self.images(), before)

    def test_engine_failure_reports_its_last_line(self):
        before = self.images()
        progress, note, result, gallery = self.call("/generate_image", "z-image-turbo", "FAILME please",
                                                    "Small 512 x 512 (fastest)", 2, 3401)
        self.assertIn("Generation failed", progress)
        self.assertIn("ggml: out of video memory", progress)
        self.assertEqual(note, "")
        self.assertTrue(noop(result) and noop(gallery), "the last good image stays on screen")
        self.assertEqual(self.images(), before)
        self.assertEqual(list((self.data / "images").glob("*-3401.*")), [])

    def test_missing_engine_is_explained(self):
        engine, away = self.sd / "sd-cli", self.sd / "sd-cli.away"
        engine.rename(away)
        self.addCleanup(lambda: away.exists() and away.rename(engine))
        before = self.images()
        _, note, _, _ = self.call("/generate_image", "z-image-turbo", "E2E no engine", "Square 1024 x 1024", 1, 3402)
        self.assertIn("The image engine is missing from", note)
        self.assertIn(str(self.sd), note)
        self.assertEqual(self.images(), before)

    def test_gallery_is_newest_first_and_escapes_prompts(self):
        _, first, _ = self.generate("z-image-turbo", 'E2E a "red" fox & <owl>', "Small 512 x 512 (fastest)", 1, 3501)
        _, second, _ = self.generate("z-image-turbo", "E2E second tile", "Small 512 x 512 (fastest)", 1, 3502)
        gallery = self.call("/gallery_html")
        self.assertLess(gallery.index(second.name), gallery.index(first.name))
        self.assertIn('title="E2E a &quot;red&quot; fox &amp; &lt;owl&gt;"', gallery)
        self.assertNotIn("<owl>", gallery)

    def test_gallery_delete_ignores_settings_files_and_unknown_names(self):
        _, png, _ = self.generate("z-image-turbo", "E2E keep me", "Small 512 x 512 (fastest)", 1, 3601)
        for name in (png.with_suffix(".json").name, "missing-e2e.png", "../" + png.with_suffix(".json").name, ""):
            gallery, note, box = self.call("/delete_gallery_image", name)
            self.assertTrue(noop(gallery) and noop(note), name)
            self.assertEqual(box, "")
        self.assertTrue(png.exists() and png.with_suffix(".json").exists())

    # ============================================================ probes and groups (10, plus 9 generated)
    def other_module(self, mod):
        other = next((v["module"] for v in CAT.values() if v["active"] and v["module"] != mod), None)
        if not other:
            self.skipTest("only one probe family is available")
        return other, filtered(module=other)

    def drop_group(self, name):
        """Delete a saved group straight from the file, as another window would."""
        groups = self.groups()
        if groups.pop(name, None) is not None:
            (self.data / "probe_groups.json").write_text(json.dumps(groups, indent=2, sort_keys=True))

    def test_filter_queries_match_names_and_descriptions_ignoring_case(self):
        self.need_catalog()
        for query in ("inject", "LEAK", "  riley  ", "base64", "zzqx-no-such-probe"):
            with self.subTest(query=query):
                probe_list, count = self.call("/filter_probes", query, "All categories", False)
                self.assertEqual(count, f"Showing {len(filtered(query))} of {len(CAT)} probes.")
                self.assertEqual(values(probe_list), filtered(query))
        self.assertEqual(filtered("zzqx-no-such-probe"), [])

    def test_show_all_adds_probes_that_are_off_by_default(self):
        self.need_catalog()
        everything, count_all = self.call("/filter_probes", "", "All categories", True)
        defaults, count_default = self.call("/filter_probes", "", "All categories", False)
        self.assertEqual(values(everything), filtered(show_all=True))
        self.assertEqual(values(defaults), filtered())
        self.assertEqual(count_all, f"Showing {len(filtered(show_all=True))} of {len(CAT)} probes.")
        for name in [n for n, v in CAT.items() if not v["active"]][:20]:
            self.assertTrue(labels(everything)[name].endswith("   [off by default]"), name)
            self.assertNotIn(name, values(defaults))
        for name in values(defaults)[:20]:
            self.assertNotIn("[off by default]", labels(defaults)[name])

    def test_filter_keeps_a_hidden_selection_and_shows_it_again(self):
        mod, probes = self.pick_module()
        other, _ = self.other_module(mod)
        self.call("/clear_selection")
        self.call("/select_shown", "", mod, False)
        probe_list, _ = self.call("/filter_probes", "", other, False)
        self.assertEqual(probe_list["value"], [], "the selection is kept, just not shown")
        probe_list, _ = self.call("/filter_probes", "", mod, False)
        self.assertEqual(probe_list["value"], probes)

    def test_select_shown_accumulates_across_families(self):
        self.need_module("dan", "encoding")
        self.call("/clear_selection")
        self.call("/select_shown", "", "dan", False)
        probe_list, summary = self.call("/select_shown", "", "encoding", False)
        both = [n for n in CAT if n in set(filtered(module="dan")) | set(filtered(module="encoding"))]
        self.assertEqual(summary, summary_of(both))
        self.assertEqual(probe_list["value"], filtered(module="encoding"), "the list ticks what it shows")
        if len(both) > 8:
            self.assertIn(f", and {len(both) - 8} more", summary)

    def test_clear_empties_the_selection_everywhere(self):
        mod, _ = self.pick_module()
        self.call("/select_shown", "", mod, False)
        probe_list, summary = self.call("/clear_selection")
        self.assertEqual((probe_list["value"], summary), ([], "No probes selected."))
        self.assertIn("No probes selected. Choose some on the Probes tab.", self.call("/scan_info", CURRENT))

    def test_saving_a_group_needs_a_selection_and_a_name(self):
        mod, _ = self.pick_module()
        before = self.groups()
        self.call("/clear_selection")
        status, pick, scan_type = self.call("/save_group", "E2E nothing selected")
        self.assertEqual(status, "Select at least one probe before saving.")
        self.assertTrue(noop(pick) and noop(scan_type))
        self.call("/select_shown", "", mod, False)
        status, *_ = self.call("/save_group", "   ")
        self.assertEqual(status, "Enter a name for the group.")
        self.assertEqual(self.groups(), before)

    def test_saving_a_group_again_updates_it(self):
        mod, probes = self.pick_module()
        other, more = self.other_module(mod)
        self.addCleanup(self.drop_group, "E2E twice")
        self.call("/clear_selection")
        self.call("/select_shown", "", mod, False)
        status, *_ = self.call("/save_group", "E2E twice")
        self.assertTrue(status.startswith(f'Saved group "E2E twice" with {len(probes)} probes.'), status)
        self.call("/select_shown", "", other, False)
        status, *_ = self.call("/save_group", "E2E twice")
        both = [n for n in CAT if n in set(probes) | set(more)]
        self.assertTrue(status.startswith(f'Updated group "E2E twice" with {len(both)} probes.'), status)
        self.assertEqual(self.groups()["E2E twice"], both)

    def test_loading_a_preset_selects_its_probes_without_naming_a_group(self):
        self.need_catalog()
        if "Preset: Quick check" not in self.config_choices(lambda p: p.get("label") == "Group"):
            self.skipTest("Quick check preset not offered")
        self.call("/clear_selection")
        probe_list, summary, name = self.call("/load_group", "Preset: Quick check", "", "All categories", False)
        quick = [n for n in CAT if n in set(expand(PRESET_PROBES["Quick check"]))]
        self.assertEqual(summary, summary_of(quick))
        self.assertEqual(name["value"], "", "a preset isn't a group name to save over")
        self.assertEqual(probe_list["value"], [n for n in quick if n in filtered()])

    def test_group_buttons_without_a_saved_group(self):
        before = self.groups()
        probe_list, summary, name = self.call("/load_group", None, "", "All categories", False)
        self.assertEqual(summary, "Choose a group to load.")
        self.assertTrue(noop(probe_list) and noop(name))
        for pick in ("Preset: Quick check", None):
            status, group_pick, scan_type = self.call("/delete_group", pick)
            self.assertIn("presets can't be deleted", status)
            self.assertTrue(noop(group_pick) and noop(scan_type))
        self.assertEqual(self.groups(), before)

    def test_group_info_for_a_preset_matches_the_scan_tab(self):
        self.need_catalog()
        if "Encoding and obfuscation" not in self.config_choices(lambda p: p.get("label") == "Scan type"):
            self.skipTest("Encoding preset not offered")
        info = self.call("/group_info", "Preset: Encoding and obfuscation")
        self.assertEqual(info, self.call("/scan_info", "Encoding and obfuscation"))
        self.assertEqual(info.count('class="pi-chip"'), len(expand(["encoding", "ansiescape"])))
        self.assertEqual(self.call("/group_info", None), "")

    # ============================================================ scan (5, plus 18 generated)
    def test_scan_info_and_run_scan_for_the_current_selection(self):
        mod, probes = self.pick_module()
        self.call("/clear_selection")
        self.assertIn("No probes selected. Choose some on the Probes tab.", self.call("/scan_info", CURRENT))
        log, progress = self.call("/run_scan", "tiny:latest", CURRENT, 1, 60, False)
        self.assertEqual(log, "No probes selected. Choose some on the Probes tab.")
        self.assertTrue(noop(progress))
        self.assertFalse((self.data / "scan_job.json").exists())
        self.call("/select_shown", "", mod, False)
        info = self.call("/scan_info", CURRENT)
        self.assertIn("The probes currently selected on the Probes tab.", info)
        self.assertIn(f"{len(probes)} probe{'s' if len(probes) != 1 else ''} in 1 family", info)
        self.call("/clear_selection")

    def test_run_scan_refuses_a_model_ollama_cannot_load(self):
        self.call("/clear_selection")  # (so nothing could start even if the check were missing)
        log, _ = self.call("/run_scan", "broken-pq2:latest", CURRENT, 1, 60, False)
        self.assertEqual(log, "broken-pq2:latest can't be scanned: Ollama can't load this model (unsupported tensor "
                              "type PQ2_0). Choose a model that runs.")
        self.assertFalse((self.data / "scan_job.json").exists())

    def test_run_scan_when_ollama_is_stopped(self):
        self.call("/clear_selection")
        E2EOllama.down = True
        try:
            log, _ = self.call("/run_scan", "tiny:latest", CURRENT, 1, 60, False)
        finally:
            E2EOllama.down = False
        self.assertEqual(log, "Ollama is not running. Start it at the top of the page.")

    def test_group_deleted_elsewhere_cannot_be_scanned(self):
        mod, _ = self.pick_module()
        self.addCleanup(self.drop_group, "E2E ghost group")
        self.call("/clear_selection")
        self.call("/select_shown", "", mod, False)
        self.call("/save_group", "E2E ghost group")
        self.call("/clear_selection")
        self.drop_group("E2E ghost group")
        self.assertNotIn("E2E ghost group", self.groups())
        self.assertIn("That group no longer exists.", self.call("/scan_info", GROUP + "E2E ghost group"))
        self.assertIn("That group no longer exists.", self.call("/group_info", "E2E ghost group"))
        log, _ = self.call("/run_scan", "tiny:latest", GROUP + "E2E ghost group", 1, 60, False)
        self.assertEqual(log, "That group no longer exists.")
        self.assertFalse((self.data / "scan_job.json").exists())

    def test_idle_scan_controls_leave_everything_as_is(self):
        job = self.data / "scan_job.json"
        self.assertFalse(job.exists())
        self.call("/toggle_scan_pause")
        self.assertTrue(all(noop(x) for x in self.call("/recover_scan", 300)))
        self.assertTrue(all(noop(x) for x in self.call("/skip_stuck_probe", 300)))
        self.assertIn(IDLE_PROGRESS, self.call("/scan_controls_after_stop"))
        self.assertFalse(job.exists())
        self.assertIn("Idle", self.call("/status_controls")[2])

    # ============================================================ reports (7)
    def test_report_list_is_newest_first_and_marks_stopped_runs(self):
        listing, select_all = self.call("/report_list_state")
        found = labels(listing)
        self.assertEqual([n for n in found if n in (ALPHA, BETA, GAMMA, DELTA)], [ALPHA, BETA, GAMMA, DELTA])
        self.assertTrue(found[ALPHA].startswith("org_alpha   "), "hf.co_ and the timestamp are dropped")
        self.assertFalse(found[ALPHA].endswith("(stopped)") or found[GAMMA].endswith("(stopped)"))
        self.assertTrue(found[BETA].startswith("beta   ") and found[BETA].endswith("   (stopped)"))
        self.assertTrue(found[DELTA].endswith("   (stopped)"))
        self.assertEqual((listing["value"], select_all["value"]), ([], False))

    def test_ticking_a_stopped_run_shows_what_it_collected(self):
        summary, view, raw, files = self.call("/reports_ticked", [BETA], FULL)
        self.assertIn("Incomplete run", summary)
        self.assertIn("stopped before garak wrote its report", view)
        self.assertIn("start_run setup", raw)
        names = json.dumps(files, default=str)
        for f in ("beta_20260917-100000.report.jsonl", "beta_20260917-100000.report.txt",
                  "beta_20260917-100000.hitlog.jsonl"):
            self.assertIn(f, names)
        self.assertNotIn(BETA, names, "there is no garak HTML report to offer")

    def test_unticking_keeps_the_shown_report_and_new_ticks_take_over(self):
        summary, *_ = self.call("/reports_ticked", [ALPHA, BETA], FULL)
        self.assertIn("Incomplete run", summary)
        summary, *_ = self.call("/reports_ticked", [ALPHA], FULL)
        self.assertIn("Incomplete run", summary, "unticking only changes what Delete removes")
        summary, *_ = self.call("/reports_ticked", [ALPHA, GAMMA], FULL)
        self.assertNotIn("Incomplete run", summary)
        self.assertIn("LLM security assessment", summary)
        summary, view, *_ = self.call("/reports_ticked", [], FULL)
        self.assertNotIn("Incomplete run", summary)
        self.assertIn("&lt;html&gt;garak&lt;/html&gt;", view)

    def test_raw_view_switches_between_report_and_hitlog(self):
        _, _, raw, _ = self.call("/reports_ticked", [GAMMA], HITLOG)
        self.assertEqual(raw, "No failing responses were recorded for this run.")
        raw = self.call("/read_raw", FULL)
        self.assertIn('"entry_type": "start_run setup"', raw)
        self.assertIn("{garbled", raw, "the raw file is shown as it is")
        self.assertEqual(self.call("/read_raw", HITLOG), "No failing responses were recorded for this run.")

    def test_clicking_a_report_name_shows_it_without_ticking(self):
        label = self.report_labels()[DELTA]
        _, view, raw, files = self.call("/view_report", "  " + label.replace("   ", "      ") + " ", FULL)
        self.assertIn("stopped before garak wrote its report", view)
        self.assertIn("start_run setup", raw)
        self.assertNotIn('"entry_type": "eval"', raw)
        self.assertIn("delta_20260917-120000.report.jsonl", json.dumps(files, default=str))
        _, _, again, _ = self.call("/view_report", "no such report", FULL)
        self.assertEqual(again, raw, "an unknown name keeps the report on screen")

    def test_delete_without_confirmation_or_names_keeps_files(self):
        for names, confirmed in (([VICTIM3], False), ([], True)):
            out = self.call("/delete_reports", names, confirmed, FULL)
            self.assertEqual(len(out), 7)
            self.assertTrue(all(noop(x) for x in out), (names, confirmed))
        self.assertEqual(len(list(self.runs.glob("victim3_20260917-150000.*"))), 3)

    def test_deleting_several_reports_removes_their_files_and_shows_the_newest(self):
        self.call("/reports_ticked", [VICTIM1, VICTIM2], FULL)
        listing, select_all, summary, view, raw, files, status = self.call("/delete_reports", [VICTIM1, VICTIM2],
                                                                            True, FULL)
        count = re.search(r"Deleted 2 reports \((\d+) files\)\.", status)
        self.assertTrue(count and int(count.group(1)) >= 6, status)
        self.assertEqual((listing["value"], select_all["value"]), ([], False))
        self.assertFalse({VICTIM1, VICTIM2} & set(values(listing)))
        self.assertIn("LLM security assessment", summary)
        self.assertNotIn("Incomplete run", summary, "the newest remaining report is shown")
        self.assertEqual(list(self.runs.glob("victim1_*")) + list(self.runs.glob("victim2_*")), [])

    # ============================================================ red team tools (2)
    def test_promptfoo_targets_list_every_chat_model(self):
        targets = self.call("/promptfoo_targets_html")
        for name, _ in MODELS:
            self.assertIn(f"<code>ollama:chat:{name}</code>", targets)
        self.assertIn(self.fake_url, targets)

    def test_red_team_tools_idle_views_are_stable(self):
        status, link = self.call("/redteam_view")
        self.assertIn("promptfoo", status)
        self.assertEqual(tuple(self.call("/stop_promptfoo")), (status, link), "Stop with nothing started changes nothing")
        status, link = self.call("/pyrit_view")
        self.assertIn("PyRIT", status)
        self.assertEqual(tuple(self.call("/stop_pyrit")), (status, link))
        self.assertIn("PyRIT documentation", self.call("/pyrit_html"))


# ================================================================ generated tests (37)
def _slug(text):
    return re.sub(r"\W+", "_", text.lower(), flags=re.ASCII).strip("_")


SIZE_CASES = [("Square 1024 x 1024", 3, 1101), ("Square 768 x 768 (faster)", 5, 1102), ("Portrait 768 x 1024", 2, 1103),
              ("Landscape 1024 x 768", 6, 1104), ("Wide 1280 x 720", 1, 1105), ("Small 512 x 512 (fastest)", 4, 1106)]

MODULE_CASES = ["dan", "encoding", "goodside", "promptinject", "lmrc", "latentinjection"]

PRESET_PROBES = {
    "Quick check": ["dan.DanInTheWild", "promptinject.HijackHateHumans", "goodside.WhoIsRiley", "lmrc.QuackMedicine"],
    "Jailbreaks and prompt injection": ["dan", "promptinject", "latentinjection"],
    "Harmful content and toxicity": ["lmrc", "realtoxicityprompts", "donotanswer"],
    "Data leakage and malware": ["leakreplay", "malwaregen", "packagehallucination", "apikey"],
    "Encoding and obfuscation": ["encoding", "ansiescape"],
    "Full scan": "ALL",
}
PRESET_CASES = [  # (scan type, words from its description)
    ("Quick check", "A fast sample of jailbreak"),
    ("Jailbreaks and prompt injection", "talked out of its instructions"),
    ("Harmful content and toxicity", "toxic, hateful, sexual"),
    ("Data leakage and malware", "training-data regurgitation"),
    ("Encoding and obfuscation", "encoded or obfuscated prompts"),
    ("OWASP Top 10 for LLMs", "OWASP Top 10 for LLM Applications"),
    ("Baseline assessment", "time-boxed pre-deployment check"),
    ("NIST AI 600-1: Information security", "hijacked or turned against"),
    ("NIST AI 600-1: Information integrity and confabulation", "states falsehoods with confidence"),
    ("NIST AI 600-1: Dangerous, violent and hateful content", "violent, hateful, harassing"),
    ("NIST AI 600-1: Data privacy and intellectual property", "memorized training data"),
    ("NIST AI 600-1: Harmful bias", "discriminates against people"),
    ("EU AI Act risk areas", "EU AI Act risk area"),
    ("Full scan", "Every default garak probe"),
]

GROUP_CASES = ["E2E jailbreak suite", "  E2E padded name  ", 'E2E Ünïcødé "quoted" <b>']

LONG_MESSAGE = "E2E long " + " ".join(["word"] * 40)
CHAT_CASES = [  # (case, message, expected chat title)
    ("unicode", "E2E unicode: Привет 👋 你好 — ¿qué tal?", "E2E unicode: Привет 👋 你好 — ¿qué tal?"),
    ("long", LONG_MESSAGE, LONG_MESSAGE[:48].rstrip() + "..."),
    ("markdown", "## **E2E bold** _markdown_ > [link]", "E2E bold markdown  link"),
    ("code_block", "```python\nprint('E2E code')\n```", "python\nprint('E2E code')"),
]


def _size_test(label, steps, seed):
    def test(self):
        self.assertIn(label, self.config_choices(lambda p: p.get("label") == "Size"))
        prompt = f"E2E size {label}"
        out, png, meta = self.generate("z-image-turbo", prompt, label, steps, seed)
        w, h = IMAGE_SIZES[label]
        self.assertEqual((meta["width"], meta["height"], meta["steps"], meta["seed"], meta["model"], meta["prompt"]),
                         (w, h, steps, seed, "Z-Image Turbo", prompt))
        self.assertTrue(png.name.endswith(f"-{seed}.png"))
        for part in ("Image ready", f"seed {seed}", f"{w} x {h}", f"{steps} steps"):
            self.assertIn(part, out[0])
        self.assertIn(png.name, json.dumps(out[2], default=str), "the new image is shown as the result")
        self.assertIn(png.name, out[3], "and added to the gallery")
    return test


def _module_test(module):
    def test(self):
        self.need_module(module)
        probe_list, count = self.call("/filter_probes", "", module, False)
        expected = filtered(module=module)
        self.assertEqual(count, f"Showing {len(expected)} of {len(CAT)} probes.")
        self.assertEqual(values(probe_list), expected)
        everything, _ = self.call("/filter_probes", "", module, True)
        self.assertEqual(values(everything), filtered(module=module, show_all=True))
        self.assertTrue(all(CAT[n]["module"] == module for n in values(everything)))
    return test


def _preset_test(name, phrase):
    def test(self):
        self.need_catalog()
        if name not in self.config_choices(lambda p: p.get("label") == "Scan type"):
            self.skipTest(f"{name} is not offered with this garak version")
        info = self.call("/scan_info", name)
        self.assertIn(phrase, info)
        found = re.search(r"(\d+) probes? in (\d+) famil(?:y|ies)", info)
        self.assertTrue(found, info[:300])
        count, families = int(found.group(1)), int(found.group(2))
        self.assertGreater(count, 0)
        self.assertEqual(info.count('class="pi-chip"'), count)
        specs = PRESET_PROBES.get(name)
        if specs is not None:
            probes = [n for n, v in CAT.items() if v["active"]] if specs == "ALL" else expand(specs)
            self.assertEqual(count, len(probes))
            self.assertEqual(families, len({CAT[p]["module"] for p in probes}))
            for p in probes[:30]:
                self.assertIn(f'data-tip-title="{html.escape(p)}"', info)
    return test


def _hint_test(model, size):
    def test(self):
        note, timeout = self.call("/model_hint", model)
        if "broken" in model:
            self.assertIn("**This model can't run, so it can't be scanned.**", note)
            self.assertIn("(unsupported tensor type PQ2_0)", note)
            self.assertNotIn("GGUF metadata", note, "the blob path means nothing to the reader")
            self.assertTrue(noop(timeout))
            return
        self.assertTrue(note.startswith(f"{size / 1e9:.1f} GB model, "), note)
        fits = "runs fully on the GPU" in note
        self.assertEqual(timeout["value"], 300 if fits else (900 if size < 25e9 else 1800))
        if not fits:
            self.assertIn("timeout has been raised", note)
        self.assertEqual("This is a reasoning model" in note, "thinker" in model)
    return test


def _group_test(name):
    def test(self):
        mod, probes = self.pick_module()
        stored = name.strip()
        self.addCleanup(self.drop_group, stored)
        self.call("/clear_selection")
        self.call("/select_shown", "", mod, False)
        status, pick, scan_type = self.call("/save_group", name)
        self.assertTrue(status.startswith(f'Saved group "{stored}" with {len(probes)} probes.'), status)
        self.assertEqual(self.groups()[stored], probes)
        self.assertEqual(pick["value"], stored)
        self.assertIn(GROUP + stored, values(scan_type))
        info = self.call("/scan_info", GROUP + stored)
        self.assertIn(html.escape(f'Your saved group "{stored}".'), info)
        self.assertEqual(self.call("/group_info", stored), info)
        self.call("/clear_selection")
        _, summary, loaded = self.call("/load_group", stored, "", "All categories", False)
        self.assertEqual((summary, loaded["value"]), (summary_of(probes), stored))
        status, _, scan_type = self.call("/delete_group", stored)
        self.assertEqual(status, f'Deleted group "{stored}".')
        self.assertNotIn(stored, self.groups())
        self.assertEqual(scan_type["value"], "Quick check")
    return test


def _chat_test(message, title):
    def test(self):
        self.call("/start_new_chat")
        box, bot, _, _ = self.call("/chat_submit", message, "tiny:latest")
        self.assertEqual(box, "", "the message box is cleared")
        self.assertEqual(bot[-1]["role"], "user")
        self.assertIn("E2E", text_of(bot[-1]))
        bot, _ = self.call("/chat_respond")
        self.assertTrue(text_of(bot[-1]).startswith("Hello there"))
        chat = self.saved_chat(message)
        self.assertEqual(chat["title"], title)
        self.assertEqual(chat["messages"][1]["content"], "Hello there")
        self.assertEqual(self.chat_calls()[-1]["messages"][-1]["content"], message, "sent to Ollama unchanged")
    return test


for _label, _steps, _seed in SIZE_CASES:
    setattr(Scenarios, f"test_image_size_{_slug(_label)}", _size_test(_label, _steps, _seed))
for _module in MODULE_CASES:
    setattr(Scenarios, f"test_probe_family_filter_{_module}", _module_test(_module))
for _name, _phrase in PRESET_CASES:
    setattr(Scenarios, f"test_scan_type_info_{_slug(_name)}", _preset_test(_name, _phrase))
for _model, _size in MODELS:
    setattr(Scenarios, f"test_model_hint_{_slug(_model)}", _hint_test(_model, _size))
for _i, _name in enumerate(GROUP_CASES, 1):
    setattr(Scenarios, f"test_group_round_trip_{_i}_{_slug(_name)}", _group_test(_name))
for _case, _message, _title in CHAT_CASES:
    setattr(Scenarios, f"test_chat_message_kind_{_case}", _chat_test(_message, _title))

assert len([n for n in dir(Scenarios) if n.startswith("test")]) == 100, "this file promises exactly 100 tests"


if __name__ == "__main__":
    unittest.main()
