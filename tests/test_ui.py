"""End-to-end tests of every page: the real app runs on a spare port against a fake Ollama and a scratch data folder,
and each test drives it the way the browser does, through the same Gradio events the buttons trigger.
Run from the app folder:  .venv/bin/python -m unittest discover -s tests -v

Deleting a chat or a chat model happens through per-row buttons that Gradio builds on the fly (not reachable through
its API); those handlers are covered in test_app.py (ChatFiles, DeleteModels)."""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR / "tests"))
from test_app import write_run  # noqa: E402

PY = str(APP_DIR / ".venv" / "bin" / "python") if (APP_DIR / ".venv").exists() else sys.executable


class FakeOllama(BaseHTTPRequestHandler):
    """Just enough of Ollama's API for the app: one small chat model that answers "Hello there"."""
    models = []
    calls = []

    def log_message(self, *args):
        pass

    def reply(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for ev in events:
            self.wfile.write(json.dumps(ev).encode() + b"\n")
            self.wfile.flush()

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def html(self, text):
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/search"):  # ollama.com library search, in the page structure the app parses
            items = "".join(f'<li x-test><a href="/library/{n}"><p class="max-w-lg break-words">{n} test model</p>'
                            f'<span class="bg-indigo-50 x"> tools </span><span >1.2M</span><span>&nbsp;Pulls</span></a></li>'
                            for n in ("slow", "smollm2", "missing"))
            return self.html(f"<ul>{items}</ul>")
        if self.path.startswith("/library/") and self.path.endswith("/tags"):
            name = self.path.split("/")[2]
            rows = "".join(f'<a href="/library/{name}:{tag}" class="md:hidden flex"><span class="font-mono"> abc123 '
                           f'</span> • {size}GB</a>' for tag, size in (("1b", "0.8"), ("2b", "1.6"), ("135m", "0.3")))
            return self.html(rows)
        if self.path == "/api/version":
            return self.reply({"version": "0.34.2"})
        if self.path == "/api/tags":
            return self.reply({"models": self.models})
        if self.path == "/api/ps":
            return self.reply({"models": []})
        self.reply({"error": "not found"}, 404)

    def do_POST(self):
        data = self.body()
        FakeOllama.calls.append((self.path, data))
        if self.path == "/api/show":
            return self.reply({"capabilities": ["completion"]})
        if self.path == "/api/generate":
            return self.reply({"done": True})
        if self.path == "/api/chat":
            return self.stream([{"message": {"content": "Hello "}}, {"message": {"content": "there"}},
                                {"message": {"content": ""}, "done": True, "prompt_eval_count": 12, "eval_count": 2,
                                 "eval_duration": 1e8, "load_duration": 0}])
        if self.path == "/api/pull":
            name = data["model"]
            if "slow" in name:  # progress arrives over a few seconds, so there is time to pause it
                events = [{"status": "pulling manifest"}]
                for i in range(1, 21):
                    events.append({"digest": "sha256:2", "total": 100, "completed": 5 * i})
                self.send_response(200)
                self.end_headers()
                for ev in events:
                    try:
                        self.wfile.write(json.dumps(ev).encode() + b"\n")
                        self.wfile.flush()
                    except OSError:  # the app stopped reading (paused)
                        return
                    time.sleep(0.25)
                FakeOllama.models.append(model_entry(name))
                self.wfile.write(json.dumps({"status": "success"}).encode() + b"\n")
                return
            if "missing" in name:
                return self.stream([{"status": "pulling manifest"}, {"error": "pull model manifest: file does not exist"}])
            FakeOllama.models.append(model_entry(name))
            return self.stream([{"status": "pulling manifest"}, {"digest": "sha256:1", "total": 100, "completed": 100},
                                {"status": "success"}])
        self.reply({"error": "not found"}, 404)

    def do_DELETE(self):
        name = self.body().get("model")
        before = len(FakeOllama.models)
        FakeOllama.models[:] = [m for m in FakeOllama.models if m["name"] != name]
        self.reply({} if len(FakeOllama.models) < before else {"error": f"model '{name}' not found"},
                   200 if len(FakeOllama.models) < before else 404)


def model_entry(name):
    return {"name": name, "model": name, "size": 270_000_000, "digest": f"d-{name}", "modified_at": "2026-09-18T10:00:00",
            "details": {"parameter_size": "135M", "quantization_level": "Q4_K_M"}}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


FAKE_ENGINE = """#!/usr/bin/env python3
import sys
out = sys.argv[sys.argv.index('-o') + 1]
steps = int(sys.argv[sys.argv.index('--steps') + 1])
for i in range(1, steps + 1):
    print(f'  |==| {i}/{steps} - 0.10s/it', flush=True)
open(out, 'wb').write(bytes.fromhex('89504e470d0a1a0a0000000d4948445200000001000000010806000000'
                                    '1f15c4890000000d4944415478da6364f8ffbf1e000502027fa1d5b40000000049454e44ae426082'))
"""


class AppPages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gradio_client import Client
        cls.tmp = Path(tempfile.mkdtemp(prefix="llm-scanner-ui-"))
        cls.data, cls.runs, sd = cls.tmp / "data", cls.tmp / "runs", cls.tmp / "sd"
        for d in (cls.data, cls.runs, sd):
            d.mkdir()
        write_run(cls.runs, "hf.co_x_tiny_20260917-100000")
        write_run(cls.runs, "tiny_20260917-110000", complete=False)
        engine = sd / "sd-cli"
        engine.write_text(FAKE_ENGINE)
        engine.chmod(0o755)
        # an installed image model: the built-in Z-Image Turbo, with placeholder files where its weights go
        for rel in ("image-models/z-image-turbo/z_image_turbo-Q4_K.gguf",
                    "image-models/shared/unsloth-qwen3-4b-instruct-2507-gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                    "image-models/shared/comfy-org-z-image-turbo/ae.safetensors"):
            (cls.data / rel).parent.mkdir(parents=True, exist_ok=True)
            (cls.data / rel).write_bytes(b"x")
        FakeOllama.models[:] = [model_entry("tiny:latest")]
        cls.ollama = ThreadingHTTPServer(("127.0.0.1", free_port()), FakeOllama)
        threading.Thread(target=cls.ollama.serve_forever, daemon=True).start()
        port = free_port()
        env = os.environ | {"LLM_SCANNER_CHECK": "1", "LLM_SCANNER_PORT": str(port), "LLM_SCANNER_DATA": str(cls.data),
                            "LLM_SCANNER_GARAK_RUNS": str(cls.runs), "LLM_SCANNER_SD_DIR": str(sd),
                            "OLLAMA_URL": f"http://127.0.0.1:{cls.ollama.server_port}",
                            "LLM_SCANNER_OLLAMA_WEB": f"http://127.0.0.1:{cls.ollama.server_port}"}
        cls.log = open(cls.tmp / "app.log", "wb")
        cls.proc = subprocess.Popen([PY, "app.py", "--no-browser"], cwd=APP_DIR, env=env, stdout=cls.log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + 120
        while True:
            try:
                cls.client = Client(f"http://127.0.0.1:{port}/", verbose=False)
                break
            except Exception:
                if time.time() > deadline or cls.proc.poll() is not None:
                    cls.tearDownClass()
                    raise RuntimeError("the app did not start: " + (cls.tmp / "app.log").read_text()[-1500:])
                time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        if cls.proc.poll() is None:
            os.killpg(cls.proc.pid, 15)
            cls.proc.wait(timeout=20)
        cls.ollama.shutdown()
        cls.log.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def call(self, api, *args):
        return self.client.predict(*args, api_name=api)

    def wait_for(self, check, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = check()
            if result:
                return result
            time.sleep(0.5)
        self.fail("timed out")

    # ------------------------------------------------------------ status bar
    def test_status_bar(self):
        status, *_, ollama, activity = self.call("/status_controls")
        self.assertIn("Ollama 0.34.2", ollama)  # shown in the control with Stop and Restart
        self.assertIn('dot ok', ollama)
        self.assertIn("RAM", status)
        self.assertIn("Idle", activity)

    # ------------------------------------------------------------ models page: downloads
    def download(self, name, tag):
        """Search the Ollama library, choose the model and a version, then press Download, as on the page."""
        self.call("/source_changed", "Ollama")
        repo_update, _, note, *_ = self.call("/search_models", "Ollama", name)
        self.assertIn("Found 3 models", note)
        versions = self.call("/list_versions", "Ollama", name)
        self.assertIn("Auto-selected", versions[1])
        return self.call("/pull_model", "Ollama", name, f"{name}:{tag}")

    def wait_downloads(self, text, timeout=20):
        """The downloads panel once it contains text."""
        return self.wait_for(lambda: next((h for h in [self.downloads()] if text in h), None), timeout)

    def downloads(self):
        return self.call("/downloads_tick", None)[0]

    def test_pause_resume_and_delete_a_download(self):
        self.download("slow", "1b")
        self.wait_for(lambda: "slow:1b" in self.downloads() and "Downloading" in self.downloads())
        html = self.downloads()
        self.assertIn('data-action="pause:slow:1b"', html)
        self.assertIn('title="Delete download"', html)
        self.call("/toggle_download", "pause:slow:1b")
        html = self.wait_downloads('data-action="resume:slow:1b"')
        self.assertIn(">Paused<", html)
        self.assertIn({"ref": "slow:1b", "paused": True}, json.loads((self.data / "downloads.json").read_text()))
        self.call("/toggle_download", "resume:slow:1b")
        self.wait_for(lambda: "Done" in self.downloads(), timeout=30)
        self.assertEqual(json.loads((self.data / "downloads.json").read_text()), [])

        self.download("slow", "2b")
        self.wait_for(lambda: "slow:2b" in self.downloads())
        self.call("/remove_download", "slow:2b")  # trash icon (after the confirm dialog)
        self.assertNotIn("slow:2b", self.downloads())
        time.sleep(1)
        self.assertNotIn("slow:2b", [m["name"] for m in FakeOllama.models], "a deleted download must stop")

    def test_download_finishes_and_clears(self):
        msg = self.download("smollm2", "135m")
        self.assertIn("Downloading smollm2:135m", msg)
        self.wait_for(lambda: "Done" in self.downloads())
        self.assertIn("smollm2:135m", [m["name"] for m in FakeOllama.models])
        self.assertIn('title="Remove from list"', self.downloads())

        self.call("/remove_download", "smollm2:135m")  # the trash icon on a finished row clears it
        self.assertNotIn("smollm2:135m", self.downloads())

    def test_failed_download_shows_reason(self):
        self.download("missing", "1b")
        html = self.wait_downloads("does not exist")
        self.assertIn("dl-state failed", html)

    def test_download_needs_repository_and_version(self):
        self.call("/source_changed", "Ollama")
        self.assertIn("Choose a repository", self.call("/pull_model", "Ollama", "", ""))

    # ------------------------------------------------------------ models page: chat
    def test_chat_reply_is_saved(self):
        self.call("/start_new_chat")
        self.call("/chat_submit", "Say hello", "tiny:latest")
        result = self.call("/chat_respond")
        content = result[0][-1]["content"]
        text = content if isinstance(content, str) else "".join(c.get("text", "") for c in content)
        self.assertTrue(text.startswith("Hello there"), text)
        chats = list((self.data / "chats").glob("*.json"))
        saved = [json.loads(c.read_text()) for c in chats]
        self.assertTrue(any(c["title"] == "Say hello" and c["messages"][-1]["content"] == "Hello there" for c in saved))

    # ------------------------------------------------------------ images page
    def test_generate_view_and_delete_image(self):
        progress, note, *_rest = self.call("/generate_image", "z-image-turbo", "a lighthouse", "Small 512 x 512 (fastest)",
                                           4, 7)
        self.assertIn("Image ready", progress)
        images = list((self.data / "images").glob("*.png"))
        self.assertEqual(len(images), 1)
        meta = json.loads(images[0].with_suffix(".json").read_text())
        self.assertEqual((meta["prompt"], meta["seed"], meta["steps"]), ("a lighthouse", 7, 4))
        self.assertIn(images[0].name, self.call("/gallery_html"))

        gallery, note, _ = self.call("/delete_gallery_image", images[0].name)
        self.assertIn("Image deleted", note)
        self.assertFalse(images[0].exists() or images[0].with_suffix(".json").exists())
        self.assertIn("No images yet", gallery)

    def test_generate_needs_a_prompt(self):
        _, note, *_rest = self.call("/generate_image", "z-image-turbo", "  ", "Square 1024 x 1024", 8, -1)
        self.assertIn("Describe the image", note)

    def test_image_model_settings_follow_the_model(self):
        steps, size = self.call("/image_model_defaults", "z-image-turbo")
        self.assertEqual((steps["value"], size["value"]), (8, "Square 1024 x 1024"))

    # ------------------------------------------------------------ probes page
    def test_filter_select_and_groups(self):
        probe_list, count = self.call("/filter_probes", "riley", "All categories", False)
        self.assertRegex(count, r"Showing \d+ of \d+ probes")
        self.call("/clear_selection")
        _, summary = self.call("/select_shown", "riley", "All categories", False)
        self.assertIn("goodside.WhoIsRiley", summary)

        status, *_ = self.call("/save_group", "UI test group")
        self.assertIn('Saved group "UI test group"', status)
        self.assertIn("UI test group", json.loads((self.data / "probe_groups.json").read_text()))
        self.assertIn("goodside.WhoIsRiley", self.call("/scan_info", "Saved group: UI test group"))

        self.call("/clear_selection")
        _, summary, name = self.call("/load_group", "UI test group", "", "All categories", False)
        self.assertIn("goodside.WhoIsRiley", summary)
        status, *_ = self.call("/delete_group", "UI test group")
        self.assertIn('Deleted group "UI test group"', status)
        self.assertNotIn("UI test group", json.loads((self.data / "probe_groups.json").read_text()))
        from gradio_client.exceptions import AppError
        with self.assertRaises(AppError):  # no longer offered on the Scan page
            self.call("/scan_info", "Saved group: UI test group")

    # ------------------------------------------------------------ scan page
    def test_scan_page(self):
        info = self.call("/scan_info", "OWASP Top 10 for LLMs")
        self.assertRegex(info, r"OWASP Top 10.*\d+ probes in \d+ families")
        self.assertIn('data-tip-title="dan.DanInTheWild"', info)
        note, timeout = self.call("/model_hint", "tiny:latest")
        self.assertIn("0.3 GB model", note)
        log, _ = self.call("/run_scan", None, "Quick check", 1, 60, False)
        self.assertIn("Select a model to scan", log)
        self.assertEqual(self.call("/stop_scan"), "No scan is running.")

    # ------------------------------------------------------------ reports page
    def test_view_and_delete_reports(self):
        choices = self.call("/refresh_reports")
        names = sorted(p.name.split(".report")[0] for p in self.runs.glob("*.report.jsonl"))
        self.assertEqual(len(names), 2)
        summary, garak_view, raw, files = self.call("/show_report", "hf.co_x_tiny_20260917-100000.report.html",
                                                     "Full report (report.jsonl)")
        self.assertIn("LLM security assessment", summary)
        self.assertIn("iframe", garak_view)
        self.assertIn("start_run setup", raw)
        self.assertIn("evil", self.call("/read_raw", "hf.co_x_tiny_20260917-100000.report.html",
                                        "Hitlog (failing responses)"))
        summary, garak_view, _, _ = self.call("/show_report", "tiny_20260917-110000.report.html",
                                              "Full report (report.jsonl)")
        self.assertIn("Incomplete run", summary)
        self.assertIn("stopped before garak wrote its report", garak_view)

        _, note = self.call("/delete_report", "tiny_20260917-110000.report.html", True)
        self.assertIn("Deleted report tiny_20260917-110000", note)
        self.assertEqual([p.name for p in self.runs.glob("tiny_20260917-110000.*")], [])
        self.assertTrue(list(self.runs.glob("hf.co_x_tiny_20260917-100000.*")), "other reports must be kept")

    def test_report_delete_rejects_unknown_names(self):
        from gradio_client.exceptions import AppError
        with self.assertRaises(AppError):  # Gradio only accepts names the Reports list offers
            self.call("/delete_report", "../../etc/passwd.report.html", True)
        self.assertTrue(list(self.runs.glob("hf.co_x_tiny_20260917-100000.*")))


if __name__ == "__main__":
    unittest.main()
