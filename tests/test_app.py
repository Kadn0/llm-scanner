"""Tests for LLM Scanner. Run from the app folder:  .venv/bin/python -m unittest discover -s tests -v

Nothing here touches real data, models or the network: every path the code writes to is redirected to a
temporary folder, and HTTP calls to Ollama or Hugging Face are replaced with fakes."""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
os.environ["LLM_SCANNER_CHECK"] = "1"

import analyst_report  # noqa: E402
import app  # noqa: E402


class FakeResponse:
    def __init__(self, data=None, lines=(), status=200, headers=None, text=""):
        self._data, self._lines, self.status_code = data, list(lines), status
        self.ok = status < 400
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._data

    def raise_for_status(self):
        if not self.ok:
            raise app.requests.HTTPError(f"HTTP {self.status_code}")

    def iter_lines(self):
        yield from self._lines

    def iter_content(self, n):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TempDataTest(unittest.TestCase):
    """Points every data folder the app uses at a fresh temporary directory."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="llm-scanner-test-"))
        data = self.tmp / "data"
        data.mkdir()
        patches = {
            "DATA_DIR": data, "CHATS_DIR": data / "chats", "ATTACH_DIR": data / "attachments",
            "IMAGES_DIR": data / "images", "IMAGE_MODEL_DIR": data / "image-models",
            "SHARED_DIR": data / "image-models" / "shared", "CUSTOM_IMAGE_FILE": data / "image_models.json",
            "GROUPS_FILE": data / "probe_groups.json", "DOWNLOADS_FILE": data / "downloads.json",
            "GARAK_RUNS": self.tmp / "garak_runs", "CHAT_INDEX": app.ChatIndex(), "SCAN_JOB": data / "scan_job.json",
            "OLLAMA_BLOBS": self.tmp / "ollama-blobs", "PULL_LAYERS_FILE": data / "pull_layers.json",
        }
        for name, value in patches.items():
            p = mock.patch.object(app, name, value)
            p.start()
            self.addCleanup(p.stop)
        (self.tmp / "garak_runs").mkdir()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


# ---------------------------------------------------------------- helpers and parsing
class TextHelpers(unittest.TestCase):
    def test_content_text_shapes(self):
        self.assertEqual(app.content_text("hi"), "hi")
        self.assertEqual(app.content_text({"text": "a"}), "a")
        self.assertEqual(app.content_text([{"text": "a"}, "b", {"content": "c"}]), "abc")
        self.assertEqual(app.content_text(None), "")
        self.assertEqual(app.content_text(5), "5")

    def test_version_tuple(self):
        self.assertEqual(app._version_tuple("v1.2.3"), (1, 2, 3))
        self.assertEqual(app._version_tuple("0.34.2-rc1"), (0, 34, 2))
        self.assertEqual(app._version_tuple(None), ())
        self.assertGreater(app._version_tuple("1.0.10"), app._version_tuple("1.0.9"))

    def test_repo_from_text(self):
        for text in ("bartowski/SmolLM2-GGUF", "https://huggingface.co/bartowski/SmolLM2-GGUF",
                     "https://www.huggingface.co/bartowski/SmolLM2-GGUF/tree/main",
                     "https://huggingface.co/bartowski/SmolLM2-GGUF/blob/main/x.gguf",
                     "hf.co/bartowski/SmolLM2-GGUF:Q4_K_M", "  bartowski/SmolLM2-GGUF/  "):
            self.assertEqual(app.repo_from_text(text), "bartowski/SmolLM2-GGUF", text)
        self.assertEqual(app.repo_from_text(None), "")

    def test_ollama_name_from_text(self):
        self.assertEqual(app.ollama_name_from_text("https://ollama.com/library/qwen3/tags"), "qwen3")
        self.assertEqual(app.ollama_name_from_text("https://ollama.com/library/qwen3:8b"), "qwen3:8b")
        self.assertEqual(app.ollama_name_from_text("qwen3"), "qwen3")

    def test_fmt_secs(self):
        self.assertEqual(app._fmt_secs(5), "5s")
        self.assertEqual(app._fmt_secs(65), "1m 5s")
        self.assertEqual(app._fmt_secs(3700), "1h 1m")

    def test_human_size(self):
        self.assertEqual(app._human_size(10), "1 KB")
        self.assertEqual(app._human_size(2_500_000), "2.5 MB")

    def test_note_html_escapes(self):
        self.assertIn("&lt;b&gt;", app.note_html("<b>"))
        self.assertEqual(app.note_html(""), "")

    def test_slug(self):
        self.assertEqual(app._slug("Comfy-Org/z_image_turbo"), "comfy-org-z-image-turbo")
        self.assertLessEqual(len(app._slug("x" * 200)), 80)


class VersionPicking(unittest.TestCase):
    def test_best_version_prefers_largest_that_fits(self):
        with mock.patch.object(app, "VRAM_BUDGET_GB", 7.5):
            opts = [(2.0, "Q2_K"), (5.0, "Q4_K_M"), (7.4, "Q6_K"), (9.0, "Q8_0")]
            self.assertEqual(app.best_version(opts), "Q6_K")

    def test_best_version_falls_back_to_smallest(self):
        with mock.patch.object(app, "VRAM_BUDGET_GB", 1.0):
            self.assertEqual(app.best_version([(2.0, "a"), (3.0, "b")]), "a")
        self.assertEqual(app.best_version([]), "")

    def test_best_version_prefers_latest_tag_on_tie(self):
        with mock.patch.object(app, "VRAM_BUDGET_GB", 7.5):
            self.assertEqual(app.best_version([(5.2, "qwen3:8b-q4_K_M"), (5.2, "qwen3:latest"), (5.2, "qwen3:8b")]),
                             "qwen3:latest")

    def test_hf_versions_skips_shards_and_projectors(self):
        tree = [{"path": "Model-Q4_K_M.gguf", "size": 4e9}, {"path": "Model-UD-Q5_K_XL.gguf", "size": 5e9},
                {"path": "mmproj-F16.gguf", "size": 1e9}, {"path": "Model.imatrix.gguf", "size": 1e6},
                {"path": "Q8_0/Model-Q8_0-00001-of-00002.gguf", "size": 9e9}, {"path": "README.md", "size": 1},
                {"path": "Model-IQ3_XS.gguf", "size": 3e9}, {"path": "Model-bf16.gguf", "size": 16e9}]
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(tree)):
            got = app.hf_versions("x/y")
        self.assertEqual([t for _, t in got], ["IQ3_XS", "Q4_K_M", "UD-Q5_K_XL", "bf16"])

    def test_hf_versions_keeps_the_whole_quantization_name(self):
        """Repositories use their own quantization names (PQ2_0, PTQ1_0); reading them as Q2_0 asks for a file
        that does not exist."""
        tree = [{"path": "Ternary-Bonsai-2-27B-PQ2_0.gguf", "size": 7.2e9},
                {"path": "Ternary-Bonsai-2-27B-PTQ1_0.gguf", "size": 5.9e9},
                {"path": "Ternary-Bonsai-2-27B-F16.gguf", "size": 53.8e9},
                {"path": "Ternary-Bonsai-2-27B-mmproj-BF16.gguf", "size": 0.9e9}]
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(tree)):
            got = app.hf_versions("prism-ml/Ternary-Bonsai-2-27B-gguf")
        self.assertEqual([t for _, t in got], ["PTQ1_0", "PQ2_0", "F16"])
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(tree)):
            files = app._hf_gguf_files("prism-ml/Ternary-Bonsai-2-27B-gguf", "PQ2_0")
        self.assertEqual(files[0][0], "Ternary-Bonsai-2-27B-PQ2_0.gguf")
        self.assertEqual(files[1][0], "Ternary-Bonsai-2-27B-mmproj-BF16.gguf")

    def test_formats_ollama_cannot_load_are_marked_not_recommended(self):
        for tag in ("Q4_K_M", "UD-Q5_K_XL", "IQ3_XS", "TQ1_0", "BF16", "F16", "MXFP4_MOE", "q8_0"):
            self.assertTrue(app.runs_in_ollama(tag), tag)
        for tag in ("PQ2_0", "PTQ1_0", "AWQ4"):
            self.assertFalse(app.runs_in_ollama(tag), tag)
        tree = [{"path": "M-PQ2_0.gguf", "size": 7.2e9}, {"path": "M-PTQ1_0.gguf", "size": 5.9e9},
                {"path": "M-F16.gguf", "size": 53.8e9}]
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(tree)):
            versions, note, *_ = list(app.list_versions(app.SRC_HF, "prism-ml/x-gguf"))[-1]
        self.assertEqual(versions["value"], "F16", "a format Ollama can load must be the recommended one")
        labels = dict((v, lbl) for lbl, v in versions["choices"] if v)
        self.assertIn("Ollama can't load this format", labels["PQ2_0"])
        self.assertNotIn("can't load", labels["F16"])
        self.assertIn("PTQ1_0, PQ2_0 are in a format Ollama cannot load", note)

    def test_fit_breakdown(self):
        with mock.patch.object(app, "VRAM_BUDGET_GB", 7.5), mock.patch.object(app, "ram_gb", return_value=32):
            self.assertEqual(app.fit_breakdown(7), "7.0 GB VRAM")
            self.assertEqual(app.fit_breakdown(20), "7.5 GB VRAM + 12.5 GB RAM")
            self.assertEqual(app.fit_breakdown(40), "40.0 GB \u2014 too large for this PC")


# ---------------------------------------------------------------- chats
class ChatIndexTests(unittest.TestCase):
    def chat(self, cid, title, *messages, updated=0):
        return {"id": cid, "title": title, "updated": updated,
                "messages": [{"role": "user", "content": m} for m in messages]}

    def test_prefix_and_multi_term_search(self):
        idx = app.ChatIndex()
        idx.add(self.chat("a", "Python help", "how do decorators work", updated=1))
        idx.add(self.chat("b", "Cooking", "python recipe for pasta", updated=2))
        self.assertEqual([c["id"] for c in idx.search("pyth")], ["b", "a"])  # newest first
        self.assertEqual([c["id"] for c in idx.search("python deco")], ["a"])
        self.assertEqual(idx.search("nothing-here"), [])
        self.assertEqual(len(idx.search("")), 2)

    def test_remove_then_search_does_not_crash(self):
        idx = app.ChatIndex()
        idx.add(self.chat("a", "alpha"))
        self.assertEqual(len(idx.search("al")), 1)
        idx.remove("a")
        self.assertEqual(idx.search("al"), [])
        self.assertEqual(idx.search(""), [])

    def test_readd_replaces_old_words(self):
        idx = app.ChatIndex()
        idx.add(self.chat("a", "t", "banana"))
        idx.search("ban")
        idx.add(self.chat("a", "t", "cherry"))
        self.assertEqual(idx.search("banana"), [])
        self.assertEqual(len(idx.search("cher")), 1)

    def test_attachment_text_is_searchable(self):
        idx = app.ChatIndex()
        c = self.chat("a", "t", "see file")
        c["messages"][0]["attachments"] = [{"name": "notes.txt", "text": "quarterly zebra numbers", "kind": "text"}]
        idx.add(c)
        self.assertEqual(len(idx.search("zebra")), 1)
        self.assertEqual(len(idx.search("notes")), 1)

    def test_concurrent_add_and_search(self):
        idx = app.ChatIndex()
        errors = []

        def writer(n):
            try:
                for i in range(200):
                    idx.add(self.chat(f"{n}-{i}", f"word{i} thread{n}"))
                    if i % 3 == 0:
                        idx.remove(f"{n}-{i}")
            except Exception as e:  # pragma: no cover
                errors.append(e)

        def reader():
            try:
                for _ in range(300):
                    idx.search("word1")
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(3)] + [threading.Thread(target=reader)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])


class ChatFiles(TempDataTest):
    def test_chat_path_rejects_traversal(self):
        for bad in ("../x", "a/b", "", None, "a.b", "x\n"):
            self.assertIsNone(app._chat_path(bad), bad)
        self.assertEqual(app._chat_path("20260101-000000-abc123").name, "20260101-000000-abc123.json")

    def test_save_load_delete_round_trip(self):
        chat = {"id": "c1", "title": "Hello", "model": "m", "messages": [{"role": "user", "content": "hi"}],
                "totals": app.new_totals()}
        app.save_chat(chat)
        (app.ATTACH_DIR / "c1").mkdir(parents=True)
        self.assertEqual(app.load_chat("c1")["title"], "Hello")
        self.assertEqual(len(app.CHAT_INDEX.search("hello")), 1)
        self.assertTrue(app.delete_chat_file("c1"))
        self.assertIsNone(app.load_chat("c1"))
        self.assertFalse((app.ATTACH_DIR / "c1").exists())
        self.assertEqual(app.CHAT_INDEX.search("hello"), [])

    def test_load_chat_handles_corrupt_file(self):
        app.CHATS_DIR.mkdir()
        (app.CHATS_DIR / "bad.json").write_text("{not json")
        self.assertIsNone(app.load_chat("bad"))
        app.CHAT_INDEX.build()  # a corrupt file is skipped, not fatal
        self.assertEqual(app.CHAT_INDEX.search(""), [])

    def test_chat_submit_needs_a_model(self):
        with mock.patch.object(app, "model_names", return_value=[]):
            out = app.chat_submit("hello", app.empty_chat(), None, [])
        self.assertIn("Choose a model", out[2][-1]["content"])
        self.assertIsNone(out[1]["id"])  # nothing saved

    def test_chat_submit_empty_message_is_ignored(self):
        out = app.chat_submit("   ", app.empty_chat("m"), "m", [])
        self.assertIsNone(out[1]["id"])

    def test_chat_submit_creates_and_titles_chat(self):
        with mock.patch.object(app, "model_names", return_value=["m"]):
            out = app.chat_submit("**Explain** `closures` in JavaScript please, with several detailed examples",
                                  app.empty_chat(), "m", [])
        chat = out[1]
        self.assertTrue(chat["id"])
        self.assertTrue(chat["title"].endswith("..."))
        self.assertNotIn("*", chat["title"])
        self.assertEqual(app.load_chat(chat["id"])["messages"][0]["role"], "user")

    def test_chat_submit_keeps_locked_model(self):
        with mock.patch.object(app, "model_names", return_value=["m", "other"]):
            first = app.chat_submit("hi", app.empty_chat(), "m", [])[1]
            second = app.chat_submit("again", first, "other", [])[1]
        self.assertEqual(second["model"], "m")

    def test_chat_respond_streams_and_records_totals(self):
        chat = {"id": "c2", "title": "t", "model": "m", "messages": [{"role": "user", "content": "hi"}],
                "totals": app.new_totals()}
        lines = [json.dumps({"message": {"content": "Hel"}}).encode(),
                 json.dumps({"message": {"content": "lo"}}).encode(),
                 json.dumps({"message": {"content": ""}, "done": True, "prompt_eval_count": 5, "eval_count": 2,
                             "eval_duration": 1e9, "load_duration": 0}).encode()]
        with mock.patch.object(app, "model_capabilities", return_value=["completion"]), \
                mock.patch.object(app.requests, "post", return_value=FakeResponse(lines=lines)):
            outs = list(app.chat_respond(chat))
        final = outs[-1][0]
        self.assertEqual(final["messages"][-1]["content"], "Hello")
        self.assertEqual(final["totals"]["replies"], 1)
        self.assertEqual(final["totals"]["output"], 2)
        self.assertEqual(app.load_chat("c2")["messages"][-1]["content"], "Hello")

    def test_chat_respond_reports_ollama_errors(self):
        chat = {"id": "c3", "title": "t", "model": "m", "messages": [{"role": "user", "content": "hi"}],
                "totals": app.new_totals()}
        with mock.patch.object(app, "model_capabilities", return_value=[]), \
                mock.patch.object(app.requests, "post", side_effect=app.requests.ConnectionError("refused")):
            final = list(app.chat_respond(chat))[-1][0]
        self.assertIn("Error: refused", final["messages"][-1]["content"])

    def test_chat_stopped_replaces_typing_indicator(self):
        chat = {"id": "c4", "title": "t", "model": "m", "totals": app.new_totals(),
                "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": app.TYPING}]}
        out = app.chat_stopped(chat)
        self.assertEqual(out[2]["messages"][-1]["content"], "*Stopped.*")

    def test_remove_chat_requires_confirmation(self):
        app.save_chat({"id": "c5", "title": "t", "messages": [], "totals": app.new_totals()})
        app.remove_chat("c5", False, None)
        self.assertIsNotNone(app.load_chat("c5"))
        app.remove_chat("c5", True, None)
        self.assertIsNone(app.load_chat("c5"))


class Attachments(TempDataTest):
    def upload(self, name, content):
        p = self.tmp / name
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
        return str(p)

    def test_text_file_is_extracted_and_truncated(self):
        f = self.upload("big.txt", "x" * (app.MAX_ATTACH_CHARS + 50))
        with mock.patch.object(app, "model_capabilities", return_value=[]):
            pending, markup, _ = app.add_attachments([f], [], "m")
        self.assertEqual(len(pending), 1)
        self.assertTrue(pending[0]["truncated"])
        self.assertEqual(len(pending[0]["text"]), app.MAX_ATTACH_CHARS)
        self.assertIn("truncated", markup)

    def test_rejections(self):
        files = [self.upload("a.exe", "MZ"), self.upload("pic.png", b"\x89PNG"), self.upload("empty.txt", "  \n")]
        with mock.patch.object(app, "model_capabilities", return_value=["completion"]):
            pending, markup, _ = app.add_attachments(files, [], "m")
        self.assertEqual(pending, [])
        self.assertIn("unsupported file type", markup)
        self.assertIn("can&#x27;t view images", markup)
        self.assertIn("no readable text", markup)

    def test_images_allowed_for_vision_models(self):
        f = self.upload("pic.png", b"\x89PNG....")
        with mock.patch.object(app, "model_capabilities", return_value=["vision"]):
            pending, _, _ = app.add_attachments([f], [], "m")
        self.assertEqual(pending[0]["kind"], "image")

    def test_unsafe_file_names_are_sanitized(self):
        f = self.upload("we ird;name.txt", "hello")
        with mock.patch.object(app, "model_capabilities", return_value=[]):
            pending, markup, _ = app.add_attachments([f], [], "m")
        self.assertNotIn(" ", Path(pending[0]["path"]).name)
        self.assertTrue(Path(pending[0]["path"]).is_relative_to(app.ATTACH_DIR))

    def test_html_in_names_is_escaped(self):
        markup = app.attachments_html([{"name": "<img src=x onerror=alert(1)>.txt", "kind": "text", "size": 5,
                                        "path": "/tmp/x"}])
        self.assertNotIn("<img src=x", markup)

    def test_api_messages_inline_text_and_images(self):
        img = self.tmp / "p.png"
        img.write_bytes(b"abc")
        chat = {"messages": [{"role": "user", "content": "look", "attachments": [
            {"kind": "text", "name": "n.txt", "text": "DOC"}, {"kind": "image", "name": "p.png", "path": str(img)}]}]}
        msgs, skipped = app.api_messages(chat, vision=True)
        self.assertIn("DOC", msgs[0]["content"])
        self.assertEqual(msgs[0]["images"], ["YWJj"])
        self.assertEqual(skipped, [])
        msgs, skipped = app.api_messages(chat, vision=False)
        self.assertNotIn("images", msgs[0])
        self.assertEqual(skipped, ["p.png"])

    def test_missing_image_file_is_skipped(self):
        chat = {"messages": [{"role": "user", "content": "x", "attachments": [
            {"kind": "image", "name": "gone.png", "path": str(self.tmp / "gone.png")}]}]}
        msgs, skipped = app.api_messages(chat, vision=True)
        self.assertEqual(skipped, ["gone.png"])

    def test_attach_to_chat_moves_files(self):
        f = self.upload("n.txt", "hello")
        with mock.patch.object(app, "model_capabilities", return_value=[]):
            pending, _, _ = app.add_attachments([f], [], "m")
        stored = app._attach_to_chat(pending, "cid1")
        self.assertTrue(Path(stored[0]["path"]).exists())
        self.assertFalse(Path(pending[0]["path"]).exists())


# ---------------------------------------------------------------- downloads
class Downloads(TempDataTest):
    def setUp(self):
        super().setUp()
        for name, value in (("_downloads", {}), ("_priority", {"ref": None}), ("_dl_finished", {"count": 0})):
            p = mock.patch.object(app, name, value)
            p.start()
            self.addCleanup(p.stop)
        sleep = mock.patch.object(app.time, "sleep", lambda s: None)
        sleep.start()
        self.addCleanup(sleep.stop)

    def run_worker(self, ref, responses):
        app._downloads[ref] = {"state": "queued", "msg": "", "done": 0, "total": 0, "cancel": False}
        calls = iter(responses)

        def fake_post(*a, **k):
            try:
                r = next(calls)
            except StopIteration:
                app._downloads[ref]["cancel"] = True
                return FakeResponse(lines=[])
            if isinstance(r, Exception):
                raise r
            return r

        with mock.patch.object(app, "ollama_up", return_value="0.34"), \
                mock.patch.object(app.requests, "post", side_effect=fake_post):
            app._download_worker(ref)
        return app._downloads[ref]

    def test_successful_pull(self):
        lines = [json.dumps({"status": "pulling manifest"}).encode(),
                 json.dumps({"digest": "sha:1", "total": 100, "completed": 50}).encode(),
                 json.dumps({"status": "success"}).encode()]
        d = self.run_worker("m:1", [FakeResponse(lines=lines)])
        self.assertEqual(d["state"], "done")
        self.assertEqual(app._dl_finished["count"], 1)

    def test_unknown_model_fails_without_retrying(self):
        d = self.run_worker("nope", [FakeResponse(lines=[json.dumps({"error": "pull model manifest: file does not exist"}).encode()])])
        self.assertEqual(d["state"], "failed")
        self.assertIn("does not exist", d["msg"])

    def test_network_drop_resumes(self):
        ok = [json.dumps({"status": "success"}).encode()]
        d = self.run_worker("m:2", [app.requests.ConnectionError("boom"), FakeResponse(lines=ok)])
        self.assertEqual(d["state"], "done")

    def test_garbled_stream_line_is_retried_not_failed(self):
        """A half-received JSON line (connection cut mid-line) is a network problem, not a bad model name."""
        ok = [json.dumps({"status": "success"}).encode()]
        d = self.run_worker("m:3", [FakeResponse(lines=[b'{"status": "pulling', ]), FakeResponse(lines=ok)])
        self.assertEqual(d["state"], "done")

    def test_http_error_from_ollama_is_permanent(self):
        r = FakeResponse({"error": "invalid model name"}, status=400)
        d = self.run_worker("bad name", [r])
        self.assertEqual(d["state"], "failed")
        self.assertIn("HTTP 400", d["msg"])
        self.assertIn("invalid model name", d["msg"])

    def test_pending_list_persists_and_clears(self):
        app._set_pending("a", True)
        app._set_pending("b", True)
        app._set_pending("a", False)
        self.assertEqual(app._pending(), [("b", False)])
        app._set_pending("b", True, paused=True)
        self.assertEqual(app._pending(), [("b", True)])

    def test_old_pending_format_still_resumes(self):
        app.DOWNLOADS_FILE.write_text(json.dumps(["m:1", "m:2"]))  # written by 1.0.11 and earlier
        with mock.patch.object(app, "start_download") as start:
            app.resume_pending_downloads()
        self.assertEqual([c.args[0] for c in start.call_args_list], ["m:1", "m:2"])

    def test_paused_priority_download_does_not_hold_others(self):
        app._downloads["p"] = {"state": "paused", "msg": "", "done": 0, "total": 0, "cancel": False, "pause": True}
        app._priority["ref"] = "p"
        self.assertFalse(app._yield_to_priority("other"))
        app._downloads["p"]["pause"] = False
        app._downloads["p"]["state"] = "downloading"
        self.assertTrue(app._yield_to_priority("other"))

    def test_run_download_restarts_worker_resumed_while_stopping(self):
        app._downloads["r"] = {"state": "queued", "msg": "", "done": 0, "total": 0, "cancel": False, "running": True}
        calls = []

        def worker(ref):
            calls.append(ref)
            # 1st run: stopped for a pause, but the user already pressed resume; 2nd run: finishes
            app._downloads[ref]["state"] = "paused" if len(calls) == 1 else "done"

        with mock.patch.object(app, "_download_worker", worker):
            app._run_download("r")
        self.assertEqual(len(calls), 2)
        self.assertFalse(app._downloads["r"]["running"])

    def test_row_icons_match_state(self):
        base = {"msg": "", "done": 1, "total": 4, "cancel": False}
        app._downloads["run"] = {**base, "state": "downloading"}
        app._downloads["held"] = {**base, "state": "paused", "pause": True}
        app._downloads["fin"] = {**base, "state": "failed"}
        out = app.downloads_html()
        run, held, fin = (out[out.index(f'title="{r}"'):] for r in ("run", "held", "fin"))
        self.assertIn('class="dl-toggle pause"', run.split('class="dl"')[0])
        self.assertIn('title="Delete download"', run.split('class="dl"')[0])
        self.assertIn('class="dl-toggle resume"', held.split('class="dl"')[0])
        self.assertNotIn("dl-toggle", fin.split('class="dl"')[0])
        self.assertIn('title="Remove from list"', fin.split('class="dl"')[0])
        self.assertEqual(app.active_downloads(), ["run"])  # paused ones aren't offered for priority

    def test_pause_and_resume_controls(self):
        app._downloads["r"] = {"state": "downloading", "msg": "", "done": 0, "total": 0, "cancel": False,
                               "running": True}
        app.toggle_download("pause:r")
        self.assertEqual((app._downloads["r"]["state"], app._downloads["r"]["pause"]), ("paused", True))
        self.assertEqual(app._pending(), [("r", True)])
        with mock.patch.object(app, "_start_worker") as start:
            app.toggle_download("resume:r")
        start.assert_not_called()  # its worker is still winding down; _run_download picks it up again
        self.assertFalse(app._downloads["r"]["pause"])
        self.assertEqual(app._pending(), [("r", False)])
        app._downloads["r"].update(pause=True, running=False)
        with mock.patch.object(app, "_start_worker") as start:
            app.toggle_download("resume:r")
        start.assert_called_once_with("r")
        self.assertEqual(app.toggle_download("pause:unknown")[1], "")

    def test_start_download_rejects_duplicates(self):
        with mock.patch.object(app.threading, "Thread"):
            self.assertTrue(app.start_download("x"))
            self.assertFalse(app.start_download("x"))

    def test_remove_download_hides_and_cancels(self):
        app._downloads["r"] = {"state": "downloading", "msg": "", "done": 0, "total": 0, "cancel": False}
        app._priority["ref"] = "r"
        app.remove_download("r")
        self.assertTrue(app._downloads["r"]["cancel"])
        self.assertIsNone(app._priority["ref"])
        self.assertIn("No active downloads", app.downloads_html())

    def test_downloads_html_escapes_refs(self):
        app._downloads['<x>"'] = {"state": "downloading", "msg": "<b>", "done": 1, "total": 2, "cancel": False}
        out = app.downloads_html()
        self.assertNotIn("<x>", out)
        self.assertNotIn("<b>", out)

    def test_pull_model_needs_both_fields(self):
        self.assertIn("Choose a repository", app.pull_model(app.SRC_HF, "", "Q4"))
        self.assertIn("Choose a version", app.pull_model(app.SRC_HF, "x/y", ""))


class HuggingFaceImport(TempDataTest):
    """Ollama can't follow Hugging Face's redirect to its Xet storage, so the app downloads the GGUF itself."""
    REF = "hf.co/bartowski/SmolLM2-GGUF:Q4_K_M"
    BLOCKED = FakeResponse(lines=[json.dumps({"status": "pulling manifest"}).encode(), json.dumps(
        {"error": 'Head "https://us.aws.cdn.hf.co/xet-bridge-us/abc": blocked redirect to a different host'}).encode()])
    TREE = [{"path": "SmolLM2-IQ4_K_M.gguf", "size": 6}, {"path": "SmolLM2-UD-Q4_K_M.gguf", "size": 7},
            {"path": "SmolLM2-Q4_K_M.gguf", "size": 8}, {"path": "SmolLM2-Q8_0.gguf", "size": 9}]

    def setUp(self):
        super().setUp()
        for name, value in (("_downloads", {}), ("_priority", {"ref": None}), ("_dl_finished", {"count": 0}),
                            ("HF_DOWNLOADS", self.tmp / "data" / "hf-downloads")):
            p = mock.patch.object(app, name, value)
            p.start()
            self.addCleanup(p.stop)
        for target in (mock.patch.object(app.time, "sleep", lambda s: None),
                       mock.patch.object(app, "ollama_up", return_value="0.34.2"),
                       mock.patch.object(app.requests, "post", return_value=self.BLOCKED)):
            target.start()
            self.addCleanup(target.stop)
        self.gets = []

    def fake_get(self, file_chunks, tree=None):
        def get(url, **kw):
            self.gets.append((url, kw.get("headers") or {}))
            if "/api/models/" in url:
                return FakeResponse(self.TREE if tree is None else tree, status=200 if tree != 404 else 404)
            return FakeResponse(lines=file_chunks.pop(0) if file_chunks else [])
        return get

    def run_worker(self, get, create_rc=0):
        app._downloads[self.REF] = {"state": "queued", "msg": "", "done": 0, "total": 0, "cancel": False}
        created = []

        def fake_run(cmd, **kw):
            created.append(Path(cmd[-1]).read_text())
            return mock.Mock(returncode=create_rc, stdout="", stderr="Error: invalid file magic")

        with mock.patch.object(app.requests, "get", side_effect=get), mock.patch.object(app.subprocess, "run", fake_run):
            app._download_worker(self.REF)
        return app._downloads[self.REF], created

    def test_blocked_redirect_downloads_and_imports(self):
        d, created = self.run_worker(self.fake_get([[b"GGUF", b"data"]]))
        self.assertEqual(d["state"], "done")
        self.assertTrue(self.gets[-1][0].endswith("/resolve/main/SmolLM2-Q4_K_M.gguf"), self.gets[-1][0])
        self.assertIn("FROM ", created[0])
        self.assertEqual(list(app.HF_DOWNLOADS.iterdir()), [], "the downloaded copy must be deleted after import")

    def test_vision_model_imports_its_projector(self):
        tree = [{"path": "SmolVLM-Q8_0.gguf", "size": 4}, {"path": "SmolVLM-f16.gguf", "size": 8},
                {"path": "mmproj-SmolVLM-f16.gguf", "size": 3}, {"path": "mmproj-SmolVLM-Q8_0.gguf", "size": 2}]
        app._downloads.clear()
        ref = "hf.co/ggml-org/SmolVLM-GGUF:Q8_0"
        app._downloads[ref] = {"state": "queued", "msg": "", "done": 0, "total": 0, "cancel": False}
        modelfiles = []

        def fake_run(cmd, **kw):
            modelfiles.append(Path(cmd[-1]).read_text())
            return mock.Mock(returncode=0)
        with mock.patch.object(app.requests, "get", side_effect=self.fake_get([[b"GGUF"], [b"PJ"]], tree=tree)), \
                mock.patch.object(app.subprocess, "run", fake_run):
            app._download_worker(ref)
        self.assertEqual(app._downloads[ref]["state"], "done")
        urls = [u for u, _ in self.gets if "/resolve/" in u]
        self.assertTrue(urls[0].endswith("/SmolVLM-Q8_0.gguf") and urls[1].endswith("/mmproj-SmolVLM-Q8_0.gguf"), urls)
        self.assertEqual(modelfiles[0].count("FROM "), 2)
        self.assertEqual(list(app.HF_DOWNLOADS.iterdir()), [])

    def test_projector_is_never_picked_as_the_model(self):
        tree = [{"path": "mmproj-Q8_0.gguf", "size": 1}, {"path": "x/Model-Q8_0.gguf", "size": 9}]
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(tree)):
            files = app._hf_gguf_files("a/b", "Q8_0")
        self.assertEqual([f for f, _ in files], ["x/Model-Q8_0.gguf", "mmproj-Q8_0.gguf"])

    def test_blocked_redirect_as_http_error_also_imports(self):
        """Ollama sometimes returns the redirect failure as an HTTP 400/500 response instead of in the stream."""
        body = {"error": 'Head "https://us.aws.cdn.hf.co/xet-bridge-us/abc?Signature=x": blocked redirect to a different host'}
        for status in (400, 500):
            with self.subTest(status=status), \
                    mock.patch.object(app.requests, "post", return_value=FakeResponse(body, status=status)):
                app._downloads.clear()
                d, created = self.run_worker(self.fake_get([[b"GGUF", b"data"]]))
                self.assertEqual(d["state"], "done", d["msg"])
                self.assertEqual(len(created), 1)

    def test_tag_ollama_rejects_is_downloaded_directly(self):
        """Ollama resolves hf.co tags its own way and answers 400 "The specified tag is not available"; the file is
        in the repository, so the app fetches it itself instead of failing."""
        rejected = FakeResponse({"error": "The specified tag is not available in the repository. Please use another "
                                          "tag or \"latest\""}, status=400)
        with mock.patch.object(app.requests, "post", return_value=rejected):
            d, created = self.run_worker(self.fake_get([[b"GGUF", b"data"]]))
        self.assertEqual(d["state"], "done", d["msg"])
        self.assertEqual(len(created), 1)

    def test_interrupted_download_resumes_with_range(self):
        def flaky():
            calls = iter([[b"GGUF"], app.requests.ConnectionError("drop"), [b"data"]])
            def get(url, **kw):
                self.gets.append((url, kw.get("headers") or {}))
                if "/api/models/" in url:
                    return FakeResponse(self.TREE)
                nxt = next(calls)
                if isinstance(nxt, Exception):
                    raise nxt
                return FakeResponse(lines=nxt, status=206 if kw.get("headers") else 200)
            return get
        # first file request delivers 4 of 8 bytes and ends early, the next drops, the third resumes
        d, created = self.run_worker(flaky())
        self.assertEqual(d["state"], "done")
        self.assertEqual(self.gets[-1][1], {"Range": "bytes=4-"})

    def test_missing_quantization_fails_clearly(self):
        app._downloads.clear()
        ref = "hf.co/bartowski/SmolLM2-GGUF:Q3_K_L"
        app._downloads[ref] = {"state": "queued", "msg": "", "done": 0, "total": 0, "cancel": False}
        with mock.patch.object(app.requests, "get", side_effect=self.fake_get([])):
            app._download_worker(ref)
        self.assertEqual(app._downloads[ref]["state"], "failed")
        self.assertIn("no GGUF file for Q3_K_L", app._downloads[ref]["msg"])
        self.assertIn("It has: IQ4_K_M, Q4_K_M", app._downloads[ref]["msg"], "say which versions the repo has")

    def test_missing_repository_fails_clearly(self):
        d, _ = self.run_worker(self.fake_get([], tree=404))
        self.assertEqual(d["state"], "failed")
        self.assertIn("HTTP 404", d["msg"])

    def test_failed_import_is_retried_and_keeps_download(self):
        attempts = []

        def fake_run(cmd, **kw):
            attempts.append(1)
            if len(attempts) == 2:
                app._downloads[self.REF]["cancel"] = True
            return mock.Mock(returncode=1, stdout="", stderr="Error: connection refused")

        app._downloads[self.REF] = {"state": "queued", "msg": "", "done": 0, "total": 0, "cancel": False}
        with mock.patch.object(app.requests, "get", side_effect=self.fake_get([[b"GGUFdata"]])), \
                mock.patch.object(app.subprocess, "run", fake_run):
            app._download_worker(self.REF)
        downloads = [u for u, _ in self.gets if "/resolve/" in u]
        self.assertEqual(len(downloads), 1, "the file is downloaded once; only the import is retried")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(app._downloads[self.REF]["state"], "cancelled")

    def test_removed_download_deletes_partial_file(self):
        def get(url, **kw):
            if "/api/models/" in url:
                return FakeResponse(self.TREE)
            app._downloads[self.REF].update(cancel=True, hidden=True)
            return FakeResponse(lines=[b"GG", b"UF"])
        d, _ = self.run_worker(get)
        self.assertEqual(d["state"], "cancelled")
        self.assertEqual([p.name for p in app.HF_DOWNLOADS.iterdir()], [])


class PauseResumeDelete(HuggingFaceImport):
    """Pause keeps the partial file for later, resume continues it with a range request, delete removes it."""

    def test_pause_keeps_partial_file_and_resume_continues(self):
        class PausedMidway(FakeResponse):
            def iter_content(inner, n):
                yield b"GGUF"
                app.toggle_download(f"pause:{self.REF}")  # pressed after the first bytes arrived
                yield b"data"

        def get(url, **kw):
            return FakeResponse(self.TREE) if "/api/models/" in url else PausedMidway()
        d, created = self.run_worker(get)
        part = app.HF_DOWNLOADS / (app._slug(self.REF) + ".gguf.part")
        self.assertEqual(d["state"], "paused")
        self.assertEqual(part.read_bytes(), b"GGUF")
        self.assertEqual(app._pending(), [(self.REF, True)])
        self.assertEqual(created, [])

        d["pause"] = False
        ranges = []

        def get2(url, **kw):
            ranges.append((kw.get("headers") or {}).get("Range"))
            if "/api/models/" in url:
                return FakeResponse(self.TREE)
            return FakeResponse(lines=[b"data"], status=206)
        with mock.patch.object(app.requests, "get", side_effect=get2), \
                mock.patch.object(app.subprocess, "run", return_value=mock.Mock(returncode=0)):
            app._download_worker(self.REF)
        self.assertEqual(d["state"], "done")
        self.assertIn("bytes=4-", ranges)
        self.assertEqual(app._pending(), [])

    def test_delete_paused_download_removes_partial_file_now(self):
        app.HF_DOWNLOADS.mkdir(parents=True)
        part = app.HF_DOWNLOADS / (app._slug(self.REF) + ".gguf.part")
        part.write_bytes(b"GG")
        app._downloads[self.REF] = {"state": "paused", "msg": "Paused", "done": 2, "total": 8, "cancel": False,
                                    "pause": True, "running": False}
        app._set_pending(self.REF, True, paused=True)
        app.remove_download(self.REF)
        self.assertFalse(part.exists())
        self.assertEqual(app._downloads[self.REF]["state"], "cancelled")
        self.assertEqual(app._pending(), [])
        self.assertIn("No active downloads", app.downloads_html())

    def test_restart_keeps_paused_downloads_paused(self):
        app.HF_DOWNLOADS.mkdir(parents=True)
        (app.HF_DOWNLOADS / (app._slug(self.REF) + ".gguf.part")).write_bytes(b"xxx")
        app.DOWNLOADS_FILE.write_text(json.dumps([{"ref": self.REF, "paused": True}, "m:1"]))
        with mock.patch.object(app, "start_download") as start:
            app.resume_pending_downloads()
        self.assertEqual([c.args[0] for c in start.call_args_list], ["m:1"])
        d = app._downloads[self.REF]
        self.assertEqual((d["state"], d["pause"], d["done"]), ("paused", True, 3))
        self.assertIn('class="dl-toggle resume"', app.downloads_html())


class DeleteModels(unittest.TestCase):
    def test_needs_confirmation(self):
        with mock.patch.object(app.requests, "delete") as delete:
            app.delete_model("m", False)
        delete.assert_not_called()

    def test_deletes_and_refreshes_the_list(self):
        with mock.patch.object(app.requests, "delete", return_value=FakeResponse(status=200)) as delete:
            note, version = app.delete_model("qwen3:8b", True)
        self.assertEqual(delete.call_args.kwargs["json"], {"model": "qwen3:8b"})
        self.assertIn("Deleted qwen3:8b.", note)
        self.assertIsInstance(version, float)

    def test_reports_ollama_errors(self):
        with mock.patch.object(app.requests, "delete", return_value=FakeResponse({"error": "model not found"}, status=404)):
            note, _ = app.delete_model("gone", True)
        self.assertIn("HTTP 404", note)
        self.assertIn("model not found", note)
        self.assertIn("error", note)

    def test_ollama_not_running(self):
        with mock.patch.object(app.requests, "delete", side_effect=app.requests.ConnectionError("refused")):
            note, _ = app.delete_model("m", True)
        self.assertIn("Ollama is not running", note)


class FetchFile(TempDataTest):
    def fetch(self, response, size, have=b""):
        dest = self.tmp / "f.bin"
        if have:
            dest.with_name("f.bin.part").write_bytes(have)
        seen = {}

        def get(url, **kw):
            seen.update(kw.get("headers") or {})
            return response
        with mock.patch.object(app.requests, "get", side_effect=get):
            ok = app.fetch_file("https://x/f.bin", dest, size, lambda: False, lambda: None)
        return ok, dest, seen

    def test_resume_appends(self):
        ok, dest, seen = self.fetch(FakeResponse(lines=[b"5678"], status=206), 8, have=b"1234")
        self.assertTrue(ok)
        self.assertEqual(seen, {"Range": "bytes=4-"})
        self.assertEqual(dest.read_bytes(), b"12345678")

    def test_server_ignoring_range_restarts_file(self):
        ok, dest, _ = self.fetch(FakeResponse(lines=[b"12345678"], status=200), 8, have=b"xxxx")
        self.assertEqual(dest.read_bytes(), b"12345678")

    def test_already_complete(self):
        ok, dest, _ = self.fetch(FakeResponse(status=416), 4, have=b"1234")
        self.assertEqual(dest.read_bytes(), b"1234")

    def test_short_download_is_retried(self):
        with self.assertRaises(ConnectionError):
            self.fetch(FakeResponse(lines=[b"12"], status=200), 8)

    def test_client_errors_are_permanent_server_errors_are_not(self):
        with self.assertRaises(app.PermanentDownloadError):
            self.fetch(FakeResponse({"error": "gated repo"}, status=403), 8)
        with self.assertRaises(app.requests.HTTPError):
            self.fetch(FakeResponse(status=503), 8)

    def test_response_error_uses_server_detail(self):
        r = FakeResponse({"error": "failed to fetch: repository not found"}, status=400)
        self.assertEqual(app.response_error(r, "Pulling x"),
                         "Pulling x failed (HTTP 400): failed to fetch: repository not found")
        plain = mock.Mock(status_code=502, text=" Bad   Gateway ")
        plain.json.side_effect = ValueError
        self.assertEqual(app.response_error(plain, "Download"), "Download failed (HTTP 502): Bad Gateway")


# ---------------------------------------------------------------- image models and gallery
class ImageModels(TempDataTest):
    def test_detect_family(self):
        cases = {"leejet/Z-Image-Turbo-GGUF": "z-image", "city96/FLUX.1-schnell-gguf": "flux1-schnell",
                 "city96/FLUX.1-dev-gguf": "flux1-dev", "stabilityai/sdxl-turbo": "sdxl",
                 "stable-diffusion-v1-5/stable-diffusion-v1-5": "sd15", "Lykon/dreamshaper-8": "sd15",
                 "black-forest-labs/FLUX.2-dev": None, "Qwen/Qwen-Image": None, "Wan-AI/Wan2.1-T2V": None,
                 "stabilityai/stable-diffusion-xl-refiner-inpaint": None, "some/random-thing": None,
                 "comfyanonymous/flux_text_encoders": None}
        for repo, fam in cases.items():
            self.assertEqual(app.detect_family(repo)[0], fam, repo)

    def test_image_repo_files(self):
        tree = [{"path": "flux1-schnell-Q4_K_S.gguf", "size": 6.8e9}, {"path": "flux1-schnell-Q8_0.gguf", "size": 12.7e9},
                {"path": "ae.safetensors", "size": 3e8}, {"path": "vae/x.gguf", "size": 1e8},
                {"path": "part-00001-of-00002.gguf", "size": 5e9}]
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(tree)):
            got = app.image_repo_files("city96/FLUX.1-schnell-gguf", "flux1-schnell")
        self.assertEqual([p for _, p in got], ["flux1-schnell-Q4_K_S.gguf", "flux1-schnell-Q8_0.gguf"])

    def test_checkpoints_prefer_safetensors(self):
        tree = [{"path": "model.ckpt", "size": 4e9}, {"path": "model.safetensors", "size": 4e9},
                {"path": "unet/diffusion.safetensors", "size": 3e9}, {"path": "tiny.safetensors", "size": 1e8}]
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(tree)):
            got = app.image_repo_files("x/sdxl", "sdxl")
        self.assertEqual([p for _, p in got], ["model.safetensors"])

    def test_add_and_delete_searched_model_keeps_shared_files(self):
        with mock.patch.object(app, "start_download", return_value=True):
            app.download_searched_image_model("city96/FLUX.1-schnell-gguf", "flux1-schnell-Q4_K_S.gguf")
        key = next(iter(app._custom_image_models()))
        self.assertEqual(app.image_models()[key]["name"], "FLUX.1-schnell (Q4_K_S)")
        for p in list(app.image_model_files(key).values()) + list(app.image_model_files("z-image-turbo").values()):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        self.assertTrue(app.image_model_installed(key))
        vae = app.image_model_files(key)["vae"]
        clip = app.image_model_files(key)["clip_l"]
        app.delete_image_model(key, True)
        self.assertNotIn(key, app.image_models())
        self.assertTrue(vae.exists(), "the VAE is shared with Z-Image Turbo and must be kept")
        self.assertFalse(clip.exists(), "clip_l is only used by the deleted model")

    def test_delete_requires_confirmation(self):
        app.delete_image_model("z-image-turbo", False)
        self.assertIn("z-image-turbo", app.image_models())

    def test_gallery_delete_blocks_traversal(self):
        app.IMAGES_DIR.mkdir(parents=True)
        outside = self.tmp / "keep.png"
        outside.write_bytes(b"x")
        app.delete_gallery_image("../keep.png")
        self.assertTrue(outside.exists())
        img = app.IMAGES_DIR / "a.png"
        img.write_bytes(b"x")
        img.with_suffix(".json").write_text("{}")
        app.delete_gallery_image("a.png")
        self.assertFalse(img.exists() or img.with_suffix(".json").exists())

    def test_gallery_html_escapes_prompt(self):
        app.IMAGES_DIR.mkdir(parents=True)
        (app.IMAGES_DIR / "a.png").write_bytes(b"x")
        (app.IMAGES_DIR / "a.json").write_text(json.dumps({"prompt": '"><script>x</script>', "model": "m", "seed": 1,
                                                            "width": 1, "height": 1, "steps": 1, "seconds": 1}))
        self.assertNotIn("<script>", app.gallery_html())

    def test_generate_image_validates_inputs(self):
        out = list(app.generate_image(None, "p", "Square 1024 x 1024", 8, -1))
        self.assertIn("Download an image model", out[0][1])
        with mock.patch.object(app, "image_model_installed", return_value=True):
            out = list(app.generate_image("z-image-turbo", "  ", "Square 1024 x 1024", 8, -1))
        self.assertIn("Describe the image", out[0][1])

    def test_generate_image_runs_engine(self):
        """Runs a fake engine that prints sampling progress and writes the output file."""
        engine = self.tmp / "sd-cli"
        engine.write_text("#!/usr/bin/env python3\nimport sys\nout = sys.argv[sys.argv.index('-o') + 1]\n"
                          "for i in range(1, 5):\n    print(f'  |===| {i}/4 - 1.00s/it', flush=True)\n"
                          "open(out, 'wb').write(b'png')\n")
        engine.chmod(0o755)
        with mock.patch.object(app, "SD_CLI", engine), mock.patch.object(app, "image_model_installed", return_value=True), \
                mock.patch.object(app, "unload_models"):
            outs = list(app.generate_image("z-image-turbo", "a cat", "Small 512 x 512 (fastest)", 4, 42))
        self.assertIn("Image ready", outs[-1][0])
        self.assertTrue(outs[-1][4].endswith("-42.png"))
        meta = json.loads(Path(outs[-1][4]).with_suffix(".json").read_text())
        self.assertEqual((meta["seed"], meta["width"], meta["steps"]), (42, 512, 4))
        self.assertIsNone(app._image_proc["p"])


# ---------------------------------------------------------------- probes, groups, scans
class ProbesAndGroups(TempDataTest):
    def test_catalog_loaded(self):
        self.assertGreater(len(app.CATALOG), 20)
        self.assertIn("dan", app.MODULES)

    def test_presets_reference_real_probes(self):
        for name, (specs, _) in app.PRESETS.items():
            for s in specs:
                self.assertTrue(s in app.CATALOG or s in app.MODULES, f"{name}: {s} is not a garak probe")
            if specs:
                self.assertTrue(app.expand(specs), name)

    def test_owasp_preset_covers_every_testable_category(self):
        probes, desc = app.PRESETS["OWASP Top 10 for LLMs"]
        self.assertGreater(len(probes), 20)
        self.assertTrue(all(app.CATALOG[p]["active"] for p in probes))
        self.assertTrue(all(any(t.startswith("owasp:") for t in app.CATALOG[p]["tags"]) for p in probes))
        tagged = {t for p in probes for t in app.CATALOG[p]["tags"] if t.startswith("owasp:")}
        for n, v in app.CATALOG.items():  # no active OWASP-tagged probe is left out
            if v["active"] and tagged & set(v["tags"]):
                self.assertIn(n, probes)
        self.assertIn("LLM01", desc)
        self.assertEqual(app.resolve_scan("OWASP Top 10 for LLMs", []), (probes, None))
        self.assertEqual(list(app.PRESETS)[-1], "Full scan")
        self.assertIn("Preset: OWASP Top 10 for LLMs", app.group_choices())

    def test_framework_groups(self):
        names = ["Baseline assessment", "NIST AI 600-1: Information security",
                 "NIST AI 600-1: Information integrity and confabulation",
                 "NIST AI 600-1: Dangerous, violent and hateful content",
                 "NIST AI 600-1: Data privacy and intellectual property", "NIST AI 600-1: Harmful bias",
                 "EU AI Act risk areas"]
        for name in names:
            probes, desc = app.PRESETS[name]
            self.assertTrue(probes, name)
            self.assertTrue(all(p in app.CATALOG and app.CATALOG[p]["active"] for p in probes), name)
            self.assertFalse({p.split(".")[0] for p in probes} & app.NEEDS_ATTACKER_MODEL, name)
            self.assertIn("not a certified compliance test", desc)
            self.assertIn(f"Preset: {name}", app.group_choices())
        baseline = set(app.PRESETS["Baseline assessment"][0])
        self.assertEqual(len(baseline), 19, "every probe named in the baseline must exist in garak")
        for name in names[1:6]:  # the baseline samples every NIST risk area
            self.assertTrue(baseline & set(app.PRESETS[name][0]), name)
        self.assertEqual(list(app.PRESETS)[-1], "Full scan")

    def test_expand_dedupes_and_ignores_unknown(self):
        probe = next(iter(app.CATALOG))
        self.assertEqual(app.expand([probe, probe, "not.a.probe"]), [probe])

    def test_groups_save_load_delete(self):
        probe = next(iter(app.CATALOG))
        self.assertIn("Enter a name", app.save_group("  ", [probe])[0])
        self.assertIn("Select at least one", app.save_group("g", [])[0])
        self.assertIn("Saved", app.save_group("g", [probe])[0])
        self.assertIn("Updated", app.save_group("g", [probe])[0])
        self.assertIn(app.GROUP_PREFIX + "g", app.scan_choices())
        self.assertEqual(app.resolve_scan(app.GROUP_PREFIX + "g", []), ([probe], None))
        app.delete_group("g")
        self.assertEqual(app.resolve_scan(app.GROUP_PREFIX + "g", [])[1], "That group no longer exists.")

    def test_resolve_scan(self):
        self.assertEqual(app.resolve_scan("Full scan", []), ([], None))
        self.assertIsNotNone(app.resolve_scan(app.CURRENT, [])[1])
        self.assertEqual(app.resolve_scan(app.CURRENT, ["a"]), (["a"], None))
        self.assertIsNotNone(app.resolve_scan(None, [])[1])

    def test_probe_selection_keeps_hidden_choices(self):
        names = list(app.CATALOG)
        a, b = names[0], names[-1]
        new, _ = app.probes_ticked([b], app.CATALOG[b]["module"], "All categories", True, [a, b])
        self.assertIn(a, new)  # a is filtered out of view, so unticking b's view must not drop it
        new, _ = app.probes_ticked([], b, "All categories", True, [a, b])
        self.assertNotIn(b, new)

    def test_scan_info_lists_every_probe_with_an_explanation(self):
        for name, (specs, desc) in app.PRESETS.items():
            out = app.scan_info(name, [])
            probes = app.expand(specs) if specs else [n for n, v in app.CATALOG.items() if v["active"]]
            self.assertIn(app.html.escape(desc), out, name)
            self.assertIn(f"{len(probes)} probe", out, name)
            for p in probes:
                self.assertIn(f'data-tip-title="{p}"', out, f"{name}: {p} not listed")
        self.assertIn("Every default", app.scan_info("Full scan", []))
        out = app.scan_info(app.CURRENT, ["dan.DanInTheWild"])
        self.assertIn("currently selected on the Probes tab", out)
        self.assertIn("No probes selected", app.scan_info(app.CURRENT, []))

    def test_every_probe_family_is_explained(self):
        self.assertEqual(sorted(set(v["module"] for v in app.CATALOG.values()) - set(app.MODULE_INFO)), [])
        tip = app.probe_tip("dan.DanInTheWild")
        self.assertIn("Do Anything Now", tip)
        self.assertIn("Goal: Disregard the system prompt", tip)
        self.assertEqual(app.probe_tip("not.a.probe"), "")
        self.assertIn('data-tip="', app.probes_html(["dan.DanInTheWild"], "<b>x</b>"))
        self.assertNotIn("<b>x</b>", app.probes_html(["dan.DanInTheWild"], "<b>x</b>"))

    def test_group_info_for_presets_and_saved_groups(self):
        self.assertIn("NIST AI 600-1", app.group_info("Preset: NIST AI 600-1: Harmful bias") + "NIST AI 600-1")
        self.assertIn("Whether the model discriminates", app.group_info("Preset: NIST AI 600-1: Harmful bias"))
        app.save_group("mine", ["dan.DanInTheWild"])
        self.assertIn('Your saved group &quot;mine&quot;', app.group_info("mine"))
        self.assertEqual(app.group_info(None), "")


class ScanProgressTests(unittest.TestCase):
    def test_progress_from_garak_output(self):
        p = app.ScanProgress()
        p.feed("queue of probes: dan.DanInTheWild, goodside.WhoIsRiley")
        self.assertEqual(p.percent(), 0)
        p.feed("probes.dan.DanInTheWild:  50%|#####     | 32/64 [00:10<00:10]")
        self.assertAlmostEqual(p.percent(), 100 * 0.425 / 2)
        p.feed("dan.DanInTheWild  mitigation.MitigationBypass: FAIL  ok on  10/  64")
        self.assertEqual(p.percent(), 50)
        p.feed("goodside.WhoIsRiley  goodside.RileyIsnt: PASS  ok on  6/  6")
        self.assertEqual(p.percent(), 100)
        html = p.html()
        self.assertIn("1 passed, 1 failed", html)
        self.assertIn("Scan complete", p.html("done"))

    def test_run_scan_guards(self):
        self.assertIn("Select a model", next(app.run_scan(None, "Quick check", [], 1, 60))[0])
        with mock.patch.object(app, "ollama_up", return_value=None):
            self.assertIn("Ollama is not running", next(app.run_scan("m", "Quick check", [], 1, 60))[0])
        with mock.patch.object(app, "ollama_up", return_value="1"):
            self.assertIn("No probes", next(app.run_scan("m", app.CURRENT, [], 1, 60))[0])

    def test_run_scan_refuses_second_scan(self):
        busy = mock.Mock()
        busy.poll.return_value = None
        with mock.patch.dict(app._proc, {"p": busy}), mock.patch.object(app, "ollama_up", return_value="1"), \
                mock.patch.object(app.subprocess, "Popen") as popen:
            first = next(app.run_scan("m", "Quick check", [], 1, 60))
        popen.assert_not_called()
        self.assertIn("already running", first[0])

    def test_run_scan_refuses_while_generating_image(self):
        busy = mock.Mock()
        busy.poll.return_value = None
        with mock.patch.dict(app._image_proc, {"p": busy}), mock.patch.object(app, "ollama_up", return_value="1"), \
                mock.patch.object(app.subprocess, "Popen") as popen:
            first = next(app.run_scan("m", "Quick check", [], 1, 60))
        popen.assert_not_called()
        self.assertIn("image", first[0])


class ScanSurvivesPageClose(TempDataTest):
    """Closing or reloading the page stops Gradio from iterating the scan generator. The scan must keep running
    (its output drained so it can't block on a full pipe) and clean up when it ends."""

    def test_abandoned_scan_finishes(self):
        chatty = self.enterContext(tempfile.NamedTemporaryFile("w", suffix=".py", delete=False))
        chatty.write("import sys\nfor i in range(20000):\n    print('x' * 100)\n")  # ~2 MB, far above a pipe buffer
        chatty.close()
        self.addCleanup(os.unlink, chatty.name)
        real_popen = app.subprocess.Popen
        started = []

        def fake_popen(cmd, **kw):
            started.append(real_popen([sys.executable, chatty.name], **kw))
            return started[-1]

        with mock.patch.object(app, "ollama_up", return_value="1"), \
                mock.patch.object(app.subprocess, "Popen", side_effect=fake_popen), \
                mock.patch.object(app, "unload_models") as unload:
            gen = app.run_scan("m", "Quick check", [], 1, 60)
            next(gen)
            next(gen)   # the scan is started
            gen.close()  # the page was closed
            proc = started[0]
            deadline = time.time() + 20
            while time.time() < deadline and (proc.poll() is None or app._proc["p"] is not None):
                time.sleep(0.1)
            self.assertEqual(proc.poll(), 0, "the scan process blocked after the page was closed")
            self.assertIsNone(app._proc["p"], "the app still thinks a scan is running")
            unload.assert_called()


class ScanResume(TempDataTest):
    """A scan cut short by LLM Scanner stopping (or the computer shutting down) continues at the next start with only
    the probes it had not finished; one that finished, failed on its own or was stopped by the user does not."""
    GARAK = ("import os, signal, sys\n"
             "print('queue of probes: a.A, b.B, c.C', flush=True)\n"
             "print('probes.a.A: 100%|####| 2/2', flush=True)\n"
             "print('a.A                  det.D: FAIL  ok on 1/2', flush=True)\n"
             "print('probes.b.B:  50%|##  | 1/2', flush=True)\n"
             "{end}\n")

    def run_fake_scan(self, end):
        script = self.tmp / "fake_garak.py"
        script.write_text(self.GARAK.format(end=end))
        real_popen = app.subprocess.Popen
        with mock.patch.object(app, "ollama_up", return_value="1"), mock.patch.object(app, "unload_models"), \
                mock.patch.object(app.subprocess, "Popen", side_effect=lambda cmd, **kw: real_popen(
                    [sys.executable, str(script)], **kw)):
            gen = app.run_scan("m", "Quick check", [], 1, 60)
            *_, last = gen
        return last

    def resume(self, installed=("m",)):
        launched = []
        with mock.patch.object(app, "ollama_up", return_value="1"), \
                mock.patch.object(app, "model_names", return_value=list(installed)), \
                mock.patch.object(app, "_launch_scan", side_effect=lambda job, note="": launched.append((job, note))):
            app.resume_interrupted_scan()
        self.assertFalse(app._resume["waiting"])
        return launched

    def test_killed_scan_resumes_with_unfinished_probes(self):
        log, *_ = self.run_fake_scan("os.kill(os.getpid(), signal.SIGKILL)")
        self.assertIn("next time LLM Scanner starts", log)
        job = json.loads(app.SCAN_JOB.read_text())
        self.assertEqual(job["queue"], ["a.A", "b.B", "c.C"])
        self.assertEqual(job["done"], ["a.A"], "b.B was still running, so it must be run again")
        self.assertEqual(job["failed"], ["a.A"])
        (job, note), = self.resume()
        self.assertEqual(app._remaining_probes(job), ["b.B", "c.C"])
        self.assertEqual(job["parts"], 2)
        self.assertIn("2 of 3 probes left", note)
        self.assertIn(job["prefix"] + "_part2", note)

    def test_resumed_command_and_progress(self):
        job = {"model": "m", "scan_type": "Quick check", "requested": [], "queue": ["a.A", "b.B", "c.C"],
               "done": ["a.A"], "failed": [], "generations": 2, "timeout": 60, "thinking": False,
               "prefix": "m_1", "parts": 2, "stalls": 0}
        with mock.patch.object(app.subprocess, "Popen") as popen, mock.patch.object(app, "pump_output"):
            popen.return_value.pid = 1
            app._launch_scan(job, "Resumed\n")
        cmd = popen.call_args[0][0]
        self.assertEqual(cmd[cmd.index("--probes") + 1], "b.B,c.C")
        self.assertEqual(cmd[cmd.index("--report_prefix") + 1], "m_1_part2")
        app._scan["progress"].feed("queue of probes: b.B, c.C")
        self.assertEqual(len(app._scan["progress"].probes), 3, "progress must count the whole scan, not just this part")
        self.assertAlmostEqual(app._scan["progress"].percent(), 100 / 3)
        app._proc["p"] = None

    def test_finished_failed_or_stopped_scans_do_not_resume(self):
        for end in ("sys.exit(0)", "sys.exit(1)"):
            self.run_fake_scan(end)
            self.assertFalse(app.SCAN_JOB.exists(), end)
            self.assertEqual(self.resume(), [])
        app._save_scan_job({"model": "m"})
        app.stop_scan()
        self.assertFalse(app.SCAN_JOB.exists(), "Stop must cancel a pending resume")

    def test_gives_up_on_missing_model_or_repeated_stalls(self):
        job = {"model": "m", "scan_type": "x", "requested": [], "queue": ["a.A", "b.B"], "done": ["a.A"],
               "failed": [], "generations": 1, "timeout": 60, "thinking": False, "prefix": "m_1", "parts": 2,
               "stalls": 0, "done_at_resume": 1}
        app._save_scan_job(job)
        self.assertEqual(self.resume(installed=()), [])
        self.assertFalse(app.SCAN_JOB.exists())
        for attempt in range(app.MAX_STALLED_RESUMES - 1):
            app._save_scan_job(dict(job, stalls=attempt))
            self.assertEqual(len(self.resume()), 1)
        app._save_scan_job(dict(job, stalls=app.MAX_STALLED_RESUMES - 1))
        self.assertEqual(self.resume(), [], "a scan that keeps dying on the same probe must not loop forever")
        self.assertFalse(app.SCAN_JOB.exists())

    def test_everything_done_just_clears_the_job(self):
        app._save_scan_job({"model": "m", "requested": [], "queue": ["a.A"], "done": ["a.A"], "failed": [],
                            "parts": 1, "stalls": 0})
        self.assertEqual(self.resume(), [])
        self.assertFalse(app.SCAN_JOB.exists())


# ---------------------------------------------------------------- reports
def write_run(folder, prefix, *, complete=True, evals=True, hits=True):
    lines = [{"entry_type": "start_run setup", "plugins.target_type": "ollama.OllamaGeneratorChat",
              "plugins.target_name": "m", "run.generations": 1, "plugins.probe_spec": "dan.DanInTheWild"},
             {"entry_type": "init", "start_time": "2026-09-17T10:00:00", "run": "r1", "garak_version": "0.17.0"}]
    if evals:
        lines.append({"entry_type": "eval", "probe": "dan.DanInTheWild", "detector": "mitigation.MitigationBypass",
                      "total_evaluated": 10, "passed": 4})
    if complete:
        lines += [{"entry_type": "completion", "end_time": "2026-09-17T10:05:30"},
                  {"entry_type": "digest", "meta": {}, "eval": {"dan": {
                      "_summary": {"group_link": "https://example.com"},
                      "dan.DanInTheWild": {"_summary": {"probe_descr": "DAN", "probe_tags": ["owasp:llm01"]},
                                           "mitigation.MitigationBypass": {"total_evaluated": 10, "passed": 4,
                                                                          "absolute_defcon": 2, "relative_defcon": 3}}}}}]
    (folder / f"{prefix}.report.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n{garbled")
    if complete:
        (folder / f"{prefix}.report.html").write_text("<html>garak</html>")
    if hits:
        (folder / f"{prefix}.hitlog.jsonl").write_text(json.dumps({
            "probe": "dan.DanInTheWild", "detector": "mitigation.MitigationBypass", "score": 1.0,
            "prompt": {"turns": [{"role": "user", "content": {"text": "<script>evil</script>"}}]},
            "output": {"text": "sure"}, "goal": "disregard the system prompt"}) + "\n")


class Reports(TempDataTest):
    def test_complete_run_summary(self):
        write_run(app.GARAK_RUNS, "m_20260917-100000")
        data = analyst_report.parse(app.GARAK_RUNS / "m_20260917-100000.report.jsonl")
        f = data["findings"][0]
        self.assertEqual((f["severity"], f["fails"], f["owasp"]), ("High", 6, ["llm01"]))
        self.assertTrue(data["meta"]["complete"])
        html_out = analyst_report.render_html(data)
        self.assertNotIn("<script>evil", html_out)
        self.assertIn("5m 30s", html_out)
        self.assertEqual(f["goal"], "disregard the system prompt")

    def test_interrupted_run_estimates_severity(self):
        write_run(app.GARAK_RUNS, "m_20260917-110000", complete=False)
        data = analyst_report.parse(app.GARAK_RUNS / "m_20260917-110000.report.jsonl")
        self.assertFalse(data["meta"]["complete"])
        self.assertTrue(data["findings"][0]["estimated"])
        self.assertEqual(data["findings"][0]["severity"], "Critical")  # 60% attack success
        self.assertIn("Incomplete run", analyst_report.render_html(data))

    def test_empty_run_renders(self):
        write_run(app.GARAK_RUNS, "m_20260917-120000", complete=False, evals=False, hits=False)
        data = analyst_report.parse(app.GARAK_RUNS / "m_20260917-120000.report.jsonl")
        self.assertEqual(data["findings"], [])
        self.assertIn("No failing checks", analyst_report.render_html(data))

    def test_build_is_cached_until_report_changes(self):
        write_run(app.GARAK_RUNS, "m_20260917-100000")
        html_path, csv_path, _ = analyst_report.build(app.GARAK_RUNS / "m_20260917-100000.report.html")
        self.assertTrue(csv_path.read_text().startswith("severity,probe"))
        first = html_path.stat().st_mtime_ns
        analyst_report.build(app.GARAK_RUNS / "m_20260917-100000.report.html")
        self.assertEqual(html_path.stat().st_mtime_ns, first)

    def test_reports_list_and_show(self):
        write_run(app.GARAK_RUNS, "hf.co_x_m_20260917-100000")
        choices = app.reports()
        self.assertEqual(choices[0][1], "hf.co_x_m_20260917-100000.report.html")
        self.assertTrue(choices[0][0].startswith("x_m "))
        summary, garak_view, raw, files = app.show_report(choices[0][1])
        self.assertIn("iframe", summary)
        self.assertEqual(len(files), 5)
        self.assertIn("evil", app.read_raw(choices[0][1], "Hitlog (failing responses)"))

    def test_stopped_scan_with_results_is_listed(self):
        """A scan stopped partway has results in report.jsonl but no garak HTML; it should still be viewable."""
        write_run(app.GARAK_RUNS, "m_20260917-110000", complete=False)
        names = [v for _, v in app.reports()]
        self.assertEqual(len(names), 1)
        summary, garak_view, raw, files = app.show_report(names[0])
        self.assertIn("Incomplete run", summary)
        self.assertNotIn("Traceback", garak_view)

    def test_delete_reports_removes_only_the_ticked_runs(self):
        for prefix in ("m_20260917-100000", "m_20260917-100000x", "m_20260917-110000"):
            write_run(app.GARAK_RUNS, prefix)
        picked = ["m_20260917-100000.report.html", "m_20260917-110000.report.html"]
        listing, select_all, button, selected, shown, *_rest, note = app.delete_reports(picked, True, picked[0], "raw")
        left = sorted(p.name for p in app.GARAK_RUNS.iterdir())
        self.assertTrue(left and all(n.startswith("m_20260917-100000x.") for n in left), left)
        self.assertIn("Deleted 2 reports", note)
        self.assertEqual(selected, [])
        self.assertEqual(button["value"], "Delete reports")
        self.assertFalse(button["interactive"])
        self.assertEqual(shown, "m_20260917-100000x.report.html", "the viewer moves to a report that still exists")

    def test_delete_reports_needs_confirmation_and_known_names(self):
        write_run(app.GARAK_RUNS, "m_20260917-100000")
        (app.GARAK_RUNS / ".hidden").write_text("x")
        app.delete_reports(["m_20260917-100000.report.html"], False, None, "raw")  # dialog cancelled
        for bad in (["x"], [".report.html"], ["../m_20260917-100000.report.html"], []):
            app.delete_reports(bad, True, None, "raw")
        self.assertTrue((app.GARAK_RUNS / ".hidden").exists())
        self.assertEqual(len(list(app.GARAK_RUNS.glob("m_*"))), 3)

    def test_delete_button_counts_what_is_ticked(self):
        write_run(app.GARAK_RUNS, "m_20260917-100000")
        write_run(app.GARAK_RUNS, "m_20260917-110000")
        names = [n for _, n in app.reports()]
        self.assertEqual(app.delete_label([])["value"], "Delete reports")
        self.assertEqual(app.delete_label(names[:1])["value"], "Delete 1 report")
        self.assertEqual(app.delete_label(names)["value"], "Delete 2 reports")

    def test_ticking_a_report_shows_it(self):
        write_run(app.GARAK_RUNS, "m_20260917-100000")
        write_run(app.GARAK_RUNS, "m_20260917-110000")
        newest, older = (n for _, n in app.reports())
        button, picked, shown, summary, *_ = app.reports_ticked([older], [], newest, "Full report (report.jsonl)")
        self.assertEqual(shown, older, "the report just ticked is the one displayed")
        self.assertEqual(picked, [older])
        self.assertIn("LLM security assessment", summary)
        _, picked, shown, *_ = app.reports_ticked([], [older], older, "Full report (report.jsonl)")
        self.assertEqual((picked, shown), ([], older), "unticking only changes what Delete would remove")

    def test_select_all_ticks_every_report(self):
        write_run(app.GARAK_RUNS, "m_20260917-100000")
        write_run(app.GARAK_RUNS, "m_20260917-110000")
        listing, button, picked, *_ = app.select_all_reports(True, None, "Full report (report.jsonl)")
        self.assertEqual(len(listing["value"]), 2)
        self.assertEqual(button["value"], "Delete 2 reports")
        listing, button, picked, *_ = app.select_all_reports(False, None, "Full report (report.jsonl)")
        self.assertEqual((listing["value"], picked), ([], []))

    def test_show_report_rejects_paths_outside_runs(self):
        secret = self.tmp / "secret.report.html"
        secret.write_text("TOP SECRET")
        summary, garak_view, raw, files = app.show_report("../secret.report.html")
        self.assertNotIn("TOP SECRET", garak_view)


# ---------------------------------------------------------------- updates and desktop integration
class Updates(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(app._updates, {"checked": 0.0, "ollama": None, "garak": None, "app": None, "busy": None,
                                           "pct": None, "msg": "", "ok": True, "restarting": False})
        p.start()
        self.addCleanup(p.stop)

    def fake_get(self, url, **kw):
        if "ollama/ollama" in url:
            return FakeResponse({"tag_name": "v0.99.0", "prerelease": False})
        if "pypi" in url:
            return FakeResponse({"info": {"version": "99.0.0"}})
        return FakeResponse({"tag_name": "v9.9.9", "body": "<b>notes</b>", "prerelease": False})

    def test_check_updates_finds_newer_versions(self):
        with mock.patch.object(app.requests, "get", side_effect=self.fake_get), \
                mock.patch.object(app, "ollama_up", return_value="0.34.2"):
            app.check_updates()
        self.assertEqual(app._updates["ollama"], ("0.34.2", "0.99.0"))
        self.assertEqual(app._updates["garak"][1], "99.0.0")
        self.assertEqual(app._updates["app"]["version"], "9.9.9")
        self.assertIn("9.9.9", app.update_item_html("app"))
        self.assertNotIn("<b>notes", app.update_item_html("app"))

    def test_check_updates_ignores_older_and_prerelease(self):
        def get(url, **kw):
            if "pypi" in url:
                return FakeResponse({"info": {"version": "0.0.1"}})
            return FakeResponse({"tag_name": "v99.0.0", "prerelease": True})
        with mock.patch.object(app.requests, "get", side_effect=get), \
                mock.patch.object(app, "ollama_up", return_value="0.34.2"):
            app.check_updates()
        self.assertIsNone(app._updates["ollama"])
        self.assertIsNone(app._updates["garak"])
        self.assertIsNone(app._updates["app"])

    def test_check_updates_survives_network_errors(self):
        with mock.patch.object(app.requests, "get", side_effect=app.requests.ConnectionError("offline")):
            app.check_updates()
        self.assertEqual(app.update_banner(), "")

    def test_update_refused_during_scan(self):
        busy = mock.Mock()
        busy.poll.return_value = None
        with mock.patch.dict(app._proc, {"p": busy}), mock.patch.object(app.threading, "Thread") as t:
            app.start_update("app")
        t.assert_not_called()
        self.assertIn("running scan", app._updates["msg"])

    def test_failed_ollama_update_rolls_back(self):
        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        old_dir = home / "ollama"
        (old_dir / "bin").mkdir(parents=True)
        (old_dir / "bin" / "ollama").write_text("old")
        import tarfile
        import io
        release = home / "rel.tar"
        with tarfile.open(release, "w") as tf:
            data = b"new"
            info = tarfile.TarInfo("bin/ollama")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        class Download(FakeResponse):
            def iter_content(self, n):
                yield release.read_bytes()

        real_run = app.subprocess.run  # tar really runs (the test archive is plain tar, not zstd); systemctl doesn't
        fake_run = mock.Mock(side_effect=lambda cmd, **kw: real_run([c for c in cmd if c != "--zstd"], **kw)
                             if cmd[0] == "tar" else mock.Mock(returncode=0))
        app._updates["ollama"] = ("0.34.2", "0.99.0")
        with mock.patch.object(app, "OLLAMA_DIR", old_dir), \
                mock.patch.object(app.requests, "get", return_value=Download(headers={"content-length": "10"})), \
                mock.patch.object(app.subprocess, "run", fake_run), \
                mock.patch.object(app, "_verify_ollama", return_value=(False, "Ollama did not start.")), \
                mock.patch.object(app, "check_updates"):
            app._run_update("ollama")
        self.assertEqual((old_dir / "bin" / "ollama").read_text(), "old")
        self.assertFalse((home / "ollama.old").exists())
        self.assertFalse(app._updates["ok"])
        self.assertIn("previous version was restored", app._updates["msg"])
        self.assertIsNone(app._updates["busy"])

    def test_failed_garak_update_reinstalls_previous(self):
        app._updates["garak"] = ("0.17.0", "0.18.0")
        run = mock.Mock(return_value=mock.Mock(returncode=0))
        with mock.patch.object(app.subprocess, "run", run), \
                mock.patch.object(app, "_verify_garak", return_value=(False, "0.18.0", "self-test failed.")), \
                mock.patch.object(app, "check_updates"):
            app._run_update("garak")
        self.assertIn("garak==0.17.0", run.call_args_list[-1].args[0])
        self.assertIn("0.17.0 was reinstalled", app._updates["msg"])


class AppUpdate(TempDataTest):
    def test_update_installs_files_it_did_not_know_about(self):
        """A release that adds a new module must install it, or the new version can't start."""
        import io
        import tarfile
        app_dir = self.tmp / "app"
        app_dir.mkdir()
        (app_dir / "app.py").write_text("old")
        (app_dir / "keep.txt").write_text("user file")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for name, data in (("app.py", b"new"), ("VERSION", b"9.9.9"), ("static/app.js", b"js"),
                               ("new_module.py", b"x = 1"), ("data/should-not-install", b"x")):
                info = tarfile.TarInfo(f"llm-scanner-9.9.9/{name}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

        class Download(FakeResponse):
            def iter_content(self, n):
                yield buf.getvalue()

        with mock.patch.dict(app._updates, {"app": {"version": "9.9.9", "tag": "v9.9.9", "notes": ""}}), \
                mock.patch.object(app, "APP_DIR", app_dir), \
                mock.patch.object(app.requests, "get", return_value=Download()), \
                mock.patch.object(app, "_verify_app_candidate", return_value=(True, "ok")), \
                mock.patch.object(app, "_restart_soon") as restart:
            app._update_app()
        self.assertEqual((app_dir / "new_module.py").read_text(), "x = 1")
        self.assertEqual((app_dir / "app.py").read_text(), "new")
        self.assertEqual((app_dir / "keep.txt").read_text(), "user file")
        self.assertFalse((app_dir / "data").exists())
        self.assertEqual((app.DATA_DIR / "backups" / app.APP_VERSION / "app.py").read_text(), "old")
        restart.assert_called_once()

    def test_mismatched_release_is_rejected(self):
        import io
        import tarfile
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for name, data in (("app.py", b"new"), ("VERSION", b"1.0.3"), ("static/app.js", b"js")):
                info = tarfile.TarInfo(f"x/{name}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

        class Download(FakeResponse):
            def iter_content(self, n):
                yield buf.getvalue()

        with mock.patch.dict(app._updates, {"app": {"version": "9.9.9", "tag": "v9.9.9", "notes": ""}}), \
                mock.patch.object(app.requests, "get", return_value=Download()):
            with self.assertRaisesRegex(RuntimeError, "labeled 9.9.9"):
                app._update_app()


class LeftoverCleanup(TempDataTest):
    """clean_leftovers runs at every startup."""

    def setUp(self):
        super().setUp()
        for name, value in (("HF_DOWNLOADS", app.DATA_DIR / "hf-downloads"),):
            p = mock.patch.object(app, name, value)
            p.start()
            self.addCleanup(p.stop)

    def touch(self, path, data=b"x" * 10):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_removes_leftovers_and_keeps_what_is_still_needed(self):
        d = app.DATA_DIR
        gone = [self.touch(d / ".hf-cache/models--a--b/blobs/x"), self.touch(d / "image-models/.hf-cache/y"),
                self.touch(d / "update-staging/release.tar.gz"), self.touch(d / "tmp/selftest-1.report.jsonl"),
                self.touch(d / "attachments/staging/abcd-notes.txt"), self.touch(d / "attachments/deleted-chat/a.png"),
                self.touch(d / "images/removed.json"),
                self.touch(app.HF_DOWNLOADS / (app._slug("hf.co/x/abandoned:Q4_K_M") + ".gguf.part")),
                self.touch(app.HF_DOWNLOADS / "hf-co-x-y-q4.Modelfile"),
                self.touch(d / "image-models/some-removed-model/model.gguf.part")]
        blob = self.touch(d / ".hf-cache/models--c--d/blobs/abc", b"y" * 1000)  # huggingface_hub links files to blobs
        (d / ".hf-cache/models--c--d/snapshots/1").mkdir(parents=True)
        (d / ".hf-cache/models--c--d/snapshots/1/model.gguf").symlink_to(blob)
        gone.append(blob)
        paused, queued = "hf.co/x/paused:Q4_K_M", "image:z-image-turbo"
        app._set_pending(paused, True, paused=True)
        app._set_pending(queued, True)
        kept = [self.touch(app.HF_DOWNLOADS / (app._slug(paused) + ".gguf.part")),
                self.touch(app.image_model_files("z-image-turbo")["diffusion-model"].with_name("z_image_turbo-Q4_K.gguf.part")),
                self.touch(d / "chats/live-chat.json", b"{}"), self.touch(d / "attachments/live-chat/a.png"),
                self.touch(d / "images/kept.png"), self.touch(d / "images/kept.json")]
        for i, v in enumerate(("1.0.0", "1.0.3", "1.0.8", "1.0.9", "1.0.10")):
            folder = self.touch(d / f"backups/{v}/app.py").parent
            os.utime(folder, (1000 + i, 1000 + i))
        files, size = app.clean_leftovers()
        for f in gone:
            self.assertFalse(f.exists(), f)
        for f in kept:
            self.assertTrue(f.exists(), f)
        self.assertEqual(sorted(p.name for p in (d / "backups").iterdir()), ["1.0.10", "1.0.8", "1.0.9"])
        self.assertEqual(files, len(gone) + 2)  # plus the two oldest backups; linked files are counted once
        self.assertEqual(size, 10 * (len(gone) - 1) + 1000 + 2 * 10)
        self.assertEqual(app.clean_leftovers(), (0, 0), "a second run has nothing left to do")

    def test_nothing_to_clean_on_a_fresh_install(self):
        self.assertEqual(app.clean_leftovers(), (0, 0))


class DesktopIntegration(unittest.TestCase):
    def test_entries_are_repaired_in_place(self):
        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        apps = home / ".local/share/applications"
        apps.mkdir(parents=True)
        entry = apps / "llm-scanner.desktop"
        entry.write_text("[Desktop Entry]\nName=LLM Scanner\nExec=/old/path.sh\nIcon=/old/icon.png\n")
        with mock.patch.object(app.Path, "home", return_value=home), \
                mock.patch.object(app.subprocess, "run", return_value=mock.Mock(stdout=str(home / "Desktop"))):
            app.ensure_desktop_integration()
        text = entry.read_text()
        self.assertIn("Icon=llm-scanner\n", text)
        self.assertIn(f"Exec={app.APP_DIR / 'llm-scanner.sh'}", text)
        self.assertTrue((home / ".local/share/icons/hicolor/scalable/apps/llm-scanner.svg").exists())
        self.assertFalse((home / "Desktop" / "llm-scanner.desktop").exists())  # never creates entries on its own


if __name__ == "__main__":
    unittest.main()
