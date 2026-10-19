"""Scenario tests for the Models tab: the Hugging Face token, searching Hugging Face and the Ollama library, the
background download queue, and the table of installed chat models.

Run alone:  .venv/bin/python -m unittest discover -s tests -p test_scenarios_models.py
Nothing here touches real data, Ollama, Hugging Face or the network: data folders point at a temporary directory
(TempDataTest), every HTTP call is replaced with a fake, time.sleep is a no-op and threading.Thread is mocked wherever
the code under test would start a download worker."""

import html
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_app import FakeResponse, TempDataTest, app  # noqa: E402

NOOP = {"__type__": "update"}
LINE_401 = "401 Client Error: Unauthorized for url: https://huggingface.co/x/resolve/main/config.json"


class BadJson(FakeResponse):
    """A response whose body is not JSON."""

    def json(self):
        raise ValueError("not json")


def dl(state, **extra):
    """One entry of app._downloads."""
    return {"state": state, "msg": state.capitalize(), "done": 0, "total": 0, "cancel": False, **extra}


def ollama_li(name, desc="", caps=(), pulls=""):
    """One result of ollama.com/search, shaped like the real page."""
    caps_html = "".join(f'<span class="bg-indigo-50 text-xs">{c}</span>' for c in caps)
    pulls_html = f'<span>{pulls}</span><span class="text-xs">&nbsp;Pulls</span>' if pulls else ""
    return (f'<li x-test-model><a href="/library/{name}"><p class="max-w-md break-words">{desc}</p>'
            f'{caps_html}{pulls_html}</a></li>')


def ollama_tag_row(tag, size):
    """One row of ollama.com/library/<name>/tags."""
    return (f'<a href="/library/{tag}" class="md:hidden group"><span class="text-neutral-500 font-mono">'
            f'abc123def</span> \u2022 {size}</a>')


class ModelsTest(TempDataTest):
    """Temporary data folder, no Hugging Face token from this machine, empty module caches, no real sleeping."""

    def setUp(self):
        super().setUp()
        env = {n: "" for n in app.HF_TOKEN_ENV}
        env["HF_HOME"] = str(self.tmp / "hfhome")  # no `huggingface-cli login` file
        self.enterContext(mock.patch.dict(os.environ, env))
        for cache in (app._hf_meta, app._hf_size_cache, app._ollama_meta, app._load_problems, app._hf_api_cache):
            self.enterContext(mock.patch.dict(cache, clear=True))
        self.enterContext(mock.patch.object(app.time, "sleep", lambda s: None))


class DownloadsTest(ModelsTest):
    """Fresh download queue; threading.Thread is a mock so no worker ever starts."""

    def setUp(self):
        super().setUp()
        for name, value in (("_downloads", {}), ("_priority", {"ref": None}), ("_dl_finished", {"count": 0}),
                            ("HF_DOWNLOADS", self.tmp / "data" / "hf-downloads")):
            self.enterContext(mock.patch.object(app, name, value))
        self.thread = self.enterContext(mock.patch.object(app.threading, "Thread"))


# ================================================================ Hugging Face token: settings file
class SettingsFile(ModelsTest):
    TOKEN = "hf_settingsfiletoken"

    def test_settings_round_trip_keeps_unicode_values(self):
        values = {"hf_token": self.TOKEN, "note": "h\u00e9llo \u2713 \u65e5\u672c"}
        app.write_settings(values)
        self.assertEqual(app.read_settings(), values)

    def test_write_settings_creates_a_missing_folder(self):
        target = self.tmp / "fresh" / "nested" / "settings.json"
        with mock.patch.object(app, "SETTINGS_FILE", target):
            app.write_settings({"a": 1})
            self.assertTrue(target.exists())
            self.assertEqual(app.read_settings(), {"a": 1})

    def test_write_settings_leaves_no_temporary_file_behind(self):
        app.write_settings({"hf_token": self.TOKEN})
        app.write_settings({"hf_token": self.TOKEN + "2"})
        names = sorted(p.name for p in app.SETTINGS_FILE.parent.iterdir() if p.name.startswith("settings"))
        self.assertEqual(names, ["settings.json"])

    def test_settings_path_that_is_a_directory_reads_as_empty(self):
        app.SETTINGS_FILE.mkdir()
        self.assertEqual(app.read_settings(), {})
        self.assertEqual(app.hf_token(), "")

    def test_blank_saved_token_falls_back_to_environment(self):
        app.write_settings({"hf_token": "   "})
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_env_value"}):
            self.assertEqual(app.hf_token(), "hf_env_value")

    def test_environment_variables_are_tried_in_documented_order(self):
        with mock.patch.dict(os.environ, {"HUGGING_FACE_HUB_TOKEN": "hf_second", "HUGGINGFACE_TOKEN": "hf_third"}):
            self.assertEqual(app.hf_token(), "hf_second")
            with mock.patch.dict(os.environ, {"HF_TOKEN": "  hf_first  "}):
                self.assertEqual(app.hf_token(), "hf_first")

    def test_empty_url_never_gets_the_token(self):
        app.write_settings({"hf_token": self.TOKEN})
        self.assertEqual(app.hf_headers(""), {})
        self.assertTrue(app.hf_headers(None)["Authorization"].endswith(self.TOKEN))


# ================================================================ Hugging Face API calls
class HfApiGet(unittest.TestCase):
    URL = "https://hf.example/api/models"

    def setUp(self):
        self.enterContext(mock.patch.dict(app._hf_api_cache, clear=True))
        self.sleep = self.enterContext(mock.patch.object(app.time, "sleep"))

    def test_gives_up_after_four_429s_and_returns_the_last_answer(self):
        answers = [FakeResponse(status=429) for _ in range(4)]
        with mock.patch.object(app.requests, "get", side_effect=answers) as get:
            res = app.hf_api_get(self.URL)
        self.assertIs(res, answers[-1])
        self.assertEqual(get.call_count, 4)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [2, 6, 18])

    def test_retry_after_is_capped_at_thirty_seconds(self):
        answers = [FakeResponse(status=429, headers={"Retry-After": "120"}), FakeResponse([1])]
        with mock.patch.object(app.requests, "get", side_effect=answers):
            self.assertEqual(app.hf_api_get(self.URL).json(), [1])
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [30.0])

    def test_unparseable_retry_after_uses_the_backoff(self):
        answers = [FakeResponse(status=429, headers={"Retry-After": "soon"}), FakeResponse([1])]
        with mock.patch.object(app.requests, "get", side_effect=answers):
            app.hf_api_get(self.URL)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [2])

    def test_server_error_is_not_retried(self):
        with mock.patch.object(app.requests, "get", side_effect=[FakeResponse(status=500)]) as get:
            res = app.hf_api_get(self.URL)
        self.assertEqual(res.status_code, 500)
        self.assertEqual(get.call_count, 1)
        self.sleep.assert_not_called()

    def test_failed_answers_are_not_cached(self):
        with mock.patch.object(app, "HF_API_CACHE_SECS", 300), \
                mock.patch.object(app.requests, "get", side_effect=[FakeResponse(status=404), FakeResponse([1])]) as get:
            self.assertEqual(app.hf_api_get(self.URL).status_code, 404)
            self.assertEqual(app.hf_api_get(self.URL).json(), [1])
        self.assertEqual(get.call_count, 2)

    def test_cache_key_includes_the_headers(self):
        with mock.patch.object(app, "HF_API_CACHE_SECS", 300), \
                mock.patch.object(app.requests, "get", return_value=FakeResponse([1])) as get:
            app.hf_api_get(self.URL, headers={"Authorization": "a"})
            app.hf_api_get(self.URL, headers={"Authorization": "b"})
            app.hf_api_get(self.URL, headers={"Authorization": "a"})
        self.assertEqual(get.call_count, 2, "a different token may see different repositories")


class HfWhoami(unittest.TestCase):
    TOKEN = "hf_checked_token_1"

    def setUp(self):
        self.enterContext(mock.patch.dict(app._hf_api_cache, clear=True))
        self.enterContext(mock.patch.object(app.time, "sleep", lambda s: None))

    def whoami(self, **kw):
        with mock.patch.object(app.requests, "get", **kw) as get:
            return app.hf_whoami(self.TOKEN), get

    def test_403_is_a_rejection(self):
        (who, error), _ = self.whoami(return_value=FakeResponse({"error": "forbidden"}, status=403))
        self.assertEqual(who, "")
        self.assertEqual(error, "Hugging Face rejected this token. Check it was copied in full and has read access.")

    def test_server_error_means_unknown(self):
        (who, error), _ = self.whoami(return_value=FakeResponse(status=500))
        self.assertEqual((who, error), ("", ""))

    def test_unparseable_answer_means_unknown(self):
        (who, error), _ = self.whoami(return_value=BadJson(status=200))
        self.assertEqual((who, error), ("", ""))

    def test_network_error_means_unknown(self):
        (who, error), _ = self.whoami(side_effect=app.requests.ConnectionError("down"))
        self.assertEqual((who, error), ("", ""))

    def test_sends_the_token_being_checked(self):
        (who, error), get = self.whoami(return_value=FakeResponse({"name": "bob"}))
        self.assertEqual((who, error), ("bob", ""))
        self.assertEqual(get.call_args.args[0], f"{app.HF_API}/whoami-v2")
        self.assertTrue(get.call_args.kwargs["headers"]["Authorization"].endswith(self.TOKEN))
        self.assertEqual(get.call_args.kwargs["timeout"], 15)


# ================================================================ token box on the Add a model card
class TokenPanel(ModelsTest):
    TOKEN = "hf_tokenpanelvalue123"

    def test_status_names_the_environment_variable_in_use(self):
        with mock.patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "hf_x"}):
            self.assertIn("Using HUGGINGFACE_TOKEN from this machine", app.hf_token_status())

    def test_status_ignores_whitespace_only_environment_token(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "   "}):
            self.assertIn("No token set", app.hf_token_status())

    def test_saving_nothing_is_refused_without_asking_hugging_face(self):
        with mock.patch.object(app.requests, "get") as get:
            box, note = app.save_hf_token(None)
        get.assert_not_called()
        self.assertEqual(box, NOOP)
        self.assertIn("Enter a token first", note)
        self.assertIn("error", note)
        self.assertFalse(app.SETTINGS_FILE.exists())

    def test_saving_clears_cached_repository_details(self):
        app._hf_meta["o/r"] = {"gated": False}
        with mock.patch.object(app.requests, "get", return_value=FakeResponse({"name": "kadn"})):
            app.save_hf_token(self.TOKEN)
        self.assertEqual(app._hf_meta, {})

    def test_rejected_token_keeps_previous_token_and_cache(self):
        app.write_settings({"hf_token": "hf_previous_good_token"})
        app._hf_meta["o/r"] = {"gated": True}
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(status=401)):
            box, note = app.save_hf_token("hf_new_bad_token_value")
        self.assertEqual(app.hf_token(), "hf_previous_good_token")
        self.assertIn("o/r", app._hf_meta)
        self.assertEqual(box, NOOP, "the box keeps the text so it can be corrected")
        self.assertIn("rejected", note)

    def test_removing_when_nothing_is_saved_writes_nothing(self):
        app._hf_meta["o/r"] = {"gated": True}
        box, note = app.remove_hf_token()
        self.assertFalse(app.SETTINGS_FILE.exists())
        self.assertIn("o/r", app._hf_meta)
        self.assertEqual(box["value"], "")
        self.assertIn("No token set", note)

    def test_gear_opens_a_closed_panel_with_an_empty_box(self):
        is_open, panel, box, status = app.toggle_hf_panel(False)
        self.assertTrue(is_open)
        self.assertTrue(panel["visible"])
        self.assertEqual(box["value"], "")
        self.assertIn("No token set", status)

    def test_gear_closes_an_open_panel_and_shows_saved_token(self):
        app.write_settings({"hf_token": self.TOKEN})
        is_open, panel, box, status = app.toggle_hf_panel(True)
        self.assertFalse(is_open)
        self.assertFalse(panel["visible"])
        self.assertEqual(box["value"], "")
        self.assertIn(app.mask_token(self.TOKEN), status)


# ================================================================ where garak's token comes from, and refusals
class TokenSource(ModelsTest):
    def test_saved_token_wins_over_environment(self):
        app.write_settings({"hf_token": "hf_saved_value"})
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_env"}):
            self.assertEqual(app.hf_token_source(), "the token saved in LLM Scanner")

    def test_names_the_environment_variable(self):
        with mock.patch.dict(os.environ, {"HUGGING_FACE_HUB_TOKEN": "hf_x"}):
            self.assertEqual(app.hf_token_source(),
                             "the HUGGING_FACE_HUB_TOKEN environment variable on this computer")

    def test_login_file_is_a_source_for_garak_but_not_for_the_app(self):
        login = self.tmp / "hfhome" / "token"
        login.parent.mkdir()
        login.write_text("hf_from_login", encoding="utf-8")
        self.assertEqual(app.hf_token_source(), f"the token `huggingface-cli login` saved in {login}")
        self.assertEqual(app.hf_failure_kind(LINE_401), "token")
        self.assertEqual(app.hf_token(), "", "the app itself only uses its own token or the environment")

    def test_no_source_makes_a_401_an_access_refusal(self):
        self.assertEqual(app.hf_token_source(), "")
        self.assertEqual(app.hf_failure_kind(LINE_401), "access")

    def test_token_reason_names_the_source(self):
        app.write_settings({"hf_token": "hf_saved_value"})
        self.assertIn("rejected the token from the token saved in LLM Scanner", app.hf_reason("token"))

    def test_token_reason_without_a_source_says_this_computer(self):
        self.assertIn("rejected the token from this computer", app.hf_reason("token"))

    def test_unknown_kind_falls_back_to_the_gated_explanation(self):
        self.assertEqual(app.hf_reason(""), app.hf_reason("access"))
        self.assertEqual(app.hf_reason("whatever"), app.hf_reason("access"))
        self.assertIn("gated or private", app.hf_reason("access"))
        self.assertNotEqual(app.hf_reason("rate"), app.hf_reason("access"))


# ================================================================ model search: Hugging Face
class HfSearch(ModelsTest):
    TREE = [{"path": "M-Q4_K_M.gguf", "size": 4e9}, {"path": "M-Q8_0.gguf", "size": 8e9}]

    def catalog(self, tree, repo="o/r"):
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(tree)):
            return app.hf_gguf_catalog(repo)

    def test_note_html_loading_adds_spinner_and_empty_text_is_nothing(self):
        self.assertEqual(app.note_html("Working", "loading"),
                         '<div class="note-line loading"><span class="spinner"></span>Working</div>')
        self.assertEqual(app.note_html("", "error"), "")

    def test_download_size_is_remembered(self):
        with mock.patch.object(app, "VRAM_BUDGET_GB", 7.5), \
                mock.patch.object(app.requests, "get", return_value=FakeResponse(self.TREE)) as get:
            self.assertEqual(app.hf_download_size("o/r"), 4.0)
            self.assertEqual(app.hf_download_size("o/r"), 4.0)
        self.assertEqual(get.call_count, 1)

    def test_download_size_failed_lookup_is_retried_next_time(self):
        with mock.patch.object(app, "VRAM_BUDGET_GB", 7.5), \
                mock.patch.object(app.requests, "get", side_effect=[OSError("down"), FakeResponse(self.TREE)]):
            self.assertEqual(app.hf_download_size("o/r"), 0.0)
            self.assertEqual(app.hf_download_size("o/r"), 4.0)

    def test_direct_gguf_link_skips_the_search(self):
        with mock.patch.object(app, "hf_download_size", return_value=2.5), \
                mock.patch.object(app.requests, "get") as get:
            repos, q = app.hf_find_repos("https://huggingface.co/o/r-GGUF/tree/main")
        get.assert_not_called()
        self.assertEqual(q, "o/r-GGUF")
        self.assertEqual(repos, [{"id": "o/r-GGUF", "size": 2.5, "gated": False, "private": False}])

    def test_search_uses_the_last_path_segment(self):
        with mock.patch.object(app, "hf_download_size", return_value=0.0), \
                mock.patch.object(app.requests, "get", return_value=FakeResponse([])) as get:
            repos, q = app.hf_find_repos("meta-llama/Llama-3.2-1B")
        self.assertEqual((repos, q), ([], "meta-llama/Llama-3.2-1B"))
        self.assertEqual(get.call_args.args[0], f"{app.HF_API}/models")
        params = get.call_args.kwargs["params"]
        self.assertEqual((params["search"], params["filter"], params["sort"], params["limit"]),
                         ("Llama-3.2-1B", "gguf", "downloads", 25))

    def test_search_server_error_raises(self):
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(status=500)):
            with self.assertRaises(app.requests.HTTPError):
                app.hf_find_repos("smollm")

    def test_repo_marks_flag_other_modified_names_case_insensitively(self):
        self.assertEqual(app.hf_repo_marks({"id": "o/Heretic-7B", "size": 2.0}),
                         "   2.0 GB   (modified: abliterated / uncensored)")
        self.assertEqual(app.hf_repo_marks({"id": "o/Model-UNCENSORED"}), "   (modified: abliterated / uncensored)")

    def test_catalog_single_file_without_quantization_keeps_its_name(self):
        self.assertEqual(self.catalog([{"path": "model.gguf", "size": 5}]), ([("model", "model.gguf", 5)], []))

    def test_catalog_drops_the_shared_name_prefix_at_a_separator(self):
        tree = [{"path": "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "size": 1},
                {"path": "Llama-3.2-1B-Instruct-Q8_0.gguf", "size": 2}]
        catalog, _ = self.catalog(tree)
        self.assertEqual(catalog, [("Q4_K_M", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", 1),
                                   ("Q8_0", "Llama-3.2-1B-Instruct-Q8_0.gguf", 2)])

    def test_catalog_reads_uppercase_gguf_extensions(self):
        catalog, _ = self.catalog([{"path": "sub/Model-Q4_K_M.GGUF", "size": 1},
                                   {"path": "sub/Model-Q8_0.GGUF", "size": 2}])
        self.assertEqual(catalog, [("Q4_K_M", "sub/Model-Q4_K_M.GGUF", 1), ("Q8_0", "sub/Model-Q8_0.GGUF", 2)])

    def test_catalog_unrelated_file_names_fall_back_to_the_quantization(self):
        catalog, _ = self.catalog([{"path": "alpha-Q4_K_M.gguf", "size": 1}, {"path": "beta-Q8_0.gguf"}])
        self.assertEqual(catalog, [("Q4_K_M", "alpha-Q4_K_M.gguf", 1), ("Q8_0", "beta-Q8_0.gguf", 0)])

    def test_catalog_lists_projectors_apart_and_ignores_other_files(self):
        tree = [{"path": "README.md", "size": 1}, {"path": "model.safetensors", "size": 9},
                {"path": "M-Q4_K_M.gguf", "size": 4}, {"path": "mmproj-M-f16.gguf", "size": 1},
                {"path": "M.imatrix.gguf", "size": 2}]
        self.assertEqual(self.catalog(tree), ([("Q4_K_M", "M-Q4_K_M.gguf", 4)], [("mmproj-M-f16.gguf", 1)]))

    def test_catalog_missing_repository_is_a_permanent_error(self):
        with mock.patch.object(app.requests, "get",
                               return_value=FakeResponse({"error": "Repository not found"}, status=404)):
            with self.assertRaises(app.PermanentDownloadError) as err:
                app.hf_gguf_catalog("o/r")
        self.assertEqual(str(err.exception), "Listing files of o/r failed (HTTP 404): Repository not found")

    def test_catalog_requests_the_clean_repository_recursively(self):
        with mock.patch.object(app.requests, "get", return_value=FakeResponse([])) as get:
            self.assertEqual(app.hf_gguf_catalog("https://huggingface.co/o/r/blob/main/x.gguf"), ([], []))
        self.assertEqual(get.call_args.args[0], f"{app.HF_API}/models/o/r/tree/main")
        self.assertEqual(get.call_args.kwargs["params"], {"recursive": "true"})

    def test_repo_info_reads_card_details(self):
        data = {"cardData": {"base_model": ["Qwen/Qwen3-8B", "other/base"], "license": "apache-2.0"},
                "downloads": 123, "likes": 4, "gated": "auto", "private": False}
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(data)) as get:
            info = app.hf_repo_info("https://huggingface.co/o/r")
        self.assertEqual(get.call_args.args[0], f"{app.HF_API}/models/o/r")
        self.assertEqual(info, {"base": "Qwen/Qwen3-8B", "license": "apache-2.0", "downloads": 123, "likes": 4,
                                "gated": True, "private": False})

    def test_repo_info_unreadable_answer_is_empty(self):
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(["not", "a", "dict"])):
            self.assertEqual(app.hf_repo_info("o/bad"), {})

    def test_repo_info_is_remembered(self):
        with mock.patch.object(app.requests, "get", return_value=FakeResponse({"likes": 7})) as get:
            first = app.hf_repo_info("o/c")
            second = app.hf_repo_info("o/c")
        self.assertEqual(get.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(first["likes"], 7)
        self.assertEqual(first["base"], "")

    def test_versions_same_tag_in_two_folders_keeps_the_first_size(self):
        tree = [{"path": "a/x-Q4_K_M.gguf", "size": 4e9}, {"path": "b/y-Q4_K_M.gguf", "size": 5e9}]
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(tree)):
            self.assertEqual(app.hf_versions("o/r"), [(4.0, "Q4_K_M")])


# ================================================================ model search: Ollama library
class OllamaSearch(ModelsTest):
    def page(self, body, status=200):
        return FakeResponse(status=status, text=body)

    def test_find_parses_names_descriptions_capabilities_and_pulls(self):
        body = "<ul>" + ollama_li("qwen3", "Qwen &amp; friends", ["tools", "thinking"], "5.1M") + \
               ollama_li("llama3.2", "Meta") + "</ul>"
        with mock.patch.object(app.requests, "get", return_value=self.page(body)):
            results, q = app.ollama_find("qwen3")
        self.assertEqual(q, "qwen3")
        self.assertEqual(results, [{"name": "qwen3", "desc": "Qwen & friends", "caps": ["tools", "thinking"],
                                    "pulls": "5.1M"},
                                   {"name": "llama3.2", "desc": "Meta", "caps": [], "pulls": ""}])
        self.assertEqual(app._ollama_meta["llama3.2"]["desc"], "Meta", "remembered for the versions step")

    def test_find_puts_the_exact_match_first_and_searches_without_the_tag(self):
        body = ollama_li("qwen3-coder") + ollama_li("qwen3")
        with mock.patch.object(app.requests, "get", return_value=self.page(body)) as get:
            results, q = app.ollama_find("https://ollama.com/library/qwen3:8b")
        self.assertEqual(get.call_args.kwargs["params"], {"q": "qwen3"})
        self.assertEqual([r["name"] for r in results], ["qwen3", "qwen3-coder"])

    def test_find_skips_items_without_a_library_link(self):
        body = '<li><a href="/blog/new-models">Blog</a></li>' + ollama_li("gemma3")
        with mock.patch.object(app.requests, "get", return_value=self.page(body)):
            results, _ = app.ollama_find("gemma")
        self.assertEqual([r["name"] for r in results], ["gemma3"])

    def test_versions_convert_units_and_sort_by_size(self):
        body = ollama_tag_row("qwen3:8b", "5.2GB") + ollama_tag_row("qwen3:0.6b", "523MB") + \
               ollama_tag_row("qwen3:tiny", "300KB")
        with mock.patch.object(app.requests, "get", return_value=self.page(body)) as get:
            got = app.ollama_versions("https://ollama.com/library/qwen3:8b")
        self.assertEqual(get.call_args.args[0], f"{app.OLLAMA_WEB}/library/qwen3/tags")
        self.assertEqual([t for _, t in got], ["qwen3:tiny", "qwen3:0.6b", "qwen3:8b"])
        for (size, _), want in zip(got, (0.0003, 0.523, 5.2)):
            self.assertAlmostEqual(size, want)

    def test_versions_skip_mlx_builds_and_keep_the_first_listing(self):
        body = ollama_tag_row("qwen3:8b", "5.2GB") + ollama_tag_row("qwen3:8b-mlx", "4.9GB") + \
               ollama_tag_row("qwen3:8b", "9.9GB")
        with mock.patch.object(app.requests, "get", return_value=self.page(body)):
            self.assertEqual(app.ollama_versions("qwen3"), [(5.2, "qwen3:8b")])

    def test_best_version_exactly_at_the_budget_fits(self):
        with mock.patch.object(app, "VRAM_BUDGET_GB", 5.0):
            self.assertEqual(app.best_version([(4.0, "a"), (5.0, "b"), (6.0, "c")]), "b")

    def test_best_version_tie_prefers_the_shorter_tag(self):
        with mock.patch.object(app, "VRAM_BUDGET_GB", 7.5):
            self.assertEqual(app.best_version([(4.0, "m:q4_K_M"), (4.0, "m:q4")]), "m:q4")


# ================================================================ search and version pickers (generators)
class SearchFlow(ModelsTest):
    def test_loading_screen_is_visible_and_escapes_text(self):
        u = app.loading_screen("<Search>", 'a "b"')
        self.assertTrue(u["visible"])
        self.assertIn("&lt;Search&gt;", u["value"])
        self.assertIn("a &quot;b&quot;", u["value"])

    def test_source_changed_resets_fields_for_each_source(self):
        o = app.source_changed(app.SRC_OLLAMA)
        self.assertEqual((o[0]["placeholder"], o[0]["value"]), ("qwen3, llama3.2, gemma3", ""))
        self.assertEqual((o[1]["label"], o[1]["choices"], o[1]["interactive"]), ("Model", [app.PLEASE_CHOOSE], False))
        self.assertEqual((o[3], o[4]["interactive"]), ("", False))
        h = app.source_changed(app.SRC_HF)
        self.assertEqual(h[0]["placeholder"], "Qwen3.8-27B, or a Hugging Face link")
        self.assertEqual(h[1]["label"], "Repository")

    def test_search_with_blank_query_asks_for_a_name(self):
        with mock.patch.object(app, "hf_find_repos") as find:
            for query in ("", "   ", None):
                out = list(app.search_models(app.SRC_HF, query))
                self.assertEqual(len(out), 1, repr(query))
                self.assertIn("Enter a model name", out[0][2])
                self.assertFalse(out[0][4]["visible"])
        find.assert_not_called()

    def test_ollama_search_lists_models_with_pulls_and_capabilities(self):
        results = [{"name": "qwen3", "desc": "", "caps": ["tools", "thinking"], "pulls": "5M"},
                   {"name": "qwen3-coder", "desc": "", "caps": [], "pulls": "1M"}]
        with mock.patch.object(app, "ollama_find", return_value=(results, "qwen3")):
            out = list(app.search_models(app.SRC_OLLAMA, " qwen3 "))
        first, last = out[0], out[-1]
        self.assertEqual(first[0]["choices"], [app.LOADING])
        self.assertTrue(first[4]["visible"])
        self.assertIn("Searching the Ollama library", first[4]["value"])
        self.assertIn("Looking for &quot;qwen3&quot;", first[4]["value"])
        self.assertEqual(last[0]["choices"], [app.PLEASE_CHOOSE, ("qwen3   5M pulls   tools, thinking", "qwen3"),
                                              ("qwen3-coder   1M pulls", "qwen3-coder")])
        self.assertTrue(last[0]["interactive"])
        self.assertIn("Found 2 models in the Ollama library", last[2])
        self.assertFalse(last[3]["interactive"], "downloading waits until a version is chosen")
        self.assertFalse(last[4]["visible"])

    def test_ollama_search_without_results_says_so(self):
        with mock.patch.object(app, "ollama_find", return_value=([], "zzz")):
            last = list(app.search_models(app.SRC_OLLAMA, "zzz"))[-1]
        self.assertIn("No models named &quot;zzz&quot; found in the Ollama library.", last[2])
        self.assertIn("note-line error", last[2])
        self.assertEqual(last[0]["choices"], [app.PLEASE_CHOOSE])

    def test_hugging_face_search_failure_is_reported(self):
        with mock.patch.object(app, "hf_find_repos", side_effect=RuntimeError("HTTP 503")):
            out = list(app.search_models(app.SRC_HF, "smollm"))
        self.assertEqual(len(out), 2)
        self.assertIn("Search failed: HTTP 503", out[-1][2])
        self.assertFalse(out[-1][4]["visible"])

    def test_search_for_a_non_gguf_repository_explains_the_conversions(self):
        repos = [{"id": "bartowski/Llama-GGUF", "size": 4.0, "gated": False, "private": False}]
        with mock.patch.object(app, "hf_find_repos", return_value=(repos, "meta-llama/Llama")):
            last = list(app.search_models(app.SRC_HF, "meta-llama/Llama"))[-1]
        self.assertIn("meta-llama/Llama is not in GGUF format, so these are converted versions of it. "
                      "Found 1 GGUF repositories", last[2])
        self.assertEqual(last[0]["choices"][1], ("bartowski/Llama-GGUF   4.0 GB", "bartowski/Llama-GGUF"))

    # ---------------------------------------------------------------- list_versions
    def versions(self, source, repo, budget=7.5):
        with mock.patch.object(app, "VRAM_BUDGET_GB", budget), mock.patch.object(app, "ram_gb", return_value=32):
            return list(app.list_versions(source, repo))

    def hf_versions(self, info, projectors=()):
        catalog = [("Q4_K_M", "M-Q4_K_M.gguf", 4e9)]
        with mock.patch.object(app, "hf_gguf_catalog", return_value=(catalog, list(projectors))), \
                mock.patch.object(app, "hf_repo_info", return_value=info):
            return self.versions(app.SRC_HF, "o/r")[-1]

    def test_list_versions_without_a_repository_stops_at_once(self):
        out = self.versions(app.SRC_OLLAMA, "")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0]["choices"], [app.PLEASE_CHOOSE])
        self.assertEqual(out[0][1], "")
        self.assertFalse(out[0][2]["interactive"])
        self.assertFalse(out[0][3]["visible"])

    def test_ollama_versions_recommend_the_largest_that_fits(self):
        app._ollama_meta["qwen3"] = {"desc": "Qwen family models."}
        opts = [(2.0, "qwen3:1.7b"), (5.2, "qwen3:8b"), (9.3, "qwen3:14b")]
        with mock.patch.object(app, "ollama_versions", return_value=opts):
            out = self.versions(app.SRC_OLLAMA, "qwen3")
        self.assertIn("Loading versions", out[0][3]["value"])
        versions, note, button, hide = out[-1]
        self.assertEqual(versions["value"], "qwen3:8b")
        labels = {v: lbl for lbl, v in versions["choices"]}
        self.assertEqual(labels["qwen3:8b"], "8b   5.2 GB VRAM   (Recommended)")
        self.assertEqual(labels["qwen3:14b"], "14b   7.5 GB VRAM + 1.8 GB RAM")
        self.assertIn("Qwen family models. Recommended: 8b (5.2 GB), the largest version that fits your 7.5 GB of "
                      "video memory, so it runs fully on the GPU.", note)
        self.assertTrue(button["interactive"])
        self.assertFalse(hide["visible"])

    def test_ollama_versions_when_nothing_fits_pick_the_smallest(self):
        with mock.patch.object(app, "ollama_versions", return_value=[(2.0, "m:a"), (3.0, "m:b")]):
            versions, note, _, _ = self.versions(app.SRC_OLLAMA, "m", budget=1.0)[-1]
        self.assertEqual(versions["value"], "m:a")
        self.assertIn("No version fits your 1.0 GB of video memory, so the smallest Ollama can load (a, 2.0 GB) is "
                      "selected", note)

    def test_gated_repository_without_a_token_points_to_the_gear(self):
        _, note, _, _ = self.hf_versions({"gated": True})
        self.assertIn("Gated: accept this model&#x27;s conditions on its Hugging Face page, then add a token with "
                      "the gear above.", note)

    def test_gated_repository_with_a_token_only_asks_to_accept(self):
        app.write_settings({"hf_token": "hf_gatedrepotoken"})
        _, note, _, _ = self.hf_versions({"gated": True})
        self.assertIn("conditions on its Hugging Face page.", note)
        self.assertNotIn("gear above", note)

    def test_repository_details_are_described_in_order(self):
        info = {"private": True, "base": "Qwen/Qwen3-8B", "license": "apache-2.0", "downloads": 1_500_000}
        _, note, _, _ = self.hf_versions(info, projectors=[("mmproj-f16.gguf", 1)])
        parts = ["Private: only your Hugging Face account can see this repository.", "Based on Qwen3-8B.",
                 "apache-2.0 licence.", "1.5M downloads on Hugging Face.",
                 "Sees images (the repository includes a vision projector).", "Recommended: Q4_K_M (4.0 GB)"]
        positions = [note.index(p) for p in parts]
        self.assertEqual(positions, sorted(positions))

    def test_listing_failure_is_reported(self):
        err = app.PermanentDownloadError("Listing files of o/r failed (HTTP 404)")
        with mock.patch.object(app, "hf_gguf_catalog", side_effect=err):
            out = self.versions(app.SRC_HF, "o/r")
        self.assertEqual(len(out), 2)
        versions, note, button, _ = out[-1]
        self.assertIn("Could not list versions: Listing files of o/r failed (HTTP 404)", note)
        self.assertEqual(versions["choices"], [app.PLEASE_CHOOSE])
        self.assertFalse(button["interactive"])

    def test_version_changed_needs_both_fields(self):
        self.assertTrue(app.version_changed("o/r", "Q4_K_M")["interactive"])
        for repo, version in (("", "Q4_K_M"), ("o/r", ""), (None, None)):
            self.assertFalse(app.version_changed(repo, version)["interactive"], (repo, version))

    # ---------------------------------------------------------------- pull_model
    def test_pull_from_ollama_uses_the_version_link(self):
        with mock.patch.object(app, "start_download", return_value=True) as start:
            note = app.pull_model(app.SRC_OLLAMA, "qwen3", "https://ollama.com/library/qwen3:8b")
        start.assert_called_once_with("qwen3:8b")
        self.assertIn("Downloading qwen3:8b.", note)

    def test_pull_from_hugging_face_builds_the_reference_from_a_link(self):
        with mock.patch.object(app, "start_download", return_value=True) as start:
            app.pull_model(app.SRC_HF, "https://huggingface.co/o/r-GGUF/tree/main", "Q4_K_M")
        start.assert_called_once_with("hf.co/o/r-GGUF:Q4_K_M")

    def test_pull_of_a_download_already_running_says_so(self):
        with mock.patch.object(app, "start_download", return_value=False):
            note = app.pull_model(app.SRC_HF, "o/r", "Q8_0")
        self.assertIn("hf.co/o/r:Q8_0 is already downloading.", note)
        self.assertNotIn("error", note)


# ================================================================ downloads: persisted queue and error text
class DownloadErrors(ModelsTest):
    def test_pending_without_a_readable_file_is_empty(self):
        self.assertEqual(app._pending(), [])
        app.DOWNLOADS_FILE.write_text("{oops")
        self.assertEqual(app._pending(), [])

    def test_pending_entry_without_paused_flag_is_active(self):
        app.DOWNLOADS_FILE.write_text(json.dumps([{"ref": "m:1"}, {"ref": "m:2", "paused": 1}, "m:3"]))
        self.assertEqual(app._pending(), [("m:1", False), ("m:2", True), ("m:3", False)])

    # BUG: _pending() only guards json.loads; a downloads.json holding valid JSON that is not a list (e.g. "null")
    # raises TypeError, which crashes resume_pending_downloads() at startup. read_settings() and _pull_layers()
    # check the type and fall back to empty; _pending() should too.
    def test_pending_file_holding_json_null_reads_as_empty(self):
        app.DOWNLOADS_FILE.write_text("null")
        self.assertEqual(app._pending(), [])

    def test_response_error_uses_the_message_field(self):
        r = FakeResponse({"message": "rate limited"}, status=429)
        self.assertEqual(app.response_error(r, "Op"), "Op failed (HTTP 429): rate limited")

    def test_response_error_for_non_object_bodies(self):
        self.assertEqual(app.response_error(FakeResponse(["a", "b"], status=500), "Op"),
                         "Op failed (HTTP 500): ['a', 'b']")
        self.assertEqual(app.response_error(FakeResponse(None, status=502), "Op"), "Op failed (HTTP 502)")

    def test_response_error_collapses_whitespace_and_truncates(self):
        r = FakeResponse({"error": "x\n\n  y " + "z" * 600}, status=400)
        self.assertEqual(app.response_error(r, "Op"), "Op failed (HTTP 400): x y " + "z" * 496)

    def test_refusal_with_an_environment_token_blames_the_account(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_env_token"}):
            msg = app._hf_denied("m.gguf", FakeResponse({"error": "restricted"}, status=403))
        self.assertTrue(msg.startswith("Downloading m.gguf failed (HTTP 403): restricted. The Hugging Face token in "
                                       "use doesn't grant access"), msg)

    def fetch(self, response, should_stop=lambda: False, size=0, url="https://example.com/m.gguf"):
        dest, calls = self.tmp / "m.gguf", []
        with mock.patch.object(app.requests, "get", return_value=response):
            ok = app.fetch_file(url, dest, size, should_stop, lambda: calls.append(1))
        return ok, dest, len(calls)

    def test_fetch_stopped_before_the_first_chunk_keeps_an_empty_partial(self):
        ok, dest, progress = self.fetch(FakeResponse(lines=[b"abcd"]), should_stop=lambda: True, size=4)
        self.assertFalse(ok)
        self.assertFalse(dest.exists())
        self.assertEqual(dest.with_name("m.gguf.part").stat().st_size, 0)
        self.assertEqual(progress, 0)

    def test_fetch_reports_progress_per_chunk_and_accepts_unknown_size(self):
        ok, dest, progress = self.fetch(FakeResponse(lines=[b"ab", b"cd", b"ef"]))
        self.assertTrue(ok)
        self.assertEqual(dest.read_bytes(), b"abcdef")
        self.assertEqual(progress, 3)
        self.assertFalse(dest.with_name("m.gguf.part").exists())

    def test_fetch_401_from_another_host_is_a_plain_permanent_error(self):
        with self.assertRaises(app.PermanentDownloadError) as err:
            self.fetch(FakeResponse({"error": "nope"}, status=401))
        self.assertEqual(str(err.exception), "Downloading m.gguf failed (HTTP 401): nope")


# ================================================================ downloads: queue controls
class DownloadQueue(DownloadsTest):
    REF = "hf.co/o/r:Q4_K_M"
    SLUG = "hf-co-o-r-q4-k-m"

    def test_start_creates_a_queued_entry_and_one_worker(self):
        self.assertTrue(app.start_download("m:1"))
        d = app._downloads["m:1"]
        self.assertEqual((d["state"], d["msg"], d["done"], d["total"], d["cancel"], d["running"]),
                         ("queued", "Queued", 0, 0, False, True))
        self.assertIs(self.thread.call_args.kwargs["target"], app._run_download)
        self.assertEqual(self.thread.call_args.kwargs["args"], ("m:1",))
        self.thread.return_value.start.assert_called_once()
        self.assertEqual(app._pending(), [("m:1", False)])

    def test_start_on_a_stopped_paused_download_resumes_it(self):
        app._downloads["m:1"] = dl("paused", pause=True, running=False, done=5, total=10)
        app._set_pending("m:1", True, paused=True)
        self.assertTrue(app.start_download("m:1"))
        d = app._downloads["m:1"]
        self.assertEqual((d["state"], d["msg"], d["pause"], d["done"]), ("queued", "Resuming", False, 5))
        self.thread.assert_called_once()
        self.assertEqual(app._pending(), [("m:1", False)])

    def test_priority_goes_to_an_active_download(self):
        app._downloads["a"] = dl("downloading")
        note = app.prioritize_download("a")
        self.assertIn("a now has priority", note)
        self.assertEqual(app._priority["ref"], "a")

    def test_priority_ignores_finished_unknown_and_empty_refs(self):
        app._downloads["a"] = dl("done")
        for ref in ("a", "missing", "", None):
            self.assertEqual(app.prioritize_download(ref), NOOP, repr(ref))
        self.assertIsNone(app._priority["ref"])

    def test_removing_an_unknown_download_changes_nothing(self):
        self.assertEqual(app.remove_download("nope"), (NOOP, NOOP))
        self.thread.assert_not_called()
        self.assertFalse(app.DOWNLOADS_FILE.exists())

    def test_removing_a_running_download_leaves_cleanup_to_its_worker(self):
        app._downloads["r"] = dl("downloading", running=True)
        app._set_pending("r", True)
        note, ref_box = app.remove_download("r")
        d = app._downloads["r"]
        self.assertEqual((d["cancel"], d["hidden"], d["state"]), (True, True, "downloading"))
        self.thread.assert_not_called()
        self.assertEqual(app._pending(), [])
        self.assertIn("Deleted r from downloads.", note)
        self.assertEqual(ref_box, "")

    def test_removing_a_failed_download_deletes_its_files_now(self):
        app.HF_DOWNLOADS.mkdir(parents=True)
        part = app.HF_DOWNLOADS / f"{self.SLUG}.gguf.part"
        proj = app.HF_DOWNLOADS / f"{self.SLUG}.mmproj.gguf"
        part.write_bytes(b"GG")
        proj.write_bytes(b"PJ")
        app._downloads[self.REF] = dl("failed", msg="boom")
        note, _ = app.remove_download(self.REF)
        self.assertFalse(part.exists())
        self.assertFalse(proj.exists())
        d = app._downloads[self.REF]
        self.assertEqual((d["state"], d["cancel"], d["hidden"]), ("failed", False, True))
        self.assertIs(self.thread.call_args.kwargs["target"], app._drop_pull_data)
        self.assertEqual(self.thread.call_args.kwargs["args"], (self.REF,))
        self.assertIn(f"Deleted {self.REF} from downloads.", note)

    def test_pausing_twice_writes_the_queue_once(self):
        app._downloads["r"] = dl("downloading", running=True)
        self.assertEqual(app.toggle_download("pause:r"), (NOOP, ""))
        self.assertEqual(app._pending(), [("r", True)])
        app.DOWNLOADS_FILE.write_text("[]")
        app.toggle_download("pause:r")
        self.assertEqual(app.DOWNLOADS_FILE.read_text(), "[]", "a second pause is a no-op")

    def test_pausing_the_priority_download_releases_priority(self):
        app._downloads["r"] = dl("downloading", running=True)
        app._priority["ref"] = "r"
        app.toggle_download("pause:r")
        self.assertIsNone(app._priority["ref"])
        self.assertEqual(app._downloads["r"]["state"], "paused")

    def test_resume_of_a_download_that_is_not_paused_does_nothing(self):
        app._downloads["r"] = dl("downloading", running=True)
        app.toggle_download("resume:r")
        self.thread.assert_not_called()
        self.assertFalse(app._downloads["r"].get("pause"))
        self.assertEqual(app._downloads["r"]["msg"], "Downloading")
        self.assertFalse(app.DOWNLOADS_FILE.exists())

    def test_restart_counts_every_partial_file_of_a_paused_import(self):
        app.HF_DOWNLOADS.mkdir(parents=True)
        (app.HF_DOWNLOADS / f"{self.SLUG}.gguf.part").write_bytes(b"abc")
        (app.HF_DOWNLOADS / f"{self.SLUG}.mmproj.gguf").write_bytes(b"12345")
        app.DOWNLOADS_FILE.write_text(json.dumps([{"ref": self.REF, "paused": True}, "m:1"]))
        with mock.patch.object(app, "start_download") as start:
            app.resume_pending_downloads()
        self.assertEqual([c.args[0] for c in start.call_args_list], ["m:1"])
        d = app._downloads[self.REF]
        self.assertEqual((d["state"], d["done"], d["running"], d["pause"]), ("paused", 8, False, True))

    def test_download_label_names_image_models(self):
        self.assertEqual(app.download_label("image:z-image-turbo"), "Z-Image Turbo (image model)")
        self.assertEqual(app.download_label("image:nope"), "image:nope")
        self.assertEqual(app.download_label("qwen3:8b"), "qwen3:8b")

    def test_finished_downloads_leave_the_list_after_thirty_seconds(self):
        app._downloads["stale"] = dl("done", finished=time.time() - 31)
        app._downloads["fresh"] = dl("done", finished=time.time() - 5)
        out = app.downloads_html()
        self.assertIn('title="fresh"', out)
        self.assertNotIn('title="stale"', out)
        self.assertIn('class="fill done" style="width:100.0%"', out)

    def test_download_row_shows_percentage_and_sizes(self):
        app._downloads["m"] = dl("downloading", done=5e8, total=1e9)
        out = app.downloads_html()
        self.assertIn('<div class="dl-detail">50%   0.5 of 1.0 GB</div>', out)
        self.assertIn('style="width:50.0%"', out)
        self.assertIn('data-active="1"', out)

    def test_paused_download_without_total_shows_what_is_downloaded(self):
        app._downloads["m"] = dl("paused", pause=True, done=2.5e9)
        out = app.downloads_html()
        self.assertIn('<div class="dl-detail">2.5 GB downloaded</div>', out)
        self.assertIn('data-action="resume:m"', out)

    def test_active_downloads_skip_hidden_user_paused_and_finished(self):
        app._downloads.update({"a": dl("queued"), "b": dl("waiting"), "c": dl("paused", pause=True),
                               "d": dl("paused"), "e": dl("done"), "f": dl("downloading", hidden=True)})
        self.assertEqual(app.active_downloads(), ["a", "b", "d"])

    def test_tick_with_nothing_running(self):
        panel, seen, refresh, picker, row = app.downloads_tick(0, None)
        self.assertIn("No active downloads", panel)
        self.assertEqual((seen, refresh), (0, NOOP))
        self.assertEqual((picker["choices"], picker["value"]), ([], None))
        self.assertFalse(row["visible"])

    def test_tick_drops_stale_priority_keeps_pick_and_reports_finished(self):
        app._downloads.update({"a": dl("downloading"), "b": dl("queued")})
        app._priority["ref"] = "gone"
        app._dl_finished["count"] = 3
        panel, seen, refresh, picker, row = app.downloads_tick(2, "b")
        self.assertIsNone(app._priority["ref"])
        self.assertEqual((picker["value"], picker["choices"]), ("b", [("a", "a"), ("b", "b")]))
        self.assertEqual(seen, 3)
        self.assertIsInstance(refresh, float)
        self.assertTrue(row["visible"])

    def test_tick_selects_and_labels_the_priority_download(self):
        app._downloads.update({"a": dl("downloading"), "b": dl("queued")})
        app._priority["ref"] = "b"
        _, seen, refresh, picker, _ = app.downloads_tick(0, "not-active")
        self.assertEqual(picker["value"], "b")
        self.assertEqual(picker["choices"], [("a", "a"), ("b   (priority)", "b")])
        self.assertEqual((seen, refresh), (0, NOOP))

    def test_partial_files_per_kind_of_reference(self):
        names = [f.name for f in app._partial_files(self.REF)]
        self.assertEqual(names, [f"{self.SLUG}.gguf.part", f"{self.SLUG}.gguf", f"{self.SLUG}.mmproj.gguf.part",
                                 f"{self.SLUG}.mmproj.gguf"])
        self.assertTrue(all(f.parent == app.HF_DOWNLOADS for f in app._partial_files(self.REF)))
        self.assertEqual(app._partial_files("qwen3:8b"), [])
        self.assertEqual(app._partial_files("image:nope"), [])


# ================================================================ downloads: Ollama's partial layers
class PullLayers(DownloadsTest):
    def test_remember_merges_sorted_and_skips_known_layers(self):
        app._remember_layers("m", ["sha256:b"])
        app._remember_layers("m", ["sha256:a", "sha256:b"])
        self.assertEqual(app._pull_layers(), {"m": ["sha256:a", "sha256:b"]})
        compact = json.dumps(app._pull_layers())
        app.PULL_LAYERS_FILE.write_text(compact)
        app._remember_layers("m", ["sha256:a"])
        self.assertEqual(app.PULL_LAYERS_FILE.read_text(), compact, "nothing new: the file is not rewritten")

    def test_forget_only_rewrites_for_a_known_ref(self):
        app.PULL_LAYERS_FILE.write_text("[1, 2]")
        self.assertEqual(app._pull_layers(), {})
        compact = json.dumps({"a": ["sha256:1"], "b": ["sha256:2"]})
        app.PULL_LAYERS_FILE.write_text(compact)
        app._forget_layers("zzz")
        self.assertEqual(app.PULL_LAYERS_FILE.read_text(), compact)
        app._forget_layers("a")
        self.assertEqual(app._pull_layers(), {"b": ["sha256:2"]})

    def blob(self, name, old):
        f = app.OLLAMA_BLOBS / name
        f.write_bytes(b"x")
        if old:
            t = time.time() - 2000
            os.utime(f, (t, t))
        return f

    def test_orphans_are_old_unowned_partial_layers_only(self):
        app.OLLAMA_BLOBS.mkdir(parents=True)
        self.blob("sha256-aaa-partial", old=True)
        self.blob("sha256-bbb-partial", old=True)
        self.blob("sha256-ccc-partial-0", old=False)
        self.blob("sha256-ddd", old=True)
        app.DOWNLOADS_FILE.write_text(json.dumps(["m:1"]))
        app.PULL_LAYERS_FILE.write_text(json.dumps({"m:1": ["sha256:aaa"]}))
        self.assertEqual([f.name for f in app.orphaned_ollama_partials()], ["sha256-bbb-partial"])

    def test_orphans_wait_while_a_queued_pull_has_unknown_layers(self):
        self.assertEqual(app.orphaned_ollama_partials(), [], "no blobs folder at all")
        app.OLLAMA_BLOBS.mkdir(parents=True)
        self.blob("sha256-eee-partial", old=True)
        app.DOWNLOADS_FILE.write_text(json.dumps([{"ref": "image:z-image-turbo", "paused": True}]))
        self.assertEqual([f.name for f in app.orphaned_ollama_partials()], ["sha256-eee-partial"],
                         "image downloads have no Ollama layers, so they don't block the cleanup")
        app.DOWNLOADS_FILE.write_text(json.dumps(["image:z-image-turbo", "m:1"]))
        self.assertEqual(app.orphaned_ollama_partials(), [])


# ================================================================ installed chat models table
class ChatModelTable(ModelsTest):
    def test_list_models_and_names_from_ollama(self):
        models = [{"name": "a"}, {"name": "b"}]
        with mock.patch.object(app.requests, "get", return_value=FakeResponse({"models": models})) as get:
            self.assertEqual(app.model_names(), ["a", "b"])
        self.assertEqual(get.call_args.args[0], f"{app.OLLAMA}/api/tags")
        with mock.patch.object(app.requests, "get", return_value=FakeResponse({})):
            self.assertEqual(app.list_models(), [])
        with mock.patch.object(app.requests, "get", side_effect=app.requests.ConnectionError("down")):
            self.assertEqual(app.list_models(), [])

    def test_readable_error_drops_the_blob_path(self):
        err = 'read GGUF metadata "/home/u/.ollama/models/blobs/sha256-ab": invalid file magic '
        self.assertEqual(app.readable_ollama_error(err), "invalid file magic")
        self.assertEqual(app.readable_ollama_error(RuntimeError("  boom  ")), "boom")

    def test_readable_error_explains_formats_ollama_cannot_read(self):
        for err, head in (('read GGUF metadata "/x": unsupported tensor type 41', "unsupported tensor type 41"),
                          ('read GGUF metadata "/x": tensor size overflows', "tensor size overflows")):
            out = app.readable_ollama_error(err)
            self.assertTrue(out.startswith(head + "\n\n"), out)
            self.assertIn("download a standard quantization instead", out)

    def test_load_problem_asks_once_for_a_model_that_loads(self):
        m = {"name": "m", "digest": "d1"}
        with mock.patch.object(app.requests, "post", return_value=FakeResponse({})) as post:
            self.assertEqual(app.load_problem(m), "")
            self.assertEqual(app.load_problem(m), "")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.args[0], f"{app.OLLAMA}/api/show")
        self.assertEqual(post.call_args.kwargs["json"], {"model": "m"})

    def test_load_problem_reports_ollamas_reason_or_the_status(self):
        answers = {"a": FakeResponse({"error": 'read GGUF metadata "/b/sha256-1": unsupported tensor type 41'},
                                     status=500),
                   "b": FakeResponse({"error": ""}, status=404), "c": BadJson(status=502)}
        with mock.patch.object(app.requests, "post", side_effect=lambda url, **kw: answers[kw["json"]["model"]]):
            self.assertEqual(app.load_problem({"name": "a"}), "unsupported tensor type 41")
            self.assertEqual(app.load_problem({"name": "b"}), "HTTP 404")
            self.assertEqual(app.load_problem({"name": "c"}), "HTTP 502")

    def test_load_problem_is_asked_again_after_ollama_was_unreachable(self):
        m = {"name": "m", "digest": "d2"}
        bad = FakeResponse({"error": "unsupported tensor"}, status=500)
        with mock.patch.object(app.requests, "post", side_effect=[app.requests.ConnectionError("down"), bad]):
            self.assertEqual(app.load_problem(m), "")
            self.assertNotIn(("m", "d2"), app._load_problems)
            self.assertEqual(app.load_problem(m), "unsupported tensor")

    def test_scan_choices_mark_models_ollama_cannot_load(self):
        models = [{"name": "x", "digest": "1"}, {"name": "y", "digest": "2"}]
        with mock.patch.object(app, "list_models", return_value=models), \
                mock.patch.object(app, "load_problem", side_effect=lambda m: "bad" if m["name"] == "x" else ""):
            self.assertEqual(app.scan_model_choices(), ([("x   Ollama can't load this format", "x"), ("y", "y")], "y"))
        with mock.patch.object(app, "list_models", return_value=models), \
                mock.patch.object(app, "load_problem", return_value="bad"):
            self.assertIsNone(app.scan_model_choices()[1])

    def test_source_icon_per_source(self):
        hf = app.source_icon(app.SRC_HF)
        self.assertIn('class="src-icon hf"', hf)
        self.assertIn('title="Hugging Face" aria-label="Hugging Face"', hf)
        ollama = app.source_icon(app.SRC_OLLAMA)
        self.assertIn('class="src-icon ollama"', ollama)
        self.assertIn(app.SOURCE_ICONS[app.SRC_OLLAMA][:40], ollama)

    def test_model_row_shows_size_parameters_quantization_and_date(self):
        m = {"name": "qwen3:8b", "size": 5.2e9, "modified_at": "2025-05-01T10:00:00Z",
             "details": {"parameter_size": "8.2B", "quantization_level": "Q4_K_M"}}
        out = app.model_row_html(m)
        self.assertEqual(out.count('class="mt-cell"'), 5)
        for cell in ("5.2 GB", "8.2B", "Q4_K_M", "2025-05-01"):
            self.assertIn(f'<div class="mt-cell">{cell}</div>', out)
        self.assertIn(f'href="{html.escape(app.OLLAMA_WEB)}/library/qwen3:8b"', out)
        self.assertIn('class="src-icon ollama"', out)

    def test_model_row_with_missing_details_and_markup_in_the_name(self):
        out = app.model_row_html({"name": "hf.co/o/<r>:Q4"})
        self.assertIn('<div class="mt-cell">0.0 GB</div>', out)
        self.assertEqual(out.count('<div class="mt-cell"></div>'), 3)
        self.assertIn('class="src-icon hf"', out)
        self.assertIn('href="https://huggingface.co/o/&lt;r&gt;"', out)
        self.assertIn(">hf.co/o/&lt;r&gt;:Q4</a>", out)
        self.assertNotIn("<r>", out)

    def test_delete_without_confirmation_changes_nothing(self):
        with mock.patch.object(app.requests, "delete") as delete:
            self.assertEqual(app.delete_model("m", False), (NOOP, NOOP))
        delete.assert_not_called()

    def test_delete_failure_with_a_plain_text_body(self):
        with mock.patch.object(app.requests, "delete", return_value=BadJson(status=500, text="  model is in use  ")):
            note, refresh = app.delete_model("m", True)
        self.assertIn("Deleting m failed (HTTP 500): model is in use", note)
        self.assertIn("note-line error", note)
        self.assertIsInstance(refresh, float)

    def test_human_size_boundaries(self):
        for n, want in ((0, "1 KB"), (999_499, "999 KB"), (1_000_000, "1.0 MB"), (1_234_567, "1.2 MB")):
            self.assertEqual(app._human_size(n), want, n)


# ================================================================ generated, table-driven cases
def _add(cls, name, fn):
    assert not hasattr(cls, name), f"duplicate test name {name}"
    fn.__name__ = name
    setattr(cls, name, fn)


class GeneratedPure(unittest.TestCase):
    """Pure helpers checked against one input each."""


RUNS_IN_OLLAMA = [
    ("q4_k_without_size_suffix", "Q4_K", True), ("q6_k_l_large_suffix", "Q6_K_L", True),
    ("iq2_xxs", "IQ2_XXS", True), ("iq4_nl", "IQ4_NL", True), ("ud_prefixed_iq2_m", "UD-IQ2_M", True),
    ("tag_after_repo_colon", "repo:Q8_0", True), ("q9_k_out_of_range", "Q9_K", False),
    ("q4_2_legacy", "Q4_2", False), ("q8_k_triple_letter_suffix", "Q8_K_XXL", False), ("empty_tag", "", False),
]
for _slug, _tag, _want in RUNS_IN_OLLAMA:
    def _t(self, tag=_tag, want=_want):
        self.assertIs(app.runs_in_ollama(tag), want, tag)
    _add(GeneratedPure, f"test_runs_in_ollama_{'accepts' if _want else 'rejects'}_{_slug}", _t)

IS_HF_URL = [
    ("uppercase_host", "https://HUGGINGFACE.CO/a", True), ("subdomain", "https://cdn-lfs.huggingface.co/x", True),
    ("userinfo_and_port", "http://user@huggingface.co:8080/x", True),
    ("lookalike_without_dot", "https://nothuggingface.co/x", False),
    ("hf_as_prefix_of_other_domain", "https://huggingface.co.evil.example/x", False),
    ("missing_scheme", "huggingface.co/a/b", False), ("short_hf_co_domain", "https://hf.co/a", False),
]
for _slug, _url, _want in IS_HF_URL:
    def _t(self, url=_url, want=_want):
        self.assertIs(app._is_hf_url(url), want, url)
    _add(GeneratedPure, f"test_is_hf_url_{'yes' if _want else 'no'}_{_slug}", _t)

REPO_FROM_TEXT = [
    ("http_scheme", "http://huggingface.co/a/b", "a/b"), ("hf_co_https", "https://hf.co/a/b", "a/b"),
    ("www_hf_co_tree", "https://www.hf.co/a/b/tree/main", "a/b"),
    ("schemeless_blob", "huggingface.co/a/b/blob/main/x.gguf", "a/b"), ("only_slashes", "///", ""),
    ("unicode_with_tag", "\u00dcn\u00ef/c\u00f8d\u00e9-GGUF:Q4_K_M", "\u00dcn\u00ef/c\u00f8d\u00e9-GGUF"),
]
for _slug, _text, _want in REPO_FROM_TEXT:
    def _t(self, text=_text, want=_want):
        self.assertEqual(app.repo_from_text(text), want)
    _add(GeneratedPure, f"test_repo_from_text_{_slug}", _t)

OLLAMA_NAME = [
    ("www_library_url", "https://www.ollama.com/library/qwen3", "qwen3"),
    ("http_tags_trailing_slash", "http://ollama.com/library/llama3.2/tags/", "llama3.2"),
    ("namespaced_user_model", "https://ollama.com/huihui_ai/qwen3-abliterated", "huihui_ai/qwen3-abliterated"),
    ("padded_tag", "  gemma3:4b  ", "gemma3:4b"), ("none", None, ""),
    ("tags_query_string", "https://ollama.com/library/qwen3/tags?sort=size", "qwen3"),
]
for _slug, _text, _want in OLLAMA_NAME:
    def _t(self, text=_text, want=_want):
        self.assertEqual(app.ollama_name_from_text(text), want)
    _add(GeneratedPure, f"test_ollama_name_from_text_{_slug}", _t)

SHORT_COUNT = [("zero", 0, "0"), ("just_below_thousand", 999, "999"), ("exact_thousand", 1000, "1k"),
               ("tens_of_thousands", 12_345, "12k"), ("exact_million", 1_000_000, "1M"),
               ("hundreds_of_millions", 123_456_789, "123.5M")]
for _slug, _n, _want in SHORT_COUNT:
    def _t(self, n=_n, want=_want):
        self.assertEqual(app._short_count(n), want)
    _add(GeneratedPure, f"test_short_count_{_slug}", _t)

MASK_TOKEN = [("empty", "", ""), ("twelve_chars_fully_hidden", "abcdefghijkl", "\u2026"),
              ("thirteen_chars_shows_ends", "abcdefghijklm", "abc\u2026jklm"),
              ("unicode_token", "t\u00f6k\u00e9n_\u00fcn\u00efc\u00f8d\u00e9_xyz", "t\u00f6k\u2026_xyz")]
for _slug, _token, _want in MASK_TOKEN:
    def _t(self, token=_token, want=_want):
        self.assertEqual(app.mask_token(token), want)
    _add(GeneratedPure, f"test_mask_token_{_slug}", _t)

CHECK_RESPONSE = [(400, True), (401, True), (410, True), (499, True), (500, False)]
for _code, _permanent in CHECK_RESPONSE:
    def _t(self, code=_code, permanent=_permanent):
        if permanent:
            with self.assertRaises(app.PermanentDownloadError) as err:
                app._check_response(FakeResponse(status=code), "Op")
            self.assertEqual(str(err.exception), f"Op failed (HTTP {code})")
        else:
            with self.assertRaises(app.requests.HTTPError):
                app._check_response(FakeResponse(status=code), "Op")
    _add(GeneratedPure, f"test_check_response_{_code}_{'is_permanent' if _permanent else 'is_retried'}", _t)

MODEL_PAGE = [("hf_ref_drops_tag", "hf.co/a/b:Q4_K_M", "https://huggingface.co/a/b"),
              ("bare_library_name", "qwen3", "/library/qwen3"),
              ("latest_tag_is_omitted", "qwen3:latest", "/library/qwen3"),
              ("explicit_tag_kept", "qwen3:8b", "/library/qwen3:8b"),
              ("namespaced_model", "user/m:tag", "/user/m:tag")]
for _slug, _name, _want in MODEL_PAGE:
    def _t(self, name=_name, want=_want):
        expected = want if want.startswith("https://") else app.OLLAMA_WEB + want
        self.assertEqual(app.model_page(name), expected)
    _add(GeneratedPure, f"test_model_page_{_slug}", _t)


class GeneratedFailureKinds(ModelsTest):
    """garak output lines, read while a token is saved in the app."""

    def setUp(self):
        super().setUp()
        app.write_settings({"hf_token": "hf_saved_token_value"})


FAILURE_KINDS = [
    ("401_on_hf_co", "OSError: 401 Client Error: Unauthorized for url: https://hf.co/api/x", "token"),
    ("invalid_credentials", "HfHubHTTPError: Invalid credentials in Authorization header", "token"),
    ("gated_repo_error", "GatedRepoError: Access to model is restricted", "access"),
    ("403_forbidden", "huggingface.co returned 403 Forbidden", "access"),
    ("too_many_requests_any_case", "HUGGINGFACE: too many requests", "rate"),
    ("rate_beats_gated", "GatedRepoError 429", "rate"),
    ("401_not_from_hf", "401 Unauthorized from api.openai.com", ""),
    ("number_containing_401", "hf.co/model downloaded 4011 bytes", ""),
]
for _slug, _line, _want in FAILURE_KINDS:
    def _t(self, line=_line, want=_want):
        self.assertEqual(app.hf_failure_kind(line), want, line)
    _add(GeneratedFailureKinds, f"test_failure_kind_{_slug}_is_{_want or 'none'}", _t)

del _t


if __name__ == "__main__":
    unittest.main()
