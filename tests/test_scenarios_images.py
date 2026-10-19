"""Scenario tests for the Images tab: image model catalog, Hugging Face image search, downloads, the gallery and
image generation with a fake stable-diffusion.cpp engine."""
import json
import os
import signal
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_app import FakeResponse, TempDataTest, app  # noqa: E402

Z = "z-image-turbo"
SQUARE = "Square 1024 x 1024"
SMALL = "Small 512 x 512 (fastest)"
FLUX_REPO, FLUX_FILE = "city96/FLUX.1-schnell-gguf", "flux1-schnell-Q4_K_S.gguf"
PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                    "1f15c4890000000d4944415478da6364f8ffbf1e000502027fa1d5b40000000049454e44ae426082")

# A stand-in for stable-diffusion.cpp's sd-cli. Its behavior is steered by FAKE_SD_* environment variables, which
# generate_image passes on to the engine.
FAKE_ENGINE = f"""#!{sys.executable}
import json, os, signal, sys, time
a = sys.argv[1:]
log = os.environ.get("FAKE_SD_LOG")
if log:
    with open(log, "w", encoding="utf-8") as fh:
        json.dump({{"argv": a, "ld": os.environ.get("LD_LIBRARY_PATH"),
                   "vk": os.environ.get("GGML_VK_VISIBLE_DEVICES")}}, fh)
out = a[a.index("-o") + 1]
steps = int(a[a.index("--steps") + 1])
for line in os.environ.get("FAKE_SD_PRE", "").split("|"):
    if line:
        print(line, flush=True)
time.sleep(float(os.environ.get("FAKE_SD_PRESLEEP", "0")))
for i in range(1, steps + 1):
    print(f"  |==| {{i}}/{{steps}} - 0.10s/it", flush=True)
if os.environ.get("FAKE_SD_POST"):
    print(os.environ["FAKE_SD_POST"], flush=True)
time.sleep(float(os.environ.get("FAKE_SD_SLEEP", "0")))
if os.environ.get("FAKE_SD_SIGNAL"):
    os.kill(os.getpid(), signal.SIGTERM)
if os.environ.get("FAKE_SD_MSG"):
    print(os.environ["FAKE_SD_MSG"], flush=True)
code = int(os.environ.get("FAKE_SD_EXIT", "0"))
if code == 0 and not os.environ.get("FAKE_SD_NOOUT"):
    open(out, "wb").write(bytes.fromhex({PNG.hex()!r}))
sys.exit(code)
"""


class ImagesBase(TempDataTest):
    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.dict(app._image_proc, {"p": None}))
        self.enterContext(mock.patch.dict(app._proc, {"p": None}))
        self.enterContext(mock.patch.object(app, "VRAM_BUDGET_GB", 7.5))
        self.enterContext(mock.patch.dict(os.environ, {n: "" for n in app.HF_TOKEN_ENV}))

    def install(self, key=Z, skip=(), data=b"xx"):
        for role, p in app.image_model_files(key).items():
            if role in skip:
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    def write_custom(self, entries):
        app.CUSTOM_IMAGE_FILE.write_text(json.dumps(entries), encoding="utf-8")

    def add_searched(self, repo=FLUX_REPO, path=FLUX_FILE):
        with mock.patch.object(app, "start_download", return_value=True):
            app.download_searched_image_model(repo, path)
        return app._slug(f"{repo}-{path.rsplit('.', 1)[0]}")


def sdxl_entry(name="My XL", repo="me/my-xl", path="my-xl.safetensors"):
    return {"name": name, "family": "sdxl", "desc": "d", "files": {"model": [repo, path, False]},
            "steps": 25, "size": SQUARE, "args": ["--cfg-scale", "6.0"]}


# ---------------------------------------------------------------- catalog
class Catalog(ImagesBase):
    def test_custom_models_empty_when_file_missing(self):
        self.assertFalse(app.CUSTOM_IMAGE_FILE.exists())
        self.assertEqual(app._custom_image_models(), {})
        self.assertEqual(list(app.image_models()), [Z])

    def test_custom_models_empty_when_json_corrupt(self):
        app.CUSTOM_IMAGE_FILE.write_text('{"broken": [', encoding="utf-8")
        self.assertEqual(app._custom_image_models(), {})
        self.assertEqual(list(app.image_models()), [Z])

    def test_custom_models_are_listed_after_builtins(self):
        self.write_custom({"my-xl": sdxl_entry(), "other": sdxl_entry("Other")})
        self.assertEqual(list(app.image_models()), [Z, "my-xl", "other"])
        self.assertEqual(app.image_models()["other"]["name"], "Other")

    def test_custom_entry_with_builtin_key_replaces_builtin(self):
        self.write_custom({Z: sdxl_entry("Replaced")})
        self.assertEqual(app.image_models()[Z]["name"], "Replaced")
        self.assertEqual(app.BUILTIN_IMAGE_MODELS[Z]["name"], "Z-Image Turbo", "the built-in table must not change")

    # BUG: _custom_image_models returns whatever valid JSON image_models.json holds; a list (or null) makes
    # image_models() raise TypeError ({**BUILTIN, **[]}), which breaks the whole Images tab instead of falling
    # back to the built-in models like a corrupt file does (app.py:2256-2264).
    def test_custom_file_holding_a_json_list_keeps_builtins(self):
        app.CUSTOM_IMAGE_FILE.write_text("[]", encoding="utf-8")
        self.assertEqual(list(app.image_models()), [Z])

    def test_builtin_file_layout_shares_companions(self):
        files = app.image_model_files(Z)
        self.assertEqual(list(files), ["diffusion-model", "llm", "vae"])
        self.assertEqual(files["diffusion-model"], app.IMAGE_MODEL_DIR / Z / "z_image_turbo-Q4_K.gguf")
        self.assertEqual(files["llm"], app.SHARED_DIR / "unsloth-qwen3-4b-instruct-2507-gguf"
                         / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
        self.assertEqual(files["vae"], app.SHARED_DIR / "comfy-org-z-image-turbo" / "ae.safetensors",
                         "only the file name is kept, not the repo's split_files/vae folders")

    def test_two_element_file_spec_lives_in_model_folder(self):
        self.write_custom({"two": {**sdxl_entry(), "files": {"model": ["me/two", "deep/dir/two.safetensors"]}}})
        self.assertEqual(app.image_model_files("two"), {"model": app.IMAGE_MODEL_DIR / "two" / "two.safetensors"})

    def test_unknown_key_files_raise_but_installed_is_false(self):
        with self.assertRaises(KeyError):
            app.image_model_files("no-such-model")
        self.assertFalse(app.image_model_installed("no-such-model"))

    def test_all_files_present_means_installed_and_listed(self):
        self.install()
        self.assertTrue(app.image_model_installed(Z))
        self.assertEqual(app.installed_image_models(), [("Z-Image Turbo", Z)])

    def test_no_files_means_not_installed(self):
        self.assertFalse(app.image_model_installed(Z))
        self.assertEqual(app.installed_image_models(), [])

    def test_installed_list_only_has_complete_custom_models(self):
        self.write_custom({"done": sdxl_entry("Done", path="done.safetensors"),
                           "half": sdxl_entry("Half", path="half.safetensors")})
        self.install("done")
        self.assertEqual(app.installed_image_models(), [("Done", "done")])
        self.install(Z)
        self.assertEqual(app.installed_image_models(), [("Z-Image Turbo", Z), ("Done", "done")])

    def test_migrate_moves_legacy_companions_to_shared(self):
        legacy = app.IMAGE_MODEL_DIR / Z
        legacy.mkdir(parents=True)
        for name in ("z_image_turbo-Q4_K.gguf", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "ae.safetensors"):
            (legacy / name).write_bytes(name.encode())
        app.migrate_shared_files()
        files = app.image_model_files(Z)
        self.assertEqual(files["llm"].read_bytes(), b"Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
        self.assertEqual(files["vae"].read_bytes(), b"ae.safetensors")
        self.assertFalse((legacy / "ae.safetensors").exists())
        self.assertTrue((legacy / "z_image_turbo-Q4_K.gguf").exists(), "the model's own file stays where it is")
        self.assertTrue(app.image_model_installed(Z))

    def test_migrate_keeps_legacy_copy_when_shared_exists(self):
        vae = app.image_model_files(Z)["vae"]
        vae.parent.mkdir(parents=True)
        vae.write_bytes(b"new")
        legacy = app.IMAGE_MODEL_DIR / Z / "ae.safetensors"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"old")
        app.migrate_shared_files()
        self.assertEqual(vae.read_bytes(), b"new", "an existing shared file is never overwritten")
        self.assertEqual(legacy.read_bytes(), b"old")

    def test_migrate_on_fresh_install_creates_nothing(self):
        app.migrate_shared_files()
        self.assertFalse(app.IMAGE_MODEL_DIR.exists())


def _missing_file_test(role):
    def test(self):
        self.install(skip=(role,))
        self.assertFalse(app.image_model_installed(Z))
        self.assertEqual(app.installed_image_models(), [])
        self.assertIn(Z, app.image_models())
    return test


for _role in ("diffusion-model", "llm", "vae"):
    setattr(Catalog, f"test_builtin_not_installed_without_{_role.replace('-', '_')}_file", _missing_file_test(_role))


# ---------------------------------------------------------------- slugs and families
class SlugAndFamily(unittest.TestCase):
    def test_slug_lowercases_and_joins_with_dashes(self):
        self.assertEqual(app._slug("City96/FLUX.1-schnell-gguf"), "city96-flux-1-schnell-gguf")

    def test_slug_drops_unicode_and_edge_dashes(self):
        self.assertEqual(app._slug("  Ünïcode Modèl!  "), "n-code-mod-l")
        self.assertEqual(app._slug("---"), "")
        self.assertEqual(app._slug(""), "")

    def test_slug_is_cut_to_80_without_trailing_dash(self):
        self.assertEqual(app._slug("a" * 200), "a" * 80)
        cut = app._slug("a" * 79 + "-bbbb")  # the 80th character is the dash
        self.assertEqual(cut, "a" * 79)


FAMILY_CASES = [
    ("tongyi_z_image_turbo", "Tongyi-MAI/Z-Image-Turbo", "z-image", ""),
    ("dotted_z_image", "someone/z.image-gguf", "z-image", ""),
    ("bfl_flux_schnell", "black-forest-labs/FLUX.1-schnell", "flux1-schnell", ""),
    ("flux_krea_defaults_to_dev", "someone/flux-krea", "flux1-dev", ""),
    ("juggernaut_xl", "RunDiffusion/Juggernaut-XL-v9", "sdxl", ""),
    ("realvis_xl", "SG161222/RealVisXL_V4.0", "sdxl", ""),
    ("illustrious", "OnomaAIResearch/Illustrious-xl-early-release-v0", "sdxl", ""),
    ("pony", "someone/pony-diffusion-v6", "sdxl", ""),
    ("uppercase_sdxl", "UPPER/SDXL-TURBO", "sdxl", ""),
    ("sdxl_base", "stabilityai/stable-diffusion-xl-base-1.0", "sdxl", ""),
    ("runwayml_v1_5", "runwayml/stable-diffusion-v1-5", "sd15", ""),
    ("deliberate", "XpucT/Deliberate", "sd15", ""),
    ("sd_1_dot_5", "someone/sd-1.5-fp16", "sd15", ""),
    ("sd3_unsupported", "stabilityai/stable-diffusion-3.5-large", None, "isn't supported yet"),
    ("hunyuan_video", "tencent/HunyuanVideo", None, "video model"),
    ("instruct_pix2pix", "timbrooks/instruct-pix2pix", None, "edits existing images"),
    ("sdxl_vae", "madebyollin/sdxl-vae-fp16-fix", None, "add-on or component"),
    ("flux2_klein", "black-forest-labs/FLUX.2-klein", None, "Hugging Face login"),
    ("empty_name", "", None, "Couldn't tell"),
]


def _family_test(repo, family, reason):
    def test(self):
        got, why = app.detect_family(repo)
        self.assertEqual(got, family, repo)
        if family:
            self.assertEqual(why, "")
        else:
            self.assertIn(reason, why)
    return test


for _name, _repo, _fam, _reason in FAMILY_CASES:
    setattr(SlugAndFamily, f"test_detect_family_{_name}", _family_test(_repo, _fam, _reason))


class FamilyFalsePositives(unittest.TestCase):
    # BUG: detect_family matches "vae" anywhere in the name, so SG161222/Realistic_Vision_V6.0_B1_noVAE, the main
    # Realistic Vision repo (a complete SD 1.5 checkpoint that the sd15 pattern explicitly names), is rejected as
    # "an add-on or component file" (app.py:2241 checked before app.py:2318).
    def test_realistic_vision_no_vae_is_sd15(self):
        self.assertEqual(app.detect_family("SG161222/Realistic_Vision_V6.0_B1_noVAE")[0], "sd15")


# ---------------------------------------------------------------- Hugging Face search
class RepoFiles(ImagesBase):
    def files(self, tree, family, repo="x/y"):
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(tree)) as get:
            return app.image_repo_files(repo, family), get

    def test_z_image_keeps_unsharded_gguf_in_subfolders(self):
        tree = [{"path": "z-Q4.gguf", "size": 5e9}, {"path": "z-Q8.safetensors", "size": 7e9},
                {"path": "split-00001-of-00003.gguf", "size": 4e9}, {"path": "sub/z-Q2.gguf", "size": 3e9},
                {"path": "text_encoder/q.gguf", "size": 2e9}]
        got, get = self.files(tree, "z-image", "leejet/Z-Image-Turbo-GGUF")
        self.assertEqual(got, [(3.0, "sub/z-Q2.gguf"), (5.0, "z-Q4.gguf")])
        self.assertEqual(get.call_args.kwargs["params"], {"recursive": "true"})
        self.assertTrue(get.call_args.args[0].endswith("/models/leejet/Z-Image-Turbo-GGUF/tree/main"))

    def test_checkpoint_size_threshold_is_0_8_gb(self):
        tree = [{"path": "a.safetensors", "size": 0.79e9}, {"path": "b.safetensors", "size": 0.8e9},
                {"path": "c.gguf", "size": 2e9}, {"path": "folder/d.safetensors", "size": 2e9}]
        got, _ = self.files(tree, "sd15")
        self.assertEqual([p for _, p in got], ["b.safetensors", "c.gguf"])

    def test_ckpt_is_kept_when_no_safetensors(self):
        got, _ = self.files([{"path": "m.ckpt", "size": 4e9}, {"path": "readme.md", "size": 2e9}], "sd15")
        self.assertEqual(got, [(4.0, "m.ckpt")])

    def test_component_files_are_skipped(self):
        names = ["refiner.safetensors", "controlnet.safetensors", "lora_x.safetensors", "inpaint.safetensors",
                 "mmproj.gguf", "base.safetensors"]
        got, _ = self.files([{"path": n, "size": 5e9} for n in names], "sdxl")
        self.assertEqual(got, [(5.0, "base.safetensors")])

    def test_listing_http_error_raises(self):
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(status=404)), \
                self.assertRaises(app.requests.HTTPError):
            app.image_repo_files("gone/repo", "sdxl")

    def test_remote_size_reads_content_length(self):
        with mock.patch.object(app.requests, "head", return_value=FakeResponse(headers={"content-length": "2500000000"})) as h:
            self.assertEqual(app._remote_size("a/b", "dir/f.gguf"), 2.5)
        self.assertEqual(h.call_args.args[0], "https://huggingface.co/a/b/resolve/main/dir/f.gguf")
        self.assertTrue(h.call_args.kwargs["allow_redirects"])

    def test_remote_size_without_header_is_zero(self):
        with mock.patch.object(app.requests, "head", return_value=FakeResponse(headers={})):
            self.assertEqual(app._remote_size("a/b", "f.gguf"), 0.0)

    def test_remote_size_network_error_is_zero(self):
        with mock.patch.object(app.requests, "head", side_effect=app.requests.ConnectionError("down")):
            self.assertEqual(app._remote_size("a/b", "f.gguf"), 0.0)


class Search(ImagesBase):
    def search(self, query, responses=None, side_effect=None):
        get = mock.Mock(side_effect=side_effect or list(responses or []))
        with mock.patch.object(app.requests, "get", get), mock.patch.object(app.time, "sleep") as sleep:
            outs = list(app.image_search(query))
        return outs, get, sleep

    def test_empty_query_asks_for_a_name(self):
        outs, get, _ = self.search("")
        self.assertEqual(len(outs), 1)
        self.assertIn("Enter a model name to search.", outs[0][2])
        self.assertFalse(outs[0][3]["interactive"])
        get.assert_not_called()

    def test_whitespace_query_asks_for_a_name(self):
        outs, get, _ = self.search("   \t ")
        self.assertIn("Enter a model name", outs[-1][2])
        get.assert_not_called()

    def test_link_with_tree_path_skips_the_api(self):
        outs, get, _ = self.search(f"https://huggingface.co/{FLUX_REPO}/tree/main")
        get.assert_not_called()
        repo = outs[-1][0]
        self.assertEqual(repo["choices"], [app.PLEASE_CHOOSE, (f"{FLUX_REPO}   (FLUX.1 schnell)", FLUX_REPO)])
        self.assertTrue(repo["interactive"])
        self.assertIn("Found 1 image models, 1 of them supported", outs[-1][2])

    def test_short_link_to_unsupported_repo_is_labelled(self):
        outs, _, _ = self.search("hf.co/Qwen/Qwen-Image")
        self.assertEqual(outs[-1][0]["choices"][1], ("Qwen/Qwen-Image   (not supported)", "Qwen/Qwen-Image"))
        self.assertIn("0 of them supported", outs[-1][2])

    def test_first_update_shows_loading_screen(self):
        outs, _, _ = self.search("flux", [FakeResponse([]), FakeResponse([])])
        self.assertEqual(outs[0][0]["choices"], [app.LOADING])
        self.assertTrue(outs[0][4]["visible"])
        self.assertIn("Looking for &quot;flux&quot; image models", outs[0][4]["value"])
        self.assertFalse(outs[-1][4]["visible"], "the loading screen is hidden at the end")

    def test_no_results_note_escapes_query(self):
        outs, get, _ = self.search("<b>zzz", [FakeResponse([]), FakeResponse([])])
        self.assertEqual(get.call_count, 2)
        self.assertIn("No image models named &quot;&lt;b&gt;zzz&quot; found", outs[-1][2])
        self.assertFalse(outs[-1][0]["interactive"])

    def test_http_500_reports_failure(self):
        outs, _, _ = self.search("sdxl", [FakeResponse(status=500)])
        self.assertIn("Search failed: HTTP 500", outs[-1][2])
        self.assertEqual(outs[-1][1]["choices"], [app.PLEASE_CHOOSE])

    def test_http_401_reports_failure_without_retry(self):
        outs, get, sleep = self.search("sdxl", [FakeResponse(status=401)])
        self.assertIn("Search failed: HTTP 401", outs[-1][2])
        self.assertEqual(get.call_count, 1)
        sleep.assert_not_called()

    def test_http_429_is_retried_then_reported(self):
        outs, get, sleep = self.search("sdxl", [FakeResponse(status=429, headers={"Retry-After": "1"})] * 4)
        self.assertEqual(get.call_count, 4)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.0, 1.0, 1.0])
        self.assertIn("Search failed: HTTP 429", outs[-1][2])

    def test_network_error_reports_failure(self):
        outs, _, _ = self.search("sdxl", side_effect=app.requests.ConnectionError("no route to host"))
        self.assertIn("Search failed: no route to host", outs[-1][2])
        self.assertFalse(outs[-1][3]["interactive"])

    def test_results_are_deduplicated_and_supported_first(self):
        first = FakeResponse([{"id": FLUX_REPO}, {"id": "some/random-thing"}])
        second = FakeResponse([{"id": FLUX_REPO}, {"id": "stabilityai/sdxl-turbo"}])
        outs, get, _ = self.search("turbo", [first, second])
        self.assertEqual([c[1] for c in outs[-1][0]["choices"][1:]],
                         [FLUX_REPO, "stabilityai/sdxl-turbo", "some/random-thing"])
        self.assertIn("Found 3 image models, 2 of them supported", outs[-1][2])
        params = [c.kwargs["params"] for c in get.call_args_list]
        self.assertEqual(params[0]["filter"], "gguf")
        self.assertNotIn("filter", params[1])
        self.assertEqual({p["pipeline_tag"] for p in params}, {"text-to-image"})
        self.assertFalse(outs[-1][3]["interactive"], "download waits until a version is chosen")


class Versions(ImagesBase):
    FLUX_TREE = [{"path": "flux1-schnell-Q4_K_S.gguf", "size": 6.8e9}, {"path": "flux1-schnell-Q8_0.gguf", "size": 12.7e9}]

    def versions(self, repo, tree=None, get_effect=None, head_len="1000000000"):
        get = mock.Mock(return_value=FakeResponse(tree), side_effect=get_effect)
        head = mock.Mock(return_value=FakeResponse(headers={"content-length": head_len}))
        with mock.patch.object(app.requests, "get", get), mock.patch.object(app.requests, "head", head):
            return list(app.image_versions(repo)), get, head

    def test_no_repo_resets_fields(self):
        outs, get, _ = self.versions("")
        self.assertEqual(len(outs), 1)
        self.assertEqual(outs[0][0]["choices"], [app.PLEASE_CHOOSE])
        get.assert_not_called()

    def test_unsupported_repo_shows_reason(self):
        outs, get, _ = self.versions("Wan-AI/Wan2.1-T2V-14B")
        self.assertIn("This is a video model", outs[-1][1])
        get.assert_not_called()

    def test_listing_error_is_reported(self):
        outs, _, _ = self.versions(FLUX_REPO, get_effect=app.requests.Timeout("timed out"))
        self.assertIn("Could not list files: timed out", outs[-1][1])
        self.assertFalse(outs[-1][2]["interactive"])

    def test_flux_repo_without_gguf(self):
        outs, _, head = self.versions(FLUX_REPO, [{"path": "flux1-schnell.safetensors", "size": 23e9}])
        self.assertIn("no GGUF model files the image engine can use", outs[-1][1])
        head.assert_not_called()

    def test_sdxl_repo_without_single_file(self):
        outs, _, _ = self.versions("stabilityai/sdxl-turbo", [{"path": "unet/model.safetensors", "size": 5e9}])
        self.assertIn("no complete single-file models", outs[-1][1])

    def test_recommends_version_and_sums_companions(self):
        outs, _, head = self.versions(FLUX_REPO, self.FLUX_TREE)
        choices, note, dl, loading = outs[-1]
        self.assertEqual(choices["value"], FLUX_FILE)
        self.assertEqual(choices["choices"][1], ("flux1-schnell-Q4_K_S   6.8 GB   (Recommended)", FLUX_FILE))
        self.assertEqual(choices["choices"][2], ("flux1-schnell-Q8_0   12.7 GB", "flux1-schnell-Q8_0.gguf"))
        self.assertIn("FLUX.1 schnell model. Auto-selected the largest version under 7.5 GB.", note)
        self.assertIn("It also needs 3 companion files (3.0 GB to download).", note)
        self.assertIn("Total download about 9.8 GB.", note)
        self.assertTrue(dl["interactive"])
        self.assertEqual(head.call_count, 3)
        self.assertTrue(outs[0][3]["visible"], "a loading screen shows while the files are listed")

    def test_installed_companions_are_not_downloaded_again(self):
        vae = app.SHARED_DIR / "comfy-org-z-image-turbo" / "ae.safetensors"
        vae.parent.mkdir(parents=True)
        vae.write_bytes(b"x")
        outs, _, head = self.versions(FLUX_REPO, self.FLUX_TREE)
        self.assertIn("(2.0 GB to download, 1 already installed)", outs[-1][1])
        self.assertEqual(head.call_count, 2)

    def test_sdxl_has_no_companion_sentence(self):
        outs, _, head = self.versions("stabilityai/sdxl-turbo", [{"path": "sd_xl_turbo_1.0_fp16.safetensors", "size": 6.9e9}])
        self.assertNotIn("companion", outs[-1][1])
        self.assertIn("Total download about 6.9 GB.", outs[-1][1])
        head.assert_not_called()

    def test_all_versions_too_big_picks_smallest(self):
        tree = [{"path": "flux1-schnell-F16.gguf", "size": 23.8e9}, {"path": "flux1-schnell-Q8_0.gguf", "size": 12.7e9}]
        outs, _, _ = self.versions(FLUX_REPO, tree)
        self.assertEqual(outs[-1][0]["value"], "flux1-schnell-Q8_0.gguf")

    def test_version_change_needs_repo_and_path(self):
        self.assertTrue(app.image_version_changed(FLUX_REPO, FLUX_FILE)["interactive"])
        self.assertFalse(app.image_version_changed(FLUX_REPO, "")["interactive"])
        self.assertFalse(app.image_version_changed("", FLUX_FILE)["interactive"])
        self.assertFalse(app.image_version_changed(None, None)["interactive"])


class SearchedDownload(ImagesBase):
    def test_missing_repo_is_refused(self):
        note, ver = app.download_searched_image_model("", FLUX_FILE)
        self.assertIn("Choose a repository before downloading.", note)
        self.assertEqual(ver, app.gr.update())
        self.assertFalse(app.CUSTOM_IMAGE_FILE.exists())

    def test_missing_version_is_refused(self):
        note, _ = app.download_searched_image_model(FLUX_REPO, None)
        self.assertIn("Choose a version before downloading.", note)
        self.assertFalse(app.CUSTOM_IMAGE_FILE.exists())

    def test_unsupported_repo_is_refused(self):
        with mock.patch.object(app, "start_download") as start:
            note, _ = app.download_searched_image_model("Qwen/Qwen-Image", "qwen-image-Q4_K_M.gguf")
        self.assertIn("Qwen-Image isn&#x27;t supported yet", note)
        start.assert_not_called()
        self.assertFalse(app.CUSTOM_IMAGE_FILE.exists())

    def test_sdxl_turbo_gets_few_steps_and_low_cfg(self):
        key = self.add_searched("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors")
        self.assertEqual(key, "stabilityai-sdxl-turbo-sd-xl-turbo-1-0-fp16")
        m = app.image_models()[key]
        self.assertEqual(m["name"], "sdxl-turbo (fp16)")
        self.assertEqual((m["steps"], m["args"]), (6, ["--cfg-scale", "1.5", "--offload-to-cpu"]))
        self.assertEqual(m["files"], {"model": ["stabilityai/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors", False]})

    def test_sd15_lightning_gets_few_steps(self):
        key = self.add_searched("someone/sd15-lightning", "model.safetensors")
        m = app.image_models()[key]
        self.assertEqual((m["name"], m["steps"], m["args"], m["size"]),
                         ("sd15-lightning", 6, ["--cfg-scale", "1.5"], SMALL))

    def test_full_sdxl_checkpoint_keeps_family_defaults(self):
        repo, path = "RunDiffusion/Juggernaut-XL-v9", "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
        with mock.patch.object(app, "start_download", return_value=True) as start:
            note, ver = app.download_searched_image_model(repo, path)
        key = "rundiffusion-juggernaut-xl-v9-juggernaut-xl-v9-rundiffusionphoto-v2"
        start.assert_called_once_with(f"image:{key}")
        m = app.image_models()[key]
        self.assertEqual((m["name"], m["steps"], m["args"]), ("Juggernaut-XL-v9", 25, ["--cfg-scale", "6.0", "--offload-to-cpu"]))
        self.assertEqual(m["desc"], f"Stable Diffusion XL model from {repo}.")
        self.assertIn("Downloading Juggernaut-XL-v9.", note)
        self.assertAlmostEqual(ver, time.time(), delta=5)

    def test_unsloth_dynamic_quant_in_name(self):
        key = self.add_searched("unsloth/FLUX.1-dev-GGUF", "flux1-dev-UD-Q4_K_XL.gguf")
        m = app.image_models()[key]
        self.assertEqual(m["name"], "FLUX.1-dev (UD-Q4_K_XL)")
        self.assertEqual(m["files"]["clip_l"], [*app.COMPANIONS["clip-l"], True])

    def test_second_model_keeps_the_first(self):
        k1 = self.add_searched()
        k2 = self.add_searched("leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q8_0.gguf")
        saved = json.loads(app.CUSTOM_IMAGE_FILE.read_text())
        self.assertEqual(list(saved), [k1, k2])
        self.assertEqual(saved[k2]["name"], "Z-Image-Turbo (Q8_0)")

    def test_already_installed_model_is_not_downloaded_again(self):
        key = self.add_searched("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors")
        self.install(key)
        with mock.patch.object(app, "start_download") as start:
            note, _ = app.download_searched_image_model("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors")
        self.assertIn("sdxl-turbo (fp16) is already installed.", note)
        start.assert_not_called()


# ---------------------------------------------------------------- defaults when switching models
class Defaults(ImagesBase):
    def test_entry_without_steps_or_size_uses_fallbacks(self):
        entry = sdxl_entry()
        del entry["steps"], entry["size"]
        self.write_custom({"bare": entry})
        steps, size = app.image_model_defaults("bare")
        self.assertEqual((steps["value"], size["value"]), (8, SQUARE))

    def test_unknown_key_changes_nothing(self):
        self.assertEqual(app.image_model_defaults("deleted-model"), (app.gr.update(), app.gr.update()))

    def test_no_model_selected_changes_nothing(self):
        self.assertEqual(app.image_model_defaults(None), (app.gr.update(), app.gr.update()))


DEFAULT_CASES = [
    ("z_image", "leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q8_0.gguf", "z-image", 8, SQUARE),
    ("flux_schnell", FLUX_REPO, FLUX_FILE, "flux1-schnell", 4, SQUARE),
    ("flux_dev", "city96/FLUX.1-dev-gguf", "flux1-dev-Q4_0.gguf", "flux1-dev", 20, SQUARE),
    ("sdxl_base", "stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", "sdxl", 25, SQUARE),
    ("sd15", "stable-diffusion-v1-5/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors", "sd15", 25, SMALL),
]


def _defaults_test(repo, path, family, steps, size):
    def test(self):
        key = self.add_searched(repo, path)
        self.assertEqual(app.image_models()[key]["family"], family)
        got_steps, got_size = app.image_model_defaults(key)
        self.assertEqual((got_steps["value"], got_size["value"]), (steps, size))
    return test


for _n, *_args in DEFAULT_CASES:
    setattr(Defaults, f"test_defaults_for_searched_{_n}_model", _defaults_test(*_args))


# ---------------------------------------------------------------- background download worker
class DownloadWorker(ImagesBase):
    SIZES = {"z_image_turbo-Q4_K.gguf": 5, "Qwen3-4B-Instruct-2507-Q4_K_M.gguf": 3, "ae.safetensors": 4}

    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.dict(app._dl_finished, {"count": 0}))
        self.enterContext(mock.patch.dict(app._priority, {"ref": None}))
        self.fetched = []

    def start(self, key=Z, extra=None):
        ref = f"image:{key}"
        d = {"state": "queued", "msg": "Queued", "done": 0, "total": 0, "cancel": False}
        self.enterContext(mock.patch.dict(app._downloads, {ref: d, **(extra or {})}))
        app._set_pending(ref, True)
        return ref, d

    def head_ok(self, url, **kw):
        return FakeResponse(headers={"content-length": str(self.SIZES[url.rsplit("/", 1)[1]])})

    def fetch_ok(self, url, dest, size, should_stop, on_progress):
        self.fetched.append(url)
        dest.write_bytes(b"x" * size)
        on_progress()
        return True

    def run_worker(self, ref, head=None, fetch=None, sleep=None):
        with mock.patch.object(app.requests, "head", side_effect=head or self.head_ok) as h, \
                mock.patch.object(app, "fetch_file", side_effect=fetch or self.fetch_ok), \
                mock.patch.object(app.time, "sleep", side_effect=sleep):
            app._image_download_worker(ref)
        return h

    def test_downloads_every_file_and_finishes(self):
        ref, d = self.start()
        self.run_worker(ref)
        self.assertEqual((d["state"], d["done"], d["total"]), ("done", 12, 12))
        self.assertTrue(app.image_model_installed(Z))
        self.assertEqual(app._pending(), [])
        self.assertEqual(app._dl_finished["count"], 1)
        self.assertIn("https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q4_K.gguf", self.fetched)
        self.assertIn("https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
                      self.fetched)

    def test_complete_files_are_skipped_wrong_sized_ones_refetched(self):
        files = app.image_model_files(Z)
        for role, data in (("vae", b"x" * 4), ("llm", b"x")):
            files[role].parent.mkdir(parents=True, exist_ok=True)
            files[role].write_bytes(data)
        ref, d = self.start()
        self.run_worker(ref)
        self.assertEqual(len(self.fetched), 2)
        self.assertFalse(any(u.endswith("ae.safetensors") for u in self.fetched))
        self.assertEqual(files["llm"].stat().st_size, 3)
        self.assertEqual(d["state"], "done")

    def test_missing_remote_file_fails_permanently(self):
        ref, d = self.start()
        self.run_worker(ref, head=lambda url, **kw: FakeResponse(status=404))
        self.assertEqual(d["state"], "failed")
        self.assertTrue(d["msg"].startswith("Checking z_image_turbo-Q4_K.gguf failed (HTTP 404)"), d["msg"])
        self.assertEqual(self.fetched, [])
        self.assertEqual(app._pending(), [])

    def test_server_error_waits_and_can_be_cancelled(self):
        ref, d = self.start()
        seen = []

        def sleep(secs):
            seen.append((secs, d["state"], d["msg"]))
            d["cancel"] = True
        self.run_worker(ref, head=lambda url, **kw: FakeResponse(status=503), sleep=sleep)
        self.assertEqual(seen, [(5, "waiting", "Connection interrupted, resuming in 5s")])
        self.assertEqual((d["state"], d["msg"]), ("cancelled", "Cancelled"))
        self.assertEqual(app._pending(), [])

    def test_repeated_connection_errors_back_off(self):
        ref, d = self.start()
        waits = []

        def sleep(secs):
            waits.append(secs)
            if len(waits) == 3:
                d["cancel"] = True
        self.run_worker(ref, head=mock.Mock(side_effect=app.requests.ConnectionError("reset")), sleep=sleep)
        self.assertEqual(waits, [5, 10, 15])

    def test_pause_during_fetch_keeps_it_paused_for_later(self):
        ref, d = self.start()

        def fetch(url, dest, size, should_stop, on_progress):
            d["pause"] = True
            return False
        self.run_worker(ref, fetch=fetch)
        self.assertEqual((d["state"], d["msg"]), ("paused", "Paused"))
        self.assertEqual(app._pending(), [(ref, True)])
        self.assertFalse(app.image_model_installed(Z))

    def test_waits_for_priority_download(self):
        ref, d = self.start(extra={"other": {"state": "downloading", "cancel": False}})
        app._priority["ref"] = "other"
        seen = []

        def sleep(secs):
            seen.append(d["msg"])
            d["cancel"] = True
        h = self.run_worker(ref, sleep=sleep)
        self.assertEqual(seen, ["Paused for priority download"])
        h.assert_not_called()

    def test_gated_file_fails_with_reason(self):
        ref, d = self.start()

        def fetch(url, dest, size, should_stop, on_progress):
            raise app.PermanentDownloadError("Downloading ae.safetensors failed (HTTP 403)")
        self.run_worker(ref, fetch=fetch)
        self.assertEqual((d["state"], d["msg"]), ("failed", "Downloading ae.safetensors failed (HTTP 403)"))

    def test_searched_flux_model_downloads_companions_to_shared_folders(self):
        key = self.add_searched()
        self.SIZES = {FLUX_FILE: 2, "clip_l.safetensors": 2, "t5-v1_1-xxl-encoder-Q4_K_M.gguf": 2, "ae.safetensors": 2}
        ref, d = self.start(key)
        self.run_worker(ref)
        self.assertEqual(d["state"], "done")
        self.assertTrue((app.SHARED_DIR / "comfyanonymous-flux-text-encoders" / "clip_l.safetensors").exists())
        self.assertTrue((app.IMAGE_MODEL_DIR / key / FLUX_FILE).exists())


# ---------------------------------------------------------------- model table, download and delete
class ModelTable(ImagesBase):
    def test_version_html_not_installed(self):
        html = app.image_models_version_html(Z)
        self.assertEqual(html.count('class="mt-cell"'), 4)
        self.assertIn('<div class="mt-cell">-</div><div class="mt-cell">Not installed</div>', html)
        self.assertIn("Z-Image Turbo", html)

    def test_version_html_installed_sums_sizes(self):
        sizes = {"diffusion-model": 1_000_000_000, "llm": 400_000_000, "vae": 100_000_000}
        for role, p in app.image_model_files(Z).items():
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as fh:
                fh.truncate(sizes[role])  # sparse: no real disk space used
        html = app.image_models_version_html(Z)
        self.assertIn('<div class="mt-cell">1.5 GB</div><div class="mt-cell">Installed</div>', html)

    def test_version_html_partial_download_shows_size_but_not_installed(self):
        p = app.image_model_files(Z)["diffusion-model"]
        p.parent.mkdir(parents=True)
        with open(p, "wb") as fh:
            fh.truncate(300_000_000)
        self.assertIn('<div class="mt-cell">0.3 GB</div><div class="mt-cell">Not installed</div>',
                      app.image_models_version_html(Z))

    def test_version_html_escapes_custom_text(self):
        self.write_custom({"evil": {**sdxl_entry('<img src=x onerror="a">'), "desc": "a & b"}})
        html = app.image_models_version_html("evil")
        self.assertNotIn("<img", html)
        self.assertIn("&lt;img src=x onerror=&quot;a&quot;&gt;", html)
        self.assertIn("a &amp; b", html)

    def test_download_installed_model_says_so(self):
        self.install()
        with mock.patch.object(app, "start_download") as start:
            self.assertIn("Z-Image Turbo is already installed.", app.download_image_model(Z))
        start.assert_not_called()

    def test_download_while_downloading_says_so(self):
        with mock.patch.object(app, "start_download", return_value=False):
            self.assertIn("Z-Image Turbo is already downloading.", app.download_image_model(Z))

    def test_download_starts_image_ref(self):
        with mock.patch.object(app, "start_download", return_value=True) as start:
            note = app.download_image_model(Z)
        start.assert_called_once_with("image:z-image-turbo")
        self.assertIn("Progress is shown under Downloads on the Models tab.", note)

    def test_delete_unknown_key_does_nothing(self):
        self.install()
        self.assertEqual(app.delete_image_model("never-existed", True), (app.gr.update(), app.gr.update()))
        self.assertTrue(app.image_model_installed(Z))

    def test_delete_unconfirmed_keeps_files(self):
        self.install()
        self.assertEqual(app.delete_image_model(Z, False), (app.gr.update(), app.gr.update()))
        self.assertTrue(all(p.exists() for p in app.image_model_files(Z).values()))

    def test_delete_builtin_removes_files_but_keeps_entry(self):
        self.install()
        files = app.image_model_files(Z)
        note, ver = app.delete_image_model(Z, True)
        self.assertIn("Deleted Z-Image Turbo.", note)
        self.assertIsInstance(ver, float)
        self.assertFalse(any(p.exists() for p in files.values()))
        self.assertFalse((app.IMAGE_MODEL_DIR / Z).exists())
        self.assertIn(Z, app.image_models(), "built-in models stay available to download again")
        self.assertFalse(app.CUSTOM_IMAGE_FILE.exists())

    def test_delete_builtin_keeps_vae_used_by_installed_flux(self):
        key = self.add_searched()
        self.install(key)
        self.install(Z)
        files = app.image_model_files(Z)
        app.delete_image_model(Z, True)
        self.assertTrue(files["vae"].exists(), "FLUX still needs the shared VAE")
        self.assertFalse(files["llm"].exists(), "only Z-Image used the Qwen3 encoder")
        self.assertTrue(app.image_model_installed(key))

    def test_delete_custom_removes_only_its_entry(self):
        self.write_custom({"a": sdxl_entry("A", path="a.safetensors"), "b": sdxl_entry("B", path="b.safetensors")})
        self.install("a")
        self.install("b")
        app.delete_image_model("a", True)
        self.assertEqual(list(json.loads(app.CUSTOM_IMAGE_FILE.read_text())), ["b"])
        self.assertFalse((app.IMAGE_MODEL_DIR / "a").exists())
        self.assertTrue(app.image_model_installed("b"))

    def test_delete_twice_second_time_is_harmless(self):
        self.write_custom({"a": sdxl_entry("A")})
        app.delete_image_model("a", True)
        self.assertEqual(app.delete_image_model("a", True), (app.gr.update(), app.gr.update()))


# ---------------------------------------------------------------- gallery and progress bar
class Gallery(ImagesBase):
    def img(self, name, mtime=None, meta=None, raw=None):
        app.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        p = app.IMAGES_DIR / name
        p.write_bytes(PNG)
        if meta is not None:
            p.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
        if raw is not None:
            p.with_suffix(".json").write_text(raw, encoding="utf-8")
        if mtime:
            os.utime(p, (mtime, mtime))
        return p

    META = {"prompt": "a red fox", "model": "Z-Image Turbo", "seed": 7, "width": 768, "height": 1024, "steps": 8,
            "seconds": 12.3}

    def test_no_images_folder_shows_empty_message(self):
        self.assertFalse(app.IMAGES_DIR.exists())
        self.assertEqual(app.gallery_html(), '<div class="gallery-empty">No images yet. Generate one above.</div>')

    def test_only_non_png_files_count_as_empty(self):
        app.IMAGES_DIR.mkdir(parents=True)
        (app.IMAGES_DIR / "a.jpg").write_bytes(b"x")
        (app.IMAGES_DIR / "orphan.json").write_text("{}")
        self.assertIn("No images yet", app.gallery_html())

    def test_shows_at_most_120_newest(self):
        for i in range(125):
            self.img(f"img-{i:03d}.png", mtime=1_000_000 + i)
        html = app.gallery_html()
        self.assertEqual(html.count('class="g-tile"'), 120)
        self.assertIn('data-name="img-124.png"', html)
        self.assertIn('data-name="img-005.png"', html)
        self.assertNotIn('data-name="img-004.png"', html)

    def test_newest_first_by_modification_time(self):
        self.img("b.png", mtime=2000)
        self.img("a.png", mtime=3000)
        self.img("c.png", mtime=1000)
        html = app.gallery_html()
        order = sorted(("a.png", "b.png", "c.png"), key=lambda n: html.index(f'data-name="{n}"'))
        self.assertEqual(order, ["a.png", "b.png", "c.png"])

    def test_missing_settings_file_shows_file_name(self):
        self.img("20260101-120000-5.png")
        html = app.gallery_html()
        self.assertIn('title=""', html)
        self.assertIn('data-detail="20260101-120000-5.png"', html)

    def test_corrupt_settings_file_shows_file_name(self):
        self.img("x.png", raw="{not json")
        self.assertIn('data-detail="x.png"', app.gallery_html())

    def test_settings_missing_a_key_falls_back_to_file_name(self):
        meta = dict(self.META)
        del meta["seconds"]
        self.img("y.png", meta=meta)
        html = app.gallery_html()
        self.assertIn('data-detail="y.png"', html)
        self.assertIn('data-prompt=""', html)

    def test_detail_lists_model_seed_size_steps_time(self):
        p = self.img("z.png", meta=self.META)
        html = app.gallery_html()
        self.assertIn('data-detail="Z-Image Turbo · seed 7 · 768 x 1024 · 8 steps · 12.3 s"', html)
        self.assertIn('title="a red fox"', html)
        self.assertIn(f'src="/gradio_api/file={p}"', html)

    def test_prompt_and_file_name_are_attribute_escaped(self):
        self.img("a&b.png", meta={**self.META, "prompt": 'say "hi" & <wave>'})
        html = app.gallery_html()
        self.assertIn('data-prompt="say &quot;hi&quot; &amp; &lt;wave&gt;"', html)
        self.assertIn('data-name="a&amp;b.png"', html)
        self.assertNotIn("<wave>", html)


class ProgressHtml(unittest.TestCase):
    def test_percent_is_rounded_and_bar_width_has_one_decimal(self):
        html = app.image_progress_html("Generating, step 1 of 3", 100 / 3, "Elapsed 1s")
        self.assertIn('<span class="sp-pct">33%</span>', html)
        self.assertIn('style="width:33.3%"', html)
        self.assertIn('class="scanprog running"', html)

    def test_title_is_escaped_but_meta_is_markup(self):
        html = app.image_progress_html("<b>x</b>", 50, "2s &nbsp;·&nbsp; seed 1", "done")
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", html)
        self.assertIn("2s &nbsp;·&nbsp; seed 1", html)
        self.assertIn('class="scanprog done"', html)


# ---------------------------------------------------------------- generation with a fake engine
class Generate(ImagesBase):
    def setUp(self):
        super().setUp()
        self.sd = self.tmp / "sd"
        self.sd.mkdir()
        engine = self.sd / "sd-cli"
        engine.write_text(FAKE_ENGINE)
        engine.chmod(0o755)
        self.enterContext(mock.patch.object(app, "SD_DIR", self.sd))
        self.enterContext(mock.patch.object(app, "SD_CLI", engine))
        self.unload = self.enterContext(mock.patch.object(app, "unload_models"))
        self.log = self.tmp / "engine-args.json"
        self.install()

    def run_gen(self, prompt="a cat", size=SMALL, steps=2, seed=7, key=Z, **env):
        env = {"FAKE_SD_LOG": str(self.log), **{f"FAKE_SD_{k.upper()}": str(v) for k, v in env.items()}}
        with mock.patch.dict(os.environ, env):
            return list(app.generate_image(key, prompt, size, steps, seed))

    def argv(self):
        return json.loads(self.log.read_text(encoding="utf-8"))["argv"]

    def arg(self, flag):
        a = self.argv()
        return a[a.index(flag) + 1]

    def meta(self, outs):
        return json.loads(Path(outs[-1][4]).with_suffix(".json").read_text(encoding="utf-8"))

    def test_missing_model_file_asks_for_download(self):
        app.image_model_files(Z)["llm"].unlink()
        outs = self.run_gen()
        self.assertEqual(outs, [(app.IMAGE_IDLE, app.note_html("Download an image model first (Image models below).",
                                                                "error"), *[app.gr.update()] * 4)])
        self.assertFalse(self.log.exists(), "the engine must not start")
        self.unload.assert_not_called()

    def test_empty_prompt_is_refused(self):
        outs = self.run_gen(prompt="")
        self.assertIn("Describe the image you want.", outs[-1][1])
        self.assertFalse(self.log.exists())

    def test_none_prompt_is_refused(self):
        outs = self.run_gen(prompt=None)
        self.assertIn("Describe the image you want.", outs[-1][1])
        self.assertEqual(len(outs), 1)

    def test_unicode_prompt_reaches_engine_and_settings(self):
        prompt = "un chat 🐱 sous la pluie — 雨の夜"
        outs = self.run_gen(prompt=prompt)
        self.assertIn("Image ready", outs[-1][0])
        self.assertEqual(self.arg("-p"), prompt)
        self.assertEqual(self.meta(outs)["prompt"], prompt)

    def test_long_prompt_is_passed_whole(self):
        prompt = "a very detailed castle, " * 800
        outs = self.run_gen(prompt=prompt)
        self.assertEqual(self.arg("-p"), prompt.strip())
        self.assertEqual(len(self.meta(outs)["prompt"]), len(prompt.strip()))

    def test_prompt_is_trimmed(self):
        outs = self.run_gen(prompt="\n  a boat at dawn \t ")
        self.assertEqual(self.arg("-p"), "a boat at dawn")
        self.assertEqual(self.meta(outs)["prompt"], "a boat at dawn")

    def test_minimum_one_step(self):
        outs = self.run_gen(steps=1)
        self.assertEqual(self.arg("--steps"), "1")
        self.assertEqual(self.meta(outs)["steps"], 1)
        self.assertIn("1 steps", outs[-1][0])

    def test_maximum_thirty_steps(self):
        outs = self.run_gen(steps=30)
        self.assertEqual(self.arg("--steps"), "30")
        self.assertEqual(self.meta(outs)["steps"], 30)

    def test_fractional_steps_are_truncated(self):
        outs = self.run_gen(steps=3.0)
        self.assertEqual(self.arg("--steps"), "3")
        self.assertIn("Image ready", outs[-1][0])

    def test_unknown_size_label_uses_1024_square(self):
        outs = self.run_gen(size="Gigantic 9999 x 9999")
        self.assertEqual((self.arg("-W"), self.arg("-H")), ("1024", "1024"))
        self.assertEqual((self.meta(outs)["width"], self.meta(outs)["height"]), (1024, 1024))

    def test_seed_minus_one_picks_random_seed(self):
        with mock.patch.object(app.os, "urandom", return_value=b"\xff\xff\xff\xff"):
            outs = self.run_gen(seed=-1)
        self.assertEqual(self.arg("--seed"), str(2**31 - 1))
        self.assertTrue(outs[-1][4].endswith(f"-{2**31 - 1}.png"))
        self.assertIn(f"seed {2**31 - 1}", outs[-1][0])

    def test_blank_seed_picks_random_seed(self):
        with mock.patch.object(app.os, "urandom", return_value=b"\x00\x00\x01\x00"):
            outs = self.run_gen(seed="")
        self.assertEqual(self.arg("--seed"), "256")
        self.assertEqual(self.meta(outs)["seed"], 256)

    def test_fixed_seed_is_used_and_named(self):
        outs = self.run_gen(seed=123456)
        self.assertEqual(self.arg("--seed"), "123456")
        self.assertRegex(Path(outs[-1][4]).name, r"^\d{8}-\d{6}-123456\.png$")

    def test_seed_zero_is_a_fixed_seed(self):
        with mock.patch.object(app.os, "urandom", side_effect=AssertionError("must not draw a random seed")):
            outs = self.run_gen(seed=0)
        self.assertEqual(self.meta(outs)["seed"], 0)

    def test_engine_failure_shows_last_line(self):
        outs = self.run_gen(exit=3, msg="error: out of memory")
        progress = outs[-1][0]
        self.assertIn("Generation failed", progress)
        self.assertIn("error: out of memory", progress)
        self.assertIn('class="scanprog stopped"', progress)
        self.assertIn("90%", progress, "progress stays where sampling ended")
        self.assertEqual(outs[-1][4], app.gr.update(), "the previous result stays")
        self.assertEqual(list(app.IMAGES_DIR.glob("*.json")), [])
        self.assertIsNone(app._image_proc["p"])

    def test_engine_success_without_image_is_a_failure(self):
        outs = self.run_gen(noout=1)
        self.assertIn("Generation failed", outs[-1][0])
        self.assertIn("2/2 - 0.10s/it", outs[-1][0])
        self.assertEqual(list(app.IMAGES_DIR.iterdir()), [])

    def test_engine_killed_by_signal_shows_stopped(self):
        outs = self.run_gen(signal=1)
        self.assertIn("Stopped", outs[-1][0])
        self.assertNotIn("Generation failed", outs[-1][0])
        self.assertTrue(outs[-1][2]["interactive"])

    def test_failure_detail_strips_ansi_and_escapes_html(self):
        outs = self.run_gen(exit=1, msg="\x1b[31m<oom> & crash\x1b[0m")
        self.assertIn("&lt;oom&gt; &amp; crash", outs[-1][0])
        self.assertNotIn("\x1b", outs[-1][0])

    def test_missing_engine_names_its_folder(self):
        missing = self.tmp / "no-engine"
        with mock.patch.object(app, "SD_DIR", missing), mock.patch.object(app, "SD_CLI", missing / "sd-cli"):
            outs = self.run_gen()
        self.assertEqual(len(outs), 1)
        self.assertIn(f"The image engine is missing from {missing}.", outs[0][1])
        self.unload.assert_not_called()

    def test_running_scan_blocks_generation(self):
        busy = mock.Mock(**{"poll.return_value": None})
        with mock.patch.dict(app._proc, {"p": busy}):
            outs = self.run_gen()
        self.assertEqual(outs[0][0], app.IMAGE_IDLE)
        self.assertIn("A scan is using the GPU", outs[0][1])
        self.assertFalse(self.log.exists())

    def test_finished_scan_does_not_block(self):
        done = mock.Mock(**{"poll.return_value": 0})
        with mock.patch.dict(app._proc, {"p": done}):
            outs = self.run_gen()
        self.assertIn("Image ready", outs[-1][0])

    def test_second_generation_is_refused_while_one_runs(self):
        app._image_proc["p"] = mock.Mock(**{"poll.return_value": None})
        outs = self.run_gen()
        self.assertEqual(outs[0][0], app.gr.update(), "the running generation's progress is left alone")
        self.assertIn("An image is already being generated.", outs[0][1])
        self.assertFalse(self.log.exists())

    def test_command_has_model_files_preset_args_and_gpu_env(self):
        self.run_gen()
        a = self.argv()
        files = app.image_model_files(Z)
        self.assertEqual(a[:6], ["--diffusion-model", str(files["diffusion-model"]), "--llm", str(files["llm"]),
                                 "--vae", str(files["vae"])])
        self.assertEqual(a[-4:], ["--cfg-scale", "1.0", "--diffusion-fa", "--offload-to-cpu"])
        logged = json.loads(self.log.read_text())
        self.assertEqual((logged["ld"], logged["vk"]), (str(self.sd), "0"))
        self.unload.assert_called_once()

    def test_custom_flux_model_passes_its_encoders(self):
        key = self.add_searched()
        self.install(key)
        outs = self.run_gen(key=key, steps=4)
        a = self.argv()
        for flag in ("--diffusion-model", "--clip_l", "--t5xxl", "--vae"):
            self.assertIn(flag, a)
        self.assertEqual(a[a.index("--sampling-method") + 1], "euler")
        self.assertEqual(self.meta(outs)["model"], "FLUX.1-schnell (Q4_K_S)")

    def test_button_states_and_gallery_update(self):
        outs = self.run_gen()
        first, last = outs[0], outs[-1]
        self.assertIn("Loading model", first[0])
        self.assertFalse(first[2]["interactive"])
        self.assertTrue(first[3]["visible"], "the stop button appears")
        self.assertTrue(last[2]["interactive"])
        self.assertFalse(last[3]["visible"])
        self.assertIn(f'data-name="{Path(last[4]).name}"', last[5])
        self.assertIn('class="scanprog done"', last[0])

    def test_decoding_after_sampling_is_shown(self):
        outs = self.run_gen(post="decode_first_stage: decoding latents", sleep=0.9)
        middle = [o[0] for o in outs[1:-1]]
        self.assertTrue(any("Decoding image" in m and "92%" in m for m in middle), middle)

    def test_messages_before_sampling_count_as_loading(self):
        outs = self.run_gen(pre="loading vae: decoding weights|  |==| 1/3 - 0.50s/it", presleep=0.9)
        middle = [o[0] for o in outs[1:-1]]
        self.assertTrue(middle, "expected progress updates while the engine loads")
        self.assertTrue(all("Decoding" not in m and "step 1 of 3" not in m for m in middle), middle)
        self.assertIn("Loading model", middle[0])
        self.assertIn("Image ready", outs[-1][0])

    def test_stop_during_generation(self):
        with mock.patch.dict(os.environ, {"FAKE_SD_SLEEP": "30", "FAKE_SD_LOG": str(self.log)}):
            gen = app.generate_image(Z, "slow", SMALL, 2, 1)
            next(gen)
            next(gen)
            p = app._image_proc["p"]
            self.addCleanup(lambda: p.poll() is None and p.kill())
            self.assertIsNotNone(p)
            app.stop_image()
            rest = list(gen)
        self.assertEqual(p.returncode, -signal.SIGTERM)
        self.assertIn("Stopped", rest[-1][0])
        self.assertIsNone(app._image_proc["p"])
        self.assertEqual(list(app.IMAGES_DIR.glob("*.png")), [])


def _size_test(label, width, height):
    def test(self):
        outs = self.run_gen(size=label)
        self.assertEqual((self.arg("-W"), self.arg("-H")), (str(width), str(height)))
        meta = self.meta(outs)
        self.assertEqual((meta["width"], meta["height"]), (width, height))
        self.assertIn(f"{width} x {height}", outs[-1][0])
    return test


for _label, (_w, _h) in app.IMAGE_SIZES.items():
    setattr(Generate, f"test_size_{_w}x{_h}_{app._slug(_label.split()[0])}", _size_test(_label, _w, _h))


class StopAndDelete(ImagesBase):
    def test_stop_without_generation_sends_nothing(self):
        with mock.patch.object(app.os, "killpg") as kill:
            app.stop_image()
        kill.assert_not_called()

    def test_stop_after_generation_finished_sends_nothing(self):
        app._image_proc["p"] = mock.Mock(pid=4242, **{"poll.return_value": 0})
        with mock.patch.object(app.os, "killpg") as kill:
            app.stop_image()
        kill.assert_not_called()

    def gallery_with(self, *names):
        app.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for n in names:
            (app.IMAGES_DIR / n).write_bytes(PNG)
            (app.IMAGES_DIR / n).with_suffix(".json").write_text("{}")

    def test_unknown_name_changes_nothing(self):
        self.gallery_with("keep.png")
        self.assertEqual(app.delete_gallery_image("nope.png"), (app.gr.update(), app.gr.update(), ""))
        self.assertTrue((app.IMAGES_DIR / "keep.png").exists())

    def test_nested_traversal_only_touches_gallery_copy(self):
        self.gallery_with("x.png")
        outside = app.IMAGES_DIR.parent / "x.png"
        outside.write_bytes(b"precious")
        gallery, note, cleared = app.delete_gallery_image("sub/../../x.png")
        self.assertTrue(outside.exists())
        self.assertFalse((app.IMAGES_DIR / "x.png").exists())
        self.assertIn("Image deleted.", note)
        self.assertIn("No images yet", gallery)
        self.assertEqual(cleared, "")

    def test_already_deleted_image(self):
        self.gallery_with("a.png", "b.png")
        gallery, _, _ = app.delete_gallery_image("a.png")
        self.assertIn('data-name="b.png"', gallery)
        self.assertNotIn('data-name="a.png"', gallery)
        self.assertEqual(app.delete_gallery_image("a.png"), (app.gr.update(), app.gr.update(), ""))

    def test_settings_file_name_is_refused(self):
        self.gallery_with("a.png")
        app.delete_gallery_image("a.json")
        self.assertTrue((app.IMAGES_DIR / "a.json").exists())
        self.assertTrue((app.IMAGES_DIR / "a.png").exists())

    def test_empty_or_missing_name_is_ignored(self):
        self.gallery_with("a.png")
        for name in (None, "", "."):
            self.assertEqual(app.delete_gallery_image(name), (app.gr.update(), app.gr.update(), ""), name)
        self.assertTrue(app.IMAGES_DIR.is_dir())
        self.assertTrue((app.IMAGES_DIR / "a.png").exists())

    def test_image_without_settings_file_is_deleted(self):
        app.IMAGES_DIR.mkdir(parents=True)
        (app.IMAGES_DIR / "solo.png").write_bytes(PNG)
        _, note, _ = app.delete_gallery_image("solo.png")
        self.assertIn("Image deleted.", note)
        self.assertFalse((app.IMAGES_DIR / "solo.png").exists())


if __name__ == "__main__":
    unittest.main()
