"""Tests for updating: LLM Scanner itself, Ollama and garak. Run from the app folder:
    .venv/bin/python -m unittest discover -s tests -v

The last class upgrades real earlier releases (from git tags) to the code in this folder with their own updater and
starts the result, the way an installed copy updates itself. It takes a minute or two."""

import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
os.environ["LLM_SCANNER_CHECK"] = "1"

import app  # noqa: E402
from test_app import FakeResponse, TempDataTest  # noqa: E402

PY = sys.executable
SKIP_IN_RELEASE = {"data", ".venv", ".git", "__pycache__", ".ruff_cache"}


def release_tarball(version, top="llm-scanner-release"):
    """This folder's files packaged like a GitHub release archive, with VERSION set to `version`."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in sorted(APP_DIR.iterdir()):
            if f.name in SKIP_IN_RELEASE or f.name.endswith((".tar.gz", ".part")):
                continue
            if f.name == "VERSION":
                data = f"{version}\n".encode()
                info = tarfile.TarInfo(f"{top}/VERSION")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            else:
                tf.add(f, arcname=f"{top}/{f.name}", filter=lambda t: None if "__pycache__" in t.name else t)
    return buf.getvalue()


class Download(FakeResponse):
    def __init__(self, payload):
        super().__init__(status=200)
        self.payload = payload

    def iter_content(self, n):
        yield self.payload


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serves(port, timeout=90):
    """The page HTML once the app on this port answers, or None."""
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}", timeout=2)
            if r.ok:
                return r.text
        except requests.RequestException:
            pass
        time.sleep(1)
    return None


class UpdateState(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(app._updates, {"checked": 0.0, "ollama": None, "garak": None, "app": None, "busy": None,
                                           "pct": None, "msg": "", "ok": True, "restarting": False})
        p.start()
        self.addCleanup(p.stop)


class Checking(UpdateState):
    def box(self):
        """update_controls as a dict: box, message, rows/labels per kind, Update all, Dismiss."""
        box, msg, row_app, row_ollama, row_garak, lbl_app, lbl_ollama, lbl_garak, all_btn, dismiss = app.update_controls()
        return {"box": box["visible"], "msg": msg, "rows": {"app": row_app["visible"], "ollama": row_ollama["visible"],
                "garak": row_garak["visible"]}, "labels": {"app": lbl_app, "ollama": lbl_ollama, "garak": lbl_garak},
                "all": all_btn["visible"], "dismiss": dismiss["visible"]}

    def test_checked_once_at_startup_then_on_request(self):
        src = (APP_DIR / "app.py").read_text()
        self.assertNotIn("_update_loop", src, "there must be no repeating background check")
        self.assertIn("threading.Thread(target=check_updates", src.split('if __name__ == "__main__":')[1])
        with mock.patch.object(app.requests, "get", side_effect=app.requests.ConnectionError) as get:
            self.assertEqual(app.check_updates(), ["Ollama", "garak", "LLM Scanner"])
        self.assertEqual(get.call_count, 3)  # LLM Scanner on GitHub, Ollama and garak

    def test_no_checks_while_restarting(self):
        app._updates["restarting"] = True
        with mock.patch.object(app.requests, "get") as get:
            app.check_updates()
        get.assert_not_called()

    def test_nothing_shown_without_updates(self):
        b = self.box()
        self.assertFalse(b["box"])
        self.assertFalse(any(b["rows"].values()) or b["all"] or b["dismiss"])

    def test_one_update_is_listed_without_update_all(self):
        app._updates["ollama"] = ("0.34.2", "0.35.0")
        b = self.box()
        self.assertTrue(b["box"])
        self.assertEqual(b["rows"], {"app": False, "ollama": True, "garak": False})
        self.assertIn('0.34.2 <span class="up-arrow">→</span> 0.35.0', b["labels"]["ollama"])
        self.assertFalse(b["all"])

    def test_several_updates_offer_update_all(self):
        app._updates.update(app={"version": "9.9.9", "tag": "v9.9.9", "notes": "Fixes <things>"},
                            ollama=("0.34.2", "0.35.0"), garak=("0.17.0", "0.18.0"))
        b = self.box()
        self.assertTrue(all(b["rows"].values()))
        self.assertTrue(b["all"])
        self.assertIn(f'{app.APP_VERSION} <span class="up-arrow">→</span> 9.9.9', b["labels"]["app"])
        self.assertIn("Fixes &lt;things&gt;", b["labels"]["app"])  # release notes on hover
        app._set_progress("Downloading Ollama 0.35.0", 42.0)
        b = self.box()
        self.assertIn("42%", b["msg"])
        self.assertFalse(any(b["rows"].values()) or b["all"], "no buttons while an update runs")

    def test_close_button_hides_until_something_new(self):
        app._updates.update(ollama=("0.34.2", "0.35.0"), garak=("0.17.0", "0.18.0"))
        self.assertTrue(self.box()["dismiss"], "the x shows whenever the box does")
        app.dismiss_update_msg()
        self.assertFalse(self.box()["box"])
        app._updates["ollama"] = ("0.34.2", "0.36.0")  # a newer release than the one closed
        self.assertEqual(self.box()["rows"], {"app": False, "ollama": True, "garak": False})
        with mock.patch.object(app, "check_updates", return_value=[]):
            app.check_updates_now()  # asking again shows everything again
        self.assertTrue(self.box()["rows"]["garak"])

    def test_button_reports_up_to_date_then_clears(self):
        def get(url, **kw):
            if "pypi" in url:
                return FakeResponse({"info": {"version": "0.0.1"}})
            return FakeResponse({"tag_name": "v0.0.1", "prerelease": False})
        with mock.patch.object(app.requests, "get", side_effect=get), mock.patch.object(app, "ollama_up", return_value="0.34.2"):
            app.check_updates_now()
        b = self.box()
        self.assertIn("Everything is up to date: LLM Scanner", b["msg"])
        self.assertIn("Ollama 0.34.2", b["msg"])
        app._updates["msg_expires"] = time.time() - 1
        self.assertFalse(self.box()["box"], "the up-to-date note goes away by itself")

    def test_button_shows_available_update(self):
        def get(url, **kw):
            if "pypi" in url:
                return FakeResponse({"info": {"version": "0.0.1"}})
            if "ollama/ollama" in url:
                return FakeResponse({"tag_name": "v0.0.1"})
            return FakeResponse({"tag_name": "v99.0.0", "body": "New things", "prerelease": False})
        with mock.patch.object(app.requests, "get", side_effect=get), mock.patch.object(app, "ollama_up", return_value="0.34.2"):
            app.check_updates_now()
        b = self.box()
        self.assertEqual(b["rows"], {"app": True, "ollama": False, "garak": False})
        self.assertEqual(b["msg"], "")

    def test_button_reports_offline_until_dismissed(self):
        with mock.patch.object(app.requests, "get", side_effect=app.requests.ConnectionError):
            app.check_updates_now()
        b = self.box()
        self.assertIn("Couldn&#x27;t check Ollama, garak, LLM Scanner for updates", b["msg"])
        self.assertTrue(b["dismiss"])
        app.dismiss_update_msg()
        self.assertFalse(self.box()["box"])


class UpdateAll(UpdateState):
    def test_runs_each_in_order_and_restarts_once(self):
        app._updates.update(app={"version": "9.9.9", "tag": "v9.9.9", "notes": ""}, ollama=("0.1", "0.2"),
                            garak=("0.17.0", "0.18.0"))
        ran, restarts = [], []

        def fake_run(kind):
            ran.append(kind)
            app._updates.update(ok=True, msg=f"{kind} done.")
            if kind in ("garak", "app"):
                app._restart_soon(kind)  # both need a restart; it must wait until the end
            else:
                app._updates[kind] = None
        with mock.patch.object(app, "_run_update", side_effect=fake_run), \
                mock.patch.object(app.threading, "Timer", side_effect=lambda *a: restarts.append(a) or mock.Mock()), \
                mock.patch.object(app, "check_updates"):
            app._run_all_updates()
        self.assertEqual(ran, ["ollama", "garak", "app"])
        self.assertEqual(len(restarts), 1)
        self.assertTrue(app._updates["restarting"])
        self.assertEqual(app._updates["msg"], "ollama done. garak done. app done.")
        self.assertFalse(app._updates["defer_restart"])

    def test_failure_is_reported_and_others_still_run(self):
        app._updates.update(ollama=("0.1", "0.2"), garak=("0.17.0", "0.18.0"))

        def fake_run(kind):
            app._updates.update(ok=kind != "ollama", msg=f"{kind} {'failed' if kind == 'ollama' else 'done'}.")
            app._updates[kind] = None
        with mock.patch.object(app, "_run_update", side_effect=fake_run), mock.patch.object(app, "check_updates") as check, \
                mock.patch.object(app.threading, "Timer") as timer:
            app._run_all_updates()
        self.assertFalse(app._updates["ok"])
        self.assertIn("ollama failed. garak done.", app._updates["msg"])
        timer.assert_not_called()
        check.assert_called_once()

    def test_update_all_starts_in_background(self):
        with mock.patch.object(app.threading, "Thread") as t:
            app.start_update("all")
        self.assertIs(t.call_args.kwargs["target"], app._run_all_updates)


class Starting(UpdateState):
    def busy(self):
        p = mock.Mock()
        p.poll.return_value = None
        return p

    def test_refused_during_scan_or_image(self):
        with mock.patch.dict(app._proc, {"p": self.busy()}), mock.patch.object(app.threading, "Thread") as t:
            app.start_update("ollama")
        t.assert_not_called()
        self.assertIn("scan", app._updates["msg"])
        with mock.patch.dict(app._image_proc, {"p": self.busy()}), mock.patch.object(app.threading, "Thread") as t:
            app.start_update("app")
        t.assert_not_called()
        self.assertIn("image", app._updates["msg"])

    def test_refused_while_busy_or_restarting(self):
        for state in ({"busy": "Downloading"}, {"restarting": True}):
            with self.subTest(state=state), mock.patch.dict(app._updates, state), \
                    mock.patch.object(app.threading, "Thread") as t:
                app.start_update("garak")
            t.assert_not_called()

    def test_starts_in_background(self):
        with mock.patch.object(app.threading, "Thread") as t:
            app.start_update("app")
        self.assertEqual(t.call_args.kwargs["args"], ("app",))
        t.return_value.start.assert_called_once()
        self.assertEqual(app._updates["busy"], "Starting update")

    def test_restart_is_scheduled(self):
        with mock.patch.object(app.threading, "Timer") as timer:
            app._restart_soon("app")
        self.assertTrue(app._updates["restarting"])
        self.assertIsNone(app._updates["app"])
        timer.return_value.start.assert_called_once()


class AppUpdates(TempDataTest):
    def setUp(self):
        super().setUp()
        self.app_dir = self.tmp / "app"
        self.app_dir.mkdir()
        for name in ("app.py", "VERSION", "requirements.txt"):
            (self.app_dir / name).write_text(f"old {name}")
        for target in (mock.patch.object(app, "APP_DIR", self.app_dir),
                       mock.patch.dict(app._updates, {"app": {"version": "9.9.9", "tag": "v9.9.9", "notes": ""},
                                                      "busy": None, "msg": "", "ok": True, "restarting": False}),
                       mock.patch.object(app, "check_updates")):
            target.start()
            self.addCleanup(target.stop)

    def run_update(self, verify=(True, "ok"), requirements=None):
        tarball = release_tarball("9.9.9")
        if requirements is not None:  # swap requirements.txt inside the archive
            src = tarfile.open(fileobj=io.BytesIO(tarball))
            out = io.BytesIO()
            with tarfile.open(fileobj=out, mode="w:gz") as tf:
                for m in src.getmembers():
                    data = requirements.encode() if m.name.endswith("/requirements.txt") else (
                        src.extractfile(m).read() if m.isfile() else None)
                    if data is not None:
                        m.size = len(data)
                        tf.addfile(m, io.BytesIO(data))
                    else:
                        tf.addfile(m)
            tarball = out.getvalue()
        real_run = subprocess.run
        runs = []

        def run(cmd, **kw):
            runs.append(cmd)
            return real_run(cmd, **kw) if cmd[0] == "tar" else mock.Mock(returncode=0)
        with mock.patch.object(app.requests, "get", return_value=Download(tarball)), \
                mock.patch.object(app, "_verify_app_candidate", return_value=verify), \
                mock.patch.object(app.subprocess, "run", run), mock.patch.object(app, "_restart_soon") as restart:
            app._run_update("app")
        return runs, restart

    def test_successful_update_installs_backs_up_and_restarts(self):
        runs, restart = self.run_update()
        self.assertEqual((self.app_dir / "VERSION").read_text().strip(), "9.9.9")
        self.assertEqual((self.app_dir / "app.py").read_bytes(), (APP_DIR / "app.py").read_bytes())
        self.assertTrue((self.app_dir / "static" / "app.js").exists())
        self.assertEqual((app.DATA_DIR / "backups" / app.APP_VERSION / "app.py").read_text(), "old app.py")
        self.assertFalse((app.DATA_DIR / "update-staging").exists())
        restart.assert_called_once_with("app")
        self.assertTrue(app._updates["ok"])
        self.assertIsNone(app._updates["busy"])
        self.assertTrue(os.access(self.app_dir / "install.sh", os.X_OK))

    def test_failed_check_changes_nothing(self):
        runs, restart = self.run_update(verify=(False, "the new version failed to start (SyntaxError)"))
        self.assertEqual((self.app_dir / "app.py").read_text(), "old app.py")
        restart.assert_not_called()
        self.assertFalse(app._updates["ok"])
        self.assertIn("was not installed", app._updates["msg"])
        self.assertFalse((app.DATA_DIR / "update-staging").exists())

    def test_new_python_packages_are_installed_first(self):
        runs, _ = self.run_update(requirements="gradio==99.0\n")
        self.assertTrue(any(cmd[:3] == [app.UV, "pip", "install"] for cmd in runs))

    def test_unchanged_packages_are_not_reinstalled(self):
        (self.app_dir / "requirements.txt").write_text((APP_DIR / "requirements.txt").read_text())
        runs, _ = self.run_update()
        self.assertFalse(any(cmd[0] == app.UV for cmd in runs))

    def test_download_error_is_reported(self):
        with mock.patch.object(app.requests, "get", return_value=FakeResponse(status=404)), \
                mock.patch.object(app, "_restart_soon") as restart:
            app._run_update("app")
        restart.assert_not_called()
        self.assertFalse(app._updates["ok"])
        self.assertIn("app update failed", app._updates["msg"])
        self.assertEqual((self.app_dir / "app.py").read_text(), "old app.py")


class OllamaAndGarakUpdates(UpdateState):
    def ollama_update(self, verify):
        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        (home / "ollama" / "bin").mkdir(parents=True)
        (home / "ollama" / "bin" / "ollama").write_text("old")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            info = tarfile.TarInfo("bin/ollama")
            info.size = 3
            tf.addfile(info, io.BytesIO(b"new"))
        real_run = subprocess.run
        run = mock.Mock(side_effect=lambda cmd, **kw: real_run([c for c in cmd if c != "--zstd"], **kw)
                        if cmd[0] == "tar" else mock.Mock(returncode=0))
        app._updates["ollama"] = ("0.34.2", "0.99.0")
        with mock.patch.object(app, "OLLAMA_DIR", home / "ollama"), \
                mock.patch.object(app.requests, "get", return_value=Download(buf.getvalue())), \
                mock.patch.object(app.subprocess, "run", run), mock.patch.object(app, "_verify_ollama", return_value=verify), \
                mock.patch.object(app, "check_updates"):
            app._run_update("ollama")
        return home, [c.args[0] for c in run.call_args_list]

    def test_ollama_update_success(self):
        home, cmds = self.ollama_update((True, "Ollama 0.99.0 verified."))
        self.assertEqual((home / "ollama" / "bin" / "ollama").read_text(), "new")
        self.assertFalse((home / "ollama.old").exists() or (home / "ollama.new").exists())
        self.assertIn(["systemctl", "--user", "stop", "ollama"], cmds)
        self.assertEqual(cmds[-1], ["systemctl", "--user", "start", "ollama"])
        self.assertTrue(app._updates["ok"])
        self.assertIn("verified", app._updates["msg"])

    def test_ollama_update_rolls_back(self):
        home, cmds = self.ollama_update((False, "Ollama did not start as version 0.99.0."))
        self.assertEqual((home / "ollama" / "bin" / "ollama").read_text(), "old")
        self.assertFalse(app._updates["ok"])
        self.assertIn("previous version was restored", app._updates["msg"])

    def test_verify_ollama(self):
        tags = {"models": [{"name": "big", "size": 9}, {"name": "small", "size": 1}]}

        def post(url, **kw):
            if url.endswith("/api/show"):
                return FakeResponse({"capabilities": ["completion"]})
            return FakeResponse(self.reply)
        with mock.patch.object(app, "ollama_up", return_value="0.99.0"), mock.patch.object(app.time, "sleep"), \
                mock.patch.object(app.requests, "get", return_value=FakeResponse(tags)), \
                mock.patch.object(app.requests, "post", side_effect=post) as p:
            self.reply = {"response": "ok"}
            ok, detail = app._verify_ollama("0.99.0")
            self.assertTrue(ok, detail)
            self.assertEqual(p.call_args.kwargs["json"]["model"], "small")  # the quickest model to test with
            self.reply = {"response": ""}
            self.assertFalse(app._verify_ollama("0.99.0")[0])
        with mock.patch.object(app, "ollama_up", return_value="0.34.2"), mock.patch.object(app.time, "sleep"):
            ok, detail = app._verify_ollama("0.99.0")
        self.assertFalse(ok)
        self.assertIn("reported 0.34.2", detail)

    def test_garak_update_success_restarts_app(self):
        app._updates["garak"] = ("0.17.0", "0.18.0")
        with mock.patch.object(app.subprocess, "run", return_value=mock.Mock(returncode=0)) as run, \
                mock.patch.object(app, "_verify_garak", return_value=(True, "0.18.0", "garak 0.18.0 verified.")), \
                mock.patch.object(app, "_restart_soon") as restart, mock.patch.object(app, "check_updates"):
            app._run_update("garak")
        self.assertIn("--upgrade", run.call_args.args[0])
        restart.assert_called_once_with("garak")
        self.assertTrue(app._updates["ok"])

    def test_garak_update_failure_reinstalls_previous(self):
        app._updates["garak"] = ("0.17.0", "0.18.0")
        with mock.patch.object(app.subprocess, "run", return_value=mock.Mock(returncode=0)) as run, \
                mock.patch.object(app, "_verify_garak", return_value=(False, "0.18.0", "self-test failed.")), \
                mock.patch.object(app, "_restart_soon") as restart, mock.patch.object(app, "check_updates"):
            app._run_update("garak")
        self.assertIn("garak==0.17.0", run.call_args.args[0])
        restart.assert_not_called()
        self.assertFalse(app._updates["ok"])


class VerifyCandidate(unittest.TestCase):
    """The real check an update runs before switching: start the new version on a spare port."""

    def candidate(self):
        folder = Path(tempfile.mkdtemp(prefix="llm-scanner-candidate-"))
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        with tarfile.open(fileobj=io.BytesIO(release_tarball(app.APP_VERSION))) as tf:
            tf.extractall(folder, filter="data")
        return folder / "llm-scanner-release"

    def test_working_version_passes(self):
        ok, detail = app._verify_app_candidate(self.candidate())
        self.assertTrue(ok, detail)

    def test_syntax_error_fails_fast(self):
        folder = self.candidate()
        (folder / "app.py").write_text((folder / "app.py").read_text() + "\ndef broken(:\n")
        ok, detail = app._verify_app_candidate(folder)
        self.assertFalse(ok)
        self.assertIn("app.py has an error", detail)

    def test_crash_at_startup_fails(self):
        folder = self.candidate()
        (folder / "app.py").write_text("import module_that_does_not_exist\n" + (folder / "app.py").read_text())
        ok, detail = app._verify_app_candidate(folder)
        self.assertFalse(ok)
        self.assertIn("failed to start", detail)


def git_tag_exists(tag):
    return subprocess.run(["git", "-C", str(APP_DIR), "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
                          capture_output=True).returncode == 0


UPGRADE_SCRIPT = r"""
import json, os, sys
sys.argv = ["app.py"]
sys.path.insert(0, os.getcwd())
import requests
payload = open(os.environ["TEST_RELEASE"], "rb").read()
real_get = requests.get

class Release:
    status_code, ok = 200, True
    def raise_for_status(self): pass
    def iter_content(self, n): yield payload
    def __enter__(self): return self
    def __exit__(self, *a): return False

requests.get = lambda url, *a, **k: Release() if "codeload.github.com" in url else real_get(url, *a, **k)
import app
app._restart_soon = lambda kind: None
app._updates["app"] = {"version": os.environ["TEST_VERSION"], "tag": "v" + os.environ["TEST_VERSION"], "notes": ""}
app._update_app()
print("RESULT " + json.dumps({"msg": app._updates["msg"], "ok": app._updates["ok"]}))
"""


class UpgradeFromEarlierReleases(unittest.TestCase):
    """Each earlier release updates to this code with its own updater (as an installed copy would), then the
    result must start. 1.0.3 through 1.0.8 install only the files they know about, which broke 1.0.9 to 1.0.11."""

    def upgrade(self, tag):
        if not git_tag_exists(tag):
            self.skipTest(f"git tag {tag} not available")
        old_reqs = subprocess.run(["git", "-C", str(APP_DIR), "show", f"{tag}:requirements.txt"],
                                  capture_output=True, text=True).stdout
        if old_reqs != (APP_DIR / "requirements.txt").read_text():
            self.skipTest(f"{tag} pins different packages; its updater would reinstall them in this environment")
        root = Path(tempfile.mkdtemp(prefix=f"llm-scanner-{tag}-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        old = root / "app"
        old.mkdir()
        archive = subprocess.run(["git", "-C", str(APP_DIR), "archive", tag], capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            tf.extractall(old, filter="data")
        version = "99.0.0"
        release = root / "release.tar.gz"
        release.write_bytes(release_tarball(version))
        env = os.environ | {"LLM_SCANNER_CHECK": "1", "TEST_RELEASE": str(release), "TEST_VERSION": version,
                            "OLLAMA_URL": "http://127.0.0.1:9", "LLM_SCANNER_DATA": str(root / "data")}
        out = subprocess.run([PY, "-c", UPGRADE_SCRIPT], cwd=old, env=env, capture_output=True, text=True, timeout=300)
        result = next((json.loads(line[7:]) for line in out.stdout.splitlines() if line.startswith("RESULT ")), None)
        self.assertIsNotNone(result, out.stdout[-800:] + out.stderr[-800:])
        self.assertTrue(result["ok"], result["msg"])
        self.assertEqual((old / "VERSION").read_text().strip(), version)
        port = free_port()
        proc = subprocess.Popen([PY, "app.py", "--no-browser"], cwd=old, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, start_new_session=True,
                                env=env | {"LLM_SCANNER_PORT": str(port)})
        try:
            page = serves(port)
            if page is None:
                proc.kill()
                self.fail(f"{tag} upgraded to this version, which then failed to start: "
                          f"{proc.stderr.read().decode(errors='replace')[-600:]}")
            self.assertIn(f"v{version}", page)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, 15)
                proc.wait(timeout=20)

    def test_from_1_0_3(self):
        self.upgrade("v1.0.3")

    def test_from_1_0_8(self):
        self.upgrade("v1.0.8")

    def test_from_1_0_11(self):
        self.upgrade("v1.0.11")

    def test_from_1_0_12(self):
        self.upgrade("v1.0.12")


if __name__ == "__main__":
    unittest.main()
