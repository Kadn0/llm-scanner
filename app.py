"""LLM Scanner - a local GUI for pulling models into Ollama and scanning them with garak."""

import base64
import bisect
import codecs
import html
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import gradio as gr
import requests

import analyst_report

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LLM_SCANNER_DATA") or APP_DIR / "data")  # overridable so tests use a scratch folder
DATA_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
HF_API = "https://huggingface.co/api"
OLLAMA_WEB = os.environ.get("LLM_SCANNER_OLLAMA_WEB", "https://ollama.com")  # overridable for tests
GARAK_RUNS = Path(os.environ.get("LLM_SCANNER_GARAK_RUNS") or Path.home() / ".local/share/garak/garak_runs")
GROUPS_FILE = DATA_DIR / "probe_groups.json"
PY = sys.executable
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]+ ?")
SRC_HF, SRC_OLLAMA = "Hugging Face", "Ollama"
CURRENT = "Current selection (Probes tab)"
GROUP_PREFIX = "Saved group: "

PRESETS = {
    "Quick check": (["dan.DanInTheWild", "promptinject.HijackHateHumans", "goodside.WhoIsRiley", "lmrc.QuackMedicine"],
                    "A fast sample of jailbreak, prompt-injection and misinformation probes. Good first scan."),
    "Jailbreaks and prompt injection": (["dan", "promptinject", "latentinjection"],
                                        "Tests whether the model can be talked out of its instructions or safety behavior."),
    "Harmful content and toxicity": (["lmrc", "realtoxicityprompts", "donotanswer"],
                                     "Tests for toxic, hateful, sexual or otherwise harmful output."),
    "Data leakage and malware": (["leakreplay", "malwaregen", "packagehallucination", "apikey"],
                                 "Tests for training-data regurgitation, malicious code generation and hallucinated packages."),
    "Encoding and obfuscation": (["encoding", "ansiescape"],
                                 "Tests whether encoded or obfuscated prompts slip past the model's safeguards."),
    "Full scan": ([], "Every default garak probe. Very thorough; expect many hours, or days on large models."),
}

_proc = {"p": None}


# ---------------------------------------------------------------- system helpers
def ollama_up():
    try:
        return requests.get(f"{OLLAMA}/api/version", timeout=2).json().get("version")
    except Exception:
        return None


def ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        return 16


def _detect_vram_budget():
    """Largest model size (GB) that fits on this machine's GPU, leaving ~0.5 GB headroom. 7.5 if unknown."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.split()
        return round(max(1.0, int(out[0]) / 1024 - 0.5), 1)
    except Exception:
        return 7.5


VRAM_BUDGET_GB = _detect_vram_budget()  # auto-selected versions aim for the largest file that stays under this


def fit_label(size_gb):
    if size_gb <= VRAM_BUDGET_GB:
        return "fits on GPU, fast"
    if size_gb <= VRAM_BUDGET_GB + ram_gb() * 0.6:
        return "GPU + system memory, slower"
    return "too large for this PC"


def gpu_live():
    """Live GPU stats: (name, util %, used GB, total GB, temp C, power W) or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        name, util, used, total, temp, power = [x.strip() for x in out.split(",")]
        num = lambda v: float(v) if re.match(r"^[\d.]+$", v) else None
        return (name.replace("NVIDIA ", ""), num(util), int(used) / 1024, int(total) / 1024, num(temp), num(power))
    except Exception:
        return None


def ram_live():
    try:
        info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        kb = lambda k: int(info[k].strip().split()[0])
        total = kb("MemTotal") / 1048576
        return total - kb("MemAvailable") / 1048576, total
    except Exception:
        return None


def _meter(pct, cls=""):
    pct = max(0, min(100, pct or 0))
    tone = "hot" if pct >= 90 else "warm" if pct >= 70 else ""
    return f'<span class="meter {cls}"><i class="{tone}" style="width:{pct:.0f}%"></i></span>'


def status_html(v=None):
    v = v or ollama_up()
    pills = [f'<span class="pill"><span class="dot ok"></span>Ollama {v}</span>' if v else
             '<span class="pill"><span class="dot bad"></span>Ollama stopped</span>']
    g = gpu_live()
    if g:
        name, util, used, total, temp, power = g
        extra = "".join(f'<span class="muted">{x}</span>' for x in (
            f"{temp:.0f}°C" if temp is not None else "", f"{power:.0f} W" if power is not None else "") if x)
        pills.append(f'<span class="pill"><span class="dot ok"></span>{html.escape(name)}'
                     f'<span class="lbl">GPU</span>{_meter(util)}<span class="val">{util or 0:.0f}%</span>'
                     f'<span class="lbl">VRAM</span>{_meter(100 * used / total)}<span class="val">{used:.1f} / {total:.1f} GB</span>'
                     f'{extra}</span>')
    else:
        pills.append('<span class="pill"><span class="dot warn"></span>No GPU detected</span>')
    r = ram_live()
    if r:
        pills.append(f'<span class="pill"><span class="lbl">RAM</span>{_meter(100 * r[0] / r[1])}'
                     f'<span class="val">{r[0]:.0f} / {r[1]:.0f} GB</span></span>')
    return f'<div class="status">{"".join(pills)}</div>{activity_html()}'


def loaded_models():
    try:
        return requests.get(f"{OLLAMA}/api/ps", timeout=2).json().get("models", [])
    except Exception:
        return []


def activity_html():
    """What LLM Scanner currently has running: loaded models, scans, downloads."""
    items = []
    for m in loaded_models():
        size, vram = m.get("size", 0), m.get("size_vram", 0)
        gpu_pct = 100 * vram / size if size else 0
        where = "GPU" if gpu_pct >= 99.5 else "CPU" if gpu_pct < 0.5 else f"{gpu_pct:.0f}% GPU / {100 - gpu_pct:.0f}% CPU"
        mins = ""
        try:
            left = datetime.fromisoformat(m["expires_at"]).timestamp() - time.time()
            mins = f", unloads in {max(0, round(left / 60))} min" if left < 86400 else ""
        except Exception:
            pass
        items.append(f'Model loaded: <b>{html.escape(m["name"])}</b> ({size / 1e9:.1f} GB, {where}{mins})')
    p = _proc.get("p")
    if p and p.poll() is None:
        items.append("Scan running")
    ip = _image_proc.get("p")
    if ip and ip.poll() is None:
        items.append("Generating an image")
    downloads = len(active_downloads())
    if downloads:
        items.append(f"{downloads} download{'s' if downloads != 1 else ''} in progress")
    if not items:
        return '<div class="activity idle">Idle: no models loaded, nothing running</div>'
    return '<div class="activity">' + " &nbsp;·&nbsp; ".join(items) + "</div>"


def unload_models():
    """Free RAM/VRAM by unloading every model Ollama has in memory."""
    for m in loaded_models():
        try:
            requests.post(f"{OLLAMA}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=30)
        except Exception:
            pass
    return status_controls()


def status_controls():
    """Status bar plus its buttons: Start when Ollama is stopped; Stop and Restart while it runs."""
    v = ollama_up()
    return (status_html(v), gr.update(visible=bool(v and loaded_models())), gr.update(visible=not v),
            gr.update(visible=bool(v)), gr.update(visible=bool(v)))


def _ollama_service(action, want_up):
    """systemctl start/stop/restart the Ollama user service, then wait until it is up (or down)."""
    subprocess.run(["systemctl", "--user", action, "ollama"], check=False)
    for _ in range(40):
        if bool(ollama_up()) == want_up:
            break
        time.sleep(0.5)
    return status_controls()


def _scan_running():
    return bool(_proc.get("p") and _proc["p"].poll() is None)


def start_ollama():
    return _ollama_service("start", True)


def stop_ollama():
    """Stop Ollama. Models unload; downloads wait and resume when it starts again."""
    if _scan_running():
        gr.Warning("A scan is using Ollama. Stop the scan first.")
        return status_controls()
    return _ollama_service("stop", False)


def restart_ollama():
    if _scan_running():
        gr.Warning("A scan is using Ollama. Stop the scan first.")
        return status_controls()
    return _ollama_service("restart", True)


def pump_output(p, on_text, on_exit):
    """Read a child process's output on a background thread, then call on_exit once it ends. Scans and image
    generation keep running when the page that started them is closed or reloaded; if nobody read their output
    they would freeze as soon as the pipe filled up."""
    def run():
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while chunk := p.stdout.read1(4096):
            try:
                on_text(decoder.decode(chunk))
            except Exception:  # a display problem must never stop the draining
                pass
        p.wait()
        on_exit()

    reader = threading.Thread(target=run, daemon=True)
    reader.start()
    return reader


# ---------------------------------------------------------------- update checks
UV = str(Path.home() / ".local/bin/uv")
APP_VERSION = (APP_DIR / "VERSION").read_text().strip() if (APP_DIR / "VERSION").exists() else "0.0.0"
UPDATE_REPO = os.environ.get("LLM_SCANNER_REPO", "Kadn0/llm-scanner")  # GitHub repo the app updates from
CHECK_ONLY = os.environ.get("LLM_SCANNER_CHECK") == "1"  # set while an update verifies itself on a spare port
APP_FILES = ["app.py", "analyst_report.py", "garak_runner.py", "llm-scanner.sh", "install.sh", "uninstall.sh",
             "export-data.sh", "README.md", "VERSION", "requirements.txt", "static"]
OLLAMA_DIR = Path.home() / ".local/ollama"
KEEP_ON_UPDATE = {"data", ".venv", ".git", ".gitignore", ".github"}  # never replaced by a release
UPDATE_CHECK_INTERVAL = 5 * 60
_updates = {"checked": 0.0, "ollama": None, "garak": None, "app": None, "busy": None, "pct": None, "msg": "",
            "ok": True}


def _version_tuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:3])


def check_updates(force=False):
    """Compare installed Ollama/garak with the latest releases. Results are cached for 5 minutes."""
    if _updates.get("restarting"):
        return
    if not force and time.time() - _updates["checked"] < UPDATE_CHECK_INTERVAL:
        return
    _updates["checked"] = time.time()
    try:
        latest = requests.get("https://api.github.com/repos/ollama/ollama/releases/latest", timeout=10).json()
        installed = ollama_up()
        tag = (latest.get("tag_name") or "").lstrip("v")
        _updates["ollama"] = (installed, tag) if installed and tag and \
            _version_tuple(tag) > _version_tuple(installed) and not latest.get("prerelease") else None
    except Exception:
        pass
    try:
        installed = importlib.metadata.version("garak")
        tag = requests.get("https://pypi.org/pypi/garak/json", timeout=10).json()["info"]["version"]
        _updates["garak"] = (installed, tag) if _version_tuple(tag) > _version_tuple(installed) else None
    except Exception:
        pass
    try:
        rel = requests.get(f"https://api.github.com/repos/{UPDATE_REPO}/releases/latest", timeout=10).json()
        tag = (rel.get("tag_name") or "").lstrip("v")
        if tag and _version_tuple(tag) > _version_tuple(APP_VERSION) and not rel.get("prerelease"):
            _updates["app"] = {"version": tag, "tag": rel["tag_name"], "notes": (rel.get("body") or "").strip()}
        else:
            _updates["app"] = None
    except Exception:
        pass


def _set_progress(phase, pct=None):
    """pct=None shows an indeterminate (animated) bar for steps that can't report a percentage."""
    _updates["busy"] = phase
    _updates["pct"] = pct


def update_banner():
    if _updates["busy"]:
        pct = _updates.get("pct")
        bar = (f'<div class="up-bar"><div style="width:{pct:.1f}%"></div></div>' if pct is not None else
               '<div class="up-bar indeterminate"><div></div></div>')
        right = f"{pct:.0f}%" if pct is not None else ""
        return (f'<div class="update-banner working"><div class="up-top"><span>{html.escape(_updates["busy"])}</span>'
                f'<span>{right}</span></div>{bar}</div>')
    items = []
    if _updates["app"]:
        items.append(f"LLM Scanner {_updates['app']['version']} is available (installed {APP_VERSION})")
    if _updates["ollama"]:
        items.append(f"Ollama {_updates['ollama'][1]} is available (installed {_updates['ollama'][0]})")
    if _updates["garak"]:
        items.append(f"garak {_updates['garak'][1]} is available (installed {_updates['garak'][0]})")
    result = ""
    if _updates["msg"]:
        ok = _updates.get("ok", True)
        result = f'<div class="up-result {"ok" if ok else "bad"}">{html.escape(_updates["msg"])}</div>'
    if not items and not result:
        return ""
    head = ('<b>Update available:</b> ' + " &nbsp;·&nbsp; ".join(html.escape(i) for i in items)) if items else ""
    notes = ""
    if _updates["app"] and _updates["app"]["notes"]:
        notes = f'<div class="up-notes">{html.escape(_updates["app"]["notes"][:400])}</div>'
    return f'<div class="update-banner">{head}{notes}{result}</div>'


def update_controls():
    busy = bool(_updates["busy"])
    return (update_banner(), gr.update(visible=bool(_updates["ollama"]) and not busy),
            gr.update(visible=bool(_updates["garak"]) and not busy),
            gr.update(visible=bool(_updates["app"]) and not busy))


def dismiss_update_msg():
    _updates["msg"] = ""
    return update_controls()


def _verify_ollama(expected):
    """The new build must report the expected version and answer a tiny prompt with an installed chat model."""
    for _ in range(60):
        if ollama_up():
            break
        time.sleep(1)
    version = ollama_up()
    if not version or _version_tuple(version) < _version_tuple(expected):
        return False, f"Ollama did not start as version {expected} (reported {version or 'nothing'})."
    models = sorted((m for m in list_models() if "completion" in model_capabilities(m["name"])),
                    key=lambda m: m.get("size", 0))  # smallest chat model keeps the check quick
    if not models:
        return True, f"Ollama {version} is running (no chat models installed to test with)."
    smallest = models[0]["name"]
    try:
        r = requests.post(f"{OLLAMA}/api/generate", timeout=300, json={
            "model": smallest, "prompt": "Reply with the word ok.", "stream": False, "keep_alive": 0,
            "options": {"num_predict": 8}})
        body = r.json()
    except Exception as e:
        return False, f"Ollama {version} started but could not run {smallest}: {e}"
    if not r.ok or "error" in body or not body.get("response", "").strip():
        return False, f"Ollama {version} started but {smallest} returned no answer: {body.get('error', 'empty reply')}"
    return True, f"Ollama {version} verified: {smallest} loaded and replied."


def _verify_garak():
    """Run garak's built-in self-test (no model needed) and make sure the probe catalog loads."""
    tmp = DATA_DIR / "tmp"
    tmp.mkdir(exist_ok=True)
    prefix = tmp / f"selftest-{os.getpid()}"
    try:
        run = subprocess.run([PY, "-m", "garak", "--target_type", "test.Blank", "--probes", "test.Test",
                              "--report_prefix", str(prefix)], capture_output=True, text=True, timeout=300)
        listed = subprocess.run([PY, "-m", "garak", "--list_probes"], capture_output=True, text=True, timeout=300)
        version = subprocess.run([PY, "-c", "import importlib.metadata as m; print(m.version('garak'))"],
                                 capture_output=True, text=True, timeout=60).stdout.strip()
    finally:
        for f in tmp.glob(prefix.name + ".*"):
            f.unlink(missing_ok=True)
    probes = ANSI.sub("", listed.stdout).count("probes: ")
    if run.returncode != 0 or "PASS" not in ANSI.sub("", run.stdout):
        return False, version, "garak's self-test scan failed."
    if probes < 20:
        return False, version, f"garak loaded only {probes} probes."
    return True, version, f"garak {version} verified: self-test scan passed and {probes} probes loaded."


def _verify_app_candidate(folder):
    """Start the downloaded version on a spare port (without touching models, downloads or updates) and make sure
    it serves its page. Returns (ok, detail)."""
    import py_compile
    for f in ("app.py", "analyst_report.py", "garak_runner.py"):
        try:
            py_compile.compile(str(folder / f), doraise=True)
        except Exception as e:
            return False, f"{f} has an error: {e}"
    port = 7899
    env = os.environ | {"LLM_SCANNER_PORT": str(port), "LLM_SCANNER_CHECK": "1", "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen([PY, str(folder / "app.py"), "--no-browser"], cwd=str(folder), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(120):
            if p.poll() is not None:
                tail = (p.stdout.read() or b"").decode("utf-8", "replace").strip().splitlines()[-1:] or ["exited"]
                return False, f"the new version failed to start ({tail[0][:200]})"
            try:
                if requests.get(f"http://127.0.0.1:{port}", timeout=2).ok:
                    return True, "started and served its page"
            except requests.RequestException:
                pass
            time.sleep(1)
        return False, "the new version didn't respond within 2 minutes"
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)


def _restart_soon(kind):
    """The update is installed and the app restarts in a few seconds; stop offering it in the meantime."""
    _updates[kind] = None
    _updates["restarting"] = True
    threading.Timer(4, lambda: subprocess.run(["systemctl", "--user", "restart", "llm-scanner"])).start()


def _update_app():
    info = _updates["app"]
    version = info["version"]
    staging = DATA_DIR / "update-staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    _set_progress(f"Downloading LLM Scanner {version}", 5)
    archive = staging / "release.tar.gz"
    with requests.get(f"https://codeload.github.com/{UPDATE_REPO}/tar.gz/refs/tags/{info['tag']}", stream=True,
                      timeout=60) as r:
        r.raise_for_status()
        with open(archive, "wb") as fh:
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(staging)], check=True)
    new = next(p for p in staging.iterdir() if p.is_dir())
    missing = [f for f in ("app.py", "VERSION", "static") if not (new / f).exists()]
    if missing:
        raise RuntimeError(f"the release is missing {', '.join(missing)}")
    packaged_version = (new / "VERSION").read_text(encoding="utf-8").strip()
    if packaged_version != version:
        raise RuntimeError(f"the release is labeled {version}, but its package contains VERSION {packaged_version}")
    old_reqs = (APP_DIR / "requirements.txt").read_text() if (APP_DIR / "requirements.txt").exists() else ""
    if (new / "requirements.txt").exists() and (new / "requirements.txt").read_text() != old_reqs:
        _set_progress(f"Installing Python packages for LLM Scanner {version}", None)
        subprocess.run([UV, "pip", "install", "--python", PY, "-r", str(new / "requirements.txt")], check=True,
                       capture_output=True)
    _set_progress(f"Checking LLM Scanner {version} before switching", 60)
    ok, detail = _verify_app_candidate(new)
    if not ok:
        shutil.rmtree(staging, ignore_errors=True)
        _updates["ok"] = False
        _updates["msg"] = f"LLM Scanner {version} was not installed: {detail}. You're still on {APP_VERSION}."
        return
    _set_progress(f"Installing LLM Scanner {version}", 85)
    backup = DATA_DIR / "backups" / APP_VERSION
    shutil.rmtree(backup, ignore_errors=True)
    backup.mkdir(parents=True)
    # Everything the release ships, not just a fixed list, so a file added in a new version is never left behind.
    installed = sorted((set(APP_FILES) | {f.name for f in new.iterdir()}) - KEEP_ON_UPDATE)
    for name in installed:
        src, dst = APP_DIR / name, backup / name
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        elif src.exists():
            shutil.copy2(src, dst)
    for name in installed:
        src, dst = new / name, APP_DIR / name
        if src.is_dir():
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
        elif src.exists():
            shutil.copy2(src, dst)
    for script in APP_DIR.glob("*.sh"):
        script.chmod(0o755)
    shutil.rmtree(staging, ignore_errors=True)
    _updates["msg"] = (f"Updated to LLM Scanner {version} ({detail}). Restarting... reload the page in a few seconds. "
                       f"The previous version is saved in data/backups/{APP_VERSION}.")
    _restart_soon("app")


def _run_update(kind):
    _updates["ok"] = True
    try:
        if kind == "app":
            _update_app()
        elif kind == "ollama":
            version = _updates["ollama"][1]
            staging = OLLAMA_DIR.with_name("ollama.new")
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            archive = staging / "ollama.tar.zst"
            _set_progress(f"Downloading Ollama {version}", 0)
            with requests.get("https://ollama.com/download/ollama-linux-amd64.tar.zst", stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                with open(archive, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        mb = f"{done / 1e6:.0f} of {total / 1e6:.0f} MB" if total else f"{done / 1e6:.0f} MB"
                        _set_progress(f"Downloading Ollama {version} ({mb})", 70 * done / total if total else None)
            _set_progress(f"Installing Ollama {version}", 75)
            subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(staging)], check=True)
            archive.unlink()
            old = OLLAMA_DIR.with_name("ollama.old")
            _set_progress(f"Restarting Ollama {version}", 82)
            subprocess.run(["systemctl", "--user", "stop", "ollama"], check=False)
            shutil.rmtree(old, ignore_errors=True)
            OLLAMA_DIR.rename(old)
            staging.rename(OLLAMA_DIR)
            subprocess.run(["systemctl", "--user", "start", "ollama"], check=False)
            _set_progress(f"Checking Ollama {version} with a test prompt", 90)
            ok, detail = _verify_ollama(version)
            if ok:
                shutil.rmtree(old, ignore_errors=True)
                _updates["msg"] = detail
            else:  # roll back to the version that worked
                _set_progress("Update failed its check, restoring the previous Ollama", 95)
                subprocess.run(["systemctl", "--user", "stop", "ollama"], check=False)
                shutil.rmtree(OLLAMA_DIR, ignore_errors=True)
                old.rename(OLLAMA_DIR)
                subprocess.run(["systemctl", "--user", "start", "ollama"], check=False)
                _updates["ok"] = False
                _updates["msg"] = f"{detail} The previous version was restored."
        else:
            installed, version = _updates["garak"]
            _set_progress(f"Downloading and installing garak {version}", None)
            subprocess.run([UV, "pip", "install", "--python", PY, "--upgrade", "garak"], check=True,
                           capture_output=True)
            _set_progress(f"Checking garak {version} with a self-test scan", 85)
            ok, now, detail = _verify_garak()
            if ok:
                _updates["msg"] = detail + " Restarting the app to load it..."
                _restart_soon("garak")
            else:
                _set_progress(f"Update failed its check, reinstalling garak {installed}", 92)
                subprocess.run([UV, "pip", "install", "--python", PY, f"garak=={installed}"], check=True,
                               capture_output=True)
                _updates["ok"] = False
                _updates["msg"] = f"{detail} garak {installed} was reinstalled."
    except Exception as e:
        _updates["ok"] = False
        _updates["msg"] = f"{kind} update failed: {e}"
    finally:
        _set_progress(None)
        if not _updates.get("restarting"):  # a restart is coming; this old process would re-offer the same update
            check_updates(force=True)


def start_update(kind):
    if _updates["busy"] or _updates.get("restarting"):
        return update_controls()
    if _proc.get("p") and _proc["p"].poll() is None:
        _updates["ok"] = False
        _updates["msg"] = "Finish or stop the running scan before updating."
        return update_controls()
    if kind == "app" and (_image_proc.get("p") and _image_proc["p"].poll() is None):
        _updates["ok"] = False
        _updates["msg"] = "Wait for the image to finish generating before updating."
        return update_controls()
    _updates["msg"] = ""
    _set_progress("Starting update", 0)
    threading.Thread(target=_run_update, args=(kind,), daemon=True).start()
    return update_controls()


def _update_loop():
    while True:
        check_updates()
        time.sleep(60)


# ---------------------------------------------------------------- installed models
def list_models():
    try:
        return requests.get(f"{OLLAMA}/api/tags", timeout=5).json().get("models", [])
    except Exception:
        return []


def model_names():
    return [m["name"] for m in list_models()]


MODEL_COLUMNS = ("Model", "Source", "Size", "Parameters", "Quantization", "Added", "")


def model_row_html(m):
    d = m.get("details", {})
    source = SRC_HF if m["name"].startswith("hf.co/") else SRC_OLLAMA
    cells = (m["name"], source, f"{m.get('size', 0) / 1e9:.1f} GB", d.get("parameter_size", ""),
             d.get("quantization_level", ""), m.get("modified_at", "")[:10])
    return "".join(f'<div class="mt-cell">{html.escape(str(c))}</div>' for c in cells)


def delete_model(name, confirmed):
    """Remove a model from Ollama; its files are deleted from disk."""
    if not confirmed:
        return gr.update(), gr.update()
    try:
        r = requests.delete(f"{OLLAMA}/api/delete", json={"model": name}, timeout=60)
    except requests.RequestException:
        msg = f"Could not delete {name}: Ollama is not running. Start it at the top of the page and try again."
        return note_html(msg, "error"), gr.update()
    msg = f"Deleted {name}." if r.ok else response_error(r, f"Deleting {name}")
    return note_html(msg, "" if r.ok else "error"), time.time()


# ---------------------------------------------------------------- chat (saved conversations)
CHATS_DIR = DATA_DIR / "chats"


def content_text(content):
    """Plain text from any chat content shape (string, Gradio content list, or dict)."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or ""
    if isinstance(content, (list, tuple)):
        return "".join(content_text(c) for c in content)
    return "" if content is None else str(content)


def new_totals():
    return {"replies": 0, "seconds": 0.0, "prompt": 0, "output": 0}


def chat_totals_md(t):
    if not t["replies"]:
        return "No messages yet."
    return (f"**This chat:** {t['replies']} {'reply' if t['replies'] == 1 else 'replies'} &nbsp;·&nbsp; {t['seconds']:.1f} s &nbsp;·&nbsp; "
            f"{t['prompt']:,} prompt tokens &nbsp;·&nbsp; {t['output']:,} response tokens &nbsp;·&nbsp; "
            f"{t['prompt'] + t['output']:,} total")


def _chat_path(cid):
    if not cid or not re.fullmatch(r"[\w-]+", cid):
        return None
    return CHATS_DIR / f"{cid}.json"


class ChatIndex:
    """In-memory inverted index over saved chats (title + every message).

    Words map to the chats that contain them; a sorted word list lets each search term match by prefix with a
    binary search, so typing in the search box never re-reads chat files. Updated on save and delete."""
    WORD = re.compile(r"[a-z0-9]+")

    def __init__(self):
        self.lock = threading.Lock()
        self.meta = {}        # chat id -> {"id", "title", "updated"}
        self.words = {}       # chat id -> set of words
        self.postings = {}    # word -> set of chat ids
        self.sorted_words = []
        self.dirty = False

    def _tokens(self, chat):
        parts = [chat.get("title", "")]
        for m in chat.get("messages", []):
            parts.append(content_text(m.get("content", "")))
            parts += [a.get("name", "") + " " + a.get("text", "") for a in m.get("attachments", [])]
        return set(self.WORD.findall(" ".join(parts).lower()))

    def _drop(self, cid):
        for w in self.words.pop(cid, ()):
            ids = self.postings.get(w)
            if ids is not None:
                ids.discard(cid)
                if not ids:
                    del self.postings[w]
                    self.dirty = True
        self.meta.pop(cid, None)

    def add(self, chat):
        with self.lock:
            cid = chat["id"]
            self._drop(cid)
            words = self._tokens(chat)
            self.words[cid] = words
            for w in words:
                if w not in self.postings:
                    self.postings[w] = set()
                    self.dirty = True
                self.postings[w].add(cid)
            self.meta[cid] = {"id": cid, "title": chat.get("title") or "New chat", "updated": chat.get("updated", 0)}

    def touch(self, chat):
        with self.lock:
            if chat["id"] in self.meta:
                self.meta[chat["id"]]["updated"] = chat.get("updated", 0)

    def remove(self, cid):
        with self.lock:
            self._drop(cid)

    def build(self):
        for f in CHATS_DIR.glob("*.json") if CHATS_DIR.exists() else []:
            try:
                self.add(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                continue

    def search(self, query=""):
        """Chats (newest first) containing every term in the query; each term matches word prefixes."""
        with self.lock:
            terms = self.WORD.findall((query or "").lower())
            if not terms:
                ids = set(self.meta)
            else:
                if self.dirty:
                    self.sorted_words = sorted(self.postings)
                    self.dirty = False
                ids = None
                for term in terms:
                    matched = set()
                    i = bisect.bisect_left(self.sorted_words, term)
                    while i < len(self.sorted_words) and self.sorted_words[i].startswith(term):
                        matched |= self.postings[self.sorted_words[i]]
                        i += 1
                    ids = matched if ids is None else ids & matched
                    if not ids:
                        break
            return sorted((self.meta[i] for i in ids or ()), key=lambda c: -c["updated"])


CHAT_INDEX = ChatIndex()


def save_chat(chat, index=True):
    """Write a chat atomically. index=False (used while a reply streams) only refreshes its timestamp."""
    CHATS_DIR.mkdir(exist_ok=True)
    chat["updated"] = time.time()
    path = _chat_path(chat["id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(chat, indent=1), encoding="utf-8")
    tmp.replace(path)
    if index:
        CHAT_INDEX.add(chat)
    else:
        CHAT_INDEX.touch(chat)


def load_chat(cid):
    path = _chat_path(cid)
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path else None
    except Exception:
        return None


def delete_chat_file(cid):
    """Permanently delete a saved chat: the file is unlinked, not moved to a trash folder."""
    path = _chat_path(cid)
    if path and path.exists():
        path.unlink()
    if path:
        shutil.rmtree(ATTACH_DIR / cid, ignore_errors=True)  # its attachments are deleted permanently too
    CHAT_INDEX.remove(cid)
    return not (path and path.exists())


def display_messages(chat):
    """Chatbot messages from a saved chat, with the stats line under each reply."""
    out = []
    for m in (chat or {}).get("messages", []):
        text = attachment_markup(m.get("attachments")) + m["content"]
        if m["role"] == "assistant" and m.get("stats"):
            text += "\n\n<sub>" + m["stats"] + "</sub>"
        out.append({"role": m["role"], "content": text})
    return out


def empty_chat(model=None):
    return {"id": None, "title": "", "model": model, "messages": [], "totals": new_totals(), "created": None}


def model_lock(chat):
    """Dropdown state for a chat: locked to its model once it has messages, free for a new chat."""
    model = (chat or {}).get("model")
    if chat and chat.get("messages") and model:
        names = model_names()
        return gr.update(value=model, choices=names if model in names else names + [model], interactive=False,
                         label="Model (locked to this chat)")
    return gr.update(interactive=True, label="Model")


def chat_submit(message, chat, model, pending):
    message = (message or "").strip()
    keep = (gr.update(), gr.update(), gr.update())
    if not message and not pending:
        return (gr.update(), chat, gr.update(), gr.update(), gr.update(), gr.update()) + keep
    chat = json.loads(json.dumps(chat or empty_chat()))
    if not (chat.get("messages") and chat.get("model")) and not model:
        return (gr.update(), chat, display_messages(chat) + [
            {"role": "assistant", "content": "Choose a model above first."}], gr.update(), gr.update(), gr.update()) + keep
    new = chat["id"] is None
    if new:
        chat["id"] = time.strftime("%Y%m%d-%H%M%S-") + os.urandom(3).hex()
        chat["created"] = time.time()
        title = re.sub(r"[*_`#>\[\]]+", "", message).strip() or (pending[0]["name"] if pending else "New chat")
        chat["title"] = (title[:48].rstrip() + "...") if len(title) > 48 else title
    if not chat["messages"] or not chat.get("model"):
        chat["model"] = model
    user_msg = {"role": "user", "content": message}
    if pending:
        user_msg["attachments"] = _attach_to_chat(pending, chat["id"])
    chat["messages"].append(user_msg)
    save_chat(chat)
    cleared = ([], "", gr.update(visible=False))
    if new:
        return ("", chat, display_messages(chat), time.time(), chat["id"], model_lock(chat)) + cleared
    return ("", chat, display_messages(chat), gr.update(), gr.update(), model_lock(chat)) + cleared


TYPING = '<span class="typing"><i></i><i></i><i></i></span>'
CHAT_KEEP_ALIVE = "3m"  # keep the model loaded between chat messages, then free the memory


def chat_respond(chat):
    """Stream the chat model's reply, with timing and token usage under the reply. Saves the chat as it goes."""
    if not chat or not chat["messages"] or chat["messages"][-1]["role"] != "user" or not chat.get("model"):
        yield chat, gr.update(), gr.update(), gr.update(), gr.update()
        return
    model = chat["model"]
    has_images = any(a["kind"] == "image" for m in chat["messages"] for a in m.get("attachments", []))
    msgs, skipped_images = api_messages(chat, vision=has_images and "vision" in model_capabilities(model))
    reply = {"role": "assistant", "content": "", "stats": "", "model": model}
    chat["messages"].append(reply)
    reply["content"] = TYPING
    yield chat, display_messages(chat), gr.update(visible=False), gr.update(visible=True), gr.update()
    answer, last, last_save, final, first_token = "", 0.0, time.time(), {}, None
    t0 = time.time()
    try:
        with requests.post(f"{OLLAMA}/api/chat", json={"model": model, "messages": msgs, "stream": True,
                                                        "keep_alive": CHAT_KEEP_ALIVE},
                           stream=True, timeout=(10, 1800)) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                ev = json.loads(line)
                if "error" in ev:
                    answer += f"\n\nError: {ev['error']}"
                    break
                msg = ev.get("message", {})
                if msg.get("content") or msg.get("thinking"):
                    first_token = first_token or time.time()
                answer += msg.get("content", "")
                if ev.get("done"):
                    final = ev
                if time.time() - last > 0.1:
                    last = time.time()
                    reply["content"] = answer or TYPING
                    reply["stats"] = ""
                    if time.time() - last_save > 2:
                        last_save = time.time()
                        save_chat(chat, index=False)
                    yield chat, display_messages(chat), gr.update(), gr.update(), gr.update()
    except Exception as e:
        answer += f"\n\nError: {e}"
    wall = time.time() - t0
    parts = [f"{wall:.1f} s"]
    if final:
        ns = 1e9
        p_tok, o_tok = final.get("prompt_eval_count", 0), final.get("eval_count", 0)
        load = final.get("load_duration", 0) / ns
        if load >= 0.5:
            parts.append(f"{load:.1f} s model load")
        if first_token:
            parts.append(f"{first_token - t0:.1f} s to first token")
        parts.append(f"{p_tok:,} prompt + {o_tok:,} response tokens")
        if final.get("eval_duration"):
            parts.append(f"{o_tok / (final['eval_duration'] / ns):.1f} tokens/s")
        t = chat["totals"]
        chat["totals"] = {"replies": t["replies"] + 1, "seconds": t["seconds"] + wall,
                          "prompt": t["prompt"] + p_tok, "output": t["output"] + o_tok}
    parts.append(model)
    if skipped_images:
        parts.append("images not sent: this model has no vision support")
    reply["content"] = answer or "*No response.*"
    reply["stats"] = " &nbsp;·&nbsp; ".join(_e for _e in parts)
    save_chat(chat)
    yield chat, display_messages(chat), gr.update(visible=True), gr.update(visible=False), chat_totals_md(chat["totals"])


def chat_stopped(chat):
    """After Stop: keep what was generated so far and mark it as stopped."""
    if chat and chat["messages"] and chat["messages"][-1]["role"] == "assistant":
        m = chat["messages"][-1]
        if m["content"] == TYPING:
            m["content"] = "*Stopped.*"
        m["stats"] = "Stopped"
        save_chat(chat)
    return gr.update(visible=True), gr.update(visible=False), chat, display_messages(chat)


def open_chat(cid):
    chat = load_chat(cid)
    if not chat:
        return empty_chat(), [], gr.update(), "This chat no longer exists.", time.time(), None
    return (chat, display_messages(chat), model_lock(chat), chat_totals_md(chat.get("totals", new_totals())),
            time.time(), chat["id"])


def start_new_chat():
    return empty_chat(), [], "No messages yet.", time.time(), None, model_lock(None)


def remove_chat(cid, confirmed, current):
    if not confirmed:
        return current, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    delete_chat_file(cid)
    if current and current.get("id") == cid:
        return empty_chat(), [], "No messages yet.", time.time(), None, model_lock(None)
    return current, gr.update(), gr.update(), time.time(), gr.update(), gr.update()


# ---------------------------------------------------------------- chat attachments
ATTACH_DIR = DATA_DIR / "attachments"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
DOC_EXTS = {".pdf", ".docx"}
TEXT_EXTS = {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".xml", ".html", ".htm", ".log", ".ini",
             ".toml", ".cfg", ".conf", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp", ".cs",
             ".go", ".rs", ".rb", ".php", ".sh", ".bash", ".ps1", ".sql", ".css", ".swift", ".kt", ".lua", ".r", ".tex"}
ATTACH_TYPES = sorted(IMAGE_EXTS | DOC_EXTS | TEXT_EXTS)
DOC_TYPES = sorted(DOC_EXTS | TEXT_EXTS)
MAX_ATTACH_CHARS = 24000  # per file; the model's context window is limited, so long files are truncated


def _human_size(n):
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{max(1, round(n / 1e3))} KB"


def extract_text(path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    if ext == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    return path.read_text(encoding="utf-8", errors="replace")


def add_attachments(files, pending, model=None):
    """Stage uploaded files for the next message. Text is extracted now so problems show before sending."""
    pending = list(pending or [])
    errors = []
    vision = bool(model) and "vision" in model_capabilities(model)
    staging = ATTACH_DIR / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    for f in files or []:
        src = Path(f if isinstance(f, str) else getattr(f, "name", ""))
        ext = src.suffix.lower()
        name = src.name.split("/")[-1]
        if ext not in IMAGE_EXTS | DOC_EXTS | TEXT_EXTS:
            errors.append(f"{name}: unsupported file type")
            continue
        if ext in IMAGE_EXTS and not vision:
            errors.append(f"{name}: this model can't view images. Documents and text files still work.")
            continue
        dest = staging / f"{os.urandom(4).hex()}-{re.sub(r'[^A-Za-z0-9._-]+', '_', name)}"
        shutil.copyfile(src, dest)
        item = {"name": name, "kind": "image" if ext in IMAGE_EXTS else "text", "path": str(dest), "size": dest.stat().st_size}
        if item["kind"] == "text":
            try:
                text = extract_text(dest).strip()
            except Exception as e:
                errors.append(f"{name}: could not read ({e})")
                dest.unlink(missing_ok=True)
                continue
            if not text:
                errors.append(f"{name}: no readable text (scanned PDFs need OCR)")
                dest.unlink(missing_ok=True)
                continue
            item["truncated"] = len(text) > MAX_ATTACH_CHARS
            item["text"] = text[:MAX_ATTACH_CHARS]
        pending.append(item)
    return pending, attachments_html(pending, errors), gr.update(visible=bool(pending))


def attachments_html(pending, errors=()):
    chips = []
    for a in pending or []:
        if a["kind"] == "image":
            thumb = f'<img src="/gradio_api/file={html.escape(a["path"])}" alt="">'
        else:
            thumb = '<span class="chip-doc"></span>'
        extra = " · truncated" if a.get("truncated") else ""
        chips.append(f'<span class="attach-chip">{thumb}<span class="chip-name">{html.escape(a["name"])}</span>'
                     f'<span class="chip-size">{_human_size(a["size"])}{extra}</span></span>')
    errs = "".join(f'<div class="note-line error">{html.escape(e)}</div>' for e in errors)
    return (f'<div class="attach-chips">{"".join(chips)}</div>' if chips else "") + errs


def clear_attachments(pending):
    for a in pending or []:
        Path(a["path"]).unlink(missing_ok=True)
    return [], "", gr.update(visible=False)


def _attach_to_chat(pending, cid):
    folder = ATTACH_DIR / cid
    folder.mkdir(parents=True, exist_ok=True)
    stored = []
    for a in pending:
        src = Path(a["path"])
        if not src.exists():
            continue
        dest = folder / src.name
        src.replace(dest)
        stored.append({**a, "path": str(dest)})
    return stored


def attachment_markup(attachments):
    parts = []
    for a in attachments or []:
        if a["kind"] == "image":
            parts.append(f'<img class="attach-img" src="/gradio_api/file={html.escape(a["path"])}" alt="{html.escape(a["name"])}">')
        else:
            parts.append(f'<span class="attach-chip in-message"><span class="chip-doc"></span><span class="chip-name">'
                         f'{html.escape(a["name"])}</span><span class="chip-size">{_human_size(a["size"])}</span></span>')
    return f'<div class="attach-row">{"".join(parts)}</div>' if parts else ""


def attach_state(model):
    """Documents can be attached for any model; images only for models that can see them (vision)."""
    vision = bool(model) and "vision" in model_capabilities(model)
    return gr.update(file_types=ATTACH_TYPES if vision else DOC_TYPES,
                     elem_classes=["attach-btn"] if vision else ["attach-btn", "docs-only"])


def model_capabilities(model):
    try:
        return requests.post(f"{OLLAMA}/api/show", json={"model": model}, timeout=10).json().get("capabilities", [])
    except Exception:
        return []


def api_messages(chat, vision):
    """Messages for Ollama: attached text is inlined; images are sent only to models that can see them."""
    msgs, skipped = [], []
    for m in chat["messages"]:
        content = content_text(m["content"])
        images = []
        for a in m.get("attachments", []):
            if a["kind"] == "text":
                note = " (truncated)" if a.get("truncated") else ""
                content += f"\n\n[Attached file: {a['name']}{note}]\n{a.get('text', '')}"
            elif vision:
                try:
                    images.append(base64.b64encode(Path(a["path"]).read_bytes()).decode())
                except OSError:
                    skipped.append(a["name"])
            else:
                skipped.append(a["name"])
                content += f"\n\n[An image named {a['name']} was attached, but this model cannot view images.]"
        msg = {"role": m["role"], "content": content}
        if images:
            msg["images"] = images
        msgs.append(msg)
    return msgs, skipped


# ---------------------------------------------------------------- model search
def repo_from_text(text):
    text = re.sub(r"^https?://(www\.)?(huggingface\.co|hf\.co)/", "", (text or "").strip())
    text = re.sub(r"^(huggingface\.co|hf\.co)/", "", text)
    return text.split("/blob/")[0].split("/tree/")[0].split(":")[0].strip("/")


def ollama_name_from_text(text):
    text = re.sub(r"^https?://(www\.)?ollama\.com/(library/)?", "", (text or "").strip())
    return text.split("/tags")[0].strip("/")


PLEASE_CHOOSE = ("Please choose", "")
LOADING = ("...", "")
UNCENSORED_RE = re.compile(r"abliterat|uncensor|heretic|obliterat", re.I)
_ollama_meta = {}


def note_html(text, kind=""):
    """Status line under the search fields. kind: '', 'loading', 'error'."""
    spinner = '<span class="spinner"></span>' if kind == "loading" else ""
    return f'<div class="note-line {kind}">{spinner}{html.escape(text)}</div>' if text else ""


def hf_find_repos(query):
    q = repo_from_text(query)
    if "/" in q and q.lower().endswith("gguf"):
        return [q], q
    res = requests.get(f"{HF_API}/models", timeout=15, params={
        "search": q.split("/")[-1], "filter": "gguf", "sort": "downloads", "limit": 25})
    res.raise_for_status()
    return [m["id"] for m in res.json()], q


def hf_versions(repo):
    """[(size_gb, tag)] for single-file GGUF quantizations in a Hugging Face repo."""
    res = requests.get(f"{HF_API}/models/{repo_from_text(repo)}/tree/main", params={"recursive": "true"}, timeout=15)
    res.raise_for_status()
    opts = {}
    for f in res.json():
        name = f.get("path", "").split("/")[-1]
        low = name.lower()
        if (not low.endswith(".gguf") or any(k in low for k in ("mmproj", "imatrix", "mtp"))
                or re.search(r"-\d{5}-of-\d{5}", low)):
            continue
        m = re.search(r"((?:UD-)?(?:I?Q\d\w*|BF16|F16|F32))\.gguf$", name, re.I)
        if m:
            opts[m.group(1)] = f.get("size", 0) / 1e9
    return sorted((size, tag) for tag, size in opts.items())


def ollama_find(query):
    q = ollama_name_from_text(query).split(":")[0]
    page = requests.get(f"{OLLAMA_WEB}/search", params={"q": q}, timeout=15)
    page.raise_for_status()
    results = []
    for li in re.findall(r"<li[^>]*>(.*?)</li>", page.text, re.S):
        m = re.search(r'href="/library/([^"/:]+)"', li)
        if not m:
            continue
        desc = re.search(r"<p[^>]*break-words[^>]*>(.*?)</p>", li, re.S)
        caps = re.findall(r'bg-indigo-50[^>]*>\s*([^<]+?)\s*<', li)
        pulls = re.search(r"<span\s*>([\d.,]+[KMB]?)</span>\s*<span[^>]*>&nbsp;Pulls", li)
        results.append({"name": m.group(1), "desc": html.unescape(desc.group(1).strip()) if desc else "",
                        "caps": caps, "pulls": pulls.group(1) if pulls else ""})
    results.sort(key=lambda r: r["name"] != q.lower())  # exact match first, otherwise keep Ollama's ranking
    _ollama_meta.update({r["name"]: r for r in results})
    return results, q


def ollama_versions(name):
    """[(size_gb, tag)] for locally runnable tags of an Ollama library model."""
    name = ollama_name_from_text(name).split(":")[0]
    page = requests.get(f"{OLLAMA_WEB}/library/{name}/tags", timeout=15)
    page.raise_for_status()
    opts = {}
    for tag, size, unit in re.findall(
            r'href="/library/([^"]+)" class="md:hidden.*?font-mono">\s*\w+\s*</span>\s*•\s*([\d.]+)\s*([KMG])B', page.text, re.S):
        if "mlx" not in tag:  # MLX builds only run on Apple Silicon
            opts.setdefault(tag, float(size) * {"K": 1e-6, "M": 1e-3, "G": 1}[unit])
    return sorted((size, tag) for tag, size in opts.items())


def best_version(opts):
    """Largest version that fits the VRAM budget; otherwise the smallest available."""
    fitting = [o for o in opts if o[0] <= VRAM_BUDGET_GB]
    if fitting:
        top = max(size for size, _ in fitting)
        return min((t for size, t in fitting if size == top), key=lambda t: (not t.endswith(":latest"), len(t)))
    return opts[0][1] if opts else ""


def loading_screen(title, detail=""):
    return gr.update(visible=True, value=(
        f'<div class="ls-inner"><div class="ls-spinner"></div><div class="ls-title">{html.escape(title)}</div>'
        f'<div class="ls-detail">{html.escape(detail)}</div></div>'))


def source_changed(source):
    ph = "qwen3, llama3.2, gemma3" if source == SRC_OLLAMA else "Qwen3.8-27B, or a Hugging Face link"
    label = "Model" if source == SRC_OLLAMA else "Repository"
    return (gr.update(placeholder=ph, value=""), gr.update(choices=[PLEASE_CHOOSE], value="", label=label,
                                                           interactive=False),
            gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False), "", gr.update(interactive=False))


def search_models(source, query):
    """Search, then ask the user to choose a repository/model. Downloads stay disabled until both fields are set."""
    where = "the Ollama library" if source == SRC_OLLAMA else "Hugging Face"
    no_version = gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False)
    hide = gr.update(visible=False)
    if not (query or "").strip():
        yield gr.update(), gr.update(), note_html("Enter a model name to search.", "error"), gr.update(interactive=False), hide
        return
    yield (gr.update(choices=[LOADING], value="", interactive=False), no_version, "", gr.update(interactive=False),
           loading_screen(f"Searching {where}", f"Looking for \"{query.strip()}\""))
    try:
        if source == SRC_OLLAMA:
            results, q = ollama_find(query)
            choices = [(f"{r['name']}   {r['pulls']} pulls" + (f"   {', '.join(r['caps'])}" if r["caps"] else ""),
                        r["name"]) for r in results]
            found = f"Found {len(results)} models in the Ollama library. Choose one to see its versions."
        else:
            repos, q = hf_find_repos(query)
            choices = [(r + ("   (modified: abliterated / uncensored)" if UNCENSORED_RE.search(r) else ""), r)
                       for r in repos]
            found = f"Found {len(repos)} GGUF repositories, sorted by downloads. Choose one to see its versions."
            if "/" in q and not q.lower().endswith("gguf"):
                found = f"{q} is not in GGUF format, so these are converted versions of it. " + found
    except Exception as e:
        yield (gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False), no_version,
               note_html(f"Search failed: {e}", "error"), gr.update(interactive=False), hide)
        return
    if not choices:
        msg = (f"No models named \"{q}\" found in the Ollama library." if source == SRC_OLLAMA else
               f"No GGUF versions of \"{q}\" found on Hugging Face. Ollama can only import GGUF models.")
        yield (gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False), no_version, note_html(msg, "error"),
               gr.update(interactive=False), hide)
        return
    yield (gr.update(choices=[PLEASE_CHOOSE] + choices, value="", interactive=True), no_version,
           note_html(found), gr.update(interactive=False), hide)


def list_versions(source, repo):
    """Load versions for the chosen repository/model and auto-select the best fit for the GPU."""
    hide = gr.update(visible=False)
    if not repo:
        yield gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False), "", gr.update(interactive=False), hide
        return
    yield (gr.update(choices=[LOADING], value="", interactive=False), "", gr.update(interactive=False),
           loading_screen("Loading versions", repo))
    try:
        opts = ollama_versions(repo) if source == SRC_OLLAMA else hf_versions(repo)
    except Exception as e:
        yield (gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False),
               note_html(f"Could not list versions: {e}", "error"), gr.update(interactive=False), hide)
        return
    if not opts:
        msg = ("No downloadable versions found (cloud-only models can't run locally)." if source == SRC_OLLAMA else
               "This repository has no single-file GGUF models Ollama can pull.")
        yield (gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False), note_html(msg, "error"),
               gr.update(interactive=False), hide)
        return
    best = best_version(opts)
    choices = [PLEASE_CHOOSE] + [
        (f"{tag.split(':', 1)[-1]}   {size:.1f} GB   {fit_label(size)}" + ("   (Recommended)" if tag == best else ""), tag)
        for size, tag in opts]
    size = next(sz for sz, t in opts if t == best)
    desc = _ollama_meta.get(repo, {}).get("desc", "") if source == SRC_OLLAMA else ""
    why = (f"Auto-selected {best.split(':', 1)[-1]} ({size:.1f} GB), the largest version under {VRAM_BUDGET_GB} GB "
           "so it runs fully on your GPU." if size <= VRAM_BUDGET_GB else
           f"No version fits under {VRAM_BUDGET_GB} GB, so the smallest ({size:.1f} GB) was selected; it will "
           "run partly in system memory.")
    yield (gr.update(choices=choices, value=best, interactive=True),
           note_html((desc + " " if desc else "") + why), gr.update(interactive=True), hide)


def version_changed(repo, version):
    return gr.update(interactive=bool(repo and version))


# ---------------------------------------------------------------- downloads (background, resumable)
DOWNLOADS_FILE = DATA_DIR / "downloads.json"
_downloads = {}  # ref -> {"state", "msg", "done", "total", "cancel", "pause", "running"}
_dl_lock = threading.Lock()
_dl_finished = {"count": 0}
ACTIVE = ("queued", "downloading", "waiting", "paused")
DONE_VISIBLE_SECS = 30  # finished downloads stay in the list this long
_priority = {"ref": None}


def _yield_to_priority(ref):
    """True while another download has priority and is still running."""
    p = _priority["ref"]
    other = _downloads.get(p, {})
    return bool(p and p != ref and other.get("state") in ACTIVE and not other.get("pause"))


def _should_stop(ref, d):
    return d["cancel"] or d.get("pause") or _yield_to_priority(ref)


def _pending():
    """[(ref, paused)] for downloads to continue after a restart."""
    try:
        entries = json.loads(DOWNLOADS_FILE.read_text())
    except Exception:
        return []
    return [(e, False) if isinstance(e, str) else (e["ref"], bool(e.get("paused"))) for e in entries]


class PermanentDownloadError(Exception):
    """The download can't succeed as requested (missing file, rejected request), so retrying won't help."""


def response_error(response, operation):
    """The server's own explanation for a failed request, e.g. 'Pulling x failed (HTTP 404): not found'."""
    detail = ""
    try:
        payload = response.json()
        detail = str(payload.get("error") or payload.get("message") or "") if isinstance(payload, dict) else str(payload or "")
    except ValueError:
        detail = (response.text or "").strip()
    detail = re.sub(r"\s+", " ", detail).strip()[:500]
    return f"{operation} failed (HTTP {response.status_code})" + (f": {detail}" if detail else "")


def _check_response(response, operation):
    """4xx means the request itself is wrong and fails for good; 5xx and network errors are retried."""
    if 400 <= response.status_code < 500:
        raise PermanentDownloadError(response_error(response, operation))
    response.raise_for_status()


def fetch_file(url, dest, size, should_stop, on_progress):
    """Download url to dest, resuming from dest.part with an HTTP range request. Returns False if should_stop()
    interrupted it (the partial file is kept), True once dest is complete."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=(15, 120), allow_redirects=True) as r:
        if r.status_code != 416:  # 416: the partial file is already complete
            _check_response(r, f"Downloading {dest.name}")
            if have and r.status_code != 206:  # server ignored the range: start this file over
                have = 0
            with open(part, "ab" if have else "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    if should_stop():
                        return False
                    fh.write(chunk)
                    on_progress()
    if size and part.stat().st_size < size:
        raise ConnectionError("download ended early")
    part.replace(dest)
    return True


HF_DOWNLOADS = DATA_DIR / "hf-downloads"  # GGUF files waiting to be imported into Ollama; deleted once imported


def _hf_gguf_files(repo, tag):
    """[(path, size)] to import for an Ollama quantization tag like Q4_K_M: the model's GGUF file, plus the image
    projector (mmproj) if the repo has one, which vision models need to see images."""
    res = requests.get(f"{HF_API}/models/{repo}/tree/main", params={"recursive": "true"}, timeout=30)
    _check_response(res, f"Listing files of {repo}")
    ggufs = [(e["path"], int(e.get("size") or 0)) for e in res.json() if e.get("path", "").lower().endswith(".gguf")]
    stem = lambda path: path.rsplit("/", 1)[-1][:-5]  # noqa: E731
    wanted = re.compile(rf"(^|[-_.]){re.escape(tag.removesuffix('.gguf'))}$", re.I)
    models = [g for g in ggufs if "mmproj" not in g[0].lower() and wanted.search(stem(g[0]))]
    if not models:
        raise PermanentDownloadError(f"{repo} has no single-file GGUF for {tag}")
    files = [min(models, key=lambda g: len(g[0]))]
    projectors = [g for g in ggufs if "mmproj" in g[0].lower()]
    if projectors:  # prefer the one matching the model's quantization, then full precision
        rank = lambda g: (not wanted.search(stem(g[0])), not re.search(r"f16|bf16", stem(g[0]), re.I), g[1])  # noqa: E731
        files.append(min(projectors, key=rank))
    return files


def _hf_import(ref, d):
    """Download a Hugging Face GGUF directly and import it into Ollama. Ollama's own pull fails for most Hugging
    Face models ("blocked redirect to a different host") because the files are served from Hugging Face's Xet
    storage on another domain. Returns True once imported, False if cancelled or paused for another download."""
    m = re.fullmatch(r"hf\.co/([^:]+):(.+)", ref)
    if not m:
        raise PermanentDownloadError(f"{ref} is not a Hugging Face model reference")
    repo, tag = m.groups()
    files = _hf_gguf_files(repo, tag)
    HF_DOWNLOADS.mkdir(parents=True, exist_ok=True)
    dests = _hf_import_files(ref)[:len(files)]
    total = sum(size for _, size in files)

    def progress():
        d.update(state="downloading", msg="Downloading", total=total, done=sum(
            f.stat().st_size for dest in dests for f in (dest, dest.with_name(dest.name + ".part")) if f.exists()))

    for (path, size), dest in zip(files, dests):
        if not dest.exists():
            progress()
            if not fetch_file(HF_RESOLVE.format(repo=repo, path=path), dest, size, lambda: _should_stop(ref, d), progress):
                return False
    d.update(state="downloading", msg="Importing into Ollama", done=total, total=total)
    ollama_cli = OLLAMA_DIR / "bin/ollama"
    if not ollama_cli.exists():
        ollama_cli = Path(shutil.which("ollama") or "ollama")
    modelfile = HF_DOWNLOADS / f"{_slug(ref)}.Modelfile"
    modelfile.write_text("".join(f"FROM {dest}\n" for dest in dests), encoding="utf-8")  # 2nd FROM: the projector
    try:
        result = subprocess.run([str(ollama_cli), "create", ref, "-f", str(modelfile)], capture_output=True, text=True,
                                timeout=3600, env=os.environ | {"OLLAMA_HOST": OLLAMA})
    finally:
        modelfile.unlink(missing_ok=True)
    if result.returncode:
        detail = ANSI.sub("", result.stderr or result.stdout or f"exit code {result.returncode}").strip()
        raise RuntimeError(f"Ollama import failed: {detail[-300:]}")
    for dest in dests:
        dest.unlink(missing_ok=True)  # Ollama keeps its own copy
    return True


def _hf_import_files(ref):
    """Where _hf_import keeps a model's GGUF and its projector until Ollama has imported them."""
    return [HF_DOWNLOADS / f"{_slug(ref)}.gguf", HF_DOWNLOADS / f"{_slug(ref)}.mmproj.gguf"]


def _set_pending(ref, add, paused=False):
    with _dl_lock:
        entries = [{"ref": r, "paused": True} if p else r for r, p in _pending() if r != ref]
        if add:
            entries.append({"ref": ref, "paused": True} if paused else ref)
        DOWNLOADS_FILE.write_text(json.dumps(entries, indent=2))


def _run_download(ref):
    """Runs a download's worker on its own thread until it finishes, fails, is deleted or is paused."""
    d = _downloads[ref]
    worker = _image_download_worker if ref.startswith("image:") else _download_worker
    while True:
        worker(ref)
        with _dl_lock:
            if d["state"] == "paused" and not d.get("pause") and not d["cancel"]:
                continue  # resumed while it was stopping
            d["running"] = False
            return


def _start_worker(ref):
    _downloads[ref]["running"] = True
    threading.Thread(target=_run_download, args=(ref,), daemon=True).start()


def start_download(ref):
    with _dl_lock:
        d = _downloads.get(ref)
        if d and d["state"] in ACTIVE:
            if not d.get("pause"):
                return False
            d["pause"] = False  # asking for a paused download again resumes it
            d.update(state="queued", msg="Resuming")
            if not d.get("running"):
                _start_worker(ref)
        else:
            _downloads[ref] = {"state": "queued", "msg": "Queued", "done": 0, "total": 0, "cancel": False}
            _start_worker(ref)
    _set_pending(ref, True)
    return True


def _partial_files(ref):
    """Files a stopped download leaves behind (Ollama discards its own partial pulls when it restarts)."""
    if ref.startswith("image:"):
        key = ref.split(":", 1)[1]
        if key not in image_models():
            return []
        return [f.with_name(f.name + ".part") for f in image_model_files(key).values()]
    if ref.startswith("hf.co/"):
        return [f for gguf in _hf_import_files(ref) for f in (gguf.with_name(gguf.name + ".part"), gguf)]
    return []


def _worker_stopped(ref, d):
    """A worker left its loop without finishing: deleted (partial data removed) or paused (kept to resume)."""
    if d["cancel"]:
        if d.get("hidden"):
            for f in _partial_files(ref):
                f.unlink(missing_ok=True)
        d.update(state="cancelled", msg="Cancelled")
        _set_pending(ref, False)
    elif d.get("pause"):  # (if it was resumed meanwhile, _run_download starts it again)
        d.update(state="paused", msg="Paused")
        _set_pending(ref, True, paused=True)
    else:
        d.update(state="paused")


def _download_worker(ref):
    """Pull a model, retrying through network drops and Ollama restarts. Ollama keeps partially
    downloaded layers, so every retry (and every app restart) continues where it left off."""
    d = _downloads[ref]
    attempt = 0
    while not (d["cancel"] or d.get("pause")):
        if _yield_to_priority(ref):
            d.update(state="paused", msg="Paused for priority download")
            time.sleep(1)
            continue
        if not ollama_up():
            d.update(state="waiting", msg="Waiting for Ollama to start")
            time.sleep(5)
            continue
        layers = {}
        try:
            if d.get("direct"):  # Ollama can't fetch this one itself
                if _hf_import(ref, d):
                    d.update(state="done", msg="Done", finished=time.time())
                    _set_pending(ref, False)
                    _dl_finished["count"] += 1
                    return
                continue  # cancelled or paused; the loop handles both
            with requests.post(f"{OLLAMA}/api/pull", json={"model": ref, "stream": True},
                               stream=True, timeout=(10, 120)) as r:
                if r.status_code >= 400:  # Ollama reports some failures, including the Xet redirect, this way
                    message = response_error(r, f"Pulling {ref}")
                    if ref.startswith("hf.co/") and "redirect" in message:
                        d["direct"] = True
                        continue
                    raise PermanentDownloadError(message)
                for line in r.iter_lines():
                    if _should_stop(ref, d):
                        break
                    if not line:
                        continue
                    ev = json.loads(line)  # a line cut off mid-transfer raises, and is retried like any drop
                    if "error" in ev:
                        if ref.startswith("hf.co/") and "redirect" in ev["error"]:
                            d["direct"] = True
                            break
                        raise PermanentDownloadError(ev["error"])
                    if ev.get("digest") and ev.get("total"):
                        layers[ev["digest"]] = (ev.get("completed", 0), ev["total"])
                        done = sum(c for c, _ in layers.values())
                        total = sum(t for _, t in layers.values())
                        d.update(state="downloading", msg="Downloading", done=done, total=total)
                    elif ev.get("status"):
                        d.update(state="downloading", msg=ev["status"].capitalize())
                    if ev.get("status") == "success":
                        d.update(state="done", msg="Done", finished=time.time())
                        _set_pending(ref, False)
                        _dl_finished["count"] += 1
                        return
            if d["cancel"] or d.get("pause") or d.get("direct"):
                continue
            if _yield_to_priority(ref):
                continue  # Ollama keeps the partial data; this resumes once the priority download finishes
            raise ConnectionError("download stream ended early")
        except PermanentDownloadError as e:  # e.g. the model or version doesn't exist
            d.update(state="failed", msg=str(e))
            _set_pending(ref, False)
            return
        except Exception:  # network drop, timeout, Ollama restarted: retry and resume
            attempt += 1
            wait = min(60, 5 * attempt)
            d.update(state="waiting", msg=f"Connection interrupted, resuming in {wait}s")
            time.sleep(wait)
    _worker_stopped(ref, d)


def prioritize_download(ref):
    if ref and _downloads.get(ref, {}).get("state") in ACTIVE:
        _priority["ref"] = ref
        return note_html(f"{ref} now has priority. Other downloads are paused until it finishes.")
    return gr.update()


def remove_download(ref):
    """Trash icon on a download: stop it and remove it from the list, deleting its partial data. (Ollama discards
    partial data of its own pulls the next time it starts, since nothing references it.)"""
    d = _downloads.get(ref)
    if not d:
        return gr.update(), gr.update()
    with _dl_lock:
        d["hidden"] = True
        if d["state"] in ACTIVE:
            d["cancel"] = True
        running = d.get("running")
    if not running:  # nothing will clean up after it, e.g. a paused download
        for f in _partial_files(ref):
            f.unlink(missing_ok=True)
        if d["state"] in ACTIVE:
            d.update(state="cancelled", msg="Cancelled")
    _set_pending(ref, False)
    if _priority["ref"] == ref:
        _priority["ref"] = None
    return note_html(f"Deleted {download_label(ref)} from downloads."), ""


def toggle_download(action):
    """Pause or resume icon on a download. action is "pause:<ref>" or "resume:<ref>"."""
    verb, _, ref = (action or "").partition(":")
    d = _downloads.get(ref)
    if not d or d["state"] not in ACTIVE:
        return gr.update(), ""
    if verb == "pause" and not d.get("pause"):
        with _dl_lock:
            d["pause"] = True
            d.update(state="paused", msg="Paused")
        _set_pending(ref, True, paused=True)
        if _priority["ref"] == ref:
            _priority["ref"] = None
    elif verb == "resume" and d.get("pause"):
        start_download(ref)
    return gr.update(), ""


def resume_pending_downloads():
    """After a restart: continue unfinished downloads, and list paused ones so they can be resumed."""
    for ref, paused in _pending():
        if paused:
            done = sum(f.stat().st_size for f in _partial_files(ref) if f.exists())  # shown until it resumes
            _downloads[ref] = {"state": "paused", "msg": "Paused", "done": done, "total": 0, "cancel": False,
                               "pause": True, "running": False}
        else:
            start_download(ref)


def download_label(ref):
    if ref.startswith("image:") and ref[6:] in image_models():
        return f"{image_models()[ref[6:]]['name']} (image model)"
    return ref


def downloads_html():
    now = time.time()
    visible = [(ref, d) for ref, d in _downloads.items()
               if not d.get("hidden") and not (d["state"] == "done" and now - d.get("finished", now) > DONE_VISIBLE_SECS)]
    if not visible:
        return '<div class="dl-empty">No active downloads.</div>'
    rows = []
    for ref, d in reversed(visible):
        pct = 100 * d["done"] / d["total"] if d["total"] else (100 if d["state"] == "done" else 0)
        size = f'{d["done"]/1e9:.1f} of {d["total"]/1e9:.1f} GB' if d["total"] else ""
        detail = f'{pct:.0f}%   {size}' if d["state"] in ("downloading", "paused") and d["total"] else ""
        if d["state"] == "paused" and not d["total"] and d["done"]:
            detail = f'{d["done"] / 1e9:.1f} GB downloaded'
        toggle = ""
        if d["state"] in ACTIVE:
            action = "resume" if d.get("pause") else "pause"
            toggle = (f'<button class="dl-toggle {action}" type="button" title="{action.capitalize()} download" '
                      f'aria-label="{action.capitalize()} download" data-action="{action}:{html.escape(ref)}"></button>')
        remove = "Delete download" if d["state"] in ACTIVE else "Remove from list"
        rows.append(
            f'<div class="dl"><div class="dl-top"><span class="dl-name" title="{html.escape(ref)}">{html.escape(download_label(ref))}</span>'
            f'<span class="dl-state {d["state"]}">{html.escape(d["msg"])}</span>{toggle}'
            f'<button class="dl-del" type="button" title="{remove}" aria-label="{remove}" data-ref="{html.escape(ref)}"'
            f' data-active="{int(d["state"] in ACTIVE)}"></button></div>'
            f'<div class="bar"><div class="fill {d["state"]}" style="width:{pct:.1f}%"></div></div>'
            f'<div class="dl-detail">{detail}</div></div>')
    return "".join(rows)


def active_downloads():
    """Downloads that are running or waiting their turn (not the ones paused with the pause icon)."""
    return [r for r, d in _downloads.items() if d["state"] in ACTIVE and not d.get("hidden") and not d.get("pause")]


def downloads_tick(seen, picked):
    """Timer callback: refresh the downloads panel, and the model lists when something finished."""
    active = active_downloads()
    if _priority["ref"] not in active:
        _priority["ref"] = None
    value = picked if picked in active else (_priority["ref"] or (active[0] if active else None))
    pick_upd = gr.update(choices=[(download_label(r) + ("   (priority)" if r == _priority["ref"] else ""), r) for r in active],
                         value=value)
    row_upd = gr.update(visible=bool(active))
    if _dl_finished["count"] == seen:
        return downloads_html(), seen, gr.update(), pick_upd, row_upd
    return downloads_html(), _dl_finished["count"], time.time(), pick_upd, row_upd


def pull_model(source, repo, version):
    if not repo or not version:
        missing = "a repository" if not repo else "a version"
        return note_html(f"Choose {missing} before downloading.", "error")
    if source == SRC_OLLAMA:
        ref = ollama_name_from_text(version)
    else:
        ref = f"hf.co/{repo_from_text(repo)}:{version}"
    if not start_download(ref):
        return note_html(f"{ref} is already downloading.")
    return note_html(f"Downloading {ref}. You can keep working or close this page; downloads continue in the "
                     "background and resume automatically if interrupted.")


# ---------------------------------------------------------------- image generation (stable-diffusion.cpp)
SD_DIR = Path(os.environ.get("LLM_SCANNER_SD_DIR") or Path.home() / ".local/sd-cpp")
SD_CLI = SD_DIR / "sd-cli"
IMAGE_MODEL_DIR = DATA_DIR / "image-models"
IMAGES_DIR = DATA_DIR / "images"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"


SHARED_DIR = IMAGE_MODEL_DIR / "shared"      # companion files (text encoders, VAEs) reused across models
CUSTOM_IMAGE_FILE = DATA_DIR / "image_models.json"

# Companion files each model family needs besides the diffusion model itself. All are ungated on Hugging Face.
COMPANIONS = {
    "qwen3-4b": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
    "flux-ae": ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
    "clip-l": ("comfyanonymous/flux_text_encoders", "clip_l.safetensors"),
    "t5xxl": ("city96/t5-v1_1-xxl-encoder-gguf", "t5-v1_1-xxl-encoder-Q4_K_M.gguf"),
}
FLUX_ARGS = ["--cfg-scale", "1.0", "--sampling-method", "euler", "--diffusion-fa", "--offload-to-cpu", "--clip-on-cpu"]
FAMILIES = {
    "z-image": {"label": "Z-Image", "role": "diffusion-model", "companions": {"llm": "qwen3-4b", "vae": "flux-ae"},
                "steps": 8, "size": "Square 1024 x 1024",
                "args": ["--cfg-scale", "1.0", "--diffusion-fa", "--offload-to-cpu"]},
    "flux1-schnell": {"label": "FLUX.1 schnell", "role": "diffusion-model",
                      "companions": {"clip_l": "clip-l", "t5xxl": "t5xxl", "vae": "flux-ae"},
                      "steps": 4, "size": "Square 1024 x 1024", "args": FLUX_ARGS},
    "flux1-dev": {"label": "FLUX.1 dev", "role": "diffusion-model",
                  "companions": {"clip_l": "clip-l", "t5xxl": "t5xxl", "vae": "flux-ae"},
                  "steps": 20, "size": "Square 1024 x 1024", "args": FLUX_ARGS},
    "sdxl": {"label": "Stable Diffusion XL", "role": "model", "companions": {}, "steps": 25,
             "size": "Square 1024 x 1024", "args": ["--cfg-scale", "6.0", "--offload-to-cpu"]},
    "sd15": {"label": "Stable Diffusion 1.5", "role": "model", "companions": {}, "steps": 25,
             "size": "Small 512 x 512 (fastest)", "args": ["--cfg-scale", "7.0"]},
}
UNSUPPORTED = [
    (r"flux[._-]?2|klein", "FLUX.2 isn't supported: its image decoder requires a Hugging Face login."),
    (r"qwen[._-]?image", "Qwen-Image isn't supported yet (it needs about 20 GB and a separate 7B encoder)."),
    (r"wan[._-]?2|hunyuan|ltx|video|animatediff", "This is a video model, not an image model."),
    (r"chroma|hidream|sd3|stable-diffusion-3|cogview|sana|pixart|kolors|lumina|omnigen|ideogram",
     "This model family isn't supported yet."),
    (r"inpaint|instruct[-_]?pix|edit\b", "This model edits existing images (inpainting), it doesn't create new ones."),
    (r"text[._-]?encoder|t5|clip|vae|lora|controlnet|ip[._-]?adapter|upscal",
     "This is an add-on or component file, not a complete image model."),
]

BUILTIN_IMAGE_MODELS = {
    "z-image-turbo": {
        "name": "Z-Image Turbo", "family": "z-image",
        "desc": "Fast, photorealistic text-to-image model (6B) from Alibaba Tongyi Lab. Handles text in images well.",
        "files": {"diffusion-model": ["leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q4_K.gguf", False],
                  "llm": [*COMPANIONS["qwen3-4b"], True], "vae": [*COMPANIONS["flux-ae"], True]},
        "steps": 8, "size": "Square 1024 x 1024", "args": FAMILIES["z-image"]["args"],
    },
}


def _custom_image_models():
    try:
        return json.loads(CUSTOM_IMAGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def image_models():
    return {**BUILTIN_IMAGE_MODELS, **_custom_image_models()}


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80].strip("-")


IMAGE_SIZES = {"Square 1024 x 1024": (1024, 1024), "Square 768 x 768 (faster)": (768, 768),
               "Portrait 768 x 1024": (768, 1024), "Landscape 1024 x 768": (1024, 768),
               "Wide 1280 x 720": (1280, 720), "Small 512 x 512 (fastest)": (512, 512)}
_image_proc = {"p": None}


def image_model_files(key):
    """Local path for each file of a model. Shared companion files live in one place for every model that uses them."""
    out = {}
    for role, spec in image_models()[key]["files"].items():
        repo, path = spec[0], spec[1]
        shared = len(spec) > 2 and spec[2]
        folder = SHARED_DIR / _slug(repo) if shared else IMAGE_MODEL_DIR / key
        out[role] = folder / Path(path).name
    return out


def image_model_installed(key):
    return key in image_models() and all(p.exists() for p in image_model_files(key).values())


def installed_image_models():
    return [(m["name"], k) for k, m in image_models().items() if image_model_installed(k)]


def migrate_shared_files():
    """Earlier versions stored Z-Image Turbo's encoder and VAE in its own folder; move them to the shared folder."""
    for key in BUILTIN_IMAGE_MODELS:
        for dest in image_model_files(key).values():
            legacy = IMAGE_MODEL_DIR / key / dest.name
            if legacy != dest and legacy.exists() and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                legacy.replace(dest)


def detect_family(repo):
    """(family key or None, reason if unsupported) from a Hugging Face repo name."""
    n = repo.lower()
    for pattern, reason in UNSUPPORTED:
        if re.search(pattern, n):
            return None, reason
    if re.search(r"z[._-]?image", n):
        return "z-image", ""
    if "flux" in n:
        return ("flux1-schnell" if "schnell" in n else "flux1-dev"), ""
    if re.search(r"sdxl|[-_]xl\b|xl[-_]|illustrious|pony|juggernaut|realvis", n):
        return "sdxl", ""
    if re.search(r"stable[-_]?diffusion|sd[-_]?1|sd15|v1[-_]5|dreamshaper|realistic[-_]?vision|deliberate", n):
        return "sd15", ""
    return None, "Couldn't tell what kind of image model this is, so it can't be set up automatically."


def image_repo_files(repo, family):
    """[(size_gb, path)] of files usable as the main model for this family."""
    res = requests.get(f"{HF_API}/models/{repo}/tree/main", params={"recursive": "true"}, timeout=15)
    res.raise_for_status()
    opts = []
    for f in res.json():
        path, size = f.get("path", ""), f.get("size", 0) / 1e9
        name = path.split("/")[-1].lower()
        if any(k in path.lower() for k in ("vae", "lora", "mmproj", "text_encoder", "inpaint", "refiner", "controlnet")):
            continue
        if FAMILIES[family]["role"] == "diffusion-model":
            if name.endswith(".gguf") and not re.search(r"-\d{5}-of-\d{5}", name):
                opts.append((size, path))
        elif name.endswith((".safetensors", ".ckpt", ".gguf")) and "/" not in path and size >= 0.8:
            opts.append((size, path))  # a complete single-file checkpoint (GGUF ones are converted by stable-diffusion.cpp)
    if any(p.endswith(".safetensors") for _, p in opts):  # prefer data-only safetensors over pickle-based .ckpt
        opts = [(sz, p) for sz, p in opts if not p.endswith(".ckpt")]
    return sorted(opts)


def _remote_size(repo, path):
    try:
        head = requests.head(HF_RESOLVE.format(repo=repo, path=path), allow_redirects=True, timeout=15)
        return int(head.headers.get("content-length", 0)) / 1e9
    except Exception:
        return 0.0


def image_search(query):
    """Search Hugging Face for text-to-image models. Fields stay disabled until a choice is made."""
    hide = gr.update(visible=False)
    no_version = gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False)
    q = repo_from_text(query)
    if not q:
        yield gr.update(), gr.update(), note_html("Enter a model name to search.", "error"), gr.update(interactive=False), hide
        return
    yield (gr.update(choices=[LOADING], value="", interactive=False), no_version, "", gr.update(interactive=False),
           loading_screen("Searching Hugging Face", f"Looking for \"{q}\" image models"))
    try:
        if "/" in q:
            repos = [q]
        else:
            seen, repos = set(), []
            for extra in ({"filter": "gguf"}, {}):
                res = requests.get(f"{HF_API}/models", timeout=15, params={
                    "search": q, "pipeline_tag": "text-to-image", "sort": "downloads", "limit": 20, **extra})
                res.raise_for_status()
                for m in res.json():
                    if m["id"] not in seen:
                        seen.add(m["id"])
                        repos.append(m["id"])
    except Exception as e:
        yield (gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False), no_version,
               note_html(f"Search failed: {e}", "error"), gr.update(interactive=False), hide)
        return
    if not repos:
        yield (gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False), no_version,
               note_html(f"No image models named \"{q}\" found on Hugging Face.", "error"), gr.update(interactive=False), hide)
        return
    choices = []
    supported = 0
    for r in repos:
        fam, _ = detect_family(r)
        supported += bool(fam)
        tag = FAMILIES[fam]["label"] if fam else "not supported"
        choices.append((f"{r}   ({tag})", r))
    choices.sort(key=lambda c: "not supported" in c[0])  # supported models first
    yield (gr.update(choices=[PLEASE_CHOOSE] + choices, value="", interactive=True), no_version,
           note_html(f"Found {len(repos)} image models, {supported} of them supported. Choose one to see its versions."),
           gr.update(interactive=False), hide)


def image_versions(repo):
    hide = gr.update(visible=False)
    if not repo:
        yield gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False), "", gr.update(interactive=False), hide
        return
    family, reason = detect_family(repo)
    if not family:
        yield (gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False), note_html(reason, "error"),
               gr.update(interactive=False), hide)
        return
    yield (gr.update(choices=[LOADING], value="", interactive=False), "", gr.update(interactive=False),
           loading_screen("Loading versions", repo))
    try:
        opts = image_repo_files(repo, family)
    except Exception as e:
        yield (gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False),
               note_html(f"Could not list files: {e}", "error"), gr.update(interactive=False), hide)
        return
    if not opts:
        kind = "GGUF model files" if FAMILIES[family]["role"] == "diffusion-model" else "complete single-file models"
        yield (gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False),
               note_html(f"This repository has no {kind} the image engine can use.", "error"),
               gr.update(interactive=False), hide)
        return
    best = best_version(opts)
    labels = []
    for size, path in opts:
        tag = re.sub(r"\.(gguf|safetensors|ckpt)$", "", path.split("/")[-1])
        labels.append((f"{tag}   {size:.1f} GB" + ("   (Recommended)" if path == best else ""), path))
    extra, have = 0.0, 0
    for comp in FAMILIES[family]["companions"].values():
        crepo, cpath = COMPANIONS[comp]
        if (SHARED_DIR / _slug(crepo) / Path(cpath).name).exists():
            have += 1
        else:
            extra += _remote_size(crepo, cpath)
    size = next(sz for sz, p in opts if p == best)
    parts = [f"{FAMILIES[family]['label']} model. Auto-selected the largest version under {VRAM_BUDGET_GB} GB."]
    if FAMILIES[family]["companions"]:
        parts.append(f"It also needs {len(FAMILIES[family]['companions'])} companion files "
                     f"({extra:.1f} GB to download" + (f", {have} already installed" if have else "") + ").")
    parts.append(f"Total download about {size + extra:.1f} GB.")
    yield (gr.update(choices=[PLEASE_CHOOSE] + labels, value=best, interactive=True), note_html(" ".join(parts)),
           gr.update(interactive=True), hide)


def image_version_changed(repo, path):
    return gr.update(interactive=bool(repo and path))


def download_searched_image_model(repo, path):
    if not repo or not path:
        return note_html(f"Choose {'a repository' if not repo else 'a version'} before downloading.", "error"), gr.update()
    family, reason = detect_family(repo)
    if not family:
        return note_html(reason, "error"), gr.update()
    fam = FAMILIES[family]
    stem = re.sub(r"\.(gguf|safetensors|ckpt)$", "", path.split("/")[-1])
    key = _slug(f"{repo}-{stem}")
    steps, args = fam["steps"], list(fam["args"])
    if family in ("sdxl", "sd15") and re.search(r"turbo|lightning|hyper|lcm", repo.lower()):
        steps, args = 6, ["--cfg-scale", "1.5"] + [a for a in args if a not in ("--cfg-scale", "6.0", "7.0")]
    quant = re.search(r"(?:UD-)?(?:I?Q\d\w*|BF16|F16|F32|fp16|fp8\w*)$", stem, re.I)
    base = re.sub(r"[-_]?gguf$", "", repo.split("/")[-1], flags=re.I)
    entry = {
        "name": f"{base} ({quant.group(0)})" if quant else base,
        "family": family, "desc": f"{fam['label']} model from {repo}.",
        "files": {fam["role"]: [repo, path, False],
                  **{role: [*COMPANIONS[c], True] for role, c in fam["companions"].items()}},
        "steps": steps, "size": fam["size"], "args": args,
    }
    custom = _custom_image_models()
    custom[key] = entry
    CUSTOM_IMAGE_FILE.write_text(json.dumps(custom, indent=1), encoding="utf-8")
    note = download_image_model(key)
    return note, time.time()


def image_model_defaults(key):
    """Steps and size to use when switching image models."""
    m = image_models().get(key)
    if not m:
        return gr.update(), gr.update()
    return gr.update(value=m.get("steps", 8)), gr.update(value=m.get("size", "Square 1024 x 1024"))


def _image_download_worker(ref):
    """Download every file of an image model from Hugging Face with HTTP range requests, so interrupted
    downloads (network drops, app restarts) resume from the last byte instead of starting over."""
    key = ref.split(":", 1)[1]
    d = _downloads[ref]
    files = [(spec[0], spec[1], dest) for spec, dest in
             zip(image_models()[key]["files"].values(), image_model_files(key).values(), strict=True)]
    stop = lambda: _should_stop(ref, d)  # noqa: E731
    attempt = 0
    while not (d["cancel"] or d.get("pause")):
        if _yield_to_priority(ref):
            d.update(state="paused", msg="Paused for priority download")
            time.sleep(1)
            continue
        try:
            sizes = []
            for repo, path, _ in files:
                head = requests.head(HF_RESOLVE.format(repo=repo, path=path), allow_redirects=True, timeout=20)
                _check_response(head, f"Checking {path}")
                sizes.append(int(head.headers.get("content-length", 0)))
            total = sum(sizes)

            def progress():
                done = sum(p.stat().st_size if p.exists() else 0 for p in
                           [f[2] for f in files] + [f[2].with_name(f[2].name + ".part") for f in files])
                d.update(state="downloading", msg="Downloading", done=done, total=total)

            for (repo, path, dest), size in zip(files, sizes, strict=True):
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and dest.stat().st_size == size:
                    continue
                if not fetch_file(HF_RESOLVE.format(repo=repo, path=path), dest, size, stop, progress):
                    break
            if stop():
                continue  # deleted, paused, or waiting for a priority download; the loop handles each
            if image_model_installed(key):
                d.update(state="done", msg="Done", finished=time.time(), done=total, total=total)
                _set_pending(ref, False)
                _dl_finished["count"] += 1
                return
        except PermanentDownloadError as e:
            d.update(state="failed", msg=str(e))
            _set_pending(ref, False)
            return
        except Exception:
            attempt += 1
            wait = min(60, 5 * attempt)
            d.update(state="waiting", msg=f"Connection interrupted, resuming in {wait}s")
            time.sleep(wait)
    _worker_stopped(ref, d)


def image_models_version_html(key):
    m = image_models()[key]
    size = sum(p.stat().st_size for p in image_model_files(key).values() if p.exists())
    state = "Installed" if image_model_installed(key) else "Not installed"
    cells = (m["name"], m["desc"], f"{size / 1e9:.1f} GB" if size else "-", state)
    return "".join(f'<div class="mt-cell">{html.escape(str(c))}</div>' for c in cells)


def download_image_model(key):
    ref = f"image:{key}"
    if image_model_installed(key):
        return note_html(f"{image_models()[key]['name']} is already installed.")
    if not start_download(ref):
        return note_html(f"{image_models()[key]['name']} is already downloading.")
    return note_html(f"Downloading {image_models()[key]['name']}. Progress is shown under Downloads on the Models tab.")


def delete_image_model(key, confirmed):
    """Permanently delete a model's files. Shared companion files are removed only if no other model needs them."""
    if not confirmed or key not in image_models():
        return gr.update(), gr.update()
    name = image_models()[key]["name"]
    mine = set(image_model_files(key).values())
    shutil.rmtree(IMAGE_MODEL_DIR / key, ignore_errors=True)
    custom = _custom_image_models()
    if custom.pop(key, None) is not None:
        CUSTOM_IMAGE_FILE.write_text(json.dumps(custom, indent=1), encoding="utf-8")
    still_needed = {p for k in image_models() if k != key and image_model_installed(k) for p in image_model_files(k).values()}
    for path in mine - still_needed:
        if path.is_relative_to(SHARED_DIR):
            path.unlink(missing_ok=True)
    return note_html(f"Deleted {name}."), time.time()


def gallery_html():
    """Thumbnail grid of generated images, newest first. Expand and delete are handled in static/app.js."""
    files = sorted(IMAGES_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:120] \
        if IMAGES_DIR.exists() else []
    if not files:
        return '<div class="gallery-empty">No images yet. Generate one above.</div>'
    tiles = []
    for f in files:
        try:
            meta = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
            detail = (f"{meta['model']} · seed {meta['seed']} · {meta['width']} x {meta['height']} · "
                      f"{meta['steps']} steps · {meta['seconds']} s")
            prompt = meta.get("prompt", "")
        except Exception:
            prompt, detail = "", f.name
        src = f"/gradio_api/file={f}"
        tiles.append(
            f'<div class="g-tile" title="{html.escape(prompt)}"><img src="{html.escape(src)}" loading="lazy" alt="">'
            f'<div class="g-actions"><button type="button" class="g-expand" title="Expand" aria-label="Expand" '
            f'data-src="{html.escape(src)}" data-prompt="{html.escape(prompt)}" data-detail="{html.escape(detail)}"></button>'
            f'<button type="button" class="g-del" title="Delete" aria-label="Delete" data-name="{html.escape(f.name)}">'
            f'</button></div></div>')
    return f'<div class="g-grid">{"".join(tiles)}</div>'


def image_progress_html(title, pct, meta, state="running"):
    return (f'<div class="scanprog {state}"><div class="sp-top"><span>{html.escape(title)}</span>'
            f'<span class="sp-pct">{pct:.0f}%</span></div><div class="sp-bar"><div style="width:{pct:.1f}%"></div></div>'
            f'<div class="sp-meta">{meta}</div></div>')


IMAGE_IDLE = '<div class="scanprog idle"><div class="sp-top"><span>Ready</span></div></div>'


def generate_image(key, prompt, size_label, steps, seed):
    """Run stable-diffusion.cpp once for this image. The model loads, generates, and exits, so nothing stays in memory."""
    keep = gr.update()
    prompt = (prompt or "").strip()
    if not key or not image_model_installed(key):
        yield IMAGE_IDLE, note_html("Download an image model first (Image models below).", "error"), keep, keep, keep, keep
        return
    if not prompt:
        yield IMAGE_IDLE, note_html("Describe the image you want.", "error"), keep, keep, keep, keep
        return
    if _proc.get("p") and _proc["p"].poll() is None:
        yield IMAGE_IDLE, note_html("A scan is using the GPU. Wait for it to finish, then generate.", "error"), keep, keep, keep, keep
        return
    if _image_proc.get("p") and _image_proc["p"].poll() is None:
        yield keep, note_html("An image is already being generated. Wait for it to finish.", "error"), keep, keep, keep, keep
        return
    if not SD_CLI.exists():
        yield IMAGE_IDLE, note_html(f"The image engine is missing from {SD_DIR}.", "error"), keep, keep, keep, keep
        return
    unload_models()  # free VRAM held by chat models
    width, height = IMAGE_SIZES.get(size_label, (1024, 1024))
    seed = int(seed) if seed not in (None, "") and int(seed) >= 0 else int.from_bytes(os.urandom(4), "big") % 2**31
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    out = IMAGES_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{seed}.png"
    files = image_model_files(key)
    preset = image_models()[key]
    cmd = [str(SD_CLI)]
    for role, path in files.items():
        cmd += [f"--{role}", str(path)]
    cmd += ["-p", prompt, "-W", str(width), "-H", str(height), "--steps", str(int(steps)), "--seed", str(seed),
            "-o", str(out)] + preset["args"]
    env = os.environ | {"LD_LIBRARY_PATH": str(SD_DIR), "GGML_VK_VISIBLE_DEVICES": "0"}  # the NVIDIA GPU, not the iGPU
    start = time.time()
    yield (image_progress_html("Loading model", 2, "Starting the image engine"), "", gr.update(interactive=False),
           gr.update(visible=True), keep, keep)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    _image_proc["p"] = p
    step_re = re.compile(r"(\d+)/(\d+)\s*-\s*[\d.]+\s*(?:s/it|it/s)")
    st = {"phase": "Loading model", "pct": 2.0, "buf": "", "tail": [], "sampled": False, "elapsed": 0.0}

    def on_text(text):
        lines = (st["buf"] + text.replace("\r", "\n")).split("\n")
        st["buf"] = lines.pop()
        for line in lines:
            line = ANSI.sub("", line).strip()
            if not line:
                continue
            st["tail"] = (st["tail"] + [line])[-12:]
            m = step_re.search(line)
            if m and int(m.group(2)) == int(steps):  # sampling progress bar (one tick per step)
                n, total = int(m.group(1)), int(m.group(2))
                st["sampled"] = True
                st["phase"], st["pct"] = f"Generating, step {n} of {total}", 10 + 80 * n / max(total, 1)
            elif st["sampled"] and "decod" in line.lower():  # decoder messages before sampling are part of loading
                st["phase"], st["pct"] = "Decoding image", max(st["pct"], 92)

    def on_exit():  # also runs if the page was closed, so the image still gets its settings file
        st["elapsed"] = time.time() - start
        if p.returncode == 0 and out.exists():
            out.with_suffix(".json").write_text(json.dumps({
                "prompt": prompt, "model": preset["name"], "width": width, "height": height, "steps": int(steps),
                "seed": seed, "seconds": round(st["elapsed"], 1)}, indent=1), encoding="utf-8")
        if _image_proc["p"] is p:
            _image_proc["p"] = None

    reader = pump_output(p, on_text, on_exit)
    while reader.is_alive():
        reader.join(0.4)
        if reader.is_alive():
            yield (image_progress_html(st["phase"], st["pct"], f"Elapsed {_fmt_secs(time.time() - start)}"),
                   keep, keep, keep, keep, keep)
    elapsed, pct, tail = st["elapsed"], st["pct"], st["tail"]
    if p.returncode == 0 and out.exists():
        yield (image_progress_html("Image ready", 100, f"{_fmt_secs(elapsed)} &nbsp;·&nbsp; seed {seed} &nbsp;·&nbsp; "
                                   f"{width} x {height} &nbsp;·&nbsp; {int(steps)} steps", "done"),
               "", gr.update(interactive=True), gr.update(visible=False), str(out), gallery_html())
    else:
        detail = html.escape(tail[-1] if tail else f"exit code {p.returncode}")
        state = "Stopped" if p.returncode and p.returncode < 0 else "Generation failed"
        yield (image_progress_html(state, pct, detail, "stopped"), "", gr.update(interactive=True),
               gr.update(visible=False), keep, keep)


def stop_image():
    p = _image_proc.get("p")
    if p and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)


def delete_gallery_image(name):
    """Permanently delete one generated image and its settings file."""
    path = IMAGES_DIR / Path(name or "").name
    if name and path.suffix == ".png" and path.exists():
        path.unlink()
        path.with_suffix(".json").unlink(missing_ok=True)
        return gallery_html(), note_html("Image deleted."), ""
    return gr.update(), gr.update(), ""


# ---------------------------------------------------------------- probe catalog and groups
def load_catalog():
    """Text-capable garak probes from garak's plugin cache: {name: {module, desc, tier, active}}."""
    paths = [Path.home() / ".cache/garak/resources/plugin_cache.json",
             Path(importlib.util.find_spec("garak").origin).parent / "resources/plugin_cache.json"]
    for p in paths:
        if p.exists():
            data = json.loads(p.read_text())["probes"]
            break
    else:
        return {}
    cat = {}
    for key, v in data.items():
        name = key.removeprefix("probes.")
        if "text" not in v.get("modality", {}).get("in", ["text"]):
            continue
        cat[name] = {"module": name.split(".")[0], "desc": (v.get("description") or "").strip().split("\n")[0],
                     "tier": v.get("tier"), "active": v.get("active", True), "tags": v.get("tags", [])}
    return dict(sorted(cat.items()))


CATALOG = load_catalog()
MODULES = sorted({v["module"] for v in CATALOG.values()})


def owasp_preset():
    """Every active garak probe mapped to the OWASP Top 10 for LLM Applications, and the categories they cover."""
    probes = [n for n, v in CATALOG.items() if v["active"] and any(t.startswith("owasp:") for t in v["tags"])]
    covered = sorted({t.split(":")[1].upper() for n in probes for t in CATALOG[n]["tags"] if t.startswith("owasp:llm")})
    missing = [f"LLM{i:02d}" for i in range(1, 11) if f"LLM{i:02d}" not in covered]
    return probes, (f"Every garak probe mapped to the OWASP Top 10 for LLM Applications ({', '.join(covered)}). "
                    + (f"garak has no tests for {', '.join(missing)} on a standalone model (they concern training data "
                       "and tool-using apps). " if missing else "") + "Thorough; expect several hours.")


_owasp_probes, _owasp_desc = owasp_preset()
if _owasp_probes:  # an empty list would mean every probe (the full scan)
    PRESETS = {k: v for k, v in PRESETS.items() if k != "Full scan"} | {
        "OWASP Top 10 for LLMs": (_owasp_probes, _owasp_desc), "Full scan": PRESETS["Full scan"]}


def expand(specs):
    """Turn module names into their active probe classes; keep class names as-is."""
    out = []
    for s in specs:
        if s in CATALOG:
            out.append(s)
        elif s in MODULES:
            out += [n for n, v in CATALOG.items() if v["module"] == s and v["active"]]
    return list(dict.fromkeys(out))


def load_groups():
    try:
        return json.loads(GROUPS_FILE.read_text())
    except Exception:
        return {}


def save_groups(groups):
    GROUPS_FILE.write_text(json.dumps(groups, indent=2, sort_keys=True))


def probe_label(n):
    v = CATALOG[n]
    d = v["desc"] if len(v["desc"]) <= 110 else v["desc"][:107].rstrip() + "..."
    extra = "" if v["active"] else "   [off by default]"
    return f"{n}   {d}{extra}"


def filtered_probes(query, module, show_all):
    q = (query or "").lower().strip()
    return [n for n, v in CATALOG.items()
            if (show_all or v["active"])
            and (module in (None, "", "All categories") or v["module"] == module)
            and (not q or q in n.lower() or q in v["desc"].lower())]


def selection_summary(selected):
    if not selected:
        return "No probes selected."
    shown = ", ".join(selected[:8]) + (f", and {len(selected) - 8} more" if len(selected) > 8 else "")
    return f"**{len(selected)} selected:** {shown}"


def filter_probes(query, module, show_all, selected):
    names = filtered_probes(query, module, show_all)
    return gr.update(choices=[(probe_label(n), n) for n in names], value=[n for n in selected if n in names]), \
        f"Showing {len(names)} of {len(CATALOG)} probes."


def probes_ticked(ticked, query, module, show_all, selected):
    visible = set(filtered_probes(query, module, show_all))
    new = [n for n in selected if n not in visible] + list(ticked or [])
    new = [n for n in CATALOG if n in set(new)]
    return new, selection_summary(new)


def select_shown(query, module, show_all, selected):
    names = filtered_probes(query, module, show_all)
    new = [n for n in CATALOG if n in set(selected) | set(names)]
    return new, gr.update(value=names), selection_summary(new)


def clear_selection():
    return [], gr.update(value=[]), selection_summary([])


def group_choices():
    return [f"Preset: {k}" for k in PRESETS if PRESETS[k][0]] + sorted(load_groups())


def scan_choices():
    return list(PRESETS) + [GROUP_PREFIX + g for g in sorted(load_groups())] + [CURRENT]


def load_group(name, query, module, show_all):
    if not name:
        return gr.update(), gr.update(), "Choose a group to load.", gr.update()
    specs = PRESETS[name.removeprefix("Preset: ")][0] if name.startswith("Preset: ") else load_groups().get(name, [])
    new = [n for n in CATALOG if n in set(expand(specs))]
    visible = filtered_probes(query, module, show_all)
    title = "" if name.startswith("Preset: ") else name
    return new, gr.update(value=[n for n in new if n in visible]), selection_summary(new), gr.update(value=title)


def save_group(name, selected):
    name = (name or "").strip()
    if not name:
        return "Enter a name for the group.", gr.update(), gr.update()
    if not selected:
        return "Select at least one probe before saving.", gr.update(), gr.update()
    groups = load_groups()
    verb = "Updated" if name in groups else "Saved"
    groups[name] = list(selected)
    save_groups(groups)
    return (f"{verb} group \"{name}\" with {len(selected)} probes. It is now available on the Scan tab.",
            gr.update(choices=group_choices(), value=name), gr.update(choices=scan_choices()))


def delete_group(name):
    groups = load_groups()
    if not name or name not in groups:
        return "Choose one of your saved groups to delete (presets can't be deleted).", gr.update(), gr.update()
    del groups[name]
    save_groups(groups)
    return f"Deleted group \"{name}\".", gr.update(choices=group_choices(), value=None), \
        gr.update(choices=scan_choices(), value="Quick check")


# ---------------------------------------------------------------- scanning
def resolve_scan(scan_type, selected):
    """Return (probe list, error). An empty list means garak's full default scan."""
    if scan_type in PRESETS:
        return expand(PRESETS[scan_type][0]), None
    if scan_type == CURRENT:
        return (list(selected), None) if selected else (None, "No probes selected. Choose some on the Probes tab.")
    if scan_type and scan_type.startswith(GROUP_PREFIX):
        g = load_groups().get(scan_type.removeprefix(GROUP_PREFIX))
        if not g:
            return None, "That group no longer exists."
        return expand(g), None
    return None, "Choose a scan type."


def scan_info(scan_type, selected):
    if scan_type in PRESETS and not PRESETS[scan_type][0]:
        return PRESETS[scan_type][1]
    probes, err = resolve_scan(scan_type, selected)
    if err:
        return err
    desc = PRESETS[scan_type][1] + " " if scan_type in PRESETS else ""
    return desc + f"{len(probes)} probes: " + ", ".join(probes[:10]) + (" ..." if len(probes) > 10 else "")


def model_info(name):
    """(size_gb, capabilities) for an installed model."""
    size = next((m.get("size", 0) / 1e9 for m in list_models() if m["name"] == name), 0)
    try:
        caps = requests.post(f"{OLLAMA}/api/show", json={"model": name}, timeout=10).json().get("capabilities", [])
    except Exception:
        caps = []
    return size, caps


def model_hint(name):
    """Explain how the chosen model will run on this hardware and pick a sensible timeout."""
    if not name:
        return "", gr.update()
    size, caps = model_info(name)
    g = gpu_live()
    vram = g[3] if g else 0
    if size <= vram * 0.85:
        text, timeout = f"{size:.1f} GB model, runs fully on the GPU.", 300
    else:
        share = min(100, int(100 * vram * 0.8 / size)) if size else 0
        text = (f"{size:.1f} GB model, about {share}% on the GPU and the rest in system memory. "
                "It will run, but responses are slower, so the timeout has been raised.")
        timeout = 900 if size < 25 else 1800
    if "thinking" in caps:
        text += " This is a reasoning model; thinking is turned off during scans unless you enable it below."
    return text, gr.update(value=timeout)


def _fmt_secs(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s_ = divmod(rem, 60)
    return f"{h}h {m}m" if h else f"{m}m {s_}s" if m else f"{s_}s"


class ScanProgress:
    """Tracks overall scan progress by reading garak's console output."""
    QUEUE = re.compile(r"queue of probes: (.+)$")
    PROMPTS = re.compile(r"^\s*probes\.([\w.]+):\s+\d+%\|.*?(\d+)/(\d+)")
    DETECT = re.compile(r"^\s*([\w.]+)/([\w.]+):\s+\d+%\|.*?(\d+)/(\d+)")
    RESULT = re.compile(r"^\s*([\w.]+)\s+[\w.]+:\s+(PASS|FAIL)\b")

    def __init__(self):
        self.start = time.time()
        self.probes, self.done, self.failed = [], set(), set()
        self.current, self.frac, self.phase = None, 0.0, "Starting garak"

    def feed(self, line):
        if m := self.QUEUE.search(line):
            self.probes = [x.strip() for x in m.group(1).split(",") if x.strip()]
            self.phase = "Loading model"
        elif m := self.PROMPTS.match(line):
            self.current, n, total = m.group(1), int(m.group(2)), int(m.group(3))
            self.frac = 0.85 * n / total if total else 0
            self.phase = f"Sending attack prompts {n} of {total}"
        elif m := self.DETECT.match(line):
            self.current, n, total = m.group(1), int(m.group(3)), int(m.group(4))
            self.frac = 0.85 + 0.15 * n / total if total else 0.85
            self.phase = f"Evaluating responses with {m.group(2)}"
        elif m := self.RESULT.match(line):
            self.done.add(m.group(1))
            if m.group(2) == "FAIL":
                self.failed.add(m.group(1))

    def percent(self):
        n = len(self.probes)
        if not n:
            return 0.0
        partial = self.frac if self.current and self.current not in self.done else 0
        return min(100.0, 100 * (len(self.done) + partial) / n)

    def html(self, state="running"):
        n, pct, elapsed = len(self.probes), self.percent(), time.time() - self.start
        if state == "done":
            pct, title = 100.0, "Scan complete"
        elif state == "stopped":
            title = "Scan stopped"
        elif self.current and n:
            idx = min(n, len(self.done) + (0 if self.current in self.done else 1))
            title = f"Probe {idx} of {n} &nbsp;·&nbsp; <span class='mono'>{html.escape(self.current)}</span>"
        else:
            title = self.phase
        meta = [f"Elapsed {_fmt_secs(elapsed)}"]
        if state == "running":
            meta.insert(0, html.escape(self.phase))
            if pct >= 2:
                meta.append(f"About {_fmt_secs(elapsed * (100 - pct) / pct)} left")
        if self.done:
            passed = len(self.done) - len(self.failed)
            meta.append(f"{passed} passed, {len(self.failed)} failed")
        return (f'<div class="scanprog {state}"><div class="sp-top"><span>{title}</span><span class="sp-pct">{pct:.0f}%</span></div>'
                f'<div class="sp-bar"><div style="width:{pct:.1f}%"></div></div>'
                f'<div class="sp-meta">{" &nbsp;·&nbsp; ".join(meta)}</div></div>')


IDLE_PROGRESS = '<div class="scanprog idle"><div class="sp-top"><span>No scan running</span></div></div>'


def run_scan(model, scan_type, selected, generations, timeout, thinking=False):
    if not model:
        yield "Select a model to scan.", gr.update(), gr.update(), gr.update()
        return
    if not ollama_up():
        yield "Ollama is not running. Start it at the top of the page.", gr.update(), gr.update(), gr.update()
        return
    probes, err = resolve_scan(scan_type, selected)
    if err:
        yield err, gr.update(), gr.update(), gr.update()
        return
    if _proc.get("p") and _proc["p"].poll() is None:  # e.g. started before this page was reloaded
        yield ("A scan is already running. Wait for it to finish, or click Stop to end it.", gr.update(),
               gr.update(interactive=True), gr.update())
        return
    if _image_proc.get("p") and _image_proc["p"].poll() is None:
        yield "An image is being generated on the GPU. Start the scan once it finishes.", gr.update(), gr.update(), gr.update()
        return
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    prefix = f"{safe}_{time.strftime('%Y%m%d-%H%M%S')}"
    gen_opts = {"timeout": int(timeout)}
    if thinking:
        gen_opts["max_tokens"] = 4096  # leave room for the reasoning before the answer
    runner = str(APP_DIR / "garak_runner.py")
    cmd = [PY, runner, "--target_type", "ollama.OllamaGeneratorChat", "--target_name", model,
           "--generations", str(int(generations)), "--report_prefix", prefix,
           "--generator_options", json.dumps({"ollama": {"OllamaGeneratorChat": gen_opts}})]
    if probes:
        cmd += ["--probes", ",".join(probes)]
    env = os.environ | {"PYTHONUNBUFFERED": "1", "TERM": "dumb", "PYTHONIOENCODING": "utf-8",
                        "LLM_SCANNER_THINK": "1" if thinking else "0", "LLM_SCANNER_MODEL": model}
    head = "garak " + " ".join(cmd[2:]) + "\n\n"
    progress = ScanProgress()
    yield head, gr.update(interactive=False), gr.update(interactive=True), progress.html()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    _proc["p"] = p
    buf, cur = [], [""]

    def on_text(text):
        for c in EMOJI.sub("", ANSI.sub("", text)):
            if c in "\n\r":
                if cur[0]:
                    progress.feed(cur[0])
                if c == "\n":
                    if "%|" in cur[0] and buf and "%|" in buf[-1]:
                        buf[-1] = cur[0]  # keep only the latest progress bar line
                    else:
                        buf.append(cur[0])
                        if len(buf) > 5000:  # a days-long full scan must not grow memory without limit
                            del buf[:1000]
                cur[0] = ""
            else:
                cur[0] += c

    def on_exit():  # also runs if the page was closed during the scan
        if _proc["p"] is p:
            _proc["p"] = None
        unload_models()  # free memory as soon as the scan is over

    reader = pump_output(p, on_text, on_exit)
    while reader.is_alive():
        reader.join(1.0)
        if reader.is_alive():
            yield head + "\n".join(buf[-400:] + cur), gr.update(), gr.update(), progress.html()
    ok = p.returncode == 0
    end = "Scan complete. Open Reports to view the results." if ok else f"Scan stopped (exit code {p.returncode})."
    yield (head + "\n".join(buf + cur) + f"\n\n{end}", gr.update(interactive=True), gr.update(interactive=False),
           progress.html("done" if ok else "stopped"))


def stop_scan():
    p = _proc.get("p")
    if p and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        return "Stopping scan..."
    return "No scan is running."


# ---------------------------------------------------------------- reports
def reports():
    """(label, name) per scan run, newest first. A run is named by its garak HTML report; a scan that was
    stopped partway has no HTML report, only report.jsonl, and is listed as stopped."""
    if not GARAK_RUNS.exists():
        return []
    runs = {}
    for f in GARAK_RUNS.iterdir():
        for suffix in (".report.html", ".report.jsonl"):
            if f.name.endswith(suffix) and len(f.name) > len(suffix):
                prefix = f.name[:-len(suffix)]
                runs[prefix] = max(runs.get(prefix, 0), f.stat().st_mtime)
    out = []
    for prefix, mtime in sorted(runs.items(), key=lambda r: r[1], reverse=True):
        label = re.sub(r"_(\d{8}-\d{6})$", "", prefix).replace("hf.co_", "")
        stopped = "" if (GARAK_RUNS / f"{prefix}.report.html").exists() else "   (stopped)"
        out.append((f"{label}   {time.strftime('%b %d, %H:%M', time.localtime(mtime))}{stopped}", prefix + ".report.html"))
    return out


def _known_report(name):
    """Only names the Reports list offers are accepted, so a crafted name can't reach other files."""
    return bool(name) and name in {n for _, n in reports()}


def refresh_reports():
    r = reports()
    return gr.update(choices=r, value=r[0][1] if r else None)


RAW_LIMIT = 2_000_000  # characters shown in the raw viewer; the full file is always downloadable


def report_paths(name):
    f = GARAK_RUNS / name
    prefix = name[:-len(".report.html")]
    return {"garak": f, "jsonl": f.with_name(prefix + ".report.jsonl"), "hitlog": f.with_name(prefix + ".hitlog.jsonl"),
            "analyst": f.with_name(prefix + ".analyst.html"), "csv": f.with_name(prefix + ".findings.csv")}


def read_raw(name, which):
    if not _known_report(name):
        return ""
    p = report_paths(name)["hitlog" if which == "Hitlog (failing responses)" else "jsonl"]
    if not p.exists():
        return "No failing responses were recorded for this run." if "hitlog" in p.name else "File not found."
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > RAW_LIMIT:
        text = text[:RAW_LIMIT] + f"\n\n[Truncated: showing {RAW_LIMIT:,} of {len(text):,} characters. Download the file for the rest.]"
    return text


def show_report(name, which="Full report (report.jsonl)"):
    empty = '<div class="empty">No reports yet. Run a scan to create one.</div>'
    if not _known_report(name):
        return empty, empty, "", None
    paths = report_paths(name)
    try:
        analyst_report.build(paths["garak"])
        summary = f'<iframe class="report dark" srcdoc="{html.escape(paths["analyst"].read_text(encoding="utf-8"))}"></iframe>'
    except Exception as e:
        summary = f'<div class="empty">Could not build the security summary: {html.escape(str(e))}</div>'
    if paths["garak"].exists():
        garak_view = f'<iframe class="report" srcdoc="{html.escape(paths["garak"].read_text(encoding="utf-8"))}"></iframe>'
    else:
        garak_view = ('<div class="empty">This scan was stopped before garak wrote its report. The security summary '
                      'and raw data show the results collected before it stopped.</div>')
    files = [str(paths[k]) for k in ("analyst", "csv", "garak", "jsonl", "hitlog") if paths[k].exists()]
    return summary, garak_view, read_raw(name, which), files


def delete_report(name, confirmed):
    """Permanently delete every file from one scan run (garak report, raw JSONL, hitlog, summary, CSV)."""
    if not confirmed or not _known_report(name):
        return gr.update(), gr.update()
    prefix = name[:-len(".report.html")]
    removed = 0
    for f in list(GARAK_RUNS.iterdir()):
        if f.is_file() and f.name.startswith(prefix + "."):
            f.unlink()
            removed += 1
    return refresh_reports(), note_html(f"Deleted report {prefix} ({removed} files).")


def open_report(name):
    if _known_report(name):
        p = report_paths(name)["analyst"]
        try:
            analyst_report.build(GARAK_RUNS / name)
        except Exception:
            pass
        target = p if p.exists() else GARAK_RUNS / name
        if not target.exists():
            return
        threading.Thread(target=webbrowser.open, args=(target.as_uri(),), daemon=True).start()


def model_choices_changed(chat):
    """Keep model dropdowns in sync after a download finishes or a model is deleted."""
    names = model_names()
    locked = bool(chat and chat.get("messages") and chat.get("model"))
    return gr.update(choices=names), (model_lock(chat) if locked else gr.update(choices=names))


def refresh_all(chat=None):
    names = model_names()
    upd = gr.update(choices=names, value=names[0] if names else None)
    locked = bool(chat and chat.get("messages") and chat.get("model"))
    chat_upd = model_lock(chat) if locked else gr.update(choices=names, value=names[0] if names else None,
                                                          interactive=True, label="Model")
    return (time.time(), upd, status_html(), gr.update(choices=scan_choices()),
            gr.update(choices=group_choices()), chat_upd)


# ---------------------------------------------------------------- look and feel
FONT = [gr.themes.GoogleFont("Inter"), "-apple-system", "BlinkMacSystemFont", "Helvetica Neue", "sans-serif"]
MONO = [gr.themes.GoogleFont("JetBrains Mono"), "SF Mono", "ui-monospace", "monospace"]
_vars = dict(
    body_background_fill="#000000", body_text_color="#f5f5f7", body_text_color_subdued="#86868b",
    background_fill_primary="#161617", background_fill_secondary="#1d1d1f",
    block_background_fill="transparent", block_border_width="0px", block_shadow="none",
    block_label_background_fill="transparent", block_label_text_color="#86868b", block_label_border_width="0px",
    block_title_text_color="#86868b", block_title_text_weight="500", block_label_text_weight="500",
    border_color_primary="#2d2d2f", border_color_accent="#2997ff",
    input_background_fill="#1d1d1f", input_border_color="#2d2d2f", input_border_color_focus="#2997ff",
    input_shadow="none", input_shadow_focus="0 0 0 3px rgba(41,151,255,.25)", input_radius="12px",
    button_primary_background_fill="#0071e3", button_primary_background_fill_hover="#0077ed",
    button_primary_text_color="#ffffff", button_primary_border_color="transparent",
    button_secondary_background_fill="#2a2a2c", button_secondary_background_fill_hover="#3a3a3c",
    button_secondary_text_color="#f5f5f7", button_secondary_border_color="transparent",
    button_cancel_background_fill="#2a2a2c", button_cancel_background_fill_hover="#3a1d1d",
    button_cancel_text_color="#ff453a", button_cancel_border_color="transparent",
    button_large_radius="980px", button_small_radius="980px", button_medium_radius="980px",
    color_accent="#2997ff", color_accent_soft="#0a2540", link_text_color="#2997ff",
    table_even_background_fill="#161617", table_odd_background_fill="#1a1a1c", table_border_color="#2d2d2f",
    table_row_focus="#1f2a38", checkbox_label_background_fill="#1d1d1f",
    checkbox_label_background_fill_selected="#0a2540", checkbox_label_border_color_selected="#2997ff",
    checkbox_background_color_selected="#0071e3", checkbox_border_color_selected="#0071e3",
    slider_color="#2997ff", accordion_text_color="#f5f5f7", panel_background_fill="transparent",
    code_background_fill="#0b0b0c", layout_gap="16px", loader_color="#2997ff",
)
_base = gr.themes.Base(primary_hue="blue", neutral_hue="zinc", radius_size="lg", font=FONT, font_mono=MONO)
THEME = _base.set(**{k: v for k, v in _vars.items() if hasattr(_base, k)},
                  **{k + "_dark": v for k, v in _vars.items() if hasattr(_base, k + "_dark")})

CSS = (APP_DIR / "static" / "style.css").read_text(encoding="utf-8")


APP_JS = (APP_DIR / "static" / "app.js").read_text(encoding="utf-8")

# ---------------------------------------------------------------- UI
with gr.Blocks(title="LLM Scanner") as ui:
    selected = gr.State([])
    gr.HTML('<div class="hero"><div class="eyebrow">Local AI security &nbsp;·&nbsp; Current version v' + APP_VERSION + '</div><h1>LLM Scanner</h1>'
            '<p>Download models from Hugging Face or Ollama, run them locally, and test them for '
            'vulnerabilities with garak.</p></div>')
    with gr.Row(elem_classes="update-row"):
        update_html = gr.HTML(update_banner())
        update_app_btn = gr.Button("Update LLM Scanner", variant="primary", size="sm", scale=0, min_width=170,
                                   visible=False)
        update_ollama_btn = gr.Button("Update Ollama", variant="primary", size="sm", scale=0, min_width=140, visible=False)
        update_garak_btn = gr.Button("Update garak", variant="primary", size="sm", scale=0, min_width=130, visible=False)
        update_dismiss = gr.Button("Dismiss", variant="secondary", size="sm", scale=0, min_width=100)
        update_timer = gr.Timer(1.0)
    with gr.Row(elem_classes="statusrow"):
        status = gr.HTML(status_html())
        status_timer = gr.Timer(5.0)
        _up = bool(ollama_up())
        with gr.Row(equal_height=True, elem_classes="ollama-controls"):  # one pill: "Ollama  Start" or "Stop  Restart"
            gr.HTML('<span class="oc-label">Ollama</span>', elem_classes="oc-label-holder", min_width=0)
            start_btn = gr.Button("Start", variant="secondary", size="sm", scale=0, min_width=0, visible=not _up,
                                  elem_classes="oc-start")
            stop_ollama_btn = gr.Button("Stop", variant="secondary", size="sm", scale=0, min_width=0, visible=_up)
            restart_ollama_btn = gr.Button("Restart", variant="secondary", size="sm", scale=0, min_width=0, visible=_up)
        unload_btn = gr.Button("Unload models", variant="secondary", size="sm", scale=0, min_width=140,
                               visible=bool(loaded_models()))
        refresh_btn = gr.Button("Refresh", variant="secondary", size="sm", scale=0, min_width=100)

    with gr.Tabs():
        # ---------------- Models
        with gr.Tab("Models"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=480):
                    with gr.Column(elem_classes="card add-model-card"):
                        gr.HTML('<h2>Add a model</h2><p class="sub">Choose where to search, then search by name or paste a link.</p>')
                        search_loading = gr.HTML(visible=False, elem_classes="search-loading")
                        source = gr.Radio([SRC_HF, SRC_OLLAMA], value=SRC_HF, show_label=False, elem_classes="segmented")
                        with gr.Row(equal_height=True):
                            query = gr.Textbox(show_label=False, placeholder="Qwen3.8-27B, or a Hugging Face link",
                                               scale=5, container=False)
                            search_btn = gr.Button("Search", variant="primary", scale=1, min_width=120)
                        with gr.Row():
                            repo = gr.Dropdown(label="Repository", choices=[PLEASE_CHOOSE], value="", scale=3,
                                               min_width=300, elem_classes="required", interactive=False)
                            version = gr.Dropdown(label="Version", choices=[PLEASE_CHOOSE], value="", scale=2,
                                                  min_width=300, interactive=False, elem_classes="required")
                        search_note = gr.HTML(elem_classes="note")
                        with gr.Row():
                            pull_btn = gr.Button("Download", variant="primary", scale=0, min_width=160, interactive=False)
                        pull_status = gr.HTML(elem_classes="note")

                    with gr.Column(elem_classes="card"):
                        gr.HTML('<h2>Downloads</h2><p class="sub">Downloads run in the background and pick up where they '
                                'left off after a restart or lost connection.</p>')
                        dl_view = gr.HTML(downloads_html())
                        dl_remove_ref = gr.Textbox(elem_id="dl-remove-ref", elem_classes="hidden-control", container=False)
                        dl_remove_btn = gr.Button("remove", elem_id="dl-remove-btn", elem_classes="hidden-control")
                        dl_toggle_ref = gr.Textbox(elem_id="dl-toggle-ref", elem_classes="hidden-control", container=False)
                        dl_toggle_btn = gr.Button("toggle", elem_id="dl-toggle-btn", elem_classes="hidden-control")
                        with gr.Row(equal_height=True, visible=False) as dl_controls:
                            dl_pick = gr.Dropdown(label="Active download (select one to give it priority)", choices=[], scale=4)
                        dl_timer = gr.Timer(2.0)
                        dl_seen = gr.State(0)


                with gr.Column(scale=3, min_width=560, elem_classes="card chat-card"):
                    chat_state = gr.State(empty_chat())
                    chats_version = gr.State(0.0)
                    active_chat = gr.State(None)
                    with gr.Row(equal_height=False, elem_classes="chat-layout"):
                        with gr.Column(scale=1, min_width=300, elem_classes="chat-main"):
                            with gr.Row(equal_height=True, elem_classes="chat-head"):
                                gr.HTML('<div class="head-icons"><button class="side-toggle" type="button" '
                                        'data-action="expand" title="Show chats" aria-label="Show chats"></button>'
                                        '<button class="new-chat-icon" type="button" title="New chat" '
                                        'aria-label="New chat"></button></div>', elem_classes="expand-holder", min_width=34)
                                chat_model = gr.Dropdown(label="Model", choices=model_names(), scale=4)
                            chatbot = gr.Chatbot(show_label=False, height="60vh", elem_classes="chat",
                                                 placeholder="Send a message to start chatting.",
                                                 buttons=["copy"], group_consecutive_messages=False)
                            chat_usage = gr.Markdown("No messages yet.", elem_classes="note")
                            pending_files = gr.State([])
                            with gr.Row(equal_height=True, elem_classes="attach-bar", visible=False) as attach_bar:
                                attach_view = gr.HTML()
                                attach_clear = gr.Button("Remove", variant="secondary", size="sm", scale=0, min_width=90)
                            with gr.Row(equal_height=True, elem_classes="composer"):
                                attach_btn = gr.UploadButton("", file_count="multiple", file_types=ATTACH_TYPES, scale=0,
                                                             min_width=40, elem_classes="attach-btn", size="sm")
                                chat_in = gr.Textbox(show_label=False, placeholder="Message", container=False,
                                                     elem_id="chat-input", scale=5, lines=1, max_lines=1)
                                chat_send = gr.Button("Send", variant="primary", scale=0, min_width=90, elem_id="chat-send")
                                chat_stop = gr.Button("Stop", variant="secondary", scale=0, min_width=80, visible=False)

                        with gr.Column(scale=0, min_width=230, elem_classes="chat-sidebar"):
                            with gr.Row(equal_height=True, elem_classes="side-head"):
                                chat_new = gr.Button("New chat", variant="secondary", size="sm", elem_classes="new-chat",
                                                     scale=1, elem_id="chat-new")
                                gr.HTML('<button class="side-toggle" type="button" data-action="collapse" '
                                        'title="Collapse sidebar" aria-label="Collapse sidebar"></button>',
                                        elem_classes="collapse-holder", min_width=34)
                            chat_search = gr.Textbox(show_label=False, placeholder="Search chats", container=False,
                                                     elem_classes="chat-search", max_lines=1)
                            gr.HTML('<div class="side-label">Recents</div>')

                            @gr.render(inputs=[chats_version, active_chat, chat_search],
                                       triggers=[chats_version.change, chat_search.change, ui.load],
                                       trigger_mode="always_last", show_progress="hidden")
                            def render_chat_list(_version, current_id, query):
                                chats = CHAT_INDEX.search(query)
                                if not chats:
                                    msg = "No chats match your search." if (query or "").strip() else "No saved chats yet."
                                    gr.HTML(f'<div class="side-empty">{msg}</div>', key="chats-empty")
                                for c in chats:
                                    active = " active" if c["id"] == current_id else ""
                                    with gr.Row(equal_height=True, elem_classes="chat-item-row" + active,
                                                key=f"row-{c['id']}"):
                                        item = gr.Button(c["title"], size="sm", elem_classes="chat-item", scale=1,
                                                         key=f"item-{c['id']}")
                                        trash = gr.Button("", size="sm", elem_classes="chat-del",
                                                          scale=0, min_width=30, key=f"del-{c['id']}")
                                        confirm = gr.Checkbox(value=False, visible=False, key=f"confirm-{c['id']}")
                                    cid = gr.State(c["id"])
                                    item.click(open_chat, cid, show_progress="hidden", outputs=[chat_state, chatbot, chat_model, chat_usage, chats_version, active_chat])
                                    trash.click(remove_chat, [cid, confirm, chat_state],
                                                [chat_state, chatbot, chat_usage, chats_version, active_chat, chat_model],
                                                js="(cid, c, cur) => [cid, true, cur]", show_progress="hidden")

            with gr.Column(elem_classes="card"):
                gr.HTML('<h2>Chat models</h2><p class="sub">Chat models available to Ollama on this computer.</p>')
                models_version = gr.State(0.0)
                manage_status = gr.HTML(elem_classes="note")

                @gr.render(inputs=models_version, triggers=[models_version.change, ui.load], show_progress="hidden")
                def render_models(_version):
                    models = list_models()
                    with gr.Column(elem_classes="models-table"):
                        gr.HTML('<div class="mt-row mt-head">' + "".join(
                            f'<div class="mt-cell">{c}</div>' for c in MODEL_COLUMNS) + "</div>", key="models-head")
                        if not models:
                            gr.HTML('<div class="mt-empty">No models installed yet. Search above to add one.</div>',
                                    key="models-empty")
                        for m in models:
                            with gr.Row(equal_height=True, elem_classes="mt-row", key=f"model-{m['digest']}"):
                                gr.HTML(model_row_html(m), elem_classes="mt-cells", key=f"cells-{m['digest']}")
                                trash = gr.Button("", size="sm", elem_classes="row-del", scale=0, min_width=34,
                                                  key=f"del-{m['digest']}")
                                confirm = gr.Checkbox(value=False, visible=False, key=f"confirm-{m['digest']}")
                            name = gr.State(m["name"])
                            trash.click(delete_model, [name, confirm], [manage_status, models_version],
                                        js="(n, c) => [n, true]",
                                        show_progress="hidden")

        # ---------------- Images
        with gr.Tab("Images"):
            images_version = gr.State(0.0)
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=420, elem_classes="card"):
                    gr.HTML('<h2>Generate an image</h2><p class="sub">Runs locally with stable-diffusion.cpp. The model '
                            'loads only while generating.</p>')
                    img_model = gr.Dropdown(label="Image model", choices=installed_image_models(),
                                            value=(installed_image_models() or [(None, None)])[0][1])
                    img_prompt = gr.Textbox(label="Prompt", lines=4, placeholder="A lighthouse on a cliff at sunset, "
                                            "dramatic clouds, photorealistic")
                    with gr.Row():
                        img_size = gr.Dropdown(label="Size", choices=list(IMAGE_SIZES), value="Square 1024 x 1024",
                                               min_width=200)
                        img_steps = gr.Slider(1, 30, value=8, step=1, label="Steps", min_width=160)
                        img_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0, min_width=140)
                    with gr.Row():
                        img_go = gr.Button("Generate", variant="primary", scale=0, min_width=160)
                        img_stop = gr.Button("Stop", variant="stop", scale=0, min_width=120, visible=False)
                    img_note = gr.HTML(elem_classes="note")
                    img_progress = gr.HTML(IMAGE_IDLE)
                with gr.Column(scale=3, min_width=420, elem_classes="card"):
                    gr.HTML('<h2>Result</h2>')
                    img_result = gr.Image(show_label=False, type="filepath", interactive=False, height=560,
                                          elem_classes="image-result")

            with gr.Column(elem_classes="card"):
                gr.HTML('<h2>Gallery</h2><p class="sub">Images you have generated, newest first. Hover an image to expand or delete it.</p>')
                img_gallery = gr.HTML(gallery_html())
                img_del_name = gr.Textbox(elem_id="img-del-name", elem_classes="hidden-control", container=False)
                img_del_btn = gr.Button("delete", elem_id="img-del-btn", elem_classes="hidden-control")
                img_gallery_note = gr.HTML(elem_classes="note")

            with gr.Column(elem_classes="card add-model-card"):
                gr.HTML('<h2>Image models</h2><p class="sub">Search Hugging Face for an image model, or use one below. '
                        'Downloads resume if interrupted and appear under Downloads on the Models tab.</p>')
                img_search_loading = gr.HTML(visible=False, elem_classes="search-loading")
                with gr.Row(equal_height=True):
                    img_query = gr.Textbox(show_label=False, placeholder="flux schnell, z-image, sdxl turbo, or a link",
                                           scale=5, container=False)
                    img_search_btn = gr.Button("Search", variant="primary", scale=1, min_width=120)
                with gr.Row():
                    img_repo = gr.Dropdown(label="Repository", choices=[PLEASE_CHOOSE], value="", scale=3, min_width=300,
                                           interactive=False, elem_classes="required")
                    img_file = gr.Dropdown(label="Version", choices=[PLEASE_CHOOSE], value="", scale=2, min_width=300,
                                           interactive=False, elem_classes="required")
                img_search_note = gr.HTML(elem_classes="note")
                with gr.Row():
                    img_search_dl = gr.Button("Download", variant="primary", scale=0, min_width=160, interactive=False)
                img_models_note = gr.HTML(elem_classes="note")

                @gr.render(inputs=images_version, triggers=[images_version.change, ui.load], show_progress="hidden")
                def render_image_models(_version):
                    with gr.Column(elem_classes="models-table image-models"):
                        gr.HTML('<div class="mt-row mt-head"><div class="mt-cell">Model</div><div class="mt-cell">About'
                                '</div><div class="mt-cell">Size</div><div class="mt-cell">Status</div><div class="mt-cell">'
                                '</div></div>', key="img-head")
                        for key in image_models():
                            installed = image_model_installed(key)
                            with gr.Row(equal_height=True, elem_classes="mt-row", key=f"img-row-{key}"):
                                gr.HTML(image_models_version_html(key), elem_classes="mt-cells", key=f"img-cells-{key}")
                                if installed:
                                    trash = gr.Button("", size="sm", elem_classes="row-del", scale=0, min_width=34,
                                                      key=f"img-del-{key}")
                                    confirm = gr.Checkbox(value=False, visible=False, key=f"img-confirm-{key}")
                                    trash.click(delete_image_model, [gr.State(key), confirm],
                                                [img_models_note, images_version], js="(k, c) => [k, true]",
                                                show_progress="hidden")
                                else:
                                    dl = gr.Button("Download", size="sm", variant="primary", scale=0, min_width=110,
                                                   key=f"img-dl-{key}")
                                    dl.click(download_image_model, gr.State(key), img_models_note, show_progress="hidden")

        # ---------------- Probes
        with gr.Tab("Probes"):
            with gr.Row(equal_height=False):
                with gr.Column(elem_classes="card", scale=3, min_width=420):
                    gr.HTML('<h2>Probes</h2><p class="sub">Each probe is a family of attack prompts. '
                            'Select the ones you want, then save them as a group or scan with them directly.</p>')
                    with gr.Row(equal_height=True):
                        probe_query = gr.Textbox(show_label=False, placeholder="Filter probes, e.g. jailbreak, xss, leak",
                                                 scale=3, container=False)
                        probe_module = gr.Dropdown(["All categories"] + MODULES, value="All categories",
                                                   show_label=False, container=False, scale=2, min_width=200)
                    with gr.Row(equal_height=True):
                        show_all = gr.Checkbox(label="Include probes that are off by default", value=False, scale=3)
                        sel_shown_btn = gr.Button("Select shown", variant="secondary", size="sm", scale=0, min_width=130)
                        clear_btn = gr.Button("Clear", variant="secondary", size="sm", scale=0, min_width=90)
                    probe_count = gr.Markdown(elem_classes="note")
                    _init = filtered_probes("", "All categories", False)
                    probe_list = gr.CheckboxGroup(choices=[(probe_label(n), n) for n in _init], show_label=False,
                                                  elem_classes="probe-list")
                    probe_count.value = f"Showing {len(_init)} of {len(CATALOG)} probes."
                    sel_summary = gr.Markdown("No probes selected.", elem_classes="note")

                with gr.Column(elem_classes="card", scale=2, min_width=360):
                    gr.HTML('<h2>Groups</h2><p class="sub">Save the current selection for later, or load a group '
                            'or preset to edit it.</p>')
                    with gr.Row(equal_height=True):
                        group_pick = gr.Dropdown(label="Group", choices=group_choices(), scale=3, min_width=240)
                        load_btn = gr.Button("Load", variant="secondary", scale=0, min_width=100)
                        del_group_btn = gr.Button("Delete", variant="stop", scale=0, min_width=100)
                    with gr.Row(equal_height=True):
                        group_name = gr.Textbox(label="Group name", placeholder="My jailbreak suite", scale=3, min_width=240)
                        save_btn = gr.Button("Save selection", variant="primary", scale=0, min_width=160)
                    group_status = gr.Markdown(elem_classes="note")

        # ---------------- Scan
        with gr.Tab("Scan"):
            with gr.Column(elem_classes="card"):
                gr.HTML('<h2>Run a scan</h2><p class="sub">garak sends attack prompts to the model and '
                        'checks its responses.</p>')
                with gr.Row():
                    scan_model = gr.Dropdown(label="Model", choices=model_names(), scale=1, min_width=280)
                    scan_type = gr.Dropdown(label="Scan type", choices=scan_choices(), value="Quick check",
                                            scale=1, min_width=280)
                model_note = gr.Markdown(elem_classes="note")
                scan_desc = gr.Markdown(scan_info("Quick check", []), elem_classes="note")
                with gr.Accordion("Advanced settings", open=False):
                    with gr.Row():
                        gens = gr.Slider(1, 10, value=3, step=1, label="Attempts per prompt",
                                         info="More attempts give more reliable results but take longer.")
                        tmo = gr.Slider(30, 1800, value=300, step=30, label="Response timeout (seconds)",
                                        info="Set automatically from the model size; increase if requests time out.")
                    thinking = gr.Checkbox(label="Let reasoning models think before answering",
                                           info="Closer to real-world use, but much slower. Off by default.", value=False)
                with gr.Row():
                    scan_btn = gr.Button("Start scan", variant="primary", scale=0, min_width=160)
                    stop_btn = gr.Button("Stop", variant="stop", interactive=False, scale=0, min_width=120)
            with gr.Column(elem_classes="card"):
                gr.HTML('<h2>Activity</h2>')
                scan_progress = gr.HTML(IDLE_PROGRESS, elem_classes="scanprog-holder")
                scan_log = gr.Code(show_label=False, language=None, lines=22, max_lines=22,
                                   value="No scan running.", elem_classes="log")

        # ---------------- Reports
        with gr.Tab("Reports"):
            with gr.Column(elem_classes="card"):
                gr.HTML('<h2>Reports</h2><p class="sub">Results from previous scans, newest first. The security '
                        'summary is written for analysts; the garak report and raw data are the unmodified originals.</p>')
                with gr.Row(equal_height=True):
                    rep = gr.Dropdown(label="Report", choices=reports(), scale=4)
                    rep_open = gr.Button("Open summary in browser", variant="secondary", scale=0, min_width=220)
                    rep_delete = gr.Button("Delete report", variant="stop", scale=0, min_width=150, elem_classes="rep-delete")
                rep_status = gr.HTML(elem_classes="note")
                rep_confirm = gr.Checkbox(value=False, visible=False)
                with gr.Accordion("Download files (summary HTML, findings CSV, garak HTML, raw JSONL)", open=False):
                    rep_files = gr.File(show_label=False, file_count="multiple")
            with gr.Tabs(elem_classes="subtabs"):
                with gr.Tab("Security summary"):
                    rep_summary = gr.HTML()
                with gr.Tab("garak report"):
                    rep_view = gr.HTML()
                with gr.Tab("Raw data"):
                    with gr.Column(elem_classes="card"):
                        raw_pick = gr.Radio(["Full report (report.jsonl)", "Hitlog (failing responses)"],
                                            value="Full report (report.jsonl)", show_label=False, elem_classes="segmented")
                        raw_view = gr.Code(show_label=False, language=None, lines=30, max_lines=30, elem_classes="log")

    # status
    status_outputs = [status, unload_btn, start_btn, stop_ollama_btn, restart_ollama_btn]
    start_btn.click(start_ollama, outputs=status_outputs, show_progress="hidden")
    stop_ollama_btn.click(stop_ollama, outputs=status_outputs, show_progress="hidden")
    restart_ollama_btn.click(restart_ollama, outputs=status_outputs, show_progress="hidden")
    status_timer.tick(status_controls, outputs=status_outputs, show_progress="hidden")
    unload_btn.click(unload_models, outputs=status_outputs, show_progress="hidden")
    ui.load(status_controls, outputs=status_outputs, show_progress="hidden")
    update_outputs = [update_html, update_ollama_btn, update_garak_btn, update_app_btn]
    update_timer.tick(update_controls, outputs=update_outputs, show_progress="hidden")
    update_app_btn.click(lambda: start_update("app"), outputs=update_outputs, show_progress="hidden")
    update_dismiss.click(dismiss_update_msg, outputs=update_outputs,
                         show_progress="hidden")
    update_ollama_btn.click(lambda: start_update("ollama"), outputs=update_outputs, show_progress="hidden")
    update_garak_btn.click(lambda: start_update("garak"), outputs=update_outputs, show_progress="hidden")
    refresh_btn.click(refresh_all, chat_state, [models_version, scan_model, status, scan_type, group_pick, chat_model],
                      show_progress="hidden")

    # models
    source.change(source_changed, source, [query, repo, version, search_note, pull_btn], show_progress="hidden")
    for trigger in (search_btn.click, query.submit):
        trigger(search_models, [source, query], [repo, version, search_note, pull_btn, search_loading],
                show_progress="hidden")
    repo.input(list_versions, [source, repo], [version, search_note, pull_btn, search_loading], show_progress="hidden")
    version.input(version_changed, [repo, version], pull_btn, show_progress="hidden")
    pull_btn.click(pull_model, [source, repo, version], pull_status, show_progress="hidden")
    dl_pick.input(prioritize_download, dl_pick, pull_status, show_progress="hidden")
    dl_timer.tick(downloads_tick, [dl_seen, dl_pick],
                  [dl_view, dl_seen, models_version, dl_pick, dl_controls],
                  show_progress="hidden")
    dl_toggle_btn.click(toggle_download, dl_toggle_ref, [pull_status, dl_toggle_ref], show_progress="hidden")
    dl_remove_btn.click(remove_download, dl_remove_ref, [pull_status, dl_remove_ref], show_progress="hidden")
    img_go.click(generate_image, [img_model, img_prompt, img_size, img_steps, img_seed],
                 [img_progress, img_note, img_go, img_stop, img_result, img_gallery], show_progress="hidden")
    img_stop.click(stop_image, show_progress="hidden")
    for trigger in (img_search_btn.click, img_query.submit):
        trigger(image_search, img_query, [img_repo, img_file, img_search_note, img_search_dl, img_search_loading],
                show_progress="hidden")
    img_repo.input(image_versions, img_repo, [img_file, img_search_note, img_search_dl, img_search_loading],
                   show_progress="hidden")
    img_file.input(image_version_changed, [img_repo, img_file], img_search_dl, show_progress="hidden")
    img_search_dl.click(download_searched_image_model, [img_repo, img_file], [img_models_note, images_version],
                        show_progress="hidden")
    img_model.change(image_model_defaults, img_model, [img_steps, img_size], show_progress="hidden")
    ui.load(gallery_html, outputs=img_gallery, show_progress="hidden")
    chat_model.change(attach_state, chat_model, attach_btn, show_progress="hidden")
    img_del_btn.click(delete_gallery_image, img_del_name, [img_gallery, img_gallery_note, img_del_name],
                      show_progress="hidden")
    images_version.change(lambda: gr.update(choices=installed_image_models(),
                                            value=(installed_image_models() or [(None, None)])[0][1]),
                          outputs=img_model, show_progress="hidden")
    models_version.change(lambda: time.time(), outputs=images_version, show_progress="hidden")
    models_version.change(model_choices_changed, chat_state, [scan_model, chat_model], show_progress="hidden")
    chat_evt = [
        trigger(chat_submit, [chat_in, chat_state, chat_model, pending_files],
                [chat_in, chat_state, chatbot, chats_version, active_chat, chat_model, pending_files, attach_view,
                 attach_bar], show_progress="hidden").then(
            chat_respond, chat_state, [chat_state, chatbot, chat_send, chat_stop, chat_usage], show_progress="hidden")
        for trigger in (chat_in.submit, chat_send.click)]
    attach_btn.upload(add_attachments, [attach_btn, pending_files, chat_model], [pending_files, attach_view, attach_bar],
                      show_progress="hidden")
    attach_clear.click(clear_attachments, pending_files, [pending_files, attach_view, attach_bar], show_progress="hidden")
    chat_stop.click(chat_stopped, chat_state, [chat_send, chat_stop, chat_state, chatbot], cancels=chat_evt,
                    show_progress="hidden")
    chat_new.click(start_new_chat, outputs=[chat_state, chatbot, chat_usage, chats_version, active_chat, chat_model],
                   show_progress="hidden")

    # probes and groups
    filters = [probe_query, probe_module, show_all]
    for comp in filters:
        comp.change(filter_probes, filters + [selected], [probe_list, probe_count], show_progress="hidden")
    probe_list.input(probes_ticked, [probe_list] + filters + [selected], [selected, sel_summary], show_progress="hidden")
    sel_shown_btn.click(select_shown, filters + [selected], [selected, probe_list, sel_summary], show_progress="hidden")
    clear_btn.click(clear_selection, outputs=[selected, probe_list, sel_summary], show_progress="hidden")
    load_btn.click(load_group, [group_pick] + filters, [selected, probe_list, sel_summary, group_name],
                   show_progress="hidden")
    save_btn.click(save_group, [group_name, selected], [group_status, group_pick, scan_type], show_progress="hidden")
    del_group_btn.click(delete_group, group_pick, [group_status, group_pick, scan_type], show_progress="hidden")
    selected.change(scan_info, [scan_type, selected], scan_desc, show_progress="hidden")

    # scan
    scan_type.change(scan_info, [scan_type, selected], scan_desc, show_progress="hidden")
    scan_model.change(model_hint, scan_model, [model_note, tmo], show_progress="hidden")
    scan_btn.click(run_scan, [scan_model, scan_type, selected, gens, tmo, thinking],
                   [scan_log, scan_btn, stop_btn, scan_progress], show_progress="hidden").then(
        refresh_reports, outputs=rep, show_progress="hidden")
    stop_btn.click(stop_scan, outputs=scan_log, show_progress="hidden")

    # reports
    rep.change(show_report, [rep, raw_pick], [rep_summary, rep_view, raw_view, rep_files], show_progress="hidden")
    raw_pick.change(read_raw, [rep, raw_pick], raw_view, show_progress="hidden")
    rep_open.click(open_report, rep, show_progress="hidden")
    rep_delete.click(delete_report, [rep, rep_confirm], [rep, rep_status],
                     js="(name, c) => [name, !!name]", show_progress="hidden")

    ui.load(refresh_all, chat_state, [models_version, scan_model, status, scan_type, group_pick, chat_model],
            show_progress="hidden").then(
        model_hint, scan_model, [model_note, tmo], show_progress="hidden").then(
        refresh_reports, outputs=rep, show_progress="hidden").then(
        show_report, [rep, raw_pick], [rep_summary, rep_view, raw_view, rep_files], show_progress="hidden")

def ensure_desktop_integration():
    """Install the app icon and keep our own launcher entries pointing at it and at this app."""
    icons = Path.home() / ".local/share/icons/hicolor"
    for src, dest in ((APP_DIR / "static/icon.svg", icons / "scalable/apps/llm-scanner.svg"),
                      (APP_DIR / "static/icon.png", icons / "256x256/apps/llm-scanner.png")):
        if src.exists() and (not dest.exists() or dest.read_bytes() != src.read_bytes()):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
    launcher = APP_DIR / "llm-scanner.sh"
    try:
        desktop_dir = Path(subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True,
                                          timeout=5).stdout.strip() or Path.home() / "Desktop")
    except Exception:
        desktop_dir = Path.home() / "Desktop"
    for entry in (Path.home() / ".local/share/applications/llm-scanner.desktop",
                  Path.home() / ".config/autostart/llm-scanner.desktop", desktop_dir / "llm-scanner.desktop"):
        try:
            text = entry.read_text(encoding="utf-8")
        except OSError:
            continue
        fixed = re.sub(r"^Icon=.*$", "Icon=llm-scanner", text, flags=re.M)
        fixed = re.sub(r"^Exec=.*$", f"Exec={launcher}", fixed, flags=re.M)
        if fixed != text:
            entry.write_text(fixed, encoding="utf-8")


if __name__ == "__main__":
    if not CHECK_ONLY:
        try:
            ensure_desktop_integration()
        except Exception:
            pass
        CHAT_INDEX.build()
        migrate_shared_files()
        unload_models()  # start with nothing in memory; models load only for a chat reply or a scan
        resume_pending_downloads()
        threading.Thread(target=_update_loop, daemon=True).start()
    ui.queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("LLM_SCANNER_PORT", 7861)),
                      inbrowser="--no-browser" not in sys.argv, allowed_paths=[str(GARAK_RUNS), str(ATTACH_DIR), str(IMAGES_DIR)],
                      theme=THEME, css=CSS, footer_links=[], js=APP_JS,
                      favicon_path=str(APP_DIR / "static/icon.png"))
