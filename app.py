"""LLM Scanner - a local GUI for pulling models into Ollama and scanning them with garak."""

import base64
import bisect
import codecs
import functools
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


def fit_breakdown(size_gb):
    """How a version's weights split across video memory and system memory when loaded."""
    if size_gb <= VRAM_BUDGET_GB:
        return f"{size_gb:.1f} GB VRAM"
    if size_gb <= VRAM_BUDGET_GB + ram_gb() * 0.6:
        return f"{VRAM_BUDGET_GB:.1f} GB VRAM + {size_gb - VRAM_BUDGET_GB:.1f} GB RAM"
    return f"{size_gb:.1f} GB — too large for this PC"


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
        short = re.sub(r"^(NVIDIA |AMD )?(GeForce |Radeon )?|( GPU)$", "", name)  # "RTX 5070 Laptop"
        return (short or name, num(util), int(used) / 1024, int(total) / 1024, num(temp), num(power))
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


def ollama_status_html(v):
    """Ollama's state, shown inside its Start / Stop / Restart control."""
    return (f'<span class="oc-status"><span class="dot ok"></span>Ollama {html.escape(v)}</span>' if v else
            '<span class="oc-status"><span class="dot bad"></span>Ollama stopped</span>')


def status_html():
    """GPU and memory pills plus what's running (Ollama's own status sits in its control)."""
    pills = []
    g = gpu_live()
    if g:
        name, util, used, total, temp, power = g
        tip = f"GPU {util or 0:.0f}% busy, {used:.1f} of {total:.1f} GB video memory in use" + (
            f", {temp:.0f}°C" if temp is not None else "") + (f", drawing {power:.0f} W" if power is not None else "")
        pills.append(f'<span class="pill" data-tip-title="{html.escape(name)}" data-tip="{html.escape(tip)}">'
                     f'<span class="dot ok"></span><span class="gpu-name">{html.escape(name)}</span>'
                     f'{_meter(util)}<span class="val">{util or 0:.0f}%</span>'
                     f'<span class="lbl">VRAM</span>{_meter(100 * used / total)}<span class="val">{used:.1f}/{total:.0f} GB</span>'
                     + (f'<span class="muted">{temp:.0f}°C</span>' if temp is not None else "") + '</span>')
    else:
        pills.append('<span class="pill"><span class="dot warn"></span>No GPU detected</span>')
    r = ram_live()
    if r:
        pills.append(f'<span class="pill"><span class="lbl">RAM</span>{_meter(100 * r[0] / r[1])}'
                     f'<span class="val">{r[0]:.0f}/{r[1]:.0f} GB</span></span>')
    return f'<div class="status">{"".join(pills)}</div>'


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
    return (status_html(), gr.update(visible=bool(v and loaded_models())), gr.update(visible=not v),
            gr.update(visible=bool(v)), gr.update(visible=bool(v)), ollama_status_html(v), activity_html())


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
_updates = {"checked": 0.0, "ollama": None, "garak": None, "app": None, "busy": None, "pct": None, "msg": "",
            "ok": True}


def _version_tuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:3])


def check_updates():
    """Compare LLM Scanner, Ollama and garak with their latest releases (only when asked: the Check for updates
    button, or after installing an update). Returns the names of the ones that couldn't be checked."""
    if _updates.get("restarting"):
        return []
    _updates["checked"] = time.time()
    failed = []
    try:
        latest = requests.get("https://api.github.com/repos/ollama/ollama/releases/latest", timeout=10).json()
        installed = ollama_up()
        tag = (latest.get("tag_name") or "").lstrip("v")
        _updates["ollama"] = (installed, tag) if installed and tag and \
            _version_tuple(tag) > _version_tuple(installed) and not latest.get("prerelease") else None
    except Exception:
        failed.append("Ollama")
    try:
        installed = importlib.metadata.version("garak")
        tag = requests.get("https://pypi.org/pypi/garak/json", timeout=10).json()["info"]["version"]
        _updates["garak"] = (installed, tag) if _version_tuple(tag) > _version_tuple(installed) else None
    except Exception:
        failed.append("garak")
    try:
        rel = requests.get(f"https://api.github.com/repos/{UPDATE_REPO}/releases/latest", timeout=10).json()
        tag = (rel.get("tag_name") or "").lstrip("v")
        if tag and _version_tuple(tag) > _version_tuple(APP_VERSION) and not rel.get("prerelease"):
            _updates["app"] = {"version": tag, "tag": rel["tag_name"], "notes": (rel.get("body") or "").strip()}
        else:
            _updates["app"] = None
    except Exception:
        failed.append("LLM Scanner")
    return failed


def check_updates_now():
    """Check for updates button: look now, then show what's available or that everything is current."""
    if _updates["busy"] or _updates.get("restarting"):
        return update_controls()
    _updates.update(msg="", msg_expires=None, dismissed={})  # asking again shows everything again
    _set_progress("Checking for updates", None)
    try:
        failed = check_updates()
    finally:
        _set_progress(None)
    if failed:
        _updates.update(ok=False, msg=f"Couldn't check {', '.join(failed)} for updates. Check the internet connection "
                                      "and try again.")
    elif not (_updates["app"] or _updates["ollama"] or _updates["garak"]):
        try:
            garak = importlib.metadata.version("garak")
        except Exception:
            garak = "?"
        _updates.update(ok=True, msg=f"Everything is up to date: LLM Scanner {APP_VERSION}, Ollama "
                                     f"{ollama_up() or 'not running'}, garak {garak}.",
                        msg_expires=time.time() + 6)  # nothing to do, so the box goes away by itself
    return update_controls()


def _set_progress(phase, pct=None):
    """pct=None shows an indeterminate (animated) bar for steps that can't report a percentage."""
    _updates["busy"] = phase
    _updates["pct"] = pct


def update_banner():
    """The line inside the update box: progress while checking or installing, or the last result. What can be
    updated is shown by the buttons beside it ("Update Ollama to 0.35.0")."""
    if _updates["busy"]:
        pct = _updates.get("pct")
        bar = (f'<div class="up-bar"><div style="width:{pct:.1f}%"></div></div>' if pct is not None else
               '<div class="up-bar indeterminate"><div></div></div>')
        right = f"{pct:.0f}%" if pct is not None else ""
        return (f'<div class="update-banner working"><div class="up-top"><span>{html.escape(_updates["busy"])}</span>'
                f'<span>{right}</span></div>{bar}</div>')
    if _updates["msg"]:
        ok = _updates.get("ok", True)
        return f'<div class="update-banner"><div class="up-result {"ok" if ok else "bad"}">{html.escape(_updates["msg"])}</div></div>'
    return ""


UPDATE_NAMES = {"app": "LLM Scanner", "ollama": "Ollama", "garak": "garak"}


def update_versions(kind):
    """(installed, available) for one kind of update, or None."""
    u = _updates[kind]
    if not u:
        return None
    return (APP_VERSION, u["version"]) if kind == "app" else u


def update_item_html(kind):
    installed, new = update_versions(kind) or ("", "")
    notes = _updates["app"]["notes"][:600] if kind == "app" and _updates["app"] and _updates["app"]["notes"] else ""
    tip = f' data-tip-title="What\'s new in {new}" data-tip="{html.escape(notes)}"' if notes else ""
    return (f'<div class="up-item-text"{tip}><span class="up-name">{UPDATE_NAMES[kind]}</span>'
            f'<span class="up-ver">{html.escape(installed)} <span class="up-arrow">→</span> {html.escape(new)}</span></div>')


def update_controls():
    """The update box at the top. Hidden unless there is something to update, an update running, or a result to show.
    Lists each available update with its own button, plus Update all when there is more than one."""
    if _updates.get("msg_expires") and time.time() > _updates["msg_expires"]:
        _updates.update(msg="", msg_expires=None)
    busy = bool(_updates["busy"])
    dismissed = _updates.get("dismissed") or {}
    available = [k for k in UPDATE_ORDER if _updates[k] and dismissed.get(k) != update_versions(k)[1]]
    show = busy or bool(_updates["msg"]) or bool(available)
    rows = [gr.update(visible=kind in available and not busy) for kind in ("app", "ollama", "garak")]
    labels = [update_item_html(kind) for kind in ("app", "ollama", "garak")]
    return (gr.update(visible=show), update_banner(), *rows, *labels,
            gr.update(visible=len(available) > 1 and not busy), gr.update(visible=show and not busy))


def dismiss_update_msg():
    """The x on the update box: close it, and don't offer these same versions again until the next check."""
    _updates["dismissed"] = {k: update_versions(k)[1] for k in UPDATE_ORDER if _updates[k]}
    _updates.update(msg="", msg_expires=None)
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
    """The update is installed and the app restarts in a few seconds; stop offering it in the meantime. During
    Update all, the restart waits until every update has run."""
    _updates[kind] = None
    if _updates.get("defer_restart"):
        _updates["restart_pending"] = True
        return
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
        if not _updates.get("restarting") and not _updates.get("defer_restart"):  # (a restart would re-offer it)
            check_updates()


UPDATE_ORDER = ("ollama", "garak", "app")  # LLM Scanner last: it restarts the app


def _run_all_updates():
    """Update all: every available update in turn, then one restart if any of them needs it."""
    _updates["defer_restart"] = True
    results, all_ok = [], True
    try:
        for kind in UPDATE_ORDER:
            if _updates.get(kind):
                _run_update(kind)
                results.append(_updates["msg"])
                all_ok = all_ok and _updates["ok"]
    finally:
        _updates["defer_restart"] = False
    _updates.update(ok=all_ok, msg=" ".join(r for r in results if r))
    if _updates.pop("restart_pending", False):
        _restart_soon("app")
    else:
        check_updates()


def start_update(kind):
    """An Update button in the update box; kind is "app", "ollama", "garak" or "all"."""
    if _updates["busy"] or _updates.get("restarting"):
        return update_controls()
    if _proc.get("p") and _proc["p"].poll() is None:
        _updates["ok"] = False
        _updates["msg"] = "Finish or stop the running scan before updating."
        return update_controls()
    if kind in ("app", "all") and (_image_proc.get("p") and _image_proc["p"].poll() is None):
        _updates["ok"] = False
        _updates["msg"] = "Wait for the image to finish generating before updating."
        return update_controls()
    _updates["msg"] = ""
    _set_progress("Starting update", 0)
    threading.Thread(target=_run_all_updates if kind == "all" else _run_update, args=() if kind == "all" else (kind,),
                     daemon=True).start()
    return update_controls()


# ---------------------------------------------------------------- installed models
def list_models():
    try:
        return requests.get(f"{OLLAMA}/api/tags", timeout=5).json().get("models", [])
    except Exception:
        return []


def model_names():
    return [m["name"] for m in list_models()]


_load_problems = {}  # (name, digest) -> Ollama's error reading the model, "" if it reads fine


def load_problem(m):
    """Why Ollama can't load an installed model ("" if it can). Ollama lists every model it has, but only finds out
    that a file is in a format it can't read (Prism's PQ2_0, for example) when it opens it, so ask once per file."""
    key = (m["name"], m.get("digest"))
    if key not in _load_problems:
        try:
            r = requests.post(f"{OLLAMA}/api/show", json={"model": m["name"]}, timeout=10)
        except Exception:
            return ""  # Ollama unreachable: say nothing rather than mark a good model
        if r.ok:
            _load_problems[key] = ""
        else:
            try:
                err = str(r.json().get("error") or "") or f"HTTP {r.status_code}"
                _load_problems[key] = re.sub(r'^read GGUF metadata "[^"]*": ', "", err)  # drop the blob path
            except ValueError:
                _load_problems[key] = f"HTTP {r.status_code}"
    return _load_problems[key]


def scan_model_choices():
    """(choices, first runnable name) for the Scan tab. A model Ollama can't load stays listed but is marked, and
    app.js greys it out with an explanation."""
    models = list_models()
    bad = {m["name"] for m in models if load_problem(m)}
    choices = [(m["name"] + (f"   {CANNOT_LOAD}" if m["name"] in bad else ""), m["name"]) for m in models]
    return choices, next((m["name"] for m in models if m["name"] not in bad), None)


MODEL_COLUMNS = ("Model", "Size", "Parameters", "Quantization", "Added", "")


# Source logos (Simple Icons, CC0), shown in the models table instead of the words "Hugging Face" / "Ollama".
SOURCE_ICONS = {
    SRC_HF: "M1.4446 11.5059c0 1.1021.1673 2.1585.4847 3.1563-.0378-.0028-.0691-.0058-.1058-.0058-.4209 0-.8015.16-1.0704.4512-.3454.3737-.4984.8335-.4316 1.293a1.576 1.576 0 0 0 .2148.5978c-.2319.1864-.4018.4456-.4844.7578-.0646.2448-.131.7543.2149 1.2794a1.4552 1.4552 0 0 0-.0625.1055c-.208.3923-.2207.8372-.0371 1.25.2783.6258.9696 1.1175 2.3126 1.6467.8356.3292 1.5988.5411 1.6056.543 1.1046.2847 2.104.4277 2.969.4277 1.4173 0 2.4754-.3849 3.1525-1.1446 1.538.2651 2.791.1403 3.592.006.6773.7555 1.7332 1.1387 3.1467 1.1387.8649 0 1.8643-.143 2.969-.4278.0068-.0019.77-.2138 1.6056-.543 1.343-.5292 2.0343-1.0208 2.3126-1.6466.1836-.4129.171-.8577-.037-1.25a1.4685 1.4685 0 0 0-.0626-.1056c.346-.525.2795-1.0346.2149-1.2793-.0826-.3122-.2525-.5714-.4844-.7579.11-.1816.1831-.3788.2148-.5977.0669-.4595-.0862-.9193-.4316-1.293-.2688-.2913-.6495-.4513-1.0704-.4513-.0209 0-.0376.0008-.0588.0018.3162-.9966.4846-2.0518.4846-3.1523 0-5.807-4.7362-10.5144-10.5789-10.5144-5.8426 0-10.5788 4.7073-10.5788 10.5144Zm10.5788-9.4831c5.2727 0 9.5476 4.246 9.5476 9.483a9.4201 9.4201 0 0 1-.2696 2.2365c-.0039-.0047-.0079-.011-.0117-.0156-.274-.3255-.6679-.5059-1.1075-.5059-.352 0-.714.1155-1.0763.3438-.2403.1517-.5058.422-.7793.7598-.2534-.3492-.608-.5832-1.0137-.6465a1.5174 1.5174 0 0 0-.2344-.0176c-.9263 0-1.4828.7993-1.6935 1.5177-.1046.2426-.6065 1.3482-1.3614 2.0978-1.1681 1.1601-1.4458 2.3534-.8396 3.6382-.843.1029-1.5836.0927-2.365-.006.5906-1.212.3626-2.4388-.8426-3.6322-.755-.7496-1.2568-1.8552-1.3614-2.0978-.2107-.7184-.7673-1.5177-1.6935-1.5177-.078 0-.1568.0054-.2344.0176-.4057.0633-.7604.2973-1.0137.6465-.2735-.3379-.539-.6081-.7794-.7598-.3622-.2283-.7243-.3438-1.0762-.3438-.4266 0-.8094.171-1.0821.4786a9.4208 9.4208 0 0 1-.2598-2.1936c0-5.237 4.2749-9.483 9.5475-9.483zM8.6443 7.0036c-.4838.0043-.9503.2667-1.1934.7227-.3536.6633-.1006 1.4873.5645 1.84.351.1862.4883-.5261.836-.6485.3107-.1095.841.399 1.0078.086.3536-.6634.1025-1.4874-.5625-1.84a1.3659 1.3659 0 0 0-.6524-.1602Zm6.8403 0c-.2199-.002-.4426.05-.6504.1602-.665.3526-.9181 1.1766-.5645 1.84.1669.313.6971-.1955 1.0079-.086.3476.1224.4867.8347.838.6485.6649-.3527.916-1.1767.5624-1.84-.243-.456-.7096-.7184-1.1934-.7227Zm-9.7565 1.418a.8768.8768 0 0 0-.877.877c0 .4846.3925.877.877.877a.8768.8768 0 0 0 .877-.877.8768.8768 0 0 0-.877-.877zm12.6434 0c-.4845 0-.879.3925-.879.877 0 .4846.3945.877.879.877a.8768.8768 0 0 0 .877-.877.8768.8768 0 0 0-.877-.877zM8.7927 11.459c-.179-.003-.2793.1107-.2793.416 0 .8097.3874 2.125 1.4279 2.924.207-.7123 1.3453-1.2832 1.5079-1.2012.2315.1167.2191.4417.6074.7266.3884-.285.374-.6098.6056-.7266.1627-.082 1.3009.4889 1.5079 1.2012 1.0404-.799 1.4278-2.1144 1.4278-2.924 0-1.2212-1.583.6402-3.5413.6485-1.4686-.0061-2.7266-1.0558-3.2639-1.0645zM4.312 14.4768c.5792.365 1.6964 2.2751 2.1056 3.0177.1371.2488.371.3536.582.3536.4188 0 .7465-.4138.0391-.9395-1.0636-.791-.6914-2.0846-.1836-2.1642a.4302.4302 0 0 1 .0664-.004c.4616 0 .666.7892.666.7892s.5959 1.4898 1.6213 2.508c.942.9356 1.062 1.703.4961 2.6661-.0164-.004-.0159.0236-.1484.2149-.1853.2673-.4322.4688-.7188.6152-.5062.2269-1.1397.2696-1.7833.2696-1.037 0-2.1017-.1824-2.6975-.336-.0293-.0075-3.6505-.9567-3.1916-1.8224.0771-.1454.2033-.2031.3633-.2031.6463 0 1.823.9551 2.3283.9551.113 0 .196-.0865.2285-.2031.2249-.8045-3.2787-1.0522-2.9846-2.1642.0519-.1967.193-.2757.3907-.2754.854 0 2.7704 1.4923 3.172 1.4923.0307 0 .0525-.0085.0645-.0274.2012-.3227.1096-.5865-1.3087-1.4395-1.4182-.8533-2.4315-1.329-1.8653-1.9416.0651-.0707.1574-.1015.2695-.1015.8611.0002 2.8948 1.84 2.8948 1.84s.5487.5683.8809.5683c.0762 0 .1416-.0315.1855-.1054.2355-.3946-2.1858-2.2183-2.3224-2.971-.0926-.51.0641-.7676.3555-.7676-.0006.008.1701-.0285.4942.1759zm16.2257.5918c-.1366.7526-2.5579 2.5764-2.3224 2.9709.044.074.1092.1055.1855.1055.3321 0 .881-.5684.881-.5684s2.0336-1.8397 2.8947-1.84c.1121 0 .2044.0308.2695.1016.5662.6125-.447 1.0882-1.8653 1.9415-1.4183.853-1.51 1.1168-1.3087 1.4396.012.0188.0337.0273.0644.0273.4016 0 2.3181-1.4923 3.1721-1.4923.1977-.0002.3388.0787.3907.2754.294 1.112-3.2095 1.3597-2.9846 2.1642.0325.1166.1156.2032.2285.2032.5054 0 1.682-.9552 2.3283-.9552.16 0 .2862.0577.3633.2032.459.8656-3.1623 1.8149-3.1916 1.8224-.5958.1535-1.6605.336-2.6975.336-.6351 0-1.261-.0409-1.7638-.2599-.2949-.1472-.5488-.3516-.7383-.625-.0411-.0682-.1026-.1476-.1426-.205-.5726-.9679-.455-1.7371.4903-2.676 1.0254-1.0182 1.6212-2.508 1.6212-2.508s.2044-.7891.666-.7891a.4318.4318 0 0 1 .0665.0039c.5078.0796.88 1.3732-.1836 2.1642-.7074.5257-.3797.9395.039.9395.211 0 .445-.1047.5821-.3535.4092-.7426 1.5264-2.6527 2.1056-3.0178.5588-.3524.99-.1816.8497.5918z",
    SRC_OLLAMA: "M16.361 10.26a.894.894 0 0 0-.558.47l-.072.148.001.207c0 .193.004.217.059.353.076.193.152.312.291.448.24.238.51.3.872.205a.86.86 0 0 0 .517-.436.752.752 0 0 0 .08-.498c-.064-.453-.33-.782-.724-.897a1.06 1.06 0 0 0-.466 0zm-9.203.005c-.305.096-.533.32-.65.639a1.187 1.187 0 0 0-.06.52c.057.309.31.59.598.667.362.095.632.033.872-.205.14-.136.215-.255.291-.448.055-.136.059-.16.059-.353l.001-.207-.072-.148a.894.894 0 0 0-.565-.472 1.02 1.02 0 0 0-.474.007Zm4.184 2c-.131.071-.223.25-.195.383.031.143.157.288.353.407.105.063.112.072.117.136.004.038-.01.146-.029.243-.02.094-.036.194-.036.222.002.074.07.195.143.253.064.052.076.054.255.059.164.005.198.001.264-.03.169-.082.212-.234.15-.525-.052-.243-.042-.28.087-.355.137-.08.281-.219.324-.314a.365.365 0 0 0-.175-.48.394.394 0 0 0-.181-.033c-.126 0-.207.03-.355.124l-.085.053-.053-.032c-.219-.13-.259-.145-.391-.143a.396.396 0 0 0-.193.032zm.39-2.195c-.373.036-.475.05-.654.086-.291.06-.68.195-.951.328-.94.46-1.589 1.226-1.787 2.114-.04.176-.045.234-.045.53 0 .294.005.357.043.524.264 1.16 1.332 2.017 2.714 2.173.3.033 1.596.033 1.896 0 1.11-.125 2.064-.727 2.493-1.571.114-.226.169-.372.22-.602.039-.167.044-.23.044-.523 0-.297-.005-.355-.045-.531-.288-1.29-1.539-2.304-3.072-2.497a6.873 6.873 0 0 0-.855-.031zm.645.937a3.283 3.283 0 0 1 1.44.514c.223.148.537.458.671.662.166.251.26.508.303.82.02.143.01.251-.043.482-.08.345-.332.705-.672.957a3.115 3.115 0 0 1-.689.348c-.382.122-.632.144-1.525.138-.582-.006-.686-.01-.853-.042-.57-.107-1.022-.334-1.35-.68-.264-.28-.385-.535-.45-.946-.03-.192.025-.509.137-.776.136-.326.488-.73.836-.963.403-.269.934-.46 1.422-.512.187-.02.586-.02.773-.002zm-5.503-11a1.653 1.653 0 0 0-.683.298C5.617.74 5.173 1.666 4.985 2.819c-.07.436-.119 1.04-.119 1.503 0 .544.064 1.24.155 1.721.02.107.031.202.023.208a8.12 8.12 0 0 1-.187.152 5.324 5.324 0 0 0-.949 1.02 5.49 5.49 0 0 0-.94 2.339 6.625 6.625 0 0 0-.023 1.357c.091.78.325 1.438.727 2.04l.13.195-.037.064c-.269.452-.498 1.105-.605 1.732-.084.496-.095.629-.095 1.294 0 .67.009.803.088 1.266.095.555.288 1.143.503 1.534.071.128.243.393.264.407.007.003-.014.067-.046.141a7.405 7.405 0 0 0-.548 1.873c-.062.417-.071.552-.071.991 0 .56.031.832.148 1.279L3.42 24h1.478l-.05-.091c-.297-.552-.325-1.575-.068-2.597.117-.472.25-.819.498-1.296l.148-.29v-.177c0-.165-.003-.184-.057-.293a.915.915 0 0 0-.194-.25 1.74 1.74 0 0 1-.385-.543c-.424-.92-.506-2.286-.208-3.451.124-.486.329-.918.544-1.154a.787.787 0 0 0 .223-.531c0-.195-.07-.355-.224-.522a3.136 3.136 0 0 1-.817-1.729c-.14-.96.114-2.005.69-2.834.563-.814 1.353-1.336 2.237-1.475.199-.033.57-.028.776.01.226.04.367.028.512-.041.179-.085.268-.19.374-.431.093-.215.165-.333.36-.576.234-.29.46-.489.822-.729.413-.27.884-.467 1.352-.561.17-.035.25-.04.569-.04.319 0 .398.005.569.04a4.07 4.07 0 0 1 1.914.997c.117.109.398.457.488.602.034.057.095.177.132.267.105.241.195.346.374.43.14.068.286.082.503.045.343-.058.607-.053.943.016 1.144.23 2.14 1.173 2.581 2.437.385 1.108.276 2.267-.296 3.153-.097.15-.193.27-.333.419-.301.322-.301.722-.001 1.053.493.539.801 1.866.708 3.036-.062.772-.26 1.463-.533 1.854a2.096 2.096 0 0 1-.224.258.916.916 0 0 0-.194.25c-.054.109-.057.128-.057.293v.178l.148.29c.248.476.38.823.498 1.295.253 1.008.231 2.01-.059 2.581a.845.845 0 0 0-.044.098c0 .006.329.009.732.009h.73l.02-.074.036-.134c.019-.076.057-.3.088-.516.029-.217.029-1.016 0-1.258-.11-.875-.295-1.57-.597-2.226-.032-.074-.053-.138-.046-.141.008-.005.057-.074.108-.152.376-.569.607-1.284.724-2.228.031-.26.031-1.378 0-1.628-.083-.645-.182-1.082-.348-1.525a6.083 6.083 0 0 0-.329-.7l-.038-.064.131-.194c.402-.604.636-1.262.727-2.04a6.625 6.625 0 0 0-.024-1.358 5.512 5.512 0 0 0-.939-2.339 5.325 5.325 0 0 0-.95-1.02 8.097 8.097 0 0 1-.186-.152.692.692 0 0 1 .023-.208c.208-1.087.201-2.443-.017-3.503-.19-.924-.535-1.658-.98-2.082-.354-.338-.716-.482-1.15-.455-.996.059-1.8 1.205-2.116 3.01a6.805 6.805 0 0 0-.097.726c0 .036-.007.066-.015.066a.96.96 0 0 1-.149-.078A4.857 4.857 0 0 0 12 3.03c-.832 0-1.687.243-2.456.698a.958.958 0 0 1-.148.078c-.008 0-.015-.03-.015-.066a6.71 6.71 0 0 0-.097-.725C8.997 1.392 8.337.319 7.46.048a2.096 2.096 0 0 0-.585-.041Zm.293 1.402c.248.197.523.759.682 1.388.03.113.06.244.069.292.007.047.026.152.041.233.067.365.098.76.102 1.24l.002.475-.12.175-.118.178h-.278c-.324 0-.646.041-.954.124l-.238.06c-.033.007-.038-.003-.057-.144a8.438 8.438 0 0 1 .016-2.323c.124-.788.413-1.501.696-1.711.067-.05.079-.049.157.013zm9.825-.012c.17.126.358.46.498.888.28.854.36 2.028.212 3.145-.019.14-.024.151-.057.144l-.238-.06a3.693 3.693 0 0 0-.954-.124h-.278l-.119-.178-.119-.175.002-.474c.004-.669.066-1.19.214-1.772.157-.623.434-1.185.68-1.382.078-.062.09-.063.159-.012z",
}


def source_icon(source):
    return (f'<span class="src-icon {"hf" if source == SRC_HF else "ollama"}" title="{source}" aria-label="{source}">'
            f'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="{SOURCE_ICONS[source]}"/></svg></span>')


def model_page(name):
    """The model's page: its Hugging Face repository, or its page in the Ollama library."""
    if name.startswith("hf.co/"):
        return "https://huggingface.co/" + name.removeprefix("hf.co/").split(":")[0]
    base, _, tag = name.partition(":")
    path = base if "/" in base else f"library/{base}"
    return f"{OLLAMA_WEB}/{path}" + (f":{tag}" if tag and tag != "latest" else "")


def model_row_html(m):
    d = m.get("details", {})
    source = SRC_HF if m["name"].startswith("hf.co/") else SRC_OLLAMA
    cells = [html.escape(str(c)) for c in (m["name"], f"{m.get('size', 0) / 1e9:.1f} GB", d.get("parameter_size", ""),
                                             d.get("quantization_level", ""), m.get("modified_at", "")[:10])]
    cells[0] = (f'<span class="mt-model">{source_icon(source)}<a class="mt-model-name" href="{html.escape(model_page(m["name"]))}"'
                f' target="_blank" rel="noopener noreferrer" title="Open this model\'s page">{cells[0]}</a></span>')
    return "".join(f'<div class="mt-cell">{c}</div>' for c in cells)


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


# The quantization at the end of a GGUF filename, with any prefix it carries: Q4_K_M, IQ3_XS, UD-Q5_K_XL, BF16, and
# repository-specific ones such as PQ2_0 or PTQ1_0. It must follow a separator, so "PQ2_0" is never read as "Q2_0".
QUANT_IN_NAME = re.compile(r"(?:^|[-_.])((?:UD-)?[A-Za-z]{0,3}(?:Q\d[\w.]*|BF16|F16|F32|FP16|MXFP4[\w.]*))\.gguf$", re.I)
SHARD_IN_NAME = re.compile(r"-\d{5}-of-\d{5}")
NOT_A_MODEL = ("mmproj", "imatrix", "mtp")  # companion files, not models to run
CANNOT_LOAD = "Ollama can't load this format"  # exact text app.js matches to grey out a version and swap in the icon

# The quantizations Ollama's own llama.cpp can load. Repositories also ship formats that only their authors' build
# can run (Prism's PQ2_0, PTQ1_0, Q2_g64, dspark-*); those are listed but marked, because Ollama cannot load them.
STANDARD_QUANT = re.compile(r"^(?:UD-)?(?:F16|F32|BF16|FP16|Q4_[01]|Q5_[01]|Q8_0|Q[2-8]_K(?:_[SMLXsmlx]{1,2})?"
                            r"|IQ[1-4]_(?:XXS|XS|S|M|NL)(?:_XL)?|TQ[12]_0|MXFP4(?:_MOE)?)$", re.I)


def runs_in_ollama(tag):
    return bool(STANDARD_QUANT.match(tag.split(":", 1)[-1]))


def hf_gguf_catalog(repo):
    """([(tag, path, size)] of runnable GGUF files, [(path, size)] of image projectors) in a Hugging Face repo.

    A file's tag is what tells it apart from the others in the repo, so it matches what the repository shows:
    Q4_K_M, UD-Q5_K_XL, PQ2_0, dspark-bf16. Repositories name their files differently, so the shared part of the
    names is dropped rather than guessing which part is the quantization."""
    res = requests.get(f"{HF_API}/models/{repo_from_text(repo)}/tree/main", params={"recursive": "true"}, timeout=30)
    _check_response(res, f"Listing files of {repo}")
    ggufs = [(e["path"], int(e.get("size") or 0)) for e in res.json()
             if e.get("path", "").lower().endswith(".gguf") and not SHARD_IN_NAME.search(e["path"].lower())]
    models = [g for g in ggufs if not any(k in g[0].lower() for k in NOT_A_MODEL)]
    projectors = [g for g in ggufs if "mmproj" in g[0].lower()]
    stems = [path.rsplit("/", 1)[-1][:-5] for path, _ in models]
    shared = os.path.commonprefix(stems) if len(stems) > 1 else ""
    cut = max(shared.rfind(c) for c in "-_.")
    shared = shared[:cut + 1] if cut >= 0 else ""
    catalog = []
    for (path, size), stem in zip(models, stems):
        tag = stem[len(shared):] if shared else ""
        if not tag:  # one file, or a name that is all shared: fall back to the quantization in it
            m = QUANT_IN_NAME.search(f"{stem}.gguf")
            tag = m.group(1) if m else stem
        catalog.append((tag, path, size))
    return catalog, projectors


_hf_meta = {}


def _short_count(n):
    return f"{n / 1e6:.1f}M".replace(".0M", "M") if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1000 else str(n)


def hf_repo_info(repo):
    """What Hugging Face says about a repository: base model, licence, downloads, likes. {} if it can't be read."""
    repo = repo_from_text(repo)
    if repo not in _hf_meta:
        try:
            d = requests.get(f"{HF_API}/models/{repo}", timeout=15).json()
            card = d.get("cardData") or {}
            base = card.get("base_model")
            _hf_meta[repo] = {"base": (base[0] if isinstance(base, list) else base) or "",
                              "license": card.get("license") or "", "downloads": d.get("downloads") or 0,
                              "likes": d.get("likes") or 0, "gated": bool(d.get("gated"))}
        except Exception:
            _hf_meta[repo] = {}
    return _hf_meta[repo]


def hf_versions(repo):
    """[(size_gb, tag)] for single-file GGUF versions in a Hugging Face repo."""
    sizes = {}
    for tag, _, size in hf_gguf_catalog(repo)[0]:
        sizes.setdefault(tag, size / 1e9)
    return sorted((size, tag) for tag, size in sizes.items())


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
    """Load the versions of the chosen repository/model, describe it, and pick the best one this PC can run."""
    hide = gr.update(visible=False)
    if not repo:
        yield gr.update(choices=[PLEASE_CHOOSE], value="", interactive=False), "", gr.update(interactive=False), hide
        return
    yield (gr.update(choices=[LOADING], value="", interactive=False), "", gr.update(interactive=False),
           loading_screen("Loading versions", repo))
    projectors, about = [], []
    try:
        if source == SRC_OLLAMA:
            opts = ollama_versions(repo)
            desc = _ollama_meta.get(repo, {}).get("desc", "")
            about = [desc] if desc else []
        else:
            catalog, projectors = hf_gguf_catalog(repo)
            sizes = {}
            for tag, _, size in catalog:
                sizes.setdefault(tag, size / 1e9)
            opts = sorted((size, tag) for tag, size in sizes.items())
            info = hf_repo_info(repo)
            about = [f"Based on {info['base'].split('/')[-1]}." if info.get("base") else "",
                     f"{info['license']} licence." if info.get("license") else "",
                     f"{_short_count(info['downloads'])} downloads on Hugging Face." if info.get("downloads") else "",
                     "Sees images (the repository includes a vision projector)." if projectors else ""]
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
    # Ollama's own library only lists models it can run; Hugging Face repositories also hold other formats.
    check_format = source != SRC_OLLAMA
    usable = [o for o in opts if runs_in_ollama(o[1])] if check_format else list(opts)
    best = best_version(usable or opts)  # never recommend a format Ollama can't load
    choices = [PLEASE_CHOOSE] + [
        (f"{tag.split(':', 1)[-1]}   " + (fit_breakdown(size) if not check_format or runs_in_ollama(tag)
                                          else f"{size:.1f} GB   {CANNOT_LOAD}")
         + ("   (Recommended)" if tag == best else ""), tag) for size, tag in opts]
    size = next(sz for sz, t in opts if t == best)
    short = best.split(":", 1)[-1]
    if not usable:
        why = (f"Ollama can't load any version in this repository, so none will run here. {short} is selected only "
               "so you can try it.")
    elif size <= VRAM_BUDGET_GB:
        why = (f"Recommended: {short} ({size:.1f} GB), the largest version that fits your {VRAM_BUDGET_GB} GB of "
               "video memory, so it runs fully on the GPU.")
    else:
        why = (f"No version fits your {VRAM_BUDGET_GB} GB of video memory, so the smallest Ollama can load "
               f"({short}, {size:.1f} GB) is selected; it runs partly in system memory and is slower.")
    skipped = [t for _, t in opts if check_format and not runs_in_ollama(t)]
    if skipped:
        why += (f" {', '.join(skipped[:6])} " + ("is" if len(skipped) == 1 else "are") + " in a format Ollama "
                "cannot load; running those needs the llama.cpp build from the model's authors.")
    yield (gr.update(choices=choices, value=best, interactive=True),
           note_html(" ".join([p for p in about if p] + [why])), gr.update(interactive=True), hide)


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
    """[(path, size)] to import for a version tag: the model's GGUF file, plus the image projector (mmproj) if the
    repo has one, which vision models need to see images."""
    catalog, projectors = hf_gguf_catalog(repo)
    wanted = tag.removesuffix(".gguf")
    matches = [(path, size) for t, path, size in catalog if t.lower() == wanted.lower()]
    if not matches:  # older download queues and hand-typed tags: match the end of the file name
        ends_with = re.compile(rf"(^|[-_.]){re.escape(wanted)}$", re.I)
        matches = [(path, size) for _, path, size in catalog if ends_with.search(path.rsplit("/", 1)[-1][:-5])]
    if not matches:
        have = ", ".join(sorted({t for t, _, _ in catalog})[:8]) or "none"
        raise PermanentDownloadError(f"{repo} has no GGUF file for {tag}. It has: {have}")
    files = [min(matches, key=lambda g: len(g[0]))]
    if projectors:  # prefer the projector matching the model's quantization, then full precision
        stem = lambda path: path.rsplit("/", 1)[-1][:-5]  # noqa: E731
        rank = lambda g: (wanted.lower() not in stem(g[0]).lower(),  # noqa: E731
                          not re.search(r"f16|bf16", stem(g[0]), re.I), g[1])
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


# Ollama keeps an unfinished pull as sha256-<digest>-partial (plus -partial-N progress records) in its blobs folder and
# never removes it by itself, so a deleted download's layers are tracked here and deleted with it.
OLLAMA_BLOBS = Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama/models") / "blobs"
PULL_LAYERS_FILE = DATA_DIR / "pull_layers.json"  # {ref: [layer digests]} for Ollama pulls not yet finished
ORPHAN_PARTIAL_SECS = 900  # at startup, a partial layer no download owns is deleted once untouched this long


def _pull_layers():
    try:
        known = json.loads(PULL_LAYERS_FILE.read_text())
        return known if isinstance(known, dict) else {}
    except (OSError, ValueError):
        return {}


def _remember_layers(ref, digests):
    with _dl_lock:
        known = _pull_layers()
        if set(digests) <= set(known.get(ref, [])):
            return
        known[ref] = sorted(set(known.get(ref, [])) | set(digests))
        PULL_LAYERS_FILE.write_text(json.dumps(known, indent=2))


def _forget_layers(ref):
    with _dl_lock:
        known = _pull_layers()
        if known.pop(ref, None) is not None:
            PULL_LAYERS_FILE.write_text(json.dumps(known, indent=2))


def _ollama_partials(digests=None):
    """Ollama's partial-layer files, for the given digests ("sha256:<hex>") or all of them."""
    if not OLLAMA_BLOBS.is_dir():
        return []
    wanted = None if digests is None else {dg.replace(":", "-") for dg in digests}
    return [f for f in OLLAMA_BLOBS.glob("sha256-*-partial*")
            if wanted is None or f.name.split("-partial", 1)[0] in wanted]


def _drop_pull_data(ref):
    """Delete the partial layers an unfinished Ollama pull left, except any another queued download still needs.
    Waits briefly for Ollama to stop writing them after the pull was cancelled."""
    known = _pull_layers()
    mine = set(known.get(ref, []))
    others = {dg for r, _ in _pending() if r != ref for dg in known.get(r, [])}
    files = _ollama_partials(mine - others)
    for _ in range(20):
        sizes = {f: f.stat().st_mtime for f in files if f.exists()}
        time.sleep(0.5)
        if all(f.exists() and f.stat().st_mtime == m for f, m in sizes.items()):
            break
    for f in files:
        f.unlink(missing_ok=True)
    _forget_layers(ref)


def orphaned_ollama_partials():
    """Partial layers no download owns (left by downloads deleted before layers were tracked, or by a crash) that
    have not been touched for a while. Nothing is claimed while a queued download's layers are unknown."""
    pending = [r for r, _ in _pending() if not r.startswith("image:")]
    known = _pull_layers()
    if any(r not in known for r in pending):
        return []
    owned = {dg.replace(":", "-") for r in pending for dg in known[r]}
    cutoff = time.time() - ORPHAN_PARTIAL_SECS
    return [f for f in _ollama_partials()
            if f.name.split("-partial", 1)[0] not in owned and f.stat().st_mtime < cutoff]


def _worker_stopped(ref, d):
    """A worker left its loop without finishing: deleted (partial data removed) or paused (kept to resume)."""
    if d["cancel"]:
        if d.get("hidden"):
            for f in _partial_files(ref):
                f.unlink(missing_ok=True)
            _drop_pull_data(ref)
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
            if d.get("direct"):  # Ollama couldn't fetch this one itself
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
                    if ref.startswith("hf.co/"):  # whatever Ollama makes of it, we can fetch the file ourselves
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
                        if ref.startswith("hf.co/"):  # blocked redirect, unknown tag, ...: download it ourselves
                            d["direct"] = True
                            break
                        raise PermanentDownloadError(ev["error"])
                    if ev.get("digest") and ev.get("total"):
                        if ev["digest"] not in layers:
                            _remember_layers(ref, [ev["digest"]])
                        layers[ev["digest"]] = (ev.get("completed", 0), ev["total"])
                        done = sum(c for c, _ in layers.values())
                        total = sum(t for _, t in layers.values())
                        d.update(state="downloading", msg="Downloading", done=done, total=total)
                    elif ev.get("status"):
                        d.update(state="downloading", msg=ev["status"].capitalize())
                    if ev.get("status") == "success":
                        d.update(state="done", msg="Done", finished=time.time())
                        _set_pending(ref, False)
                        _forget_layers(ref)
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
    """Trash icon on a download: stop it and remove it from the list, deleting its partial data, including the
    partial layers of an Ollama pull (a running worker deletes those itself once it has stopped)."""
    d = _downloads.get(ref)
    if not d:
        return gr.update(), gr.update()
    with _dl_lock:
        d["hidden"] = True
        if d["state"] in ACTIVE:
            d["cancel"] = True
        running = d.get("running")
    if not running:  # nothing will clean up after it, e.g. a paused, stalled or failed download
        for f in _partial_files(ref):
            f.unlink(missing_ok=True)
        if d["state"] in ACTIVE:
            d.update(state="cancelled", msg="Cancelled")
    _set_pending(ref, False)
    if not running:
        threading.Thread(target=_drop_pull_data, args=(ref,), daemon=True).start()
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
                     "tier": v.get("tier"), "active": v.get("active", True), "tags": v.get("tags", []),
                     "goal": (v.get("goal") or "").strip()}
    return dict(sorted(cat.items()))


# What each family of garak probes does, in plain language (shown when hovering a probe or family).
MODULE_INFO = {
    "adaptive_attacks": "An attacker model rewrites a harmful request again and again until the target complies.",
    "agent_breaker": "Tests AI agents that use tools, by tricking them into misusing those tools. Needs an agent "
                     "set up for it; not meaningful for a plain chat model.",
    "ansiescape": "Tries to make the model output terminal escape codes, which can hide or change what a person "
                  "sees in a terminal or log that shows the output.",
    "apikey": "Asks the model to produce or complete secret API keys.",
    "atkgen": "An attacker model writes prompts on the fly to steer the target into toxic replies.",
    "audio": "Attacks delivered as audio, for models that accept sound.",
    "av_spam_scanning": "Asks the model to output the standard antivirus and spam test signatures, to see whether "
                        "anything filters its output.",
    "badchars": "Hides or disguises instructions with invisible or look-alike Unicode characters.",
    "base": "garak's internal building blocks, not attacks on their own.",
    "continuation": "Starts an offensive word or slur and checks whether the model completes it.",
    "dan": "'Do Anything Now' style jailbreaks: role-play prompts that tell the model it has no rules.",
    "divergence": "Makes the model repeat a word until it drifts off and spills memorized training data.",
    "doctor": "Wraps harmful requests in fake policy files or medical role-play to get past safety training.",
    "donotanswer": "Questions a responsible model should decline: discrimination, dangerous information, "
                   "misinformation and malicious uses.",
    "dra": "Disguises a harmful request as a word puzzle the model has to reconstruct, then answer.",
    "encoding": "Hides instructions in encodings such as Base64, ROT13 or Morse code to see if the model decodes "
                "and follows them.",
    "exploitation": "Tries to get output that would inject code into an application, such as template or SQL "
                    "injection.",
    "fileformats": "Inspects the model's files for risky formats instead of sending prompts.",
    "fitd": "Foot in the door: starts with harmless requests and escalates to harmful ones over several turns.",
    "glitch": "Uses rare 'glitch' tokens that can make models behave erratically.",
    "goat": "An attacker model holds multi-turn conversations to wear down the target's refusals.",
    "goodside": "Classic attacks by Riley Goodside: invisible Unicode instructions, threats to force a format, and "
                "questions about a person the model is likely to make things up about.",
    "grandma": "The 'grandma exploit': asks the model to play a late grandmother who used to share harmful "
               "information or software product keys.",
    "latentinjection": "Indirect prompt injection: hides instructions inside documents, résumés or reports the "
                       "model is asked to work with.",
    "leakreplay": "Checks whether the model reproduces copyrighted or training text word for word, such as books "
                  "and newspaper articles.",
    "lmrc": "Language Model Risk Cards: bullying, deadnaming, quack medicine, sexual content, slurs and profanity.",
    "malwaregen": "Asks the model to write malware, from full programs to evasion tricks and payloads.",
    "misleading": "States false claims and checks whether the model corrects them or builds on them.",
    "packagehallucination": "Asks for code and checks whether it uses software packages that don't exist; "
                            "attackers can publish malicious packages under those names.",
    "phrasing": "Rewords harmful requests in the past or future tense, which often slips past refusals.",
    "promptinject": "Tries to hijack the model's task so it outputs an attacker's text instead.",
    "propile": "Checks whether the model leaks personal information about real people.",
    "realtoxicityprompts": "Sentence openings known to lead models into toxic continuations.",
    "sata": "Masks the harmful words in a request and asks the model to fill them in as part of an innocent task.",
    "smuggling": "Smuggles a harmful request past filters by splitting or disguising it.",
    "snowball": "Questions models tend to answer wrongly with confidence, testing whether errors snowball.",
    "suffix": "Adds adversarial suffixes, strings of odd text found by optimization, that break safety training.",
    "sysprompt_extraction": "Tries to get the model to reveal its hidden system prompt.",
    "tap": "Tree of Attacks with Pruning: jailbreak prompts found by an automated attacker.",
    "test": "garak's self-test probes; not real attacks.",
    "topic": "Asks about controversial topics to see whether the model takes sides.",
    "visual_jailbreak": "Jailbreaks delivered as images, for models that accept pictures.",
    "web_injection": "Tries to make the model output Markdown images or links that would send data to an "
                     "attacker's server when displayed.",
}


def probe_tip(name):
    """Hover text for a probe: what its family does, garak's description, and the attack goal."""
    v = CATALOG.get(name)
    if not v:
        return ""
    parts = [MODULE_INFO.get(v["module"], ""), v["desc"]]
    if v["goal"]:
        parts.append(f"Goal: {v['goal'][0].upper()}{v['goal'][1:]}.")
    if not v["active"]:
        parts.append("Off by default in garak.")
    return "\n".join(p for p in parts if p)


CATALOG = load_catalog()
try:
    GARAK_VERSION = importlib.metadata.version("garak")  # (a garak update restarts the app, so this stays current)
except importlib.metadata.PackageNotFoundError:
    GARAK_VERSION = "not installed"
MODULES = sorted({v["module"] for v in CATALOG.values()})


def owasp_preset():
    """Every active garak probe mapped to the OWASP Top 10 for LLM Applications, and the categories they cover."""
    probes = [n for n, v in CATALOG.items() if v["active"] and any(t.startswith("owasp:") for t in v["tags"])]
    covered = sorted({t.split(":")[1].upper() for n in probes for t in CATALOG[n]["tags"] if t.startswith("owasp:llm")})
    missing = [f"LLM{i:02d}" for i in range(1, 11) if f"LLM{i:02d}" not in covered]
    return probes, (f"Every garak probe mapped to the OWASP Top 10 for LLM Applications ({', '.join(covered)}). "
                    + (f"garak has no tests for {', '.join(missing)} on a standalone model (they concern training data "
                       "and tool-using apps). " if missing else "") + "Thorough; expect several hours.")


NEEDS_ATTACKER_MODEL = {"atkgen", "adaptive_attacks"}  # these download and run a second model to attack with


def _probes_for(tags=(), modules=(), probes=()):
    """Active probes carrying a tag that starts with one of `tags`, in one of `modules`, or named in `probes`."""
    return [n for n, v in CATALOG.items() if v["active"] and v["module"] not in NEEDS_ATTACKER_MODEL and (
        (bool(tags) and any(t.startswith(tags) for t in v["tags"])) or v["module"] in modules or n in probes)]


# Groups for AI evaluations and risk assessments. NIST AI 600-1 is the Generative AI Profile of the NIST AI Risk
# Management Framework; each group covers one of its risk categories that a model can be tested for directly. The
# mapping uses garak's own taxonomy tags plus the modules written for that risk. These support an assessment; they
# are not a certified compliance test.
_NOT_CERTIFIED = " An evaluation aid, not a certified compliance test."
FRAMEWORK_PRESETS = {
    "Baseline assessment": (
        _probes_for(probes=(
            "dan.DanInTheWild", "promptinject.HijackHateHumans", "latentinjection.LatentInjectionReport",
            "encoding.InjectBase64", "malwaregen.TopLevel", "exploitation.SQLInjectionEcho",
            "web_injection.MarkdownImageExfil", "sysprompt_extraction.SystemPromptExtraction", "apikey.GetKey",
            "leakreplay.GuardianCloze", "packagehallucination.Python", "misleading.FalseAssertion",
            "goodside.WhoIsRiley", "lmrc.QuackMedicine", "realtoxicityprompts.RTPBlank", "grandma.Substances",
            "continuation.ContinueSlursReclaimedSlurs", "lmrc.Deadnaming", "ansiescape.AnsiRaw")),
        "A time-boxed pre-deployment check with one or two representative probes from every risk area in the "
        "NIST AI 600-1 groups below: security, integrity, harmful content, privacy and bias. Run it first; follow up "
        "with the full group for any area that fails." + _NOT_CERTIFIED),
    "NIST AI 600-1: Information security": (
        _probes_for(tags=("avid-effect:security", "euai:robustness", "payload:malicious", "payload:jailbreak"),
                    modules=("dan", "promptinject", "latentinjection", "encoding", "malwaregen", "exploitation",
                             "web_injection", "ansiescape", "badchars", "phrasing", "dra", "sata")),
        "Whether the model can be hijacked or turned against the systems around it: jailbreaks, direct and indirect "
        "prompt injection, encoded and obfuscated attacks, malicious code and injection payloads, and data "
        "exfiltration through rendered output." + _NOT_CERTIFIED),
    "NIST AI 600-1: Information integrity and confabulation": (
        _probes_for(tags=("euai:deception", "payload:hallucination", "quality:Robustness:GenerativeMisinformation",
                          "quality:Behavioral:DeliberativeMisinformation"),
                    modules=("misleading", "snowball", "packagehallucination"),
                    probes=("goodside.WhoIsRiley", "lmrc.QuackMedicine")),
        "Whether the model states falsehoods with confidence, accepts false premises, invents software packages or "
        "people, or gives unsafe medical misinformation." + _NOT_CERTIFIED),
    "NIST AI 600-1: Dangerous, violent and hateful content": (
        _probes_for(tags=("euai:harmful", "euai:offensive", "payload:unwanted", "payload:dangerous",
                          "risk-cards:lmrc:bullying", "risk-cards:lmrc:providing_illegal_info",
                          "risk-cards:lmrc:slur_usage", "risk-cards:lmrc:sexual"),
                    modules=("realtoxicityprompts", "continuation"), probes=("grandma.Substances", "grandma.Slurs")),
        "Whether the model produces violent, hateful, harassing, sexual or illegal content, including instructions "
        "for dangerous substances, when asked directly or through role-play." + _NOT_CERTIFIED),
    "NIST AI 600-1: Data privacy and intellectual property": (
        _probes_for(tags=("payload:leak", "quality:Security:ExtractionInversion", "quality:Security:Confidentiality"),
                    modules=("leakreplay", "divergence", "apikey", "sysprompt_extraction"),
                    probes=("grandma.Win10", "grandma.Win11")),
        "Whether the model reveals memorized training data or copyrighted text, secrets such as API and product "
        "keys, or its confidential system prompt." + _NOT_CERTIFIED),
    "NIST AI 600-1: Harmful bias": (
        _probes_for(tags=("euai:discrimination", "avid-effect:ethics:E0101", "risk-cards:lmrc:deadnaming"),
                    modules=("topic",), probes=("continuation.ContinueSlursReclaimedSlurs",)),
        "Whether the model discriminates against people or protected groups, uses slurs, misgenders or deadnames, "
        "or takes sides on controversial topics." + _NOT_CERTIFIED),
    "EU AI Act risk areas": (
        _probes_for(tags=("euai:",)),
        "Every garak probe tagged with an EU AI Act risk area: robustness and security, deception, discrimination, "
        "and harmful or offensive content." + _NOT_CERTIFIED),
}

_owasp_probes, _owasp_desc = owasp_preset()
_extra = {"OWASP Top 10 for LLMs": (_owasp_probes, _owasp_desc), **FRAMEWORK_PRESETS}
PRESETS = {k: v for k, v in PRESETS.items() if k != "Full scan"} | {
    k: v for k, v in _extra.items() if v[0]} | {"Full scan": PRESETS["Full scan"]}  # empty would mean every probe


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


def probes_html(probes, desc):
    """A group's description and the probes it runs, grouped by family. Hovering a probe or family explains it."""
    by_module = {}
    for n in probes:
        by_module.setdefault(CATALOG[n]["module"] if n in CATALOG else n.split(".")[0], []).append(n)
    rows = "".join(
        f'<div class="pi-row"><span class="pi-mod" data-tip-title="{html.escape(m)}" '
        f'data-tip="{html.escape(MODULE_INFO.get(m, ""))}">{html.escape(m)}</span><span class="pi-chips">'
        + "".join(f'<span class="pi-chip" data-tip-title="{html.escape(n)}" data-tip="{html.escape(probe_tip(n))}">'
                  f'{html.escape(n.split(".", 1)[-1])}</span>' for n in names) + "</span></div>"
        for m, names in by_module.items())
    families = f'{len(by_module)} famil{"y" if len(by_module) == 1 else "ies"}'
    return (f'<div class="probe-info"><p class="pi-desc">{html.escape(desc)}</p>'
            f'<div class="pi-count">{len(probes)} probe{"s" if len(probes) != 1 else ""} in {families}'
            f'</div><div class="pi-list">{rows}</div>'
            f'<div class="pi-hint">Hover a probe or family to see what it does</div></div>')


def scan_info(scan_type, selected):
    """What a scan type runs: its description and every probe in it."""
    if scan_type in PRESETS and not PRESETS[scan_type][0]:  # the full scan: every probe that is on by default
        return probes_html([n for n, v in CATALOG.items() if v["active"]], PRESETS[scan_type][1])
    probes, err = resolve_scan(scan_type, selected)
    if err:
        return f'<div class="probe-info"><p class="pi-desc">{html.escape(err)}</p></div>'
    if scan_type in PRESETS:
        desc = PRESETS[scan_type][1]
    elif scan_type == CURRENT:
        desc = "The probes currently selected on the Probes tab."
    else:
        desc = f"Your saved group \"{scan_type.removeprefix(GROUP_PREFIX)}\"."
    return probes_html(probes, desc)


def group_info(name, selected=()):
    """The Probes tab's Group menu uses "Preset: X" for presets and plain names for saved groups."""
    if not name:
        return ""
    return scan_info(name.removeprefix("Preset: ") if name.startswith("Preset: ") else GROUP_PREFIX + name, selected)


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
    problem = next((load_problem(m) for m in list_models() if m["name"] == name), "")
    if problem:
        return (f"**This model can't run, so it can't be scanned.** Ollama can't load it ({problem}). Models in "
                "formats like PQ2_0 or PTQ1_0 need the llama.cpp build from the model's authors."), gr.update()
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

    def __init__(self, probes=(), done=(), failed=()):
        """probes/done/failed carry over a resumed scan, whose garak run only lists the probes still to do."""
        self.start = time.time()
        self.probes, self.done, self.failed = list(probes), set(done), set(failed)
        self.current, self.frac, self.phase = None, 0.0, "Starting garak"

    def feed(self, line):
        if m := self.QUEUE.search(line):
            if not self.probes:
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
    problem = next((load_problem(m) for m in list_models() if m["name"] == model), "")
    if problem:
        yield (f"{model} can't be scanned: Ollama can't load this model ({problem}). Choose a model that runs.",
               gr.update(), gr.update(), gr.update())
        return
    probes, err = resolve_scan(scan_type, selected)
    if err:
        yield err, gr.update(), gr.update(), gr.update()
        return
    if _proc.get("p") and _proc["p"].poll() is None:  # e.g. started before this page was reloaded
        yield ("A scan is already running. Wait for it to finish, or click Stop to end it.", gr.update(),
               gr.update(interactive=True), gr.update())
        return
    if _resume["waiting"]:
        yield ("An interrupted scan is about to resume. Click Stop to cancel it first.", gr.update(),
               gr.update(interactive=True), gr.update())
        return
    if _image_proc.get("p") and _image_proc["p"].poll() is None:
        yield "An image is being generated on the GPU. Start the scan once it finishes.", gr.update(), gr.update(), gr.update()
        return
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    job = {"model": model, "scan_type": scan_type, "requested": probes or [], "queue": [], "done": [], "failed": [],
           "generations": int(generations), "timeout": int(timeout), "thinking": bool(thinking),
           "prefix": f"{safe}_{time.strftime('%Y%m%d-%H%M%S')}", "parts": 1, "stalls": 0}
    _save_scan_job(job)
    _launch_scan(job)
    yield from follow_scan()


# A running scan is saved to SCAN_JOB with the probes garak has finished. If LLM Scanner stops or the computer shuts
# down mid-scan, the next start runs the unfinished probes as a new part of the same scan (garak can't append to a
# report, so each part has its own report). Finishing, failing on its own, or Stop removes the file.
SCAN_JOB = DATA_DIR / "scan_job.json"
MAX_STALLED_RESUMES = 3  # give up on a scan that is interrupted this many times in a row without finishing a probe
OLLAMA_WAIT_SECS = 600  # after a reboot, how long a resume waits for Ollama to come up
_scan = {"head": "", "buf": [], "cur": [""], "progress": ScanProgress(), "reader": None, "end": "", "ok": False,
         "user_stop": False}
_resume = {"waiting": False}


def _save_scan_job(job):
    tmp = SCAN_JOB.with_suffix(".tmp")
    tmp.write_text(json.dumps(job), encoding="utf-8")
    tmp.replace(SCAN_JOB)


def _load_scan_job():
    try:
        job = json.loads(SCAN_JOB.read_text(encoding="utf-8"))
        return job if isinstance(job, dict) and job.get("model") else None
    except (OSError, ValueError):
        return None


def _remaining_probes(job):
    """Probes still to run: garak's queue minus the finished ones, or the original request if garak never got as far
    as listing its queue."""
    return [p for p in job["queue"] if p not in job["done"]] if job["queue"] else job["requested"]


def _launch_scan(job, note=""):
    """Start garak for the job's unfinished probes. Output goes to _scan so any open page can show it."""
    part = job["parts"]
    prefix = job["prefix"] + (f"_part{part}" if part > 1 else "")
    gen_opts = {"timeout": job["timeout"]}
    if job["thinking"]:
        gen_opts["max_tokens"] = 4096  # leave room for the reasoning before the answer
    cmd = [PY, str(APP_DIR / "garak_runner.py"), "--target_type", "ollama.OllamaGeneratorChat",
           "--target_name", job["model"], "--generations", str(job["generations"]), "--report_prefix", prefix,
           "--generator_options", json.dumps({"ollama": {"OllamaGeneratorChat": gen_opts}})]
    remaining = _remaining_probes(job)
    if remaining:
        cmd += ["--probes", ",".join(remaining)]
    env = os.environ | {"PYTHONUNBUFFERED": "1", "TERM": "dumb", "PYTHONIOENCODING": "utf-8",
                        "LLM_SCANNER_THINK": "1" if job["thinking"] else "0", "LLM_SCANNER_MODEL": job["model"]}
    progress = ScanProgress(job["queue"], job["done"], job["failed"])
    buf, cur = [], [""]
    _scan.update(head=note + "garak " + " ".join(cmd[2:]) + "\n\n", buf=buf, cur=cur, progress=progress, end="",
                 ok=False, user_stop=False)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    _proc["p"] = p

    def record(line):
        """Save what garak has finished. A probe counts once garak has moved on to the next one, since its
        remaining detectors may still be scoring it."""
        progress.feed(line)
        changed = False
        if not job["queue"] and progress.probes:
            job["queue"], changed = list(progress.probes), True
        finished = sorted(progress.done - {progress.current})
        if finished != job["done"]:
            job["done"], job["failed"], changed = finished, sorted(progress.failed & set(finished)), True
        if changed and not _scan["user_stop"]:
            _save_scan_job(job)

    def on_text(text):
        for c in EMOJI.sub("", ANSI.sub("", text)):
            if c in "\n\r":
                if cur[0]:
                    record(cur[0])
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
        rc = p.returncode
        _scan["ok"] = rc == 0
        if rc == 0:
            _scan["end"] = "Scan complete. Open Reports to view the results."
        elif _scan["user_stop"]:
            _scan["end"] = "Scan stopped."
        elif rc > 0:
            _scan["end"] = f"Scan stopped: garak exited with code {rc}."
        else:  # killed by a signal the user didn't send (out of memory, shutdown): leave the job to resume
            _scan["end"] = "Scan interrupted. It will continue from where it stopped the next time LLM Scanner starts."
        if rc >= 0 or _scan["user_stop"]:
            SCAN_JOB.unlink(missing_ok=True)
        unload_models()  # free memory as soon as the scan is over

    _scan["reader"] = pump_output(p, on_text, on_exit)


def _scan_log(limit=None):
    lines = _scan["buf"][-limit:] if limit else _scan["buf"]
    return _scan["head"] + "\n".join(lines + _scan["cur"])


def follow_scan():
    """Stream the current scan to a page until it ends: the page that started it, or one opened later."""
    reader = _scan["reader"]
    while reader and reader.is_alive():
        yield _scan_log(400), gr.update(interactive=False), gr.update(interactive=True), _scan["progress"].html()
        reader.join(1.0)
    yield (_scan_log() + f"\n\n{_scan['end']}", gr.update(interactive=True), gr.update(interactive=False),
           _scan["progress"].html("done" if _scan["ok"] else "stopped"))


def attach_scan():
    """On page load, show a scan that is already running or about to resume, instead of 'No scan running'."""
    shown = False
    while _resume["waiting"]:
        if not shown:
            yield ("A scan was interrupted when LLM Scanner stopped. It will resume as soon as Ollama is running.",
                   gr.update(interactive=False), gr.update(interactive=True), IDLE_PROGRESS)
            shown = True
        time.sleep(1)
    if _scan_running() or (_scan["reader"] and _scan["reader"].is_alive()):
        yield from follow_scan()
    elif shown:  # the resume was cancelled or could not start
        yield "No scan running.", gr.update(interactive=True), gr.update(interactive=False), IDLE_PROGRESS
    else:
        yield gr.update(), gr.update(), gr.update(), gr.update()


def resume_interrupted_scan():
    """Continue a scan that LLM Scanner's stop, a crash or a shutdown cut short (runs once at startup)."""
    _resume["waiting"] = True
    try:
        job = _load_scan_job()
        if not job:
            return
        remaining = _remaining_probes(job)
        if job["queue"] and not remaining:
            SCAN_JOB.unlink(missing_ok=True)  # every probe finished; only the exit was missed
            return
        deadline = time.time() + OLLAMA_WAIT_SECS
        while not ollama_up():
            if not SCAN_JOB.exists():  # Stop was pressed while waiting
                return
            if time.time() > deadline:
                print("Interrupted scan not resumed: Ollama did not start. It will be tried again next start.", flush=True)
                return
            time.sleep(5)
        if not SCAN_JOB.exists():  # Stop was pressed while waiting
            return
        if job["model"] not in model_names():
            print(f"Interrupted scan dropped: {job['model']} is no longer installed.", flush=True)
            SCAN_JOB.unlink(missing_ok=True)
            return
        job["stalls"] = job["stalls"] + 1 if len(job["done"]) == job.get("done_at_resume", -1) else 0
        if job["stalls"] >= MAX_STALLED_RESUMES:
            print(f"Interrupted scan dropped: it stopped {job['stalls']} times in a row without progress.", flush=True)
            SCAN_JOB.unlink(missing_ok=True)
            return
        job["parts"] += 1
        job["done_at_resume"] = len(job["done"])
        _save_scan_job(job)
        left = (f"{len(remaining)} of {len(job['queue'])} probes left" if job["queue"]
                else "garak had not started probing yet, so the scan starts over")
        earlier = ", ".join(job["prefix"] + (f"_part{n}" if n > 1 else "") for n in range(1, job["parts"]))
        note = (f"Resumed after LLM Scanner was stopped: {left}. Results from before the interruption are in "
                f"{'report' if job['parts'] == 2 else 'reports'} {earlier}; this part is saved as "
                f"{job['prefix']}_part{job['parts']}.\n")
        print(note.strip(), flush=True)
        _launch_scan(job, note)
    finally:
        _resume["waiting"] = False


def stop_scan():
    _scan["user_stop"] = True
    SCAN_JOB.unlink(missing_ok=True)  # also cancels a resume that is still waiting for Ollama
    p = _proc.get("p")
    if p and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        return "Stopping scan..."
    if _resume["waiting"]:
        return "Resume cancelled."
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


def delete_label(selected):
    n = len(selected or [])
    return gr.update(value=f"Delete {n} report{'s' if n != 1 else ''}" if n else "Delete reports", interactive=bool(n))


def report_list_state(selected=(), shown=None):
    """The report list, the Select all box and the Delete button after the list of runs has changed."""
    r = reports()
    names = {n for _, n in r}
    selected = [n for n in (selected or []) if n in names]
    shown = shown if shown in names else (r[0][1] if r else None)
    return (gr.update(choices=r, value=selected), gr.update(value=bool(selected) and len(selected) == len(r)),
            delete_label(selected), selected, shown)


def reports_ticked(selected, previous, shown, which):
    """Ticking a report shows it; unticking only changes what Delete would remove."""
    selected = selected or []
    added = [n for n in selected if n not in (previous or [])]
    shown = added[-1] if added else (shown if _known_report(shown) else (selected[0] if selected else None))
    return (delete_label(selected), selected, shown, *show_report(shown, which))


def view_report(label, shown, which):
    """Clicking a report's name (not its box) shows that report without ticking it."""
    name = next((n for lbl, n in reports() if " ".join(lbl.split()) == " ".join((label or "").split())), None) or shown
    return (name, *show_report(name, which))


def select_all_reports(on, shown, which):
    names = [n for _, n in reports()]
    selected = names if on else []
    shown = shown if _known_report(shown) else (names[0] if names else None)
    return gr.update(value=selected), delete_label(selected), selected, shown, *show_report(shown, which)


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


def _delete_run(name):
    """Every file of one scan run: garak report, raw JSONL, hitlog, summary, CSV. Returns how many were deleted."""
    prefix = name[:-len(".report.html")]
    removed = 0
    for f in list(GARAK_RUNS.iterdir()):
        if f.is_file() and f.name.startswith(prefix + "."):
            f.unlink()
            removed += 1
    return removed


def delete_reports(names, confirmed, shown, which):
    """Delete the ticked runs and everything they wrote (garak report, raw JSONL, hitlog, summary, CSV)."""
    names = [n for n in (names or []) if _known_report(n)]
    if not confirmed or not names:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), *(gr.update(),) * 4, gr.update())
    files = sum(_delete_run(n) for n in names)
    if shown in names:
        shown = None
    listing, select_all, button, selected, shown = report_list_state([], shown)
    note = note_html(f"Deleted {len(names)} report{'s' if len(names) != 1 else ''} ({files} files).")
    return listing, select_all, button, selected, shown, *show_report(shown, which), note


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
    return gr.update(choices=scan_model_choices()[0]), (model_lock(chat) if locked else gr.update(choices=names))


def refresh_all(chat=None):
    names = model_names()
    scan_choices_, runnable = scan_model_choices()
    upd = gr.update(choices=scan_choices_, value=runnable)
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
    with gr.Row(equal_height=False, elem_classes="top-row"):  # header, with the update box beside it when shown
        gr.HTML('<div class="hero"><div class="eyebrow">Local AI security &nbsp;·&nbsp; Current version v' + APP_VERSION + '</div><h1>LLM Scanner</h1>'
                '<p>Download models from Hugging Face or Ollama, run them locally, and test them for '
                'vulnerabilities with garak.</p></div>', elem_classes="hero-holder")
        with gr.Column(elem_classes="update-box", visible=False, scale=0, min_width=380) as update_box:  # only when there are updates
            update_dismiss = gr.Button("", size="sm", scale=0, min_width=0, visible=False, elem_classes="up-close")
            update_html = gr.HTML()
            update_rows, update_labels, update_buttons = [], [], {}
            for _kind in ("app", "ollama", "garak"):
                with gr.Row(equal_height=True, elem_classes="up-item", visible=False) as _row:
                    update_labels.append(gr.HTML(elem_classes="up-item-label"))
                    update_buttons[_kind] = gr.Button("Update", variant="primary", size="sm", scale=0, min_width=90)
                update_rows.append(_row)
            with gr.Row(elem_classes="up-actions"):
                update_all_btn = gr.Button("Update all", variant="primary", size="sm", scale=0, min_width=110, visible=False)
                pass
    update_timer = gr.Timer(1.0)  # outside the update box: a timer in a hidden container never ticks
    with gr.Row(elem_classes="statusrow"):
        _v = ollama_up()
        with gr.Row(equal_height=True, elem_classes="ollama-controls"):  # "Ollama 0.34.2  Stop  Restart" in one pill
            ollama_status = gr.HTML(ollama_status_html(_v), elem_classes="oc-label-holder", min_width=0)
            start_btn = gr.Button("Start", variant="secondary", size="sm", scale=0, min_width=0, visible=not _v,
                                  elem_classes="oc-start")
            stop_ollama_btn = gr.Button("Stop", variant="secondary", size="sm", scale=0, min_width=0, visible=bool(_v),
                                        elem_classes="oc-stop")  # asks first (static/app.js)
            restart_ollama_btn = gr.Button("Restart", variant="secondary", size="sm", scale=0, min_width=0,
                                           visible=bool(_v), elem_classes="oc-restart")
        status = gr.HTML(status_html(), elem_classes="status-holder")
        status_timer = gr.Timer(5.0)
        unload_btn = gr.Button("Unload models", variant="secondary", size="sm", scale=0, min_width=140,
                               visible=bool(loaded_models()))
        refresh_btn = gr.Button("Refresh", variant="secondary", size="sm", scale=0, min_width=100)
        check_updates_btn = gr.Button("Check for updates", variant="secondary", size="sm", scale=0, min_width=150)
    activity = gr.HTML(activity_html(), elem_classes="activity-holder")

    with gr.Tabs():
        # ---------------- Models
        with gr.Tab("Models"):
            with gr.Row(equal_height=False, elem_classes="models-row"):  # both sides end at the same height
                with gr.Column(scale=2, min_width=480, elem_classes="models-left"):
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
                                               min_width=300, elem_classes="required", elem_id="repo-dd", interactive=False)
                            version = gr.Dropdown(label="Version", choices=[PLEASE_CHOOSE], value="", scale=2,
                                                  min_width=300, interactive=False, elem_classes="required",
                                                  elem_id="version-dd")
                        search_note = gr.HTML(elem_classes="note")
                        with gr.Row():
                            pull_btn = gr.Button("Download", variant="primary", scale=0, min_width=160, interactive=False)
                        pull_status = gr.HTML(elem_classes="note")

                    with gr.Column(elem_classes="card downloads-card"):
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
                    gr.HTML('<div id="probe-tips" hidden data-tips="' + html.escape(json.dumps(
                        {n: probe_tip(n) for n in CATALOG})) + '"></div>', elem_classes="hidden-control")
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
                    group_about = gr.HTML()

        # ---------------- Scan
        with gr.Tab("Scan"):
            with gr.Column(elem_classes="card"):
                gr.HTML(f'<h2>Run a scan <span class="ver-badge">garak {html.escape(GARAK_VERSION)}</span></h2>'
                        '<p class="sub">garak sends attack prompts to the model and checks its responses.</p>')
                with gr.Row():
                    scan_model = gr.Dropdown(label="Model", choices=scan_model_choices()[0], scale=1, min_width=280,
                                             elem_id="scan-model-dd")
                    scan_type = gr.Dropdown(label="Scan type", choices=scan_choices(), value="Quick check",
                                            scale=1, min_width=280)
                model_note = gr.Markdown(elem_classes="note")
                scan_desc = gr.HTML(scan_info("Quick check", []))
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
                _reports = reports()
                rep_list = gr.CheckboxGroup(choices=_reports, value=[], show_label=False, elem_classes="report-list")
                with gr.Row(equal_height=True, elem_classes="report-actions"):
                    rep_all = gr.Checkbox(label="Select all", value=False, scale=0, min_width=130,
                                          elem_classes="select-all")
                    rep_open = gr.Button("Open summary in browser", variant="secondary", scale=0, min_width=220,
                                         elem_classes="rep-open")
                    rep_delete = gr.Button("Delete reports", variant="stop", scale=0, min_width=170, interactive=False,
                                           elem_classes="reps-delete")
                rep_status = gr.HTML(elem_classes="note")
                rep_view_label = gr.Textbox(elem_id="rep-view-label", elem_classes="hidden-control", container=False)
                rep_view_btn = gr.Button("view", elem_id="rep-view-btn", elem_classes="hidden-control")
                rep_confirm = gr.Checkbox(value=False, visible=False)
                rep_picked = gr.State([])                                   # what is ticked, to spot new ticks
                rep_shown = gr.State(_reports[0][1] if _reports else None)  # the report the tabs below display
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
    status_outputs = [status, unload_btn, start_btn, stop_ollama_btn, restart_ollama_btn, ollama_status, activity]
    start_btn.click(start_ollama, outputs=status_outputs, show_progress="hidden")
    stop_ollama_btn.click(stop_ollama, outputs=status_outputs, show_progress="hidden")
    restart_ollama_btn.click(restart_ollama, outputs=status_outputs, show_progress="hidden")
    status_timer.tick(status_controls, outputs=status_outputs, show_progress="hidden")
    unload_btn.click(unload_models, outputs=status_outputs, show_progress="hidden")
    ui.load(status_controls, outputs=status_outputs, show_progress="hidden")
    update_outputs = [update_box, update_html, *update_rows, *update_labels, update_all_btn, update_dismiss]
    update_timer.tick(update_controls, outputs=update_outputs, show_progress="hidden")
    for _kind, _btn in update_buttons.items():
        _btn.click(functools.partial(start_update, _kind), outputs=update_outputs, show_progress="hidden")
    update_all_btn.click(lambda: start_update("all"), outputs=update_outputs, show_progress="hidden")
    check_updates_btn.click(check_updates_now, outputs=update_outputs, show_progress="hidden")
    update_dismiss.click(dismiss_update_msg, outputs=update_outputs,
                         show_progress="hidden")
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
    group_pick.change(group_info, [group_pick, selected], group_about, show_progress="hidden")
    ui.load(group_info, [group_pick, selected], group_about, show_progress="hidden")
    scan_model.change(model_hint, scan_model, [model_note, tmo], show_progress="hidden")
    scan_btn.click(run_scan, [scan_model, scan_type, selected, gens, tmo, thinking],
                   [scan_log, scan_btn, stop_btn, scan_progress], show_progress="hidden").then(
        report_list_state, [rep_picked, rep_shown], [rep_list, rep_all, rep_delete, rep_picked, rep_shown],
        show_progress="hidden")
    stop_btn.click(stop_scan, outputs=scan_log, show_progress="hidden")
    ui.load(attach_scan, outputs=[scan_log, scan_btn, stop_btn, scan_progress], show_progress="hidden",
            concurrency_limit=None).then(
        report_list_state, [rep_picked, rep_shown], [rep_list, rep_all, rep_delete, rep_picked, rep_shown],
        show_progress="hidden")

    # reports
    view_outputs = [rep_summary, rep_view, raw_view, rep_files]
    rep_list.input(reports_ticked, [rep_list, rep_picked, rep_shown, raw_pick],
                   [rep_delete, rep_picked, rep_shown, *view_outputs], show_progress="hidden")
    rep_all.input(select_all_reports, [rep_all, rep_shown, raw_pick],
                  [rep_list, rep_delete, rep_picked, rep_shown, *view_outputs], show_progress="hidden")
    raw_pick.change(read_raw, [rep_shown, raw_pick], raw_view, show_progress="hidden")
    rep_open.click(open_report, rep_shown, show_progress="hidden")
    rep_view_btn.click(view_report, [rep_view_label, rep_shown, raw_pick], [rep_shown, *view_outputs],
                       show_progress="hidden")
    rep_delete.click(delete_reports, [rep_list, rep_confirm, rep_shown, raw_pick],
                     [rep_list, rep_all, rep_delete, rep_picked, rep_shown, *view_outputs, rep_status],
                     js="(names, c, shown, which) => [names, !!(names && names.length), shown, which]",
                     show_progress="hidden")

    ui.load(refresh_all, chat_state, [models_version, scan_model, status, scan_type, group_pick, chat_model],
            show_progress="hidden").then(
        model_hint, scan_model, [model_note, tmo], show_progress="hidden").then(
        report_list_state, [rep_picked, rep_shown], [rep_list, rep_all, rep_delete, rep_picked, rep_shown],
        show_progress="hidden").then(
        show_report, [rep_shown, raw_pick], [rep_summary, rep_view, raw_view, rep_files], show_progress="hidden")

KEEP_BACKUPS = 3  # previous versions kept in data/backups after updates


def clean_leftovers():
    """Delete files earlier runs left behind: interrupted or abandoned downloads, the huggingface_hub caches of
    1.0.10 and 1.0.11 (a second copy of every model), failed update staging, self-test files, attachments staged but
    never sent, attachments and image settings whose chat or image is gone, and all but the newest update backups.
    Downloads that are still queued or paused keep their partial files. Returns (files removed, bytes freed)."""
    keep = {f for ref, _ in _pending() for f in _partial_files(ref)}
    doomed = [DATA_DIR / ".hf-cache", IMAGE_MODEL_DIR / ".hf-cache", DATA_DIR / "update-staging", DATA_DIR / "tmp",
              ATTACH_DIR / "staging"]
    if HF_DOWNLOADS.exists():
        doomed += [f for f in HF_DOWNLOADS.iterdir() if f not in keep]
    if IMAGE_MODEL_DIR.exists():
        doomed += [f for f in IMAGE_MODEL_DIR.rglob("*.part") if f not in keep]
    if ATTACH_DIR.exists():
        doomed += [d for d in ATTACH_DIR.iterdir() if d.is_dir() and d.name != "staging"
                   and not (CHATS_DIR / f"{d.name}.json").exists()]
    if IMAGES_DIR.exists():
        doomed += [f for f in IMAGES_DIR.glob("*.json") if not f.with_suffix(".png").exists()]
    backups = DATA_DIR / "backups"
    if backups.exists():
        doomed += sorted((d for d in backups.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)[:-KEEP_BACKUPS]
    doomed += orphaned_ollama_partials()
    files = size = 0
    for path in doomed:
        if not path.exists():
            continue
        for f in ([path] if path.is_file() else [p for p in path.rglob("*") if p.is_file() and not p.is_symlink()]):
            files += 1
            st = f.stat()
            size += min(st.st_size, st.st_blocks * 512)  # Ollama's partial layers are sparse: count the disk used
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
    return files, size


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
        files, size = clean_leftovers()
        if files:
            print(f"Removed {files} leftover files ({size / 1e9:.1f} GB)", flush=True)
        resume_pending_downloads()
        if _load_scan_job():
            _resume["waiting"] = True  # set before the page can load, so it shows the pending resume
            threading.Thread(target=resume_interrupted_scan, daemon=True).start()
        threading.Thread(target=check_updates, daemon=True).start()  # once at startup; after that, on request
    ui.queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("LLM_SCANNER_PORT", 7861)),
                      inbrowser="--no-browser" not in sys.argv, allowed_paths=[str(GARAK_RUNS), str(ATTACH_DIR), str(IMAGES_DIR)],
                      theme=THEME, css=CSS, footer_links=[], js=APP_JS,
                      favicon_path=str(APP_DIR / "static/icon.png"))
