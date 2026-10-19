"""Visual checks in a real browser: every page, at several window sizes, with the computer set to dark or light mode,
has every button and every piece of text visible, unclipped, inside the window, not covered by something else, and
readable against what is behind it (WCAG contrast: 4.5:1 for normal text, 3:1 for large text and placeholders).

The real app runs on a spare port against the fake Ollama of test_ui.py, with a scratch data folder holding a saved
chat, a generated image, a probe group and two scan reports, so the pages show content rather than empty states.
Needs Playwright with Chromium (`.venv/bin/python -m pip install playwright && .venv/bin/python -m playwright install
chromium`); without it these tests are skipped. Screenshots of every page go to $LLM_SCANNER_SHOTS when it is set.

Run from the app folder:  .venv/bin/python -m unittest discover -s tests -p test_visual.py -v"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR / "tests"))
from test_ui import PY, FAKE_ENGINE, FakeOllama, free_port, model_entry, write_run  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # optional: only needed for these checks
    sync_playwright = None

TABS = ["Models", "Images", "Probes", "Scan", "Reports", "Red team tools"]
VIEWPORTS = {"desktop": (1920, 1080), "laptop": (1366, 768), "small": (1024, 768)}
SCHEMES = ["dark", "light"]
SHOTS = os.environ.get("LLM_SCANNER_SHOTS")
PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                    "1f15c4890000000d4944415478da6364f8ffbf1e000502027fa1d5b40000000049454e44ae426082")

# Runs in the page. Returns the problems found among visible buttons (kind "button") or visible text ("text") inside
# the given root, each as a short description. Elements the app hides on purpose (.hidden-control) are skipped.
AUDIT_JS = r"""
([kind, rootSel]) => {
  const root = (rootSel && document.querySelector(rootSel)) || document.body;
  const problems = [];
  const parse = (c) => {
    const m = c && c.match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const p = m[1].split(/[\s,\/]+/).filter(Boolean).map(Number);
    return {r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1};
  };
  const lum = (c) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b);
  };
  const over = (top, under) => ({r: top.r * top.a + under.r * (1 - top.a), g: top.g * top.a + under.g * (1 - top.a),
                                 b: top.b * top.a + under.b * (1 - top.a), a: 1});
  const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
  const shown = (el) => {
    if (!el.getClientRects().length) return false;
    for (let p = el; p && p.nodeType === 1; p = p.parentElement) {
      const s = getComputedStyle(p);
      if (s.display === 'none' || s.visibility === 'hidden' || parseFloat(s.opacity) === 0) return false;
      if (p.classList.contains('hidden-control') || p.hidden || p.getAttribute('aria-hidden') === 'true') return false;
      // screen-reader-only text (Gradio's hidden labels, its tab-width measuring copy): not meant to be seen
      if (/(^|\s)(sr-only|visually-hidden|hide)(\s|$)/.test(p.className.toString())) return false;
      if (s.clip.startsWith('rect(0') || s.clipPath === 'inset(50%)') return false;
    }
    const r = el.getBoundingClientRect();
    return r.width > 1 && r.height > 1;
  };
  const background = (el) => {   // the colour behind el: its own and its ancestors' backgrounds, composited
    const layers = [];
    for (let p = el; p && p.nodeType === 1; p = p.parentElement) {
      const c = parse(getComputedStyle(p).backgroundColor);
      if (c && c.a > 0) { layers.push(c); if (c.a >= 1) break; }
      // the selected tab / switch option is transparent over a sliding highlight (.glider, a sibling placed by
      // static/app.js), so the highlight is what sits behind its text
      const g = p.parentElement && p.parentElement.querySelector(':scope > .glider');
      if (g && g !== p && parseFloat(getComputedStyle(g).opacity) > 0) {
        const gr = g.getBoundingClientRect(), pr = p.getBoundingClientRect();
        const x = pr.left + pr.width / 2, y = pr.top + pr.height / 2;
        const gc = parse(getComputedStyle(g).backgroundColor);
        if (gc && gc.a > 0 && x >= gr.left && x <= gr.right && y >= gr.top && y <= gr.bottom) {
          layers.push(gc); if (gc.a >= 1) break;
        }
      }
    }
    let col = {r: 255, g: 255, b: 255, a: 1};
    for (let i = layers.length - 1; i >= 0; i--) col = over(layers[i], col);
    return col;
  };
  const name = (el) => {
    const t = (el.innerText || el.getAttribute('aria-label') || el.title || el.className || el.tagName).toString();
    return el.tagName.toLowerCase() + ' "' + t.trim().replace(/\s+/g, ' ').slice(0, 50) + '"';
  };
  const disabled = (el) => !!el.closest('[disabled], [aria-disabled="true"], .disabled');
  const clipped = (el) => {      // text cut off by el or the box around it (an ellipsis is a deliberate cut)
    for (let p = el, i = 0; p && i < 3; p = p.parentElement, i++) {
      const s = getComputedStyle(p);
      if (s.textOverflow === 'ellipsis' && p.title) continue;
      if ((s.overflowX === 'hidden' || s.overflowX === 'clip') && p.scrollWidth > p.clientWidth + 2) return 'width';
      if ((s.overflowY === 'hidden' || s.overflowY === 'clip') && p.scrollHeight > p.clientHeight + 4
          && p.clientHeight > 0 && p.clientHeight < 40) return 'height';
    }
    return null;
  };
  const inWindow = (r) => r.left >= -1 && r.right <= innerWidth + 1;
  const covered = (el) => {
    el.scrollIntoView({block: 'center', inline: 'nearest'});
    const r = el.getBoundingClientRect();
    const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return hit && !(el === hit || el.contains(hit) || hit.contains(el)) ? hit : null;
  };
  if (document.documentElement.scrollWidth > innerWidth + 1)
    problems.push(`page scrolls sideways: ${document.documentElement.scrollWidth}px wide in a ${innerWidth}px window`);
  if (kind === 'button') {
    const els = root.querySelectorAll('button, [role="button"], input[type="button"], input[type="submit"]');
    for (const el of els) {
      if (!shown(el) || el.closest('.tip-box')) continue;
      const r = el.getBoundingClientRect();
      const text = (el.innerText || '').trim();
      if (r.width < 16 || r.height < 16) problems.push(`${name(el)} is too small to use (${Math.round(r.width)}x${Math.round(r.height)})`);
      const hit = covered(el);
      if (hit) problems.push(`${name(el)} is covered by ${name(hit)}`);
      const r2 = el.getBoundingClientRect();
      if (!inWindow(r2)) problems.push(`${name(el)} sticks out of the window (${Math.round(r2.left)}..${Math.round(r2.right)})`);
      if (text) {
        const s = getComputedStyle(el);
        const bg = background(el), fg = parse(s.color);
        const need = disabled(el) ? 2 : 4.5;
        const cr = fg ? ratio(over(fg, bg), bg) : 0;
        if (cr < need) problems.push(`${name(el)} text contrast ${cr.toFixed(2)} < ${need} (${s.color} on rgb(${Math.round(bg.r)},${Math.round(bg.g)},${Math.round(bg.b)}))`);
        const c = clipped(el);
        if (c) problems.push(`${name(el)} label is cut off (${c})`);
      } else {
        const label = el.getAttribute('aria-label') || el.title;
        const s = getComputedStyle(el), b = getComputedStyle(el, '::before'), a = getComputedStyle(el, '::after');
        const icon = el.querySelector('svg, img') || s.backgroundImage !== 'none' || s.maskImage !== 'none'
          || s.webkitMaskImage !== 'none'
          || [b, a].some((p) => p.content !== 'none' && p.content !== 'normal'
                              && (p.backgroundImage !== 'none' || p.maskImage !== 'none' || p.webkitMaskImage !== 'none'
                                  || p.content.replace(/"/g, '').trim() || parse(p.backgroundColor)?.a > 0));
        if (!icon) problems.push(`${name(el)} has no text and no visible icon`);
        if (!label && !el.closest('label') && !el.id && !(el.className || '').toString().trim())
          problems.push(`${name(el)} has no text, label or title`);
      }
    }
  } else {
    const seen = new Set();
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const el = n.parentElement;
      if (!el || seen.has(el) || !n.textContent.trim()) continue;
      seen.add(el);
      if (['SCRIPT', 'STYLE', 'NOSCRIPT', 'OPTION', 'TEMPLATE'].includes(el.tagName) || !shown(el)) continue;
      if (el.closest('.tip-box, button, [role="button"], iframe, svg')) continue;  // buttons have their own check
      const s = getComputedStyle(el);
      const size = parseFloat(s.fontSize), weight = parseInt(s.fontWeight) || 400;
      const large = size >= 24 || (size >= 18.66 && weight >= 700);
      const bg = background(el), fg = parse(s.color);
      const need = disabled(el) ? 2 : large ? 3 : 4.5;
      const cr = fg ? ratio(over(fg, bg), bg) : 0;
      const what = `"${n.textContent.trim().replace(/\s+/g, ' ').slice(0, 50)}" (${el.tagName.toLowerCase()}.${(el.className || '').toString().split(' ')[0]})`;
      if (cr < need) problems.push(`${what} contrast ${cr.toFixed(2)} < ${need} (${s.color} on rgb(${Math.round(bg.r)},${Math.round(bg.g)},${Math.round(bg.b)}))`);
      if (size < 10) problems.push(`${what} font too small (${size}px)`);
      const c = clipped(el);
      if (c) problems.push(`${what} is cut off (${c})`);
      const r = el.getBoundingClientRect();
      const scroller = (() => { for (let p = el.parentElement; p && p !== document.body; p = p.parentElement)
        if (getComputedStyle(p).overflowX !== 'visible') return true; return false; })();
      if (r.right > innerWidth + 1 && !scroller)
        problems.push(`${what} runs past the window edge (${Math.round(r.right)} > ${innerWidth})`);
    }
    for (const el of root.querySelectorAll('input[placeholder], textarea[placeholder]')) {
      if (!shown(el) || el.value || !el.placeholder) continue;
      const bg = background(el), fg = parse(getComputedStyle(el, '::placeholder').color);
      const cr = fg ? ratio(over(fg, bg), bg) : 0;
      if (cr < 3) problems.push(`placeholder "${el.placeholder.slice(0, 40)}" contrast ${cr.toFixed(2)} < 3`);
      const s = getComputedStyle(el), fg2 = parse(s.color), cr2 = fg2 ? ratio(over(fg2, bg), bg) : 0;
      if (cr2 < 4.5) problems.push(`typed text in "${el.placeholder.slice(0, 40)}" would have contrast ${cr2.toFixed(2)} < 4.5`);
    }
  }
  return [...new Set(problems)];
}
"""


def seed_data(data, runs, sd):
    """What a used install has: two reports, a chat, an image, a probe group and an installed image model."""
    write_run(runs, "hf.co_x_tiny_20260917-100000")
    write_run(runs, "tiny_20260917-110000", complete=False)
    (sd / "sd-cli").write_text(FAKE_ENGINE)
    (sd / "sd-cli").chmod(0o755)
    for rel in ("image-models/z-image-turbo/z_image_turbo-Q4_K.gguf",
                "image-models/shared/unsloth-qwen3-4b-instruct-2507-gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                "image-models/shared/comfy-org-z-image-turbo/ae.safetensors"):
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_bytes(b"x")
    (data / "images").mkdir()
    (data / "images" / "20260917-100000.png").write_bytes(PNG)
    (data / "images" / "20260917-100000.json").write_text(json.dumps({
        "prompt": "a lighthouse on a cliff at sunset", "model": "z-image-turbo", "seed": 7, "steps": 4,
        "size": "Small 512 x 512 (fastest)"}))
    (data / "probe_groups.json").write_text(json.dumps({"My jailbreak suite": ["dan.DanInTheWild",
                                                                              "goodside.WhoIsRiley"]}))


class AppInBrowser(unittest.TestCase):
    """Starts the app once for every visual test, and one Chromium shared by them."""

    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("Playwright is not installed")
        cls.tmp = Path(tempfile.mkdtemp(prefix="llm-scanner-visual-"))
        cls.data, cls.runs, sd = cls.tmp / "data", cls.tmp / "runs", cls.tmp / "sd"
        for d in (cls.data, cls.runs, sd):
            d.mkdir()
        seed_data(cls.data, cls.runs, sd)
        FakeOllama.models[:] = [model_entry("tiny:latest"), model_entry("hf.co/x/tiny-GGUF:Q4_K_M")]
        cls.ollama = ThreadingHTTPServer(("127.0.0.1", free_port()), FakeOllama)
        threading.Thread(target=cls.ollama.serve_forever, daemon=True).start()
        fake = f"http://127.0.0.1:{cls.ollama.server_port}"
        port = free_port()
        cls.url = f"http://127.0.0.1:{port}/"
        env = os.environ | {"LLM_SCANNER_CHECK": "1", "LLM_SCANNER_PORT": str(port), "LLM_SCANNER_DATA": str(cls.data),
                            "LLM_SCANNER_GARAK_RUNS": str(cls.runs), "LLM_SCANNER_SD_DIR": str(sd),
                            "OLLAMA_URL": fake, "LLM_SCANNER_HF_API": fake + "/api", "LLM_SCANNER_OLLAMA_WEB": fake,
                            "HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": "", "HUGGINGFACE_TOKEN": ""}
        cls.log = open(cls.tmp / "app.log", "wb")
        cls.proc = subprocess.Popen([PY, "app.py", "--no-browser"], cwd=APP_DIR, env=env, stdout=cls.log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        cls.pw = cls.browser = None
        try:
            import urllib.request
            deadline = time.time() + 120
            while True:
                try:
                    urllib.request.urlopen(cls.url, timeout=2).read()
                    break
                except Exception:
                    if time.time() > deadline or cls.proc.poll() is not None:
                        raise RuntimeError("the app did not start: " + (cls.tmp / "app.log").read_text()[-1500:])
                    time.sleep(0.5)
            cls.pw = sync_playwright().start()
            cls.browser = cls.pw.chromium.launch()
            from gradio_client import Client
            client = Client(cls.url, verbose=False)  # a saved chat, made the way the page makes one
            client.predict(api_name="/start_new_chat")
            client.predict("Explain prompt injection in one paragraph, with an example of a hidden instruction",
                           "tiny:latest", api_name="/chat_submit")
            client.predict(api_name="/chat_respond")
        except Exception:
            cls.tearDownClass()
            raise
        cls.pages = {}

    @classmethod
    def tearDownClass(cls):
        for page in getattr(cls, "pages", {}).values():
            page.context.close()
        if cls.browser:
            cls.browser.close()
        if cls.pw:
            cls.pw.stop()
        if cls.proc.poll() is None:
            os.killpg(cls.proc.pid, 15)
            cls.proc.wait(timeout=20)
        cls.ollama.shutdown()
        cls.log.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def page(self, viewport, scheme):
        """A loaded page for this window size and colour scheme, opened once and reused."""
        key = (viewport, scheme)
        if key not in self.pages:
            w, h = VIEWPORTS[viewport]
            ctx = self.browser.new_context(viewport={"width": w, "height": h}, color_scheme=scheme)
            page = ctx.new_page()
            page.goto(self.url)
            page.wait_for_selector("button[role=tab]", timeout=60000)
            page.wait_for_function("document.querySelectorAll('.mt-row').length > 1", timeout=30000)
            page.wait_for_timeout(1500)  # the page's load events (status, reports, tags) finish
            self.pages[key] = page
        return self.pages[key]

    def open_tab(self, page, tab):
        page.locator("button[role=tab]", has_text=tab).first.click()
        page.wait_for_timeout(500)
        page.evaluate("window.scrollTo(0, 0)")

    def shot(self, page, name):
        if SHOTS:
            Path(SHOTS).mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(Path(SHOTS) / f"{name}.png"), full_page=True)

    def audit(self, page, kind, root=None):
        return page.evaluate(AUDIT_JS, [kind, root])

    def assertReadable(self, page, kind, root=None):
        problems = self.audit(page, kind, root)
        self.assertEqual(problems, [], "\n" + "\n".join(problems))


class EveryPage(AppInBrowser):
    """Every tab x window size x colour scheme: one test for its buttons, one for its text."""


def _page_test(tab, viewport, scheme, kind):
    def test(self):
        page = self.page(viewport, scheme)
        self.open_tab(page, tab)
        if kind == "button":
            self.shot(page, f"{tab.replace(' ', '-').lower()}-{viewport}-{scheme}")
        self.assertReadable(page, kind)
    test.__doc__ = f"{tab} tab, {viewport} window {VIEWPORTS[viewport]}, {scheme} mode: every {kind} is readable"
    return test


for _tab in TABS:
    for _vp in VIEWPORTS:
        for _scheme in SCHEMES:
            for _kind in ("button", "text"):
                _name = f"test_{_tab.replace(' ', '_').lower()}_{_vp}_{_scheme}_{_kind}s_readable"
                setattr(EveryPage, _name, _page_test(_tab, _vp, _scheme, _kind))


class Interactions(AppInBrowser):
    """States that only appear after doing something: panels, menus, hovers, replies and confirmations."""

    def fresh(self, scheme="dark", viewport="laptop"):
        w, h = VIEWPORTS[viewport]
        ctx = self.browser.new_context(viewport={"width": w, "height": h}, color_scheme=scheme)
        self.addCleanup(ctx.close)
        page = ctx.new_page()
        page.goto(self.url)
        page.wait_for_selector("button[role=tab]", timeout=60000)
        page.wait_for_function("document.querySelectorAll('.mt-row').length > 1", timeout=30000)
        page.wait_for_timeout(1500)
        return page

    def test_hugging_face_token_panel_is_readable(self):
        page = self.fresh()
        page.locator(".card-gear").first.click()
        page.wait_for_selector(".hf-token-panel", state="visible")
        self.shot(page, "hf-token-panel")
        self.assertReadable(page, "button", ".hf-token-panel")
        self.assertReadable(page, "text", ".hf-token-panel")

    def test_probe_time_tag_hover_is_readable(self):
        page = self.fresh()
        self.open_tab(page, "Probes")
        page.wait_for_selector(".probe-list .probe-cost")
        tag = page.locator(".probe-list .probe-cost").first
        tag.scroll_into_view_if_needed()
        page.wait_for_timeout(400)  # the tip closes on any scroll, so hover only once scrolling has settled
        tag.hover()
        page.wait_for_selector(".tip-box.show", timeout=5000)
        self.shot(page, "probe-tag-tip")
        self.assertReadable(page, "text", ".tip-box")
        box = page.locator(".tip-box").bounding_box()
        self.assertTrue(box["x"] >= 0 and box["x"] + box["width"] <= VIEWPORTS["laptop"][0], box)
        self.assertGreater(len(page.locator(".tip-box .tip-body").inner_text()), 20)

    def test_every_time_tag_level_is_readable(self):
        page = self.fresh()
        self.open_tab(page, "Probes")
        page.locator("label", has_text="Include probes that are off by default").first.click()
        page.wait_for_timeout(1000)
        levels = page.evaluate("[...new Set([...document.querySelectorAll('.probe-cost')].map(t => t.className))]")
        self.assertGreaterEqual(len(levels), 4, levels)
        self.assertReadable(page, "text", ".probe-list")

    def test_chat_reply_and_composer_are_readable(self):
        page = self.fresh()
        page.locator(".chat-item", has_text="Explain prompt injection").first.click()
        page.wait_for_function("document.querySelectorAll('.chat .message').length >= 2", timeout=10000)
        page.locator("#chat-new").click()
        page.locator("#chat-input textarea, #chat-input input").first.fill("Say hello")
        page.locator("#chat-send").click()
        page.wait_for_function("[...document.querySelectorAll('.chat .message')].some(m => "
                               "m.innerText.includes('Hello there'))", timeout=20000)
        page.wait_for_timeout(500)
        self.shot(page, "chat-reply")
        self.assertReadable(page, "text", ".chat-card")
        self.assertReadable(page, "button", ".chat-card")

    def test_chat_sidebar_collapses_and_expands(self):
        page = self.fresh()
        page.locator(".collapse-holder .side-toggle").click()
        page.wait_for_timeout(500)
        self.assertFalse(page.locator(".chat-sidebar .new-chat").is_visible())
        self.assertReadable(page, "button", ".chat-card")
        page.locator(".expand-holder .side-toggle").click()
        page.wait_for_timeout(500)
        self.assertTrue(page.locator(".chat-sidebar .new-chat").is_visible())
        self.assertReadable(page, "button", ".chat-card")

    def test_scan_advanced_settings_are_readable(self):
        page = self.fresh()
        self.open_tab(page, "Scan")
        page.locator("button", has_text="Advanced settings").first.click()
        page.wait_for_timeout(500)
        self.shot(page, "scan-advanced")
        self.assertReadable(page, "text")
        self.assertReadable(page, "button")

    def test_report_tabs_and_raw_data_are_readable(self):
        page = self.fresh()
        self.open_tab(page, "Reports")
        for sub in ("garak report", "Raw data", "Security summary"):
            page.locator(".subtabs button[role=tab]", has_text=sub).first.click()
            page.wait_for_timeout(700)
            self.shot(page, "report-" + sub.replace(" ", "-").lower())
            self.assertReadable(page, "text")
            self.assertReadable(page, "button")

    def test_delete_confirmation_is_readable(self):
        page = self.fresh()
        self.open_tab(page, "Reports")
        page.locator(".report-list input[type=checkbox]").first.check()  # (clicking the label text views it)
        page.wait_for_function("!document.querySelector('.reps-delete').disabled", timeout=10000)
        page.locator(".reps-delete").click()
        page.wait_for_selector(".confirm-box", timeout=5000)
        dialog = ".confirm-box"
        self.shot(page, "delete-confirm")
        self.assertIn("permanently deleted", page.locator(".confirm-box p").inner_text())
        self.assertReadable(page, "button", dialog)
        self.assertReadable(page, "text", dialog)
        page.locator(".confirm-cancel").click()
        self.assertEqual(page.locator(".confirm-box").count(), 0, "Cancel closes it")


if __name__ == "__main__":
    unittest.main()
