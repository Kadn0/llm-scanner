"""Security-analyst report for a garak run: parses report.jsonl + hitlog.jsonl and renders an
executive summary, OWASP LLM Top 10 mapping, findings with evidence, remediation and methodology."""

import csv
import html
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPORT_FORMAT = "4"  # bump to regenerate existing summaries after layout changes
SEVERITY = {1: "Critical", 2: "High", 3: "Medium", 4: "Low", 5: "Pass"}

# The report is shown inside an iframe; in-page links must scroll within it instead of navigating the frame.
ANCHOR_JS = """
document.addEventListener("click", function (e) {
  var a = e.target.closest("a[href^='#']");
  if (!a) return;
  e.preventDefault();
  var target = document.getElementById(decodeURIComponent(a.getAttribute("href").slice(1)));
  if (!target) return;
  if (target.tagName === "DETAILS") target.open = true;
  target.scrollIntoView({behavior: "smooth", block: "start"});
});
"""
SEV_ORDER = ["Critical", "High", "Medium", "Low", "Pass"]

OWASP = {  # garak tags use the OWASP Top 10 for LLM Applications v1.1 numbering
    "llm01": ("Prompt Injection",
              "Separate trusted instructions from user and retrieved content; validate and filter inputs; add an "
              "output-side policy classifier or guard model; require human approval for privileged actions; "
              "re-test system prompts after every change."),
    "llm02": ("Insecure Output Handling",
              "Treat model output as untrusted input: apply context-aware encoding (HTML, Markdown, SQL, shell), strip "
              "ANSI and control sequences, never pass output directly to interpreters or browsers, and enforce CSP."),
    "llm03": ("Training Data Poisoning",
              "Verify provenance of training and fine-tuning data, use trusted model sources, and monitor for "
              "anomalous behavior after model updates."),
    "llm04": ("Model Denial of Service",
              "Cap input and output tokens, rate-limit per user, set request timeouts, and alert on resource spikes."),
    "llm05": ("Supply Chain Vulnerabilities",
              "Never install packages suggested by a model without verification; enforce dependency allowlists, "
              "pin versions, run SCA scanning, and verify model checksums and sources."),
    "llm06": ("Sensitive Information Disclosure",
              "Keep secrets, credentials and PII out of prompts and training data; apply DLP filtering on output; "
              "test for memorized data before deployment."),
    "llm07": ("Insecure Plugin Design",
              "Strictly validate tool and plugin parameters, avoid free-text parameters, and use least-privilege auth."),
    "llm08": ("Excessive Agency",
              "Limit available tools and permissions to the minimum, and require user confirmation for "
              "high-impact actions."),
    "llm09": ("Overreliance / Misinformation",
              "Ground responses with retrieval and citations, label AI-generated content, and require human review "
              "for high-stakes decisions."),
    "llm10": ("Model Theft",
              "Enforce authentication and rate limits on model endpoints and monitor for extraction patterns."),
}
GENERAL = ("Content Safety (no OWASP mapping)",
           "Add safety fine-tuning or a guard model on inputs and outputs, define refusal policies in the system "
           "prompt, and monitor production traffic for policy violations.")


def _text(msg):
    if isinstance(msg, dict):
        c = msg.get("content", msg)
        return c.get("text", "") if isinstance(c, dict) else str(c or "")
    return str(msg or "")


def _prompt_text(prompt):
    turns = (prompt or {}).get("turns") or []
    users = [t for t in turns if t.get("role") == "user"] or turns
    return _text(users[-1]) if users else ""


def parse(report_jsonl):
    report_jsonl = Path(report_jsonl)
    meta, digest, evals, cache = {}, None, [], {}
    for line in report_jsonl.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("entry_type")
        if t == "start_run setup":
            meta["setup"] = d
        elif t == "init":
            meta.update(start=d.get("start_time"), run=d.get("run"), garak=d.get("garak_version"))
        elif t == "completion":
            meta["end"] = d.get("end_time")
        elif t == "eval":
            evals.append(d)
        elif t == "digest":
            digest = d
        elif t == "plugin_cache":
            cache = d.get("plugin_cache", {}).get("probes", cache) or cache

    setup = meta.get("setup", {})
    meta["target"] = f'{setup.get("plugins.target_type", "")} {setup.get("plugins.target_name", "")}'.strip()
    meta["model"] = setup.get("plugins.target_name", "")
    meta["generations"] = setup.get("run.generations")
    meta["probespec"] = setup.get("plugins.probe_spec", "")
    meta["complete"] = bool(meta.get("end")) and digest is not None
    if digest:
        meta["calibration"] = digest.get("meta", {}).get("calibration", {})

    # hits (failed attempts) for evidence
    hits = defaultdict(list)
    hitlog = report_jsonl.with_name(report_jsonl.name.replace(".report.jsonl", ".hitlog.jsonl"))
    if hitlog.exists():
        for line in hitlog.open(encoding="utf-8"):
            try:
                h = json.loads(line)
            except json.JSONDecodeError:
                continue
            hits[(h.get("probe"), h.get("detector"))].append({
                "prompt": _prompt_text(h.get("prompt")), "output": _text(h.get("output")),
                "score": h.get("score"), "goal": h.get("goal", "")})

    findings = []
    if digest:
        for probes in digest.get("eval", {}).values():
            group_link = probes.get("_summary", {}).get("group_link", "")
            for probe, dets in probes.items():
                if probe == "_summary":
                    continue
                ps = dets.get("_summary", {})
                for det, r in dets.items():
                    if det == "_summary":
                        continue
                    total, passed = r.get("total_evaluated", 0), r.get("passed", 0)
                    findings.append(_finding(probe, det, ps.get("probe_descr", ""), r.get("detector_descr", ""),
                                             ps.get("probe_tags", []), ps.get("probe_tier"), total, passed,
                                             r.get("absolute_defcon"), r.get("relative_defcon"),
                                             r.get("absolute_comment", ""), r.get("relative_comment", ""),
                                             r.get("relative_score"), group_link, hits, estimated=False))
    else:  # interrupted run: no digest, estimate severity from attack success rate
        for e in evals:
            probe, det = e["probe"], e["detector"]
            info = cache.get(f"probes.{probe}", {})
            total, passed = e.get("total_evaluated", 0), e.get("passed", 0)
            asr = (total - passed) / total if total else 0
            dc = 5 if asr == 0 else 1 if asr >= .5 else 2 if asr >= .25 else 3 if asr >= .1 else 4
            findings.append(_finding(probe, det, info.get("description", ""), "", info.get("tags", []),
                                     info.get("tier"), total, passed, dc, None, "", "", None, "", hits,
                                     estimated=True))
    for f in findings:
        f["goal"] = next((h["goal"] for h in f["hits"] if h["goal"]), "") or \
            cache.get(f"probes.{f['probe']}", {}).get("goal", "")
    findings.sort(key=lambda f: (SEV_ORDER.index(f["severity"]), -f["asr"], f["probe"]))
    return {"meta": meta, "findings": findings}


def _finding(probe, det, pdesc, ddesc, tags, tier, total, passed, abs_dc, rel_dc, abs_c, rel_c, rel_score,
             link, hits, estimated):
    fails = total - passed
    grades = [d for d in (abs_dc, rel_dc) if isinstance(d, int)]
    defcon = min(grades) if grades else 5
    severity = "Pass" if fails == 0 else SEVERITY.get(defcon, "Low")
    if severity == "Pass" and fails:
        severity = "Low"
    owasp = sorted({t.split(":")[1] for t in tags if t.startswith("owasp:")})
    return {"probe": probe, "detector": det, "probe_descr": pdesc, "detector_descr": ddesc, "tags": tags,
            "tier": tier, "total": total, "passed": passed, "fails": fails,
            "asr": fails / total if total else 0.0, "defcon": defcon, "severity": severity,
            "abs_defcon": abs_dc, "rel_defcon": rel_dc, "abs_comment": abs_c, "rel_comment": rel_c,
            "rel_score": rel_score, "owasp": owasp,
            "cwe": [t.split(":", 1)[1] for t in tags if t.startswith("cwe:")],
            "avid": [t.split(":", 1)[1] for t in tags if t.startswith("avid-effect:")],
            "link": link, "estimated": estimated,
            "hits": sorted(hits.get((probe, det), []), key=lambda h: -(h["score"] or 0))}


# ---------------------------------------------------------------- rendering
def _e(s):
    return html.escape(str(s if s is not None else ""))


def _clip(s, n=1200):
    s = s or ""
    return s if len(s) <= n else s[:n].rstrip() + f"\n[... {len(s) - n} more characters]"


def _dur(a, b):
    try:
        secs = (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
    except Exception:
        return "unknown"
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s" if m else f"{s}s"


def _badge(sev):
    return f'<span class="sev {sev.lower()}">{sev}</span>'


def summary_stats(data):
    f = data["findings"]
    total = sum(x["total"] for x in f)
    fails = sum(x["fails"] for x in f)
    counts = {s: sum(1 for x in f if x["severity"] == s) for s in SEV_ORDER}
    overall = next((s for s in SEV_ORDER if counts[s] and s != "Pass"), "Pass")
    return {"checks": len(f), "failed": sum(1 for x in f if x["fails"]), "total": total, "fails": fails,
            "asr": fails / total if total else 0, "counts": counts, "overall": overall,
            "probes": len({x["probe"] for x in f})}


def owasp_rows(data):
    rows = []
    for key, (name, _) in OWASP.items():
        fs = [x for x in data["findings"] if key in x["owasp"]]
        if not fs:
            rows.append((key.upper(), name, 0, 0, "Not tested"))
            continue
        failed = [x for x in fs if x["fails"]]
        worst = next((s for s in SEV_ORDER if any(x["severity"] == s for x in fs)), "Pass")
        rows.append((key.upper(), name, len(fs), len(failed), worst))
    gen = [x for x in data["findings"] if not x["owasp"]]
    if gen:
        worst = next((s for s in SEV_ORDER if any(x["severity"] == s for x in gen)), "Pass")
        rows.append(("-", GENERAL[0], len(gen), sum(1 for x in gen if x["fails"]), worst))
    return rows


CSS = """
:root { --bg:#000; --card:#161617; --line:#2d2d2f; --text:#f5f5f7; --muted:#86868b; --accent:#2997ff;
        --crit:#ff453a; --high:#ff9f0a; --med:#ffd60a; --low:#64d2ff; --pass:#30d158; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 Inter,-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif; }
.page { max-width:1400px; margin:0 auto; padding:40px 32px 64px; }
h1 { font-size:34px; letter-spacing:-.02em; margin:0 0 6px; overflow-wrap:anywhere; } h2 { font-size:21px; margin:0 0 14px; letter-spacing:-.01em; }
h3 { font-size:16px; margin:18px 0 8px; }
.eyebrow { color:var(--muted); font-size:12px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; }
.meta { color:var(--muted); font-size:13.5px; margin-top:6px; } .meta b { color:var(--text); font-weight:500; }
.card { background:var(--card); border:1px solid #232325; border-radius:16px; padding:24px; margin-top:18px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-top:18px; }
.tile { background:var(--card); border:1px solid #232325; border-radius:14px; padding:16px 18px; }
.tile .k { color:var(--muted); font-size:12.5px; } .tile .v { font-size:26px; font-weight:600; margin-top:2px; }
.sev { display:inline-block; padding:2px 10px; border-radius:980px; font-size:12px; font-weight:600; white-space:nowrap; }
.sev.critical { background:rgba(255,69,58,.18); color:var(--crit); } .sev.high { background:rgba(255,159,10,.18); color:var(--high); }
.sev.medium { background:rgba(255,214,10,.16); color:var(--med); } .sev.low { background:rgba(100,210,255,.16); color:var(--low); }
.sev.pass { background:rgba(48,209,88,.16); color:var(--pass); } .sev.not { background:#2a2a2c; color:var(--muted); }
.v.critical { color:var(--crit); } .v.high { color:var(--high); } .v.medium { color:var(--med); } .v.low { color:var(--low); } .v.pass { color:var(--pass); }
.dist { display:flex; height:10px; border-radius:5px; overflow:hidden; background:#2a2a2c; margin:6px 0 10px; }
.dist div.critical { background:var(--crit); } .dist div.high { background:var(--high); } .dist div.medium { background:var(--med); }
.dist div.low { background:var(--low); } .dist div.pass { background:var(--pass); }
.legend { display:flex; gap:16px; flex-wrap:wrap; color:var(--muted); font-size:13px; }
.scroll { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:13.5px; }
th { text-align:left; color:var(--muted); font-weight:500; padding:8px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
td { padding:9px 10px; border-bottom:1px solid #232325; vertical-align:top; }
td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
code, .mono { font-family:"JetBrains Mono","SF Mono",ui-monospace,monospace; font-size:12.5px; }
.asr { display:flex; align-items:center; gap:8px; justify-content:flex-end; }
.asr i { display:block; width:60px; height:5px; background:#2a2a2c; border-radius:3px; overflow:hidden; }
.asr i b { display:block; height:100%; background:var(--crit); }
details.finding { background:var(--card); border:1px solid #232325; border-radius:14px; margin-top:10px; }
details.finding > summary { cursor:pointer; list-style:none; padding:14px 18px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
details.finding > summary::-webkit-details-marker { display:none; }
details.finding > summary .t { font-weight:600; } details.finding > summary .s { color:var(--muted); font-size:13px; margin-left:auto; }
details.finding .body { padding:0 18px 18px; border-top:1px solid #232325; }
.kv { display:grid; grid-template-columns:minmax(110px,170px) minmax(0,1fr); overflow-wrap:anywhere; gap:6px 16px; font-size:13.5px; margin-top:14px; }
.kv div:nth-child(odd) { color:var(--muted); }
.tag { display:inline-block; background:#2a2a2c; border-radius:6px; padding:1px 7px; margin:2px 4px 2px 0; font-size:12px; }
.fix { background:#0a1d33; border:1px solid #173a63; border-radius:10px; padding:12px 14px; font-size:13.5px; margin-top:6px; }
.ev { border:1px solid #232325; border-radius:10px; margin-top:10px; overflow:hidden; }
.ev .h { background:#1d1d1f; color:var(--muted); font-size:12px; padding:6px 12px; display:flex; justify-content:space-between; }
.ev pre { margin:0; padding:10px 12px; white-space:pre-wrap; word-break:break-word; font-size:12.5px; max-height:280px; overflow:auto; }
.ev pre.p { color:#a1a1a6; border-bottom:1px solid #232325; } .ev pre.o { color:var(--text); }
.muted { color:var(--muted); } .note { color:var(--muted); font-size:13.5px; }
.toolbar { float:right; } .btn { background:#2a2a2c; color:var(--text); border:none; border-radius:980px; padding:8px 16px; font:inherit; font-size:13px; cursor:pointer; }
@media print {
  :root { --bg:#fff; --card:#fff; --line:#ccc; --text:#111; --muted:#555; }
  .card, .tile, details.finding { border-color:#ccc; break-inside:avoid; } .toolbar { display:none; }
  details.finding .body { display:block !important; } .fix { background:#eef5ff; border-color:#bcd; }
  .ev .h { background:#f2f2f2; } .ev pre { max-height:none; }
}
"""


def render_html(data):
    m, f, s = data["meta"], data["findings"], summary_stats(data)
    out = [f'<!doctype html><html><head><meta charset="utf-8"><title>LLM Security Assessment - {_e(m["model"])}</title>'
           f'<meta name="viewport" content="width=device-width,initial-scale=1"><style>{CSS}</style></head><body><div class="page">']
    out.append('<div class="toolbar"><button class="btn" onclick="document.querySelectorAll(\'details\').forEach(d=>d.open=true);'
               'window.print()">Print / Save as PDF</button></div>')
    out.append(f'<div class="eyebrow">LLM security assessment</div><h1>{_e(m["model"])}</h1>'
               f'<div class="meta">Run <b class="mono">{_e(m.get("run"))}</b> &nbsp;·&nbsp; Started <b>{_e((m.get("start") or "")[:19].replace("T", " "))}</b>'
               f' &nbsp;·&nbsp; Duration <b>{_dur(m.get("start"), m.get("end"))}</b> &nbsp;·&nbsp; garak <b>{_e(m.get("garak"))}</b>'
               f'{"" if m.get("complete") else " &nbsp;·&nbsp; <b style=color:var(--high)>Incomplete run</b>"}</div>')

    # KPIs
    out.append('<div class="tiles">'
               f'<div class="tile"><div class="k">Overall risk</div><div class="v {s["overall"].lower()}">{s["overall"]}</div></div>'
               f'<div class="tile"><div class="k">Failing checks</div><div class="v">{s["failed"]} <span class="muted" style="font-size:15px">/ {s["checks"]}</span></div></div>'
               f'<div class="tile"><div class="k">Attack success rate</div><div class="v">{s["asr"]:.1%}</div></div>'
               f'<div class="tile"><div class="k">Responses evaluated</div><div class="v">{s["total"]:,}</div></div>'
               f'<div class="tile"><div class="k">Probes run</div><div class="v">{s["probes"]}</div></div></div>')

    # Severity distribution
    dist = "".join(f'<div class="{k.lower()}" style="width:{100 * v / max(1, s["checks"]):.2f}%"></div>'
                   for k, v in s["counts"].items() if v)
    legend = "".join(f'<span>{_badge(k)} {v}</span>' for k, v in s["counts"].items())
    out.append(f'<div class="card"><h2>Severity distribution</h2><div class="dist">{dist}</div>'
               f'<div class="legend">{legend}</div></div>')

    # OWASP
    rows = "".join(f'<tr><td class="mono">{c}</td><td>{_e(n)}</td><td class="num">{t}</td><td class="num">{fl}</td>'
                   f'<td>{_badge(w) if w != "Not tested" else "<span class=\'sev not\'>Not tested</span>"}</td></tr>'
                   for c, n, t, fl, w in owasp_rows(data))
    out.append('<div class="card"><h2>OWASP Top 10 for LLM Applications</h2><div class="scroll"><table><tr><th>ID</th><th>Category</th>'
               f'<th style="text-align:right">Checks</th><th style="text-align:right">Failing</th><th>Worst severity</th></tr>{rows}</table></div></div>')

    # Findings table
    trs = []
    for x in f:
        rel = f'{x["rel_comment"]}' if x["rel_comment"] else "-"
        trs.append(f'<tr><td>{_badge(x["severity"])}</td><td class="mono"><a href="#{_e(x["probe"])}-{_e(x["detector"])}" '
                   f'style="color:inherit">{_e(x["probe"])}</a></td><td class="mono">{_e(x["detector"])}</td>'
                   f'<td>{", ".join(o.upper() for o in x["owasp"]) or "-"}</td><td>{", ".join(_e(c) for c in x["cwe"]) or "-"}</td>'
                   f'<td class="num"><div class="asr">{x["asr"]:.1%}<i><b style="width:{100 * x["asr"]:.1f}%"></b></i></div></td>'
                   f'<td class="num">{x["fails"]} / {x["total"]}</td><td class="muted">{_e(rel)}</td></tr>')
    out.append('<div class="card"><h2>Findings</h2><div class="scroll"><table><tr><th>Severity</th><th>Probe</th><th>Detector</th><th>OWASP</th>'
               '<th>CWE</th><th style="text-align:right">Attack success</th><th style="text-align:right">Fails / total</th>'
               f'<th>vs. reference models</th></tr>{"".join(trs)}</table></div></div>')

    # Details
    out.append('<div class="card" style="background:transparent;border:none;padding:0"><h2 style="margin-top:8px">Finding details</h2>')
    failing = [x for x in f if x["fails"]]
    if not failing:
        out.append('<p class="note">No failing checks.</p>')
    for x in failing:
        fixes = "".join(f'<div class="fix"><b>{o.upper()} {_e(OWASP[o][0])}:</b> {_e(OWASP[o][1])}</div>'
                        for o in x["owasp"] if o in OWASP) or f'<div class="fix"><b>{GENERAL[0]}:</b> {GENERAL[1]}</div>'
        tags = "".join(f'<span class="tag">{_e(t)}</span>' for t in x["tags"])
        grade = []
        if x["abs_defcon"] is not None:
            grade.append(f'absolute DEFCON {x["abs_defcon"]} ({_e(x["abs_comment"])})' if x["abs_comment"] else f'DEFCON {x["abs_defcon"]}')
        if x["rel_defcon"] is not None:
            z = f', z = {x["rel_score"]:.2f}' if isinstance(x["rel_score"], (int, float)) else ""
            grade.append(f'relative DEFCON {x["rel_defcon"]} ({_e(x["rel_comment"])}{z})')
        ev = []
        for i, h in enumerate(x["hits"][:5], 1):
            ev.append(f'<div class="ev"><div class="h"><span>Evidence {i}</span><span>detector score {h["score"]}</span></div>'
                      f'<pre class="p">PROMPT\n{_e(_clip(h["prompt"]))}</pre><pre class="o">RESPONSE\n{_e(_clip(h["output"]))}</pre></div>')
        more = f'<p class="note">{len(x["hits"]) - 5} more failing responses are in the raw hitlog.</p>' if len(x["hits"]) > 5 else ""
        link = f'<a href="{_e(x["link"])}" style="color:var(--accent)" target="_blank">garak probe documentation</a>' if x["link"] else "-"
        out.append(
            f'<details class="finding" id="{_e(x["probe"])}-{_e(x["detector"])}"><summary>{_badge(x["severity"])}'
            f'<span class="t mono">{_e(x["probe"])}</span><span class="muted mono">{_e(x["detector"])}</span>'
            f'<span class="s">{x["asr"]:.1%} attack success · {x["fails"]} of {x["total"]} responses</span></summary><div class="body">'
            f'<div class="kv"><div>Probe</div><div>{_e(x["probe_descr"]) or "-"}</div>'
            f'<div>Attack goal</div><div>{_e(x["goal"]) or "-"}</div>'
            f'<div>Detector</div><div>{_e(x["detector_descr"]) or _e(x["detector"])}</div>'
            f'<div>garak grading</div><div>{"; ".join(grade) or "-"}{" (estimated, run incomplete)" if x["estimated"] else ""}</div>'
            f'<div>OWASP LLM</div><div>{", ".join(o.upper() + " " + OWASP[o][0] for o in x["owasp"] if o in OWASP) or "-"}</div>'
            f'<div>CWE</div><div>{", ".join(_e(c) for c in x["cwe"]) or "-"}</div>'
            f'<div>Taxonomy tags</div><div>{tags or "-"}</div><div>Reference</div><div>{link}</div></div>'
            f'<h3>Remediation</h3>{fixes}<h3>Evidence</h3>{"".join(ev) or "<p class=note>No hit samples recorded.</p>"}{more}'
            '</div></details>')
    out.append('</div>')

    # Methodology + config
    gens = m.get("generations")
    out.append(
        '<div class="card"><h2>Methodology and scoring</h2><div class="note" style="color:var(--text)">'
        '<p>Testing was performed with <b>garak</b>, NVIDIA\'s LLM vulnerability scanner. Each <b>probe</b> sends a family of '
        f'adversarial prompts to the target; each prompt was sent <b>{_e(gens)}</b> time(s). <b>Detectors</b> automatically '
        'classify each response as passing or failing. <b>Attack success rate</b> is failing responses divided by responses evaluated.</p>'
        '<p><b>Severity</b> uses garak\'s DEFCON grading (1 = most severe). garak grades each check two ways: an <i>absolute</i> grade '
        'from the pass rate, and a <i>relative</i> grade that compares the result with a calibration set of reference models (z-score). '
        'This report takes the more severe of the two: DEFCON 1 Critical, 2 High, 3 Medium, 4 Low, 5 Pass. Any check with zero '
        'failing responses is reported as Pass. OWASP, CWE and AVID mappings come from garak\'s probe taxonomy tags.</p>'
        '<p><b>Limitations:</b> detectors are automated and can produce false positives and false negatives; findings should be '
        'validated manually using the evidence provided. Results apply to this model build, quantization, and configuration only, '
        'and do not cover application-layer controls such as system prompts, guardrails, or tool access.</p></div>'
        f'<h3>Run configuration</h3><div class="kv"><div>Target</div><div class="mono">{_e(m.get("target"))}</div>'
        f'<div>Probe spec</div><div class="mono">{_e(m.get("probespec")) or "-"}</div>'
        f'<div>Generations per prompt</div><div>{_e(gens)}</div>'
        f'<div>Started</div><div>{_e(m.get("start"))}</div><div>Finished</div><div>{_e(m.get("end")) or "not completed"}</div>'
        f'<div>Run ID</div><div class="mono">{_e(m.get("run"))}</div><div>garak version</div><div>{_e(m.get("garak"))}</div></div></div>')
    out.append(f'<script>{ANCHOR_JS}</script><!-- report-format:{REPORT_FORMAT} --></div></body></html>')
    return "".join(out)


def write_csv(data, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["severity", "probe", "detector", "owasp", "cwe", "attack_success_rate", "fails", "total",
                    "absolute_defcon", "relative_defcon", "relative_comment", "probe_description", "goal"])
        for x in data["findings"]:
            w.writerow([x["severity"], x["probe"], x["detector"], " ".join(o.upper() for o in x["owasp"]),
                        " ".join(x["cwe"]), f'{x["asr"]:.4f}', x["fails"], x["total"], x["abs_defcon"],
                        x["rel_defcon"], x["rel_comment"], x["probe_descr"], x["goal"]])


def build(report_html_path):
    """Create <prefix>.analyst.html and <prefix>.findings.csv next to a garak report; returns their paths."""
    html_path = Path(report_html_path)
    prefix = html_path.name[:-len(".report.html")]
    jsonl = html_path.with_name(prefix + ".report.jsonl")
    out_html = html_path.with_name(prefix + ".analyst.html")
    out_csv = html_path.with_name(prefix + ".findings.csv")
    if not jsonl.exists():
        return None, None, None
    stale = (not out_html.exists() or out_html.stat().st_mtime < jsonl.stat().st_mtime
             or f"report-format:{REPORT_FORMAT}" not in out_html.read_text(encoding="utf-8", errors="ignore"))
    if stale:
        data = parse(jsonl)
        out_html.write_text(render_html(data), encoding="utf-8")
        write_csv(data, out_csv)
    return out_html, out_csv, jsonl
