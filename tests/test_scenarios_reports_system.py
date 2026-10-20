"""Scenario tests for the Reports tab, the status bar and Ollama controls, update checks, the Red team tools page and
startup housekeeping.

Run alone:  .venv/bin/python -m unittest discover -s tests -p test_scenarios_reports_system.py
Nothing here touches real data, Ollama, Hugging Face, GitHub, the home folder or a browser: data folders point at a
temporary directory (TempDataTest), HTTP calls, subprocesses, process groups and threads are fakes, and every app
global that a test changes is restored afterwards."""

import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_app import FakeResponse, TempDataTest, analyst_report, app, write_run  # noqa: E402

JSONL_PICK = "Full report (report.jsonl)"
HITLOG_PICK = "Hitlog (failing responses)"
NOOP = {"__type__": "update"}


def upd(**kw):
    """What gr.update(**kw) returns."""
    return {"__type__": "update", **kw}


def running_proc(pid=4242, poll=None):
    p = mock.Mock(pid=pid)
    p.poll.return_value = poll
    return p


# ---------------------------------------------------------------- Reports tab
class RunsTest(TempDataTest):
    def setUp(self):
        super().setUp()
        self.runs = app.GARAK_RUNS

    def run_files(self, prefix):
        return sorted(p.name for p in self.runs.iterdir() if p.name.startswith(prefix + "."))

    def make_run(self, prefix, mtime=None, **kw):
        write_run(self.runs, prefix, **kw)
        if mtime is not None:
            for name in self.run_files(prefix):
                os.utime(self.runs / name, (mtime, mtime))
        return prefix + ".report.html"

    def put(self, name, text="", mtime=None):
        f = self.runs / name
        f.write_text(text, encoding="utf-8")
        if mtime is not None:
            os.utime(f, (mtime, mtime))
        return f


class ReportListing(RunsTest):
    def test_missing_runs_folder_lists_nothing(self):
        shutil.rmtree(self.runs)
        self.assertEqual(app.reports(), [])

    def test_runs_are_ordered_newest_first_by_mtime(self):
        a = self.make_run("a_20260917-100000", mtime=1_000_000)
        b = self.make_run("b_20260917-100000", mtime=3_000_000)
        c = self.make_run("c_20260917-100000", mtime=2_000_000)
        self.assertEqual([n for _, n in app.reports()], [b, c, a])

    def test_label_strips_hf_prefix_and_timestamp(self):
        self.make_run("hf.co_org_repo_20260917-100000")
        label, name = app.reports()[0]
        self.assertEqual(name, "hf.co_org_repo_20260917-100000.report.html")
        self.assertEqual(label.split("   ")[0], "org_repo")

    def test_label_shows_local_time_of_the_run(self):
        self.make_run("llama3_20260917-100000", mtime=1_700_000_000)
        when = time.strftime("%b %d, %H:%M", time.localtime(1_700_000_000))
        self.assertEqual(app.reports()[0][0], f"llama3   {when}")

    def test_only_the_trailing_timestamp_is_stripped(self):
        self.make_run("m_20260101-000000_20260917-100000")
        self.assertTrue(app.reports()[0][0].startswith("m_20260101-000000   "))

    def test_stopped_run_is_marked_and_still_named_by_its_html(self):
        self.make_run("m_20260917-110000", complete=False)
        label, name = app.reports()[0]
        self.assertTrue(label.endswith("   (stopped)"), label)
        self.assertEqual(name, "m_20260917-110000.report.html")
        self.assertFalse((self.runs / name).exists())

    def test_report_html_and_jsonl_of_one_run_are_listed_once(self):
        self.make_run("m_20260917-100000")
        listing = app.reports()
        self.assertEqual(len(listing), 1)
        self.assertNotIn("(stopped)", listing[0][0])

    def test_the_newest_file_of_a_run_decides_its_place(self):
        old = self.make_run("old_20260917-100000", mtime=1_000_000)
        os.utime(self.runs / "old_20260917-100000.report.jsonl", (5_000_000, 5_000_000))
        mid = self.make_run("mid_20260917-100000", mtime=3_000_000)
        self.assertEqual([n for _, n in app.reports()], [old, mid])

    def test_bare_suffixes_and_other_files_are_not_runs(self):
        for name in (".report.html", ".report.jsonl", "x.hitlog.jsonl", "notes.txt", "y.analyst.html"):
            self.put(name, "x")
        self.assertEqual(app.reports(), [])

    def test_empty_and_corrupt_jsonl_runs_are_still_listed(self):
        self.put("empty_20260917-100000.report.jsonl", "", mtime=2_000_000)
        self.put("bad_20260917-100000.report.jsonl", "{not json\n\x00", mtime=1_000_000)
        labels = [lbl for lbl, _ in app.reports()]
        self.assertEqual(len(labels), 2)
        self.assertTrue(labels[0].startswith("empty   ") and labels[0].endswith("(stopped)"), labels)
        self.assertTrue(labels[1].startswith("bad   "), labels)


class ReportNames(RunsTest):
    def test_known_report_accepts_a_listed_name(self):
        name = self.make_run("m_20260917-100000")
        self.assertTrue(app._known_report(name))
        self.assertFalse(app._known_report("m_20260917-100000.report.jsonl"), "only the run's html name is offered")

    def test_known_report_rejects_empty_none_and_labels(self):
        self.make_run("m_20260917-100000")
        label = app.reports()[0][0]
        for bad in ("", None, label, "other_20260917-100000.report.html"):
            self.assertFalse(app._known_report(bad), bad)

    def test_known_report_rejects_traversal_to_an_existing_file(self):
        (self.tmp / "secret.report.html").write_text("TOP SECRET")
        self.make_run("m_20260917-100000")
        self.assertFalse(app._known_report("../secret.report.html"))

    def test_delete_label_for_nothing_ticked_is_disabled(self):
        self.assertEqual(app.delete_label(None), upd(value="Delete reports", interactive=False))

    def test_list_state_drops_selections_that_no_longer_exist(self):
        a = self.make_run("a_20260917-100000", mtime=2_000_000)
        self.make_run("b_20260917-100000", mtime=1_000_000)
        listing, select_all, button, selected, shown = app.report_list_state([a, "gone.report.html"], a)
        self.assertEqual(selected, [a])
        self.assertEqual(listing["value"], [a])
        self.assertEqual(listing["choices"], app.reports())
        self.assertFalse(select_all["value"])
        self.assertEqual(button["value"], "Delete 1 report")

    def test_list_state_shows_the_newest_when_the_shown_report_is_gone(self):
        self.make_run("a_20260917-100000", mtime=1_000_000)
        b = self.make_run("b_20260917-100000", mtime=2_000_000)
        *_, shown = app.report_list_state([], "deleted_20260917-100000.report.html")
        self.assertEqual(shown, b)

    def test_list_state_ticks_select_all_when_every_run_is_ticked(self):
        a = self.make_run("a_20260917-100000", mtime=1_000_000)
        b = self.make_run("b_20260917-100000", mtime=2_000_000)
        _, select_all, button, selected, shown = app.report_list_state([a, b], a)
        self.assertTrue(select_all["value"])
        self.assertEqual(button, upd(value="Delete 2 reports", interactive=True))
        self.assertEqual(shown, a)


class ReportTicking(RunsTest):
    def test_ticking_several_at_once_shows_the_last_one_added(self):
        a = self.make_run("a_20260917-100000", mtime=1_000_000)
        b = self.make_run("b_20260917-100000", mtime=2_000_000)
        button, picked, shown, summary, garak_view, raw, files = app.reports_ticked([a, b], [], None, JSONL_PICK)
        self.assertEqual((picked, shown), ([a, b], b))
        self.assertEqual(button["value"], "Delete 2 reports")
        self.assertIn(str(self.runs / "b_20260917-100000.analyst.html"), files)

    def test_unticking_after_the_shown_report_vanished_shows_the_first_ticked(self):
        a = self.make_run("a_20260917-100000", mtime=1_000_000)
        b = self.make_run("b_20260917-100000", mtime=2_000_000)
        _, picked, shown, *_ = app.reports_ticked([a], [a, b], "gone_20260917-100000.report.html", JSONL_PICK)
        self.assertEqual((picked, shown), ([a], a))

    def test_nothing_ticked_and_an_unknown_shown_report_empties_the_viewer(self):
        a = self.make_run("a_20260917-100000")
        button, picked, shown, summary, garak_view, raw, files = app.reports_ticked(None, [a], "gone.report.html",
                                                                                     JSONL_PICK)
        self.assertEqual((picked, shown, files), ([], None, None))
        self.assertIn("No reports yet", summary)
        self.assertFalse(button["interactive"])

    def test_view_report_matches_a_label_ignoring_extra_spacing(self):
        a = self.make_run("a_20260917-100000", mtime=1_000_000)
        b = self.make_run("b_20260917-100000", mtime=2_000_000)
        label = next(lbl for lbl, n in app.reports() if n == a)
        name, summary, garak_view, raw, files = app.view_report("  " + " ".join(label.split()) + " ", b, JSONL_PICK)
        self.assertEqual(name, a)
        self.assertIn("LLM security assessment", summary)

    def test_select_all_without_reports(self):
        listing, button, picked, shown, summary, *_ = app.select_all_reports(True, None, JSONL_PICK)
        self.assertEqual((listing["value"], picked, shown), ([], [], None))
        self.assertEqual(button["value"], "Delete reports")
        self.assertIn("No reports yet", summary)

    def test_select_all_keeps_the_report_already_shown(self):
        older = self.make_run("a_20260917-100000", mtime=1_000_000)
        self.make_run("b_20260917-100000", mtime=2_000_000)
        listing, button, picked, shown, *_ = app.select_all_reports(True, older, JSONL_PICK)
        self.assertEqual(shown, older)
        self.assertEqual(len(picked), 2)
        self.assertEqual(listing["value"], picked)


class ReportFiles(RunsTest):
    def test_report_paths_derives_every_file_from_the_prefix(self):
        p = app.report_paths("x_1.report.html")
        self.assertEqual({k: v.name for k, v in p.items()},
                         {"garak": "x_1.report.html", "jsonl": "x_1.report.jsonl", "hitlog": "x_1.hitlog.jsonl",
                          "analyst": "x_1.analyst.html", "csv": "x_1.findings.csv", "txt": "x_1.report.txt",
                          "table": "x_1.report.csv", "hits": "x_1.hitlog.csv"})
        self.assertTrue(all(v.parent == self.runs for v in p.values()))

    def test_report_paths_keep_dots_inside_the_prefix(self):
        p = app.report_paths("hf.co_org_repo.v2_20260917-100000.report.html")
        self.assertEqual(p["jsonl"].name, "hf.co_org_repo.v2_20260917-100000.report.jsonl")
        self.assertEqual(p["hitlog"].name, "hf.co_org_repo.v2_20260917-100000.hitlog.jsonl")

    def test_write_report_text_does_nothing_without_the_jsonl(self):
        txt = self.runs / "x.report.txt"
        app.write_report_text(self.runs / "x.report.jsonl", txt)
        self.assertFalse(txt.exists())

    def test_write_report_text_numbers_records_by_line_and_skips_blank_lines(self):
        jsonl = self.put("x.report.jsonl", '{"a": 1}\n\n   \n{"b": 2}\n')
        txt = self.runs / "x.report.txt"
        app.write_report_text(jsonl, txt)
        text = txt.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("garak report: x.report.jsonl\n"))
        self.assertIn('--- record 1 ---\n{\n  "a": 1\n}\n', text)
        self.assertIn('--- record 4 ---\n{\n  "b": 2\n}\n', text)
        self.assertNotIn("record 2", text)
        self.assertNotIn("record 3", text)
        self.assertFalse((self.runs / "x.report.txt.tmp").exists())

    def test_write_report_text_keeps_unicode_and_replaces_bad_bytes(self):
        jsonl = self.runs / "x.report.jsonl"
        jsonl.write_bytes('{"t": "héllo ✓"}\n'.encode() + b'{"u": "\xff"}\n')
        txt = self.runs / "x.report.txt"
        app.write_report_text(jsonl, txt)
        text = txt.read_text(encoding="utf-8")
        self.assertIn('"t": "héllo ✓"', text)
        self.assertIn('"u": "\ufffd"', text)

    def test_write_report_text_rewrites_a_copy_older_than_the_report(self):
        jsonl = self.put("x.report.jsonl", '{"new": true}\n', mtime=2_000_000)
        txt = self.put("x.report.txt", "stale", mtime=1_000_000)
        app.write_report_text(jsonl, txt)
        self.assertIn('"new": true', txt.read_text(encoding="utf-8"))
        self.assertNotIn("stale", txt.read_text(encoding="utf-8"))

    def test_read_raw_full_report_returns_the_jsonl(self):
        name = self.make_run("m_20260917-100000")
        raw = app.read_raw(name, JSONL_PICK)
        self.assertEqual(raw, (self.runs / "m_20260917-100000.report.jsonl").read_text(encoding="utf-8"))
        self.assertTrue(raw.endswith("{garbled"))

    def test_read_raw_hitlog_pick_returns_the_failing_responses(self):
        name = self.make_run("m_20260917-100000")
        self.assertIn("<script>evil</script>", app.read_raw(name, HITLOG_PICK))

    def test_read_raw_missing_hitlog_says_nothing_failed(self):
        name = self.make_run("m_20260917-100000", hits=False)
        self.assertEqual(app.read_raw(name, HITLOG_PICK), "No failing responses were recorded for this run.")

    def test_read_raw_missing_jsonl_says_file_not_found(self):
        self.put("m_20260917-100000.report.html", "<html>x</html>")
        self.assertEqual(app.read_raw("m_20260917-100000.report.html", JSONL_PICK), "File not found.")

    def test_read_raw_truncates_past_the_limit(self):
        self.put("m_1.report.jsonl", "a" * 25)
        with mock.patch.object(app, "RAW_LIMIT", 10):
            raw = app.read_raw("m_1.report.html", JSONL_PICK)
        self.assertEqual(raw, "a" * 10 + "\n\n[Truncated: showing 10 of 25 characters. Download the file for the rest.]")


class ShowReport(RunsTest):
    def test_show_report_with_nothing_chosen_shows_the_empty_state(self):
        summary, garak_view, raw, files = app.show_report(None)
        self.assertIn("No reports yet. Run a scan to create one.", summary)
        self.assertEqual(summary, garak_view)
        self.assertEqual(raw, "")
        self.assertIsNone(files)

    def test_show_report_for_a_run_with_only_garak_html(self):
        f = self.put("m_20260917-100000.report.html", "<html>garak only</html>")
        summary, garak_view, raw, files = app.show_report(f.name)
        self.assertIn("Could not build the security summary", summary)
        self.assertIn("&lt;html&gt;garak only", garak_view)
        self.assertEqual(raw, "File not found.")
        self.assertEqual(files, [str(f)])

    def test_show_report_for_a_stopped_run(self):
        name = self.make_run("m_20260917-110000", complete=False)
        summary, garak_view, raw, files = app.show_report(name, HITLOG_PICK)
        self.assertIn("stopped before garak wrote its report", garak_view)
        p = "m_20260917-110000"
        self.assertEqual([Path(f).name for f in files], [f"{p}.analyst.html", f"{p}.findings.csv", f"{p}.report.jsonl",
                                                         f"{p}.report.csv", f"{p}.report.txt", f"{p}.hitlog.jsonl",
                                                         f"{p}.hitlog.csv"])
        self.assertIn("evil", raw)

    def test_show_report_survives_a_record_that_is_not_an_object(self):
        self.put("m_1.report.jsonl", "null\n")
        summary, garak_view, raw, files = app.show_report("m_1.report.html")
        self.assertTrue(summary.startswith('<div class="empty">Could not build the security summary:'), summary)
        self.assertEqual(raw, "null\n")

    def test_show_report_escapes_the_summary_into_srcdoc(self):
        name = self.make_run("m_20260917-100000")
        summary, *_ = app.show_report(name)
        self.assertTrue(summary.startswith('<iframe class="report dark" srcdoc="&lt;!doctype html&gt;'), summary[:80])
        self.assertNotIn("<script>evil", summary)

    def test_show_report_for_an_empty_jsonl(self):
        self.put("m_1.report.jsonl", "")
        summary, garak_view, raw, files = app.show_report("m_1.report.html")
        self.assertIn("No failing checks.", summary)
        self.assertIn("Incomplete run", summary)
        self.assertEqual(raw, "")


class DeleteAndOpenReports(RunsTest):
    def test_delete_run_counts_its_files_and_spares_a_longer_prefix(self):
        name = self.make_run("m_20260917-100000")
        self.make_run("m_20260917-100000x")
        self.assertEqual(app._delete_run(name), 3)
        self.assertEqual(sorted(p.name for p in self.runs.iterdir()),
                         ["m_20260917-100000x.hitlog.jsonl", "m_20260917-100000x.report.html",
                          "m_20260917-100000x.report.jsonl"])

    def test_delete_reports_cancelled_changes_nothing(self):
        name = self.make_run("m_20260917-100000")
        self.assertEqual(app.delete_reports([name], False, name, JSONL_PICK), (NOOP,) * 10)
        self.assertEqual(len(self.run_files("m_20260917-100000")), 3)

    def test_delete_reports_with_nothing_ticked_changes_nothing(self):
        name = self.make_run("m_20260917-100000")
        for names in ([], None):
            self.assertEqual(app.delete_reports(names, True, name, JSONL_PICK), (NOOP,) * 10)
        self.assertTrue((self.runs / name).exists())

    def test_delete_reports_moves_the_viewer_off_the_deleted_report(self):
        keep = self.make_run("keep_20260917-100000", mtime=1_000_000)
        gone = self.make_run("gone_20260917-100000", mtime=2_000_000)
        listing, select_all, button, selected, shown, summary, garak_view, raw, files, note = \
            app.delete_reports([gone], True, gone, JSONL_PICK)
        self.assertEqual(shown, keep)
        self.assertEqual([n for _, n in listing["choices"]], [keep])
        self.assertTrue(files and all(Path(f).name.startswith("keep_20260917-100000.") for f in files), files)
        self.assertIn("Deleted 1 report (3 files).", note)

    def test_deleting_the_last_report_empties_the_viewer(self):
        name = self.make_run("m_20260917-100000")
        app.show_report(name)  # the summary, findings CSV and every CSV and text copy now exist too
        out = app.delete_reports([name], True, name, HITLOG_PICK)
        self.assertEqual(len(out), 10)
        self.assertIsNone(out[4])
        self.assertIn("No reports yet", out[5])
        self.assertIsNone(out[8])
        self.assertIn("Deleted 1 report (8 files).", out[9])
        self.assertFalse(out[1]["value"])
        self.assertEqual(list(self.runs.iterdir()), [])

    def test_delete_reports_skips_unknown_names_among_known_ones(self):
        a = self.make_run("a_20260917-100000", mtime=1_000_000)
        b = self.make_run("b_20260917-100000", mtime=2_000_000)
        out = app.delete_reports([a, "../escape.report.html", "nope.report.html"], True, b, JSONL_PICK)
        self.assertIn("Deleted 1 report (3 files).", out[9])
        self.assertEqual(out[4], b, "the report on screen was not deleted, so it stays")
        self.assertFalse((self.runs / a).exists())
        self.assertTrue((self.runs / b).exists())

    def test_open_report_opens_the_built_summary_in_the_browser(self):
        name = self.make_run("m_20260917-100000")
        with mock.patch.object(app.webbrowser, "open") as browser, mock.patch.object(app.threading, "Thread") as thread:
            app.open_report(name)
        analyst = self.runs / "m_20260917-100000.analyst.html"
        self.assertTrue(analyst.exists())
        self.assertIs(thread.call_args.kwargs["target"], browser)
        self.assertEqual(thread.call_args.kwargs["args"], (analyst.as_uri(),))
        thread.return_value.start.assert_called_once()
        browser.assert_not_called()

    def test_open_report_unknown_name_opens_nothing(self):
        (self.tmp / "secret.report.html").write_text("x")
        with mock.patch.object(app.webbrowser, "open"), mock.patch.object(app.threading, "Thread") as thread:
            app.open_report("../secret.report.html")
        thread.assert_not_called()

    def test_open_report_for_a_run_with_only_garak_html_opens_that(self):
        f = self.put("m_20260917-100000.report.html", "<html/>")
        with mock.patch.object(app.webbrowser, "open"), mock.patch.object(app.threading, "Thread") as thread:
            app.open_report(f.name)
        self.assertEqual(thread.call_args.kwargs["args"], (f.as_uri(),))

    def test_open_report_falls_back_to_garak_html_when_the_summary_fails(self):
        name = self.make_run("m_20260917-100000")
        with mock.patch.object(app.analyst_report, "build", side_effect=RuntimeError("broken")), \
                mock.patch.object(app.webbrowser, "open"), mock.patch.object(app.threading, "Thread") as thread:
            app.open_report(name)
        self.assertEqual(thread.call_args.kwargs["args"], ((self.runs / name).as_uri(),))


# ---------------------------------------------------------------- security summary (analyst_report.py)
SETUP = {"entry_type": "start_run setup", "plugins.target_type": "ollama", "plugins.target_name": "m",
         "run.generations": 1, "plugins.probe_spec": "x"}
INIT = {"entry_type": "init", "start_time": "2026-09-17T10:00:00", "run": "r1", "garak_version": "0.17.0"}


def hit(probe, score, prompt="p", output="o", goal="", turns=None, det="d.D"):
    return {"probe": probe, "detector": det, "score": score, "goal": goal, "output": {"text": output},
            "prompt": {"turns": turns or [{"role": "user", "content": {"text": prompt}}]}}


def write_hits(folder, prefix, hits):
    if hits:
        (folder / f"{prefix}.hitlog.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hits), encoding="utf-8")


def digest_report(folder, prefix, checks, hits=(), extra=()):
    """A finished run's report.jsonl. Each check: probe, total, passed and optionally det, abs, rel, tags, descr."""
    groups = {}
    for c in checks:
        group = groups.setdefault(c["probe"].split(".")[0], {"_summary": {"group_link": ""}})
        probe = group.setdefault(c["probe"], {"_summary": {"probe_descr": c.get("descr", ""),
                                                           "probe_tags": c.get("tags", [])}})
        probe[c.get("det", "d.D")] = {"total_evaluated": c["total"], "passed": c["passed"],
                                      "absolute_defcon": c.get("abs"), "relative_defcon": c.get("rel")}
    lines = [SETUP, INIT, *extra, {"entry_type": "completion", "end_time": "2026-09-17T10:00:45"},
             {"entry_type": "digest", "meta": {}, "eval": groups}]
    path = folder / f"{prefix}.report.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    write_hits(folder, prefix, hits)
    return path


def eval_report(folder, prefix, evals, extra=()):
    """An interrupted run's report.jsonl: eval records only, no completion or digest."""
    lines = [SETUP, INIT, *extra] + [{"entry_type": "eval", "probe": p, "detector": d, "total_evaluated": t,
                                      "passed": ok} for p, d, t, ok in evals]
    path = folder / f"{prefix}.report.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return path


def by_probe(findings):
    return sorted(findings, key=lambda f: f["probe"])


class AnalystSummary(RunsTest):
    def parse_data(self, *checks, hits=()):
        return analyst_report.parse(digest_report(self.runs, "m_1", checks, hits=hits))

    def parse_checks(self, *checks, hits=()):
        return self.parse_data(*checks, hits=hits)["findings"]

    def test_no_failures_is_pass_even_at_defcon_one(self):
        f = self.parse_checks({"probe": "x.A", "total": 10, "passed": 10, "abs": 1})[0]
        self.assertEqual((f["severity"], f["fails"], f["asr"]), ("Pass", 0, 0.0))

    def test_failures_graded_defcon_five_or_ungraded_are_low(self):
        graded, ungraded = by_probe(self.parse_checks({"probe": "x.A", "total": 10, "passed": 9, "abs": 5},
                                                      {"probe": "y.B", "total": 10, "passed": 8}))
        self.assertEqual((graded["severity"], ungraded["severity"]), ("Low", "Low"))
        self.assertEqual(ungraded["defcon"], 5)

    def test_the_more_severe_grade_wins_and_non_numbers_are_ignored(self):
        a, b = by_probe(self.parse_checks({"probe": "x.A", "total": 10, "passed": 9, "abs": 4, "rel": 2},
                                          {"probe": "y.B", "total": 10, "passed": 9, "abs": "1", "rel": 3}))
        self.assertEqual((a["severity"], a["defcon"]), ("High", 2))
        self.assertEqual((b["severity"], b["defcon"]), ("Medium", 3))

    def test_tags_map_to_sorted_owasp_cwe_and_avid(self):
        f = self.parse_checks({"probe": "x.A", "total": 10, "passed": 5, "abs": 2,
                               "tags": ["owasp:llm06", "owasp:llm01", "owasp:llm01", "cwe:CWE-200",
                                        "avid-effect:security:S0301", "payload:jailbreak"]})[0]
        self.assertEqual(f["owasp"], ["llm01", "llm06"])
        self.assertEqual(f["cwe"], ["CWE-200"])
        self.assertEqual(f["avid"], ["security:S0301"])

    def test_owasp_rows_list_every_category_and_mark_untested(self):
        data = self.parse_data({"probe": "x.A", "total": 10, "passed": 4, "abs": 2, "tags": ["owasp:llm01"]},
                               {"probe": "y.B", "total": 10, "passed": 10, "abs": 5, "tags": ["owasp:llm01"]})
        rows = analyst_report.owasp_rows(data)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0], ("LLM01", "Prompt Injection", 2, 1, "High"))
        self.assertEqual(rows[1], ("LLM02", "Insecure Output Handling", 0, 0, "Not tested"))

    def test_untagged_findings_get_a_content_safety_row(self):
        rows = analyst_report.owasp_rows(self.parse_data({"probe": "x.A", "total": 10, "passed": 4, "abs": 2}))
        self.assertEqual(len(rows), 11)
        self.assertEqual(rows[-1], ("-", analyst_report.GENERAL[0], 1, 1, "High"))

    def test_summary_stats_counts_checks_and_picks_the_worst_failure(self):
        s = analyst_report.summary_stats(self.parse_data({"probe": "x.A", "total": 10, "passed": 4, "abs": 2},
                                                         {"probe": "x.B", "total": 10, "passed": 10, "abs": 5}))
        self.assertEqual((s["checks"], s["failed"], s["total"], s["fails"], s["overall"], s["probes"]),
                         (2, 1, 20, 6, "High", 2))
        self.assertAlmostEqual(s["asr"], 0.3)
        self.assertEqual(s["counts"], {"Critical": 0, "High": 1, "Medium": 0, "Low": 0, "Pass": 1})

    def test_findings_sort_by_severity_then_attack_success_then_name(self):
        fs = self.parse_checks({"probe": "b.B", "total": 10, "passed": 4, "abs": 2},
                               {"probe": "d.D", "total": 10, "passed": 10, "abs": 5},
                               {"probe": "a.A", "total": 10, "passed": 4, "abs": 2},
                               {"probe": "e.E", "total": 10, "passed": 2, "abs": 2},
                               {"probe": "c.C", "total": 10, "passed": 8, "abs": 1})
        self.assertEqual([f["probe"] for f in fs], ["c.C", "e.E", "a.A", "b.B", "d.D"])

    def test_interrupted_run_takes_details_from_the_plugin_cache(self):
        cache = {"entry_type": "plugin_cache", "plugin_cache": {"probes": {"probes.x.A": {
            "description": "Cached D", "tags": ["owasp:llm09"], "goal": "make it lie", "tier": 2}}}}
        path = eval_report(self.runs, "m_1", [("x.A", "d.D", 10, 5)], extra=[cache])
        f = analyst_report.parse(path)["findings"][0]
        self.assertEqual((f["probe_descr"], f["owasp"], f["goal"], f["tier"]), ("Cached D", ["llm09"], "make it lie", 2))

    def test_evidence_is_sorted_by_score_and_quotes_the_last_user_turn(self):
        turns = [{"role": "user", "content": {"text": "first"}}, {"role": "assistant", "content": {"text": "a"}},
                 {"role": "user", "content": {"text": "second"}}]
        hits = [hit("x.A", 0.2, "mild"), hit("x.A", 0.9, turns=turns, output="worst", goal="steal"),
                hit("x.A", 1.0, "other detector", det="other.D")]
        f = self.parse_checks({"probe": "x.A", "total": 10, "passed": 5, "abs": 2}, hits=hits)[0]
        self.assertEqual([h["score"] for h in f["hits"]], [0.9, 0.2])
        self.assertEqual((f["hits"][0]["prompt"], f["hits"][0]["output"]), ("second", "worst"))
        self.assertEqual(f["goal"], "steal")

    def test_render_escapes_model_output_in_evidence(self):
        data = self.parse_data({"probe": "x.A", "total": 10, "passed": 5, "abs": 2},
                               hits=[hit("x.A", 1.0, "hi", output="<img src=x onerror=alert(1)>")])
        out = analyst_report.render_html(data)
        self.assertNotIn("<img src=x", out)
        self.assertIn("RESPONSE\n&lt;img src=x onerror=alert(1)&gt;", out)

    def test_render_shows_five_samples_and_counts_the_rest(self):
        hits = [hit("x.A", i / 10, f"prompt {i}") for i in range(7)]
        out = analyst_report.render_html(self.parse_data({"probe": "x.A", "total": 10, "passed": 3, "abs": 2},
                                                         hits=hits))
        self.assertIn("Evidence 5", out)
        self.assertNotIn("Evidence 6", out)
        self.assertIn("2 more failing responses are in the raw hitlog.", out)

    def test_csv_round_trips_commas_quotes_and_newlines(self):
        descr = 'Says "hi", then\nleaves; =1+1'
        data = self.parse_data({"probe": "x.A", "total": 10, "passed": 4, "abs": 2, "descr": descr})
        path = self.runs / "out.csv"
        analyst_report.write_csv(data, path)
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        self.assertEqual(rows[0], ["severity", "probe", "detector", "owasp", "cwe", "attack_success_rate", "fails",
                                   "total", "absolute_defcon", "relative_defcon", "relative_comment",
                                   "probe_description", "goal"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][11], descr)

    def test_csv_columns_for_an_estimated_finding(self):
        cache = {"entry_type": "plugin_cache", "plugin_cache": {"probes": {"probes.x.A": {
            "tags": ["owasp:llm06", "owasp:llm01", "cwe:CWE-79"]}}}}
        data = analyst_report.parse(eval_report(self.runs, "m_1", [("x.A", "d.D", 10, 4)], extra=[cache]))
        path = self.runs / "out.csv"
        analyst_report.write_csv(data, path)
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        self.assertEqual(rows[1], ["Critical", "x.A", "d.D", "LLM01 LLM06", "CWE-79", "0.6000", "6", "10", "1", "",
                                   "", "", ""])

    def test_build_regenerates_a_summary_from_an_older_format(self):
        digest_report(self.runs, "m_1", [{"probe": "x.A", "total": 10, "passed": 4, "abs": 2}])
        old = self.runs / "m_1.analyst.html"
        old.write_text("<html>old layout</html>")
        os.utime(old, (4_000_000_000, 4_000_000_000))  # newer than the report: only the format marker is missing
        html_path, csv_path, jsonl = analyst_report.build(self.runs / "m_1.report.html")
        self.assertEqual(html_path, old)
        self.assertIn(f"report-format:{analyst_report.REPORT_FORMAT}", old.read_text(encoding="utf-8"))
        self.assertTrue(csv_path.exists())


def _severity_case(defcon, severity):
    def test(self):
        f = self.parse_checks({"probe": "x.A", "total": 10, "passed": 9, "abs": defcon})[0]
        self.assertEqual((f["severity"], f["defcon"]), (severity, defcon))
    return test


for _dc, _sev in ((1, "Critical"), (2, "High"), (3, "Medium"), (4, "Low")):
    setattr(AnalystSummary, f"test_one_failure_graded_defcon_{_dc}_is_{_sev.lower()}", _severity_case(_dc, _sev))


def _estimate_case(total, passed, severity):
    def test(self):
        f = analyst_report.parse(eval_report(self.runs, "m_1", [("x.A", "d.D", total, passed)]))["findings"][0]
        self.assertEqual(f["severity"], severity)
        self.assertTrue(f["estimated"])
        self.assertIsNone(f["rel_defcon"])
    return test


for _total, _passed, _sev in ((10, 9, "Medium"), (4, 3, "High"), (2, 1, "Critical")):
    setattr(AnalystSummary, f"test_interrupted_run_{_passed}_of_{_total}_passing_is_estimated_{_sev.lower()}",
            _estimate_case(_total, _passed, _sev))


# ---------------------------------------------------------------- status bar and Ollama controls
class SystemReadings(unittest.TestCase):
    def fit(self, size):
        with mock.patch.object(app, "VRAM_BUDGET_GB", 8.0), mock.patch.object(app, "ram_gb", return_value=10.0):
            return app.fit_breakdown(size)

    def gpu(self, stdout=None, error=None):
        run = mock.Mock(side_effect=error) if error else mock.Mock(return_value=mock.Mock(stdout=stdout))
        with mock.patch.object(app.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch.object(app.subprocess, "run", run):
            return app.gpu_live()

    def test_ram_gb_multiplies_page_size_by_page_count(self):
        sizes = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 4_000_000}
        with mock.patch.object(app.os, "sysconf", side_effect=sizes.__getitem__):
            self.assertAlmostEqual(app.ram_gb(), 16.384)

    def test_ram_gb_falls_back_to_16_when_unknown(self):
        with mock.patch.object(app.os, "sysconf", side_effect=ValueError("unsupported")):
            self.assertEqual(app.ram_gb(), 16)

    def test_fit_breakdown_exactly_the_budget_stays_in_vram(self):
        self.assertEqual(self.fit(8.0), "8.0 GB VRAM")

    def test_fit_breakdown_spills_into_ram_up_to_sixty_percent_of_it(self):
        self.assertEqual(self.fit(14.0), "8.0 GB VRAM + 6.0 GB RAM")

    def test_fit_breakdown_beyond_that_is_too_large(self):
        self.assertEqual(self.fit(14.5), "14.5 GB — too large for this PC")

    def test_gpu_live_is_none_without_nvidia_smi(self):
        with mock.patch.object(app.shutil, "which", return_value=None), mock.patch.object(app.subprocess, "run") as run:
            self.assertIsNone(app.gpu_live())
        run.assert_not_called()

    def test_gpu_live_parses_and_shortens_the_name(self):
        self.assertEqual(self.gpu("NVIDIA GeForce RTX 5070 Laptop GPU, 37, 2048, 8192, 61, 45.30\n"),
                         ("RTX 5070 Laptop", 37.0, 2.0, 8.0, 61.0, 45.3))

    def test_gpu_live_reports_unavailable_readings_as_none(self):
        self.assertEqual(self.gpu("Tesla T4, [N/A], 100, 15360, [N/A], [N/A]\n"),
                         ("Tesla T4", None, 100 / 1024, 15.0, None, None))

    def test_gpu_live_is_none_when_nvidia_smi_hangs(self):
        self.assertIsNone(self.gpu(error=subprocess.TimeoutExpired("nvidia-smi", 5)))

    def test_ram_live_reads_used_and_total_from_meminfo(self):
        meminfo = "MemTotal:       16777216 kB\nMemFree:          524288 kB\nMemAvailable:    4194304 kB\n"
        with mock.patch.object(app, "open", mock.mock_open(read_data=meminfo), create=True):
            self.assertEqual(app.ram_live(), (12.0, 16.0))

    def test_ram_live_is_none_without_available_memory(self):
        with mock.patch.object(app, "open", mock.mock_open(read_data="MemTotal: 16777216 kB\n"), create=True):
            self.assertIsNone(app.ram_live())

    def test_meter_clamps_above_100_and_runs_hot(self):
        self.assertEqual(app._meter(150), '<span class="meter "><i class="hot" style="width:100%"></i></span>')

    def test_meter_just_below_seventy_stays_calm_though_drawn_at_70(self):
        self.assertEqual(app._meter(69.6, "ram"), '<span class="meter ram"><i class="" style="width:70%"></i></span>')

    def test_ollama_status_when_running_escapes_the_version(self):
        self.assertEqual(app.ollama_status_html("0.5<1>"),
                         '<span class="oc-status"><span class="dot ok"></span>Ollama 0.5&lt;1&gt;</span>')

    def test_ollama_status_when_stopped(self):
        stopped = '<span class="oc-status"><span class="dot bad"></span>Ollama stopped</span>'
        self.assertEqual(app.ollama_status_html(None), stopped)
        self.assertEqual(app.ollama_status_html(""), stopped)

    def test_status_html_shows_gpu_and_ram_pills(self):
        with mock.patch.object(app, "gpu_live", return_value=("RTX <5070>", 50.0, 4.0, 8.0, 65.4, 120.0)), \
                mock.patch.object(app, "ram_live", return_value=(6.0, 16.0)):
            out = app.status_html()
        self.assertTrue(out.startswith('<div class="status">'))
        self.assertIn('data-tip-title="RTX &lt;5070&gt;"', out)
        self.assertIn('data-tip="GPU 50% busy, 4.0 of 8.0 GB video memory in use, 65°C, drawing 120 W"', out)
        self.assertIn('<span class="val">4.0/8 GB</span>', out)
        self.assertIn('<span class="muted">65°C</span>', out)
        self.assertIn('<span class="val">6/16 GB</span>', out)

    def test_status_html_without_gpu_or_memory_readings(self):
        with mock.patch.object(app, "gpu_live", return_value=None), mock.patch.object(app, "ram_live", return_value=None):
            self.assertEqual(app.status_html(),
                             '<div class="status"><span class="pill"><span class="dot warn"></span>No GPU detected</span></div>')

    def test_loaded_models_is_empty_when_ollama_is_unreachable(self):
        with mock.patch.object(app.requests, "get", side_effect=app.requests.ConnectionError("refused")):
            self.assertEqual(app.loaded_models(), [])


class StatusBase(unittest.TestCase):
    def setUp(self):
        for patch in (mock.patch.dict(app._proc, {"p": None}), mock.patch.dict(app._image_proc, {"p": None}),
                      mock.patch.dict(app._scan, {"action": None, "paused": False, "job": None}),
                      mock.patch.dict(app._downloads, clear=True)):
            self.enterContext(patch)
        self.scan_job = self.enterContext(mock.patch.object(app, "_load_scan_job", return_value=None))
        self.loaded = self.enterContext(mock.patch.object(app, "loaded_models", return_value=[]))


class Activity(StatusBase):
    def test_idle_when_nothing_runs(self):
        self.assertEqual(app.activity_html(), '<div class="activity idle">Idle: no models loaded, nothing running</div>')

    def test_model_fully_on_gpu_with_unload_countdown(self):
        expires = (datetime.now(timezone.utc) + timedelta(minutes=10, seconds=20)).isoformat()
        self.loaded.return_value = [{"name": "llama3:latest", "size": 4_000_000_000, "size_vram": 4_000_000_000,
                                     "expires_at": expires}]
        self.assertEqual(app.activity_html(), '<div class="activity">Model loaded: <b>llama3:latest</b> '
                                              '(4.0 GB, GPU, unloads in 10 min)</div>')

    def test_model_split_between_gpu_and_cpu_has_its_name_escaped(self):
        self.loaded.return_value = [{"name": "<x>", "size": 10_000_000_000, "size_vram": 2_500_000_000}]
        self.assertEqual(app.activity_html(),
                         '<div class="activity">Model loaded: <b>&lt;x&gt;</b> (10.0 GB, 25% GPU / 75% CPU)</div>')

    def test_running_scan(self):
        app._proc["p"] = running_proc()
        self.assertEqual(app.activity_html(), '<div class="activity">Scan running</div>')

    def test_stuck_scan_being_restarted_is_a_warning(self):
        app._scan["action"] = "restart"
        self.assertEqual(app.activity_html(), '<div class="activity"><span class="warn">Scan stuck, restarting</span></div>')

    def test_halted_scan_asks_to_continue(self):
        self.scan_job.return_value = {"model": "m", "halted": True}
        self.assertIn('<span class="warn">Scan halted, open Scan to continue</span>', app.activity_html())

    def test_generating_an_image(self):
        app._image_proc["p"] = running_proc()
        self.assertEqual(app.activity_html(), '<div class="activity">Generating an image</div>')

    def test_one_download_counted_ignoring_paused_hidden_and_finished(self):
        app._downloads.update({"a": {"state": "downloading"}, "b": {"state": "paused", "pause": True},
                               "c": {"state": "queued", "hidden": True}, "d": {"state": "done"}})
        self.assertEqual(app.activity_html(), '<div class="activity">1 download in progress</div>')


class OllamaControls(StatusBase):
    def test_model_tag_adds_latest_only_when_no_tag(self):
        self.assertEqual(app._model_tag("llama3"), "llama3:latest")
        self.assertEqual(app._model_tag("llama3:8b"), "llama3:8b")
        self.assertEqual(app._model_tag("hf.co/org/repo"), "hf.co/org/repo:latest")
        self.assertEqual(app._model_tag("registry:5000/ns/model"), "registry:5000/ns/model:latest")

    def test_scanning_model_only_while_the_scan_runs_unstuck(self):
        app._scan["job"] = {"model": "llama3"}
        app._proc["p"] = running_proc()
        self.assertEqual(app.scanning_model(), "llama3:latest")
        app._scan["action"] = "restart"
        self.assertIsNone(app.scanning_model())

    def test_unload_models_keeps_the_model_a_scan_uses(self):
        app._scan["job"] = {"model": "b"}
        app._proc["p"] = running_proc()
        self.loaded.return_value = [{"name": "a:latest"}, {"name": "b"}]
        with mock.patch.object(app.requests, "post") as post, mock.patch.object(app, "status_controls", return_value="ctl"):
            self.assertEqual(app.unload_models(), "ctl")
        post.assert_called_once_with(f"{app.OLLAMA}/api/generate", json={"model": "a:latest", "keep_alive": 0},
                                     timeout=30)

    def test_unload_button_explains_why_the_scan_model_stays(self):
        app._scan["job"] = {"model": "b"}
        app._proc["p"] = running_proc()
        self.loaded.return_value = [{"name": "b:latest"}]
        with mock.patch.object(app.gr, "Info") as info, mock.patch.object(app, "unload_models", return_value="done"):
            self.assertEqual(app.unload_button(), "done")
        info.assert_called_once_with("b:latest stays loaded: a scan is using it. It unloads when the scan ends.")

    def test_status_controls_while_ollama_runs_with_a_model_loaded(self):
        self.loaded.return_value = [{"name": "a"}]
        with mock.patch.object(app, "ollama_up", return_value="0.5.1"), \
                mock.patch.object(app, "status_html", return_value="S"), \
                mock.patch.object(app, "activity_html", return_value="A"):
            out = app.status_controls()
        self.assertEqual(out, ("S", upd(visible=True), upd(visible=False), upd(visible=True), upd(visible=True),
                               app.ollama_status_html("0.5.1"), "A"))

    def test_status_controls_while_ollama_is_stopped(self):
        with mock.patch.object(app, "ollama_up", return_value=None), \
                mock.patch.object(app, "status_html", return_value="S"), \
                mock.patch.object(app, "activity_html", return_value="A"):
            out = app.status_controls()
        self.assertEqual(out[1:5], (upd(visible=False), upd(visible=True), upd(visible=False), upd(visible=False)))
        self.assertIn("Ollama stopped", out[5])

    def test_ollama_service_runs_systemctl_and_waits_until_up(self):
        with mock.patch.object(app.subprocess, "run") as run, \
                mock.patch.object(app, "ollama_up", side_effect=[None, None, "0.5.1"]), \
                mock.patch.object(app.time, "sleep") as sleep, mock.patch.object(app, "status_controls", return_value="ctl"):
            self.assertEqual(app._ollama_service("start", True), "ctl")
        run.assert_called_once_with(["systemctl", "--user", "start", "ollama"], check=False)
        self.assertEqual(sleep.call_count, 2)

    def test_stop_ollama_is_refused_during_a_scan(self):
        app._proc["p"] = running_proc()
        with mock.patch.object(app.gr, "Warning") as warn, mock.patch.object(app, "_ollama_service") as service, \
                mock.patch.object(app, "status_controls", return_value="ctl"):
            self.assertEqual(app.stop_ollama(), "ctl")
        warn.assert_called_once_with("A scan is using Ollama. Stop the scan first.")
        service.assert_not_called()

    def test_restart_ollama_without_a_scan_restarts_the_service(self):
        with mock.patch.object(app, "_ollama_service", return_value="ctl") as service:
            self.assertEqual(app.restart_ollama(), "ctl")
        service.assert_called_once_with("restart", True)


# ---------------------------------------------------------------- update checks
class VersionTuple(unittest.TestCase):
    def test_a_newer_patch_release_with_two_digits_compares_higher(self):
        self.assertGreater(app._version_tuple("1.0.10"), app._version_tuple("1.0.9"))


def _version_case(text, expected):
    def test(self):
        self.assertEqual(app._version_tuple(text), expected)
    return test


for _name, _text, _expected in (("plain_release", "1.2.3", (1, 2, 3)), ("v_prefixed_tag", "v0.34.2", (0, 34, 2)),
                                ("fourth_part_dropped", "1.2.3.4", (1, 2, 3)),
                                ("prerelease_suffix_ignored", "2.0.0-rc1", (2, 0, 0)),
                                ("empty_string", "", ()), ("none", None, ())):
    setattr(VersionTuple, f"test_version_tuple_{_name}", _version_case(_text, _expected))


class UpdatesBase(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.dict(app._updates, {
            "checked": 0.0, "ollama": None, "garak": None, "app": None, "busy": None, "pct": None, "msg": "",
            "ok": True, "restarting": False, "msg_expires": None, "dismissed": {}}))
        self.enterContext(mock.patch.object(app, "APP_VERSION", "1.0.0"))
        self.garak_version = self.enterContext(mock.patch.object(app.importlib.metadata, "version",
                                                                 return_value="0.17.0"))
        self.enterContext(mock.patch.dict(app._proc, {"p": None}))
        self.enterContext(mock.patch.dict(app._image_proc, {"p": None}))

    @staticmethod
    def releases(ollama="v0.5.0", garak="0.18.0", app_tag="v1.1.0", body="", fail=()):
        def get(url, **kw):
            kind = "ollama" if "ollama/ollama" in url else "garak" if "pypi" in url else "app"
            if kind in fail:
                raise app.requests.ConnectionError("offline")
            if kind == "ollama":
                return FakeResponse({"tag_name": ollama, "prerelease": False})
            if kind == "garak":
                return FakeResponse({"info": {"version": garak}})
            return FakeResponse({"tag_name": app_tag, "body": body, "prerelease": False})
        return get

    def check(self, get, installed_ollama="0.4.0"):
        with mock.patch.object(app.requests, "get", side_effect=get), \
                mock.patch.object(app, "ollama_up", return_value=installed_ollama):
            return app.check_updates()

    def controls(self):
        box, msg, row_app, row_ollama, row_garak, lbl_app, lbl_ollama, lbl_garak, all_btn, dismiss = app.update_controls()
        return {"box": box["visible"], "msg": msg, "all": all_btn["visible"], "dismiss": dismiss["visible"],
                "rows": {"app": row_app["visible"], "ollama": row_ollama["visible"], "garak": row_garak["visible"]}}


class CheckUpdates(UpdatesBase):
    def test_newer_releases_of_everything_are_found(self):
        self.assertEqual(self.check(self.releases(body="  Fixes things \n")), [])
        self.assertEqual(app._updates["ollama"], ("0.4.0", "0.5.0"))
        self.assertEqual(app._updates["garak"], ("0.17.0", "0.18.0"))
        self.assertEqual(app._updates["app"], {"version": "1.1.0", "tag": "v1.1.0", "notes": "Fixes things"})
        self.assertGreater(app._updates["checked"], 0)

    def test_same_versions_clear_earlier_findings(self):
        app._updates.update(ollama=("0.1", "0.2"), garak=("0.1", "0.2"), app={"version": "9", "tag": "v9", "notes": ""})
        self.assertEqual(self.check(self.releases(ollama="v0.4.0", garak="0.17.0", app_tag="v1.0.0")), [])
        self.assertEqual((app._updates["ollama"], app._updates["garak"], app._updates["app"]), (None, None, None))

    def test_older_releases_are_not_offered(self):
        self.check(self.releases(ollama="v0.3.9", garak="0.16.2", app_tag="v0.9.9"))
        self.assertEqual((app._updates["ollama"], app._updates["garak"], app._updates["app"]), (None, None, None))

    def test_ollama_update_needs_ollama_running_to_know_its_version(self):
        self.assertEqual(self.check(self.releases(), installed_ollama=None), [])
        self.assertIsNone(app._updates["ollama"])
        self.assertEqual(app._updates["app"]["version"], "1.1.0")

    def test_github_unreachable_for_the_app_only(self):
        self.assertEqual(self.check(self.releases(fail=("app",))), ["LLM Scanner"])
        self.assertEqual(app._updates["ollama"], ("0.4.0", "0.5.0"))

    def test_pypi_rate_limit_counts_as_garak_not_checked(self):
        app._updates["garak"] = ("0.17.0", "0.18.0")
        normal = self.releases(ollama="v0.4.0", app_tag="v1.0.0")

        def get(url, **kw):
            return FakeResponse({"message": "Too many requests"}, status=429) if "pypi" in url else normal(url)
        self.assertEqual(self.check(get), ["garak"])
        self.assertEqual(app._updates["garak"], ("0.17.0", "0.18.0"), "an earlier finding is kept, not wiped")

    def test_github_rate_limit_counts_as_not_checked(self):
        # BUG: check_updates never looks at the HTTP status of GitHub's answers. A 403 "API rate limit exceeded" reply
        # has no tag_name, so it reads as "no newer release" and check_updates_now then says "Everything is up to date".
        limited = FakeResponse({"message": "API rate limit exceeded for 203.0.113.7."}, status=403)

        def get(url, **kw):
            return FakeResponse({"info": {"version": "0.17.0"}}) if "pypi" in url else limited
        self.assertEqual(self.check(get), ["Ollama", "LLM Scanner"])

    def test_garak_not_installed_counts_as_not_checked(self):
        self.garak_version.side_effect = app.importlib.metadata.PackageNotFoundError("garak")
        self.assertEqual(self.check(self.releases()), ["garak"])

    def test_check_now_does_nothing_while_an_update_runs(self):
        app._updates["busy"] = "Downloading Ollama 0.5.0"
        with mock.patch.object(app, "check_updates") as check:
            out = app.check_updates_now()
        check.assert_not_called()
        self.assertIn("Downloading Ollama 0.5.0", out[1])

    def test_check_now_reports_what_could_not_be_checked(self):
        with mock.patch.object(app, "check_updates", return_value=["garak"]):
            out = app.check_updates_now()
        self.assertFalse(app._updates["ok"])
        self.assertEqual(app._updates["msg"],
                         "Couldn't check garak for updates. Check the internet connection and try again.")
        self.assertIn("Couldn&#x27;t check garak", out[1])
        self.assertIsNone(app._updates["busy"])

    def test_check_now_up_to_date_names_versions_and_expires(self):
        with mock.patch.object(app, "check_updates", return_value=[]), \
                mock.patch.object(app, "ollama_up", return_value=None), \
                mock.patch.object(app.time, "time", return_value=1000.0):
            app.check_updates_now()
        self.assertEqual(app._updates["msg"],
                         "Everything is up to date: LLM Scanner 1.0.0, Ollama not running, garak 0.17.0.")
        self.assertEqual(app._updates["msg_expires"], 1006.0)
        self.assertTrue(app._updates["ok"])

    def test_check_now_clears_the_progress_bar_even_when_checking_crashes(self):
        with mock.patch.object(app, "check_updates", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                app.check_updates_now()
        self.assertIsNone(app._updates["busy"])
        self.assertIsNone(app._updates["pct"])


class UpdateBox(UpdatesBase):
    def test_banner_shows_percentage_progress(self):
        app._updates.update(busy="Downloading", pct=42.4)
        banner = app.update_banner()
        self.assertIn('<div style="width:42.4%"></div>', banner)
        self.assertIn("<span>42%</span>", banner)

    def test_banner_without_percentage_is_indeterminate_and_escaped(self):
        app._updates.update(busy="<Checking>", pct=None)
        banner = app.update_banner()
        self.assertIn('<div class="up-bar indeterminate">', banner)
        self.assertIn("<span>&lt;Checking&gt;</span><span></span>", banner)

    def test_banner_shows_a_failed_result_then_nothing(self):
        app._updates.update(msg="ollama update failed: x", ok=False)
        self.assertEqual(app.update_banner(),
                         '<div class="update-banner"><div class="up-result bad">ollama update failed: x</div></div>')
        app._updates["msg"] = ""
        self.assertEqual(app.update_banner(), "")

    def test_update_versions_per_kind(self):
        app._updates.update(app={"version": "9.9.9", "tag": "v9.9.9", "notes": ""}, ollama=("0.1", "0.2"))
        self.assertEqual(app.update_versions("app"), ("1.0.0", "9.9.9"))
        self.assertEqual(app.update_versions("ollama"), ("0.1", "0.2"))
        self.assertIsNone(app.update_versions("garak"))

    def test_item_html_cuts_long_release_notes_to_600_characters(self):
        app._updates["app"] = {"version": "9.9.9", "tag": "v9.9.9", "notes": "a" * 700}
        self.assertIn('data-tip-title="What\'s new in 9.9.9" data-tip="' + "a" * 600 + '"', app.update_item_html("app"))

    def test_item_html_without_notes_or_update(self):
        app._updates["app"] = {"version": "9.9.9", "tag": "v9.9.9", "notes": ""}
        item = app.update_item_html("app")
        self.assertNotIn("data-tip", item)
        self.assertIn('<span class="up-name">LLM Scanner</span><span class="up-ver">1.0.0 <span class="up-arrow">→</span>'
                      ' 9.9.9</span>', item)
        self.assertEqual(app.update_item_html("garak"), '<div class="up-item-text"><span class="up-name">garak</span>'
                                                        '<span class="up-ver"> <span class="up-arrow">→</span> </span></div>')

    def test_expired_message_hides_the_box(self):
        app._updates.update(msg="done", msg_expires=time.time() - 1)
        self.assertFalse(self.controls()["box"])
        self.assertEqual(app._updates["msg"], "")

    def test_running_update_hides_every_button(self):
        app._updates.update(busy="Installing", ollama=("0.1", "0.2"), garak=("0.17.0", "0.18.0"))
        c = self.controls()
        self.assertTrue(c["box"])
        self.assertEqual(c["rows"], {"app": False, "ollama": False, "garak": False})
        self.assertFalse(c["all"] or c["dismiss"])

    def test_dismiss_remembers_versions_and_a_new_kind_still_shows(self):
        app._updates.update(app={"version": "9.9.9", "tag": "v9.9.9", "notes": ""}, ollama=("0.34.2", "0.35.0"))
        box = app.dismiss_update_msg()[0]
        self.assertEqual(app._updates["dismissed"], {"ollama": "0.35.0", "app": "9.9.9"})
        self.assertFalse(box["visible"])
        app._updates["garak"] = ("0.17.0", "0.18.0")
        c = self.controls()
        self.assertEqual(c["rows"], {"app": False, "ollama": False, "garak": True})
        self.assertFalse(c["all"], "Update all needs more than one update on offer")

    def test_start_update_while_one_runs_changes_nothing(self):
        app._updates.update(busy="Downloading Ollama 0.35.0", msg="earlier")
        with mock.patch.object(app.threading, "Thread") as thread:
            app.start_update("garak")
        thread.assert_not_called()
        self.assertEqual((app._updates["busy"], app._updates["msg"]), ("Downloading Ollama 0.35.0", "earlier"))

    def test_generating_an_image_does_not_block_an_ollama_update(self):
        app._image_proc["p"] = running_proc()
        with mock.patch.object(app.threading, "Thread") as thread:
            app.start_update("ollama")
        self.assertEqual(thread.call_args.kwargs, {"target": app._run_update, "args": ("ollama",), "daemon": True})
        thread.return_value.start.assert_called_once()
        self.assertEqual((app._updates["busy"], app._updates["pct"]), ("Starting update", 0))

    def test_generating_an_image_blocks_update_all(self):
        app._image_proc["p"] = running_proc()
        with mock.patch.object(app.threading, "Thread") as thread:
            app.start_update("all")
        thread.assert_not_called()
        self.assertFalse(app._updates["ok"])
        self.assertEqual(app._updates["msg"], "Wait for the image to finish generating before updating.")


# ---------------------------------------------------------------- Red team tools
class RedTeamBase(TempDataTest):
    def setUp(self):
        super().setUp()
        for name, value in {"NODE_BIN": self.tmp / "node/bin", "PROMPTFOO_DIR": self.tmp / "promptfoo",
                            "PROMPTFOO_BIN": self.tmp / "promptfoo/node_modules/.bin/promptfoo",
                            "PROMPTFOO_DATA": self.tmp / "data/promptfoo", "TOOLS_STATE_FILE": self.tmp / "data/tools.json",
                            "PYRIT_BACKEND": self.tmp / "pyrit/bin/pyrit_backend", "PYRIT_PY": self.tmp / "pyrit/bin/python",
                            "PYRIT_DATA": self.tmp / "data/pyrit"}.items():
            self.enterContext(mock.patch.object(app, name, value))
        self.enterContext(mock.patch.dict(app._tools, {"state": "", "msg": ""}))
        self.enterContext(mock.patch.dict(app._promptfoo, {"p": None}))
        self.enterContext(mock.patch.dict(app._pyrit, {"p": None}))
        self.pf_answers = self.enterContext(mock.patch.object(app, "_promptfoo_answers", return_value=False))
        self.py_answers = self.enterContext(mock.patch.object(app, "_pyrit_answers", return_value=False))
        self.sleep = self.enterContext(mock.patch.object(app.time, "sleep"))

    def install_promptfoo(self):
        for f in (app.NODE_BIN / "node", app.PROMPTFOO_BIN):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("")

    def install_pyrit(self):
        app.PYRIT_BACKEND.parent.mkdir(parents=True, exist_ok=True)
        app.PYRIT_BACKEND.write_text("")


class Promptfoo(RedTeamBase):
    def test_promptfoo_version_comes_from_its_package_json(self):
        pkg = app.PROMPTFOO_DIR / "node_modules/promptfoo/package.json"
        pkg.parent.mkdir(parents=True)
        pkg.write_text(json.dumps({"name": "promptfoo", "version": "0.124.1"}))
        self.assertEqual(app.promptfoo_version(), "0.124.1")

    def test_missing_broken_or_incomplete_package_json_has_no_version(self):
        self.assertEqual(app.promptfoo_version(), "")
        pkg = app.PROMPTFOO_DIR / "node_modules/promptfoo/package.json"
        pkg.parent.mkdir(parents=True)
        pkg.write_text("{broken")
        self.assertEqual(app.promptfoo_version(), "")
        pkg.write_text(json.dumps({"name": "promptfoo"}))
        self.assertEqual(app.promptfoo_version(), "")

    def test_promptfoo_needs_node_as_well_as_its_own_program(self):
        app.PROMPTFOO_BIN.parent.mkdir(parents=True)
        app.PROMPTFOO_BIN.write_text("")
        self.assertFalse(app.promptfoo_installed())
        app.NODE_BIN.mkdir(parents=True)
        (app.NODE_BIN / "node").write_text("")
        self.assertTrue(app.promptfoo_installed())

    def test_view_after_a_failed_install_offers_to_retry(self):
        app._tools.update(state="failed", msg="no network")
        status, start, stop, link = app.redteam_view()
        self.assertIn('class="note-line error"', status)
        self.assertIn("The last attempt failed: no network", status)
        self.assertEqual(start, upd(value="Install promptfoo", visible=True, interactive=True))
        self.assertFalse(stop["visible"])
        self.assertEqual(link, "")

    def test_view_while_our_promptfoo_answers_links_to_it(self):
        self.install_promptfoo()
        app._promptfoo["p"] = running_proc()
        self.pf_answers.return_value = True
        status, start, stop, link = app.redteam_view()
        self.assertIn(f"promptfoo is running on this computer only, at {app.PROMPTFOO_URL}.", status)
        self.assertEqual((start["visible"], stop["visible"]), (False, True))
        self.assertIn(f'href="{app.PROMPTFOO_URL}"', link)
        self.assertIn("Open promptfoo", link)

    def test_view_when_something_else_serves_the_port(self):
        self.install_promptfoo()
        self.pf_answers.return_value = True
        status, start, stop, link = app.redteam_view()
        self.assertIn("Something else is already serving", status)
        self.assertEqual(start, upd(visible=True, interactive=False))
        self.assertIn("Open promptfoo", link)

    def test_start_when_missing_installs_in_the_background(self):
        with mock.patch.object(app.threading, "Thread") as thread, mock.patch.object(app.subprocess, "Popen") as popen:
            views = list(app.start_promptfoo())
        self.assertIs(thread.call_args.kwargs["target"], app.install_red_team_tools)
        self.assertEqual(thread.call_args.kwargs["kwargs"], {"force": True})
        thread.return_value.start.assert_called_once()
        popen.assert_not_called()
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0][1]["value"], "Install promptfoo")

    def test_start_while_installing_waits_for_the_install(self):
        app._tools.update(state="installing", msg="Downloading Node.js")
        self.sleep.side_effect = lambda secs: app._tools.update(state="", msg="")
        with mock.patch.object(app.threading, "Thread") as thread:
            views = list(app.start_promptfoo())
        thread.assert_not_called()
        self.assertEqual(len(views), 2)
        self.assertIn("Installing the red team tools (once). Downloading Node.js", views[0][0])
        self.assertIn("installed yet", views[1][0])

    def test_start_launches_promptfoo_locally_with_telemetry_off(self):
        self.install_promptfoo()
        answers = iter([False, True])
        self.pf_answers.side_effect = lambda: next(answers, True)
        with mock.patch.object(app.subprocess, "Popen", return_value=running_proc()) as popen:
            views = list(app.start_promptfoo())
        cmd, kw = popen.call_args.args[0], popen.call_args.kwargs
        self.assertEqual(cmd, [str(app.PROMPTFOO_BIN), "view", "--port", str(app.PROMPTFOO_PORT), "--no"])
        self.assertEqual(kw["cwd"], app.PROMPTFOO_DATA)
        self.assertTrue(kw["start_new_session"])
        self.assertEqual((kw["env"]["PROMPTFOO_DISABLE_TELEMETRY"], kw["env"]["PROMPTFOO_DISABLE_SHARING"]), ("1", "1"))
        self.assertIn("promptfoo_localhost.cjs", kw["env"]["NODE_OPTIONS"])
        self.assertEqual(kw["env"]["PROMPTFOO_CONFIG_DIR"], str(app.PROMPTFOO_DATA))
        self.assertIs(app._promptfoo["p"], popen.return_value)
        self.assertIn("running on this computer only", views[-1][0])

    def test_start_shows_the_log_tail_when_promptfoo_exits_at_once(self):
        self.install_promptfoo()

        def popen(cmd, **kw):
            kw["stdout"].write("starting\nloading\nlistening\nError: port 15500 in use\n")
            return running_proc(poll=1)
        with mock.patch.object(app.subprocess, "Popen", side_effect=popen):
            views = list(app.start_promptfoo())
        status, start, stop, link = views[-1]
        self.assertIn("promptfoo stopped right after starting: loading listening Error: port 15500 in use", status)
        self.assertEqual((start["visible"], stop["visible"], link), (True, False, ""))

    def test_stop_ends_the_whole_process_group(self):
        p = running_proc()
        app._promptfoo["p"] = p
        with mock.patch.object(app.os, "killpg") as kill:
            app.stop_promptfoo()
        kill.assert_called_once_with(4242, signal.SIGTERM)
        p.wait.assert_called_once_with(10)
        self.assertIsNone(app._promptfoo["p"])

    def test_stop_kills_promptfoo_when_it_ignores_the_request(self):
        p = running_proc()
        p.wait.side_effect = subprocess.TimeoutExpired("promptfoo", 10)
        app._promptfoo["p"] = p
        with mock.patch.object(app.os, "killpg") as kill:
            app.stop_promptfoo()
        self.assertEqual(kill.call_args_list, [mock.call(4242, signal.SIGTERM), mock.call(4242, signal.SIGKILL)])
        self.assertIsNone(app._promptfoo["p"])

    def test_targets_list_installed_models_as_escaped_provider_ids(self):
        with mock.patch.object(app, "model_names", return_value=["llama3:8b", "x<y>"]):
            out = app.promptfoo_targets_html()
        self.assertIn("<li><code>ollama:chat:llama3:8b</code></li>", out)
        self.assertIn("<code>ollama:chat:x&lt;y&gt;</code>", out)
        self.assertIn(f"<code>{app.OLLAMA}</code>", out)


class Pyrit(RedTeamBase):
    def make_export(self, home, name, mtime, conf, old_home="/home/old"):
        stage = self.tmp / f"stage-{name}/llm-scanner-data/pyrit-settings"
        (stage / "dot-pyrit").mkdir(parents=True)
        (stage / "pyrit").mkdir()
        (stage / "home.txt").write_text(old_home + "\n")
        (stage / "dot-pyrit/.pyrit_conf").write_text(conf)
        (stage / "pyrit/setup.py").write_text(f"# {old_home}/pyrit\n")
        archive = home / f"llm-scanner-data-{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(stage.parent, arcname="llm-scanner-data")
        os.utime(archive, (mtime, mtime))

    def test_pyrit_version_is_read_from_its_metadata_in_the_home_folder(self):
        home = self.tmp / "home"
        with mock.patch.object(app.Path, "home", return_value=home):
            self.assertEqual(app.pyrit_version(), "")
            meta = home / ".local/pyrit/lib/python3.12/site-packages/pyrit-1.1.0.dist-info/METADATA"
            meta.parent.mkdir(parents=True)
            meta.write_text("Metadata-Version: 2.1\nName: pyrit\nVersion: 1.1.0\nSummary: x\n")
            self.assertEqual(app.pyrit_version(), "1.1.0")

    def test_pyrit_html_names_the_installed_version(self):
        with mock.patch.object(app, "pyrit_version", return_value="1.1.0"):
            out = app.pyrit_html()
        self.assertIn("PyRIT 1.1.0's own web interface", out)
        self.assertIn(f'href="{app.PYRIT_DOCS}"', out)

    def test_start_when_missing_installs_in_the_background(self):
        with mock.patch.object(app.threading, "Thread") as thread, mock.patch.object(app.subprocess, "Popen") as popen:
            views = list(app.start_pyrit())
        self.assertEqual(thread.call_args.kwargs, {"target": app.install_red_team_tools, "kwargs": {"force": True},
                                                   "daemon": True})
        popen.assert_not_called()
        self.sleep.assert_called_once_with(0.5)
        self.assertEqual(views[-1][1]["value"], "Install PyRIT")

    def test_start_shows_the_log_tail_when_pyrit_exits_at_once(self):
        self.install_pyrit()

        def popen(cmd, **kw):
            kw["stdout"].write("Traceback\nImportError: no module named fastapi\n")
            return running_proc(poll=1)
        with mock.patch.object(app.subprocess, "Popen", side_effect=popen) as launched:
            views = list(app.start_pyrit())
        self.assertEqual(len(views), 1)
        self.assertIn("PyRIT stopped right after starting: Traceback ImportError: no module named fastapi", views[0][0])
        cmd = launched.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--port") + 1], str(app.PYRIT_PORT))

    def test_stop_kills_pyrit_when_it_ignores_the_request(self):
        p = running_proc(pid=777)
        p.wait.side_effect = subprocess.TimeoutExpired("pyrit", 15)
        app._pyrit["p"] = p
        with mock.patch.object(app.os, "killpg") as kill:
            app.stop_pyrit()
        self.assertEqual(kill.call_args_list, [mock.call(777, signal.SIGTERM), mock.call(777, signal.SIGKILL)])
        p.wait.assert_called_once_with(15)
        self.assertIsNone(app._pyrit["p"])

    def test_restore_uses_the_newest_export_and_this_home_folder(self):
        home = self.tmp / "home"
        home.mkdir()
        self.make_export(home, "20260101-0900", 1_000_000, "old: /home/old/a\n")
        self.make_export(home, "20261019-0900", 2_000_000, "new: /home/old/b\n")
        with mock.patch.object(app, "print", create=True):
            self.assertTrue(app.restore_pyrit_settings(home))
        self.assertEqual((home / ".pyrit/.pyrit_conf").read_text(), f"new: {home}/b\n")
        self.assertEqual((home / "pyrit/setup.py").read_text(), f"# {home}/pyrit\n")

    def test_restore_keeps_files_already_in_the_home_folder(self):
        home = self.tmp / "home"
        (home / "pyrit").mkdir(parents=True)
        (home / "pyrit/setup.py").write_text("mine")
        self.make_export(home, "20261019-0900", 1_000_000, f"conf: {home}\n", old_home=str(home))
        with mock.patch.object(app, "print", create=True):
            self.assertTrue(app.restore_pyrit_settings(home))
        self.assertEqual((home / "pyrit/setup.py").read_text(), "mine")
        self.assertEqual((home / ".pyrit/.pyrit_conf").read_text(), f"conf: {home}\n")


# ---------------------------------------------------------------- startup housekeeping
class LeftoverKinds(TempDataTest):
    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.object(app, "HF_DOWNLOADS", app.DATA_DIR / "hf-downloads"))

    def touch(self, path, data=b"x" * 10):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_partial_file_of_a_queued_download_is_kept(self):
        ref = "hf.co/x/queued:Q4_K_M"
        app._set_pending(ref, True)
        kept = self.touch(app.HF_DOWNLOADS / (app._slug(ref) + ".gguf.part"))
        gone = self.touch(app.HF_DOWNLOADS / "hf-co-x-other-q4-k-m.gguf.part")
        self.assertEqual(app.clean_leftovers(), (1, 10))
        self.assertTrue(kept.exists())
        self.assertFalse(gone.exists())

    def test_files_of_a_paused_download_are_kept(self):
        ref = "hf.co/x/paused:Q8_0"
        app._set_pending(ref, True, paused=True)
        kept = [self.touch(app.HF_DOWNLOADS / (app._slug(ref) + ".gguf")),
                self.touch(app.HF_DOWNLOADS / (app._slug(ref) + ".mmproj.gguf.part"))]
        self.assertEqual(app.clean_leftovers(), (0, 0))
        self.assertTrue(all(f.exists() for f in kept))

    def test_keep_backups_decides_how_many_versions_stay(self):
        for i, v in enumerate(("1.0.1", "1.0.2", "1.0.3")):
            folder = self.touch(app.DATA_DIR / f"backups/{v}/app.py").parent
            os.utime(folder, (1000 + i, 1000 + i))
        with mock.patch.object(app, "KEEP_BACKUPS", 1):
            files, size = app.clean_leftovers()
        self.assertEqual(sorted(p.name for p in (app.DATA_DIR / "backups").iterdir()), ["1.0.3"])
        self.assertEqual(files, 2)

    def test_old_ollama_partial_layers_nobody_owns_are_removed(self):
        blobs = app.OLLAMA_BLOBS
        old = [self.touch(blobs / "sha256-aaa-partial"), self.touch(blobs / "sha256-aaa-partial-0")]
        done = self.touch(blobs / "sha256-ccc")
        for f in (*old, done):
            os.utime(f, (1000, 1000))
        fresh = self.touch(blobs / "sha256-bbb-partial")
        files, _ = app.clean_leftovers()
        self.assertEqual(files, 2)
        self.assertFalse(any(f.exists() for f in old))
        self.assertTrue(fresh.exists() and done.exists(), "recent partials and finished layers stay")


def _leftover_case(where):
    def test(self):
        f = self.touch(where())
        self.assertEqual(app.clean_leftovers(), (1, 10))
        self.assertFalse(f.exists())
    return test


for _kind, _where in (("huggingface_hub_cache", lambda: app.DATA_DIR / ".hf-cache/models--a--b/blobs/x"),
                      ("attachment_staged_but_never_sent", lambda: app.ATTACH_DIR / "staging/abcd-notes.txt"),
                      ("stray_hugging_face_download", lambda: app.HF_DOWNLOADS / "hf-co-x-gone-q4.gguf.part"),
                      ("image_settings_whose_image_is_gone", lambda: app.IMAGES_DIR / "removed.json")):
    setattr(LeftoverKinds, f"test_removes_{_kind}", _leftover_case(_where))


class DesktopEntries(TempDataTest):
    def test_icons_are_installed_and_no_launchers_are_created(self):
        home = self.tmp / "home"
        home.mkdir()
        with mock.patch.object(app.Path, "home", return_value=home), \
                mock.patch.object(app.subprocess, "run", return_value=mock.Mock(stdout="")):
            app.ensure_desktop_integration()
        icons = home / ".local/share/icons/hicolor"
        self.assertEqual((icons / "scalable/apps/llm-scanner.svg").read_bytes(),
                         (app.APP_DIR / "static/icon.svg").read_bytes())
        self.assertEqual((icons / "256x256/apps/llm-scanner.png").read_bytes(),
                         (app.APP_DIR / "static/icon.png").read_bytes())
        self.assertFalse((home / ".local/share/applications").exists())
        self.assertFalse((home / "Desktop").exists())

    def test_autostart_and_desktop_entries_are_fixed_without_xdg_user_dir(self):
        home = self.tmp / "home"
        entries = [home / ".config/autostart/llm-scanner.desktop", home / "Desktop/llm-scanner.desktop"]
        for e in entries:
            e.parent.mkdir(parents=True)
            e.write_text("[Desktop Entry]\nName=LLM Scanner\nExec=/old/llm-scanner.sh --x\nIcon=/old/icon.png\n"
                         "Terminal=false\n")
        with mock.patch.object(app.Path, "home", return_value=home), \
                mock.patch.object(app.subprocess, "run", side_effect=FileNotFoundError("xdg-user-dir")):
            app.ensure_desktop_integration()
        for e in entries:
            self.assertEqual(e.read_text(), f"[Desktop Entry]\nName=LLM Scanner\nExec={app.APP_DIR / 'llm-scanner.sh'}\n"
                                            "Icon=llm-scanner\nTerminal=false\n")


class ModelListsRefresh(TempDataTest):
    def test_model_choices_changed_keeps_a_chat_locked_to_its_removed_model(self):
        chat = {"model": "gone:latest", "messages": [{"role": "user", "content": "hi"}]}
        with mock.patch.object(app, "model_names", return_value=["a", "b"]), \
                mock.patch.object(app, "scan_model_choices", return_value=([("a", "a"), ("b", "b")], "a")):
            scan_upd, chat_upd = app.model_choices_changed(chat)
        self.assertEqual(scan_upd, upd(choices=[("a", "a"), ("b", "b")]))
        self.assertEqual(chat_upd, upd(value="gone:latest", choices=["a", "b", "gone:latest"], interactive=False,
                                       label="Model (locked to this chat)"))

    def test_refresh_all_without_a_chat(self):
        choices = [("a", "a"), ("b   cannot load", "b")]
        with mock.patch.object(app, "model_names", return_value=["a", "b"]), \
                mock.patch.object(app, "scan_model_choices", return_value=(choices, "a")), \
                mock.patch.object(app, "status_html", return_value="S"), \
                mock.patch.object(app, "load_groups", return_value={"mine": ["dan"]}), \
                mock.patch.object(app.time, "time", return_value=123.0):
            stamp, scan_upd, status, scan_type, groups, chat_upd = app.refresh_all(None)
        self.assertEqual((stamp, status), (123.0, "S"))
        self.assertEqual(scan_upd, upd(choices=choices, value="a"))
        self.assertEqual(scan_type["choices"][-2:], [app.GROUP_PREFIX + "mine", app.CURRENT])
        self.assertIn("mine", groups["choices"])
        self.assertEqual(chat_upd, upd(choices=["a", "b"], value="a", interactive=True, label="Model"))


if __name__ == "__main__":
    unittest.main()
