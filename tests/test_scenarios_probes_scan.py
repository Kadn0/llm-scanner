"""Scenario tests for the Probes and Scan tabs: the probe catalog, time tags, presets and saved groups, filtering and
selecting probes, and the scan controls (validation, saved scan jobs, progress parsing, stuck probes, pause, stop,
resume and continue).

Run alone:  .venv/bin/python -m unittest discover -s tests -p test_scenarios_probes_scan.py
Nothing here runs garak or touches real data, Ollama or the network: data folders point at a temporary directory
(TempDataTest), subprocess.Popen is replaced so no garak process can start, and HTTP calls are faked. Scan state lives
in module globals, which every test restores to idle."""

import copy
import itertools
import json
import re
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_app import FakeResponse, TempDataTest, app  # noqa: E402

NOOP = app.gr.update()

FAKE_CATALOG = {
    "alpha.One": {"module": "alpha", "desc": "First alpha probe", "tier": 1, "active": True,
                  "tags": ["owasp:llm01", "avid-effect:security"], "goal": "make it say hi"},
    "alpha.Two": {"module": "alpha", "desc": "Second alpha probe", "tier": 2, "active": False,
                  "tags": ["owasp:llm05"], "goal": ""},
    "atkgen.Tox": {"module": "atkgen", "desc": "Attacker model writes prompts", "tier": 1, "active": True,
                   "tags": ["owasp:llm01"], "goal": "make toxic"},
    "beta.Four": {"module": "beta", "desc": "Fourth one, plain", "tier": 1, "active": True, "tags": [], "goal": ""},
    "beta.Three": {"module": "beta", "desc": "Hides text in an ENCODING", "tier": 1, "active": True,
                   "tags": ["euai:robustness", "owasp:llm02"], "goal": "decode it"},
    "gamma.Five": {"module": "gamma", "desc": "Gamma probe, off", "tier": 3, "active": False, "tags": [], "goal": ""},
}
ACTIVE = ["alpha.One", "atkgen.Tox", "beta.Four", "beta.Three"]
CHOOSE_GROUP = "Choose one of your saved groups to delete (presets can't be deleted)."

MODULE_SAMPLE = 8  # real garak modules checked one by one (the first ones in name order)
SAMPLED_MODULES = sorted(app.MODULES)[:MODULE_SAMPLE]


def _slug(text):
    return re.sub(r"\W+", "_", text.lower()).strip("_")


def _add_tests(cls, cases, make):
    """Attach one generated test per case: cases are (test name, *arguments for make)."""
    for name, *args in cases:
        test = make(*args)
        test.__name__ = name
        assert not hasattr(cls, name), name
        setattr(cls, name, test)


class FakeCatalogTest(TempDataTest):
    """A small, fixed probe catalog instead of the installed garak's."""

    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.dict(app.CATALOG, copy.deepcopy(FAKE_CATALOG), clear=True))
        self.enterContext(mock.patch.object(app, "MODULES", ["alpha", "atkgen", "beta", "gamma"]))


class ProbeCostBase(TempDataTest):
    def setUp(self):
        super().setUp()
        for name in ("PROBE_SIZES", "PROBE_REPLY_TOKENS", "PROBE_PROMPT_CHARS", "PROBE_TURNS", "PROBE_UNCAPPED"):
            self.enterContext(mock.patch.dict(getattr(app, name), clear=True))
        for name in ("PROBE_JUDGED", "PROBE_SCORING_MODELS"):
            self.enterContext(mock.patch.object(app, name, set()))
        self.enterContext(mock.patch.object(app, "PROMPT_CAP", [256]))


# ---------------------------------------------------------------- catalog
class CatalogLoading(TempDataTest):
    def setUp(self):
        super().setUp()
        self.home, self.pkg = self.tmp / "home", self.tmp / "pkg"
        self.enterContext(mock.patch.object(app.Path, "home", return_value=self.home))
        self.enterContext(mock.patch.object(app.importlib.util, "find_spec",
                                            return_value=types.SimpleNamespace(origin=str(self.pkg / "__init__.py"))))
        self.user_cache = self.home / ".cache/garak/resources/plugin_cache.json"
        self.pkg_cache = self.pkg / "resources/plugin_cache.json"

    def write(self, path, probes):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"probes": probes}))

    def test_load_catalog_without_any_plugin_cache_is_empty(self):
        self.assertEqual(app.load_catalog(), {})

    def test_load_catalog_strips_prefix_derives_module_and_sorts(self):
        self.write(self.pkg_cache, {"probes.zed.Last": {"description": "z"},
                                    "probes.abc.First": {"description": "a", "tier": 2, "active": False,
                                                         "tags": ["t"], "goal": "  g  "}})
        cat = app.load_catalog()
        self.assertEqual(list(cat), ["abc.First", "zed.Last"])
        self.assertEqual(cat["abc.First"], {"module": "abc", "desc": "a", "tier": 2, "active": False, "tags": ["t"],
                                            "goal": "g"})
        self.assertEqual(cat["zed.Last"], {"module": "zed", "desc": "z", "tier": None, "active": True, "tags": [],
                                           "goal": ""})

    def test_load_catalog_skips_probes_that_take_no_text(self):
        self.write(self.pkg_cache, {"probes.img.Only": {"modality": {"in": ["image"]}},
                                    "probes.mix.Both": {"modality": {"in": ["text", "image"]}},
                                    "probes.none.Given": {}, "probes.nolist.In": {"modality": {}},
                                    "probes.empty.In": {"modality": {"in": []}}})
        self.assertEqual(list(app.load_catalog()), ["mix.Both", "nolist.In", "none.Given"])

    def test_load_catalog_keeps_first_line_of_description_and_tolerates_none(self):
        self.write(self.pkg_cache, {"probes.a.B": {"description": "  Line one\nLine two  "},
                                    "probes.a.C": {"description": None, "goal": None}})
        cat = app.load_catalog()
        self.assertEqual(cat["a.B"]["desc"], "Line one")
        self.assertEqual((cat["a.C"]["desc"], cat["a.C"]["goal"]), ("", ""))

    def test_load_catalog_prefers_user_cache_over_package_cache(self):
        self.write(self.user_cache, {"probes.user.Probe": {}})
        self.write(self.pkg_cache, {"probes.pkg.Probe": {}})
        self.assertEqual(list(app.load_catalog()), ["user.Probe"])


class RealCatalog(unittest.TestCase):
    def test_real_catalog_entries_are_well_formed_and_modules_sampled(self):
        self.assertEqual(list(app.CATALOG), sorted(app.CATALOG))
        for n, v in app.CATALOG.items():
            self.assertFalse(n.startswith("probes."), n)
            self.assertTrue(n.startswith(v["module"] + "."), n)
            self.assertEqual(set(v), {"module", "desc", "tier", "active", "tags", "goal"}, n)
            self.assertNotIn("\n", v["desc"], n)
            self.assertIn(v["module"], app.MODULES)
        self.assertEqual(app.MODULES, sorted(set(app.MODULES)))
        self.assertGreaterEqual(len(app.MODULES), MODULE_SAMPLE, "the per-module tests need this many families")
        self.assertEqual(len(SAMPLED_MODULES), MODULE_SAMPLE)


# ---------------------------------------------------------------- probe_tip and probe_label
class ProbeTipAndLabel(FakeCatalogTest):
    def test_probe_tip_joins_family_description_and_capitalised_goal(self):
        with mock.patch.dict(app.MODULE_INFO, {"alpha": "Alpha family."}):
            self.assertEqual(app.probe_tip("alpha.One"), "Alpha family.\nFirst alpha probe\nGoal: Make it say hi.")

    def test_probe_tip_inactive_probe_without_goal_says_off_by_default(self):
        with mock.patch.dict(app.MODULE_INFO, {"alpha": "Alpha family."}):
            self.assertEqual(app.probe_tip("alpha.Two"), "Alpha family.\nSecond alpha probe\nOff by default in garak.")

    def test_probe_tip_unknown_family_and_empty_description_leave_no_blank_lines(self):
        app.CATALOG["gamma.Five"]["desc"] = ""
        with mock.patch.dict(app.MODULE_INFO):
            app.MODULE_INFO.pop("gamma", None)
            self.assertEqual(app.probe_tip("gamma.Five"), "Off by default in garak.")

    def test_probe_label_keeps_a_110_character_description_whole(self):
        app.CATALOG["beta.Four"]["desc"] = "x" * 110
        self.assertEqual(app.probe_label("beta.Four"), "beta.Four   " + "x" * 110)

    def test_probe_label_truncates_a_111_character_description(self):
        app.CATALOG["beta.Four"]["desc"] = "y" * 111
        self.assertEqual(app.probe_label("beta.Four"), "beta.Four   " + "y" * 107 + "...")

    def test_probe_label_truncation_drops_trailing_space_before_ellipsis(self):
        app.CATALOG["beta.Four"]["desc"] = "a" * 106 + " " + "b" * 10
        self.assertEqual(app.probe_label("beta.Four"), "beta.Four   " + "a" * 106 + "...")

    def test_probe_label_marks_inactive_probe_and_rejects_unknown(self):
        self.assertEqual(app.probe_label("alpha.Two"), "alpha.Two   Second alpha probe   [off by default]")
        with self.assertRaises(KeyError):
            app.probe_label("nope.X")


# ---------------------------------------------------------------- probe_work / probe_costs
class ProbeWorkAndCosts(ProbeCostBase):
    def test_probe_work_of_unmeasured_probe_is_zero(self):
        self.assertEqual(app.probe_work("never.Measured"), 0)

    def test_probe_work_long_prompt_adds_reading_time(self):
        app.PROBE_SIZES["x.Long"] = 155
        app.PROBE_PROMPT_CHARS["x.Long"] = 1200  # 300 tokens to read, a tenth the cost of writing: +30 per prompt
        self.assertAlmostEqual(app.probe_work("x.Long"), 180.0)

    def test_probe_work_short_prompt_never_counts_below_one_prompt(self):
        app.PROBE_SIZES["x.Short"] = 40
        app.PROBE_PROMPT_CHARS["x.Short"] = 4
        self.assertEqual(app.probe_work("x.Short"), 40)

    def test_probe_work_zero_reply_limit_falls_back_to_garak_default(self):
        app.PROBE_SIZES["x.Zero"] = 10
        app.PROBE_REPLY_TOKENS["x.Zero"] = 0
        self.assertEqual(app.probe_work("x.Zero"), 10)

    def test_probe_work_fractional_turns_scale_and_below_one_count_as_one(self):
        app.PROBE_SIZES.update({"x.Half": 10, "x.Low": 10})
        app.PROBE_TURNS.update({"x.Half": 1.5, "x.Low": 0.5})
        self.assertEqual(app.probe_work("x.Half"), 15)
        self.assertEqual(app.probe_work("x.Low"), 10)

    def test_probe_costs_generated_probe_is_very_high_with_zero_work(self):
        app.PROBE_SIZES["zz.Generated"] = 0
        c = app.probe_costs()["zz.Generated"]
        self.assertEqual((c["level"], c["generated"], c["files"], c["work"], c["prompts"]),
                         ("Very high", True, False, 0, 0))

    def test_probe_costs_tokenizer_probe_inspects_files_only(self):
        app.PROBE_SIZES["ansiescape.AnsiRawTokenizerHF"] = 0
        c = app.probe_costs()["ansiescape.AnsiRawTokenizerHF"]
        self.assertEqual((c["level"], c["files"], c["generated"]), ("Low", True, False))

    def test_probe_costs_fileformats_probe_with_prompts_is_rated_by_work(self):
        app.PROBE_SIZES["fileformats.Sized"] = 40
        with mock.patch.dict(app.CATALOG, {"fileformats.Sized": {"module": "fileformats"}}):
            c = app.probe_costs()["fileformats.Sized"]
        self.assertEqual((c["level"], c["files"], c["generated"]), ("Medium", False, False))

    def test_probe_costs_judging_doubles_work_into_next_level(self):
        app.PROBE_SIZES.update({"x.Judged": 20, "x.Plain": 20})
        app.PROBE_JUDGED.add("x.Judged")
        costs = app.probe_costs()
        self.assertEqual((costs["x.Judged"]["level"], costs["x.Judged"]["work"], costs["x.Judged"]["judged"]),
                         ("Medium", 40, True))
        self.assertEqual((costs["x.Plain"]["level"], costs["x.Plain"]["work"], costs["x.Plain"]["judged"]),
                         ("Low", 20, False))

    def test_probe_costs_reports_custom_prompt_cap_uncapped_and_zero_defaults(self):
        app.PROMPT_CAP[0] = 100
        app.PROBE_SIZES["x.C"] = 100
        app.PROBE_UNCAPPED["x.C"] = 999
        c = app.probe_costs()["x.C"]
        self.assertEqual((c["cap"], c["uncapped"], c["reply_tokens"], c["prompt_chars"], c["downloads"]),
                         (100, 999, 0, 0, False))


class ProbeLevels(ProbeCostBase):
    """One test per prompt count around the Low / Medium / High / Very high / Extreme boundaries (30/150/500/2000)."""


def _make_level_test(n, level):
    def test(self):
        name = f"plain.P{n}"
        app.PROBE_SIZES[name] = n
        self.assertEqual(app.probe_work(name), n, "a plain probe costs one normal prompt per prompt")
        self.assertEqual(app.probe_costs()[name]["level"], level)
    return test


_add_tests(ProbeLevels, [
    (f"test_level_for_{n}_plain_prompts_is_{_slug(level)}", n, level) for n, level in [
        (1, "Low"), (29, "Low"), (30, "Low"), (45, "Medium"), (150, "Medium"), (152, "High"), (499, "High"),
        (500, "High"), (502, "Very high"), (2000, "Very high"), (2002, "Extreme"), (50000, "Extreme")]],
    _make_level_test)

# ---------------------------------------------------------------- OWASP, _probes_for, expand
class PresetBuilding(FakeCatalogTest):
    def test_owasp_preset_takes_active_owasp_tagged_probes_including_attacker_ones(self):
        self.assertEqual(app.owasp_preset()[0], ["alpha.One", "atkgen.Tox", "beta.Three"])

    def test_owasp_preset_description_names_covered_and_missing_categories(self):
        self.assertEqual(app.owasp_preset()[1],
                         "Every garak probe mapped to the OWASP Top 10 for LLM Applications (LLM01, LLM02). garak has "
                         "no tests for LLM03, LLM04, LLM05, LLM06, LLM07, LLM08, LLM09, LLM10 on a standalone model "
                         "(they concern training data and tool-using apps). Thorough; expect several hours.")

    def test_owasp_preset_with_every_category_has_no_missing_sentence(self):
        app.CATALOG["beta.Four"]["tags"] = [f"owasp:llm{i:02d}" for i in range(1, 11)]
        probes, desc = app.owasp_preset()
        self.assertIn("beta.Four", probes)
        self.assertNotIn("garak has no tests", desc)
        self.assertIn("(LLM01, LLM02, LLM03, LLM04, LLM05, LLM06, LLM07, LLM08, LLM09, LLM10)", desc)
        self.assertTrue(desc.endswith("Thorough; expect several hours."))

    def test_probes_for_tag_prefix_matches_only_active_non_attacker_probes(self):
        self.assertEqual(app._probes_for(tags=("owasp:",)), ["alpha.One", "beta.Three"])

    def test_probes_for_never_includes_attacker_or_inactive_probes_even_by_name(self):
        self.assertEqual(app._probes_for(modules=("atkgen",), probes=("atkgen.Tox", "alpha.Two")), [])

    def test_probes_for_without_criteria_selects_nothing(self):
        self.assertEqual(app._probes_for(), [])

    def test_expand_keeps_inactive_probe_named_in_full(self):
        self.assertEqual(app.expand(["alpha.Two"]), ["alpha.Two"])

    def test_expand_follows_spec_order_and_dedupes_module_members(self):
        self.assertEqual(app.expand(["beta", "alpha.One", "alpha", "beta.Three"]),
                         ["beta.Four", "beta.Three", "alpha.One"])

    def test_expand_module_without_active_probes_and_wrong_case_give_nothing(self):
        self.assertEqual(app.expand(["gamma", "ALPHA", "Alpha.One"]), [])


# ---------------------------------------------------------------- PRESETS (installed garak)
class PresetResolution(TempDataTest):
    """Every preset named in app.py resolves to real probes (the full scan to garak's defaults)."""


PRESET_NAMES = ["Quick check", "Jailbreaks and prompt injection", "Harmful content and toxicity",
                "Data leakage and malware", "Encoding and obfuscation", "OWASP Top 10 for LLMs", "Baseline assessment",
                "NIST AI 600-1: Information security", "NIST AI 600-1: Information integrity and confabulation",
                "NIST AI 600-1: Dangerous, violent and hateful content",
                "NIST AI 600-1: Data privacy and intellectual property", "NIST AI 600-1: Harmful bias",
                "EU AI Act risk areas", "Full scan"]


def _make_preset_test(name):
    def test(self):
        self.assertIn(name, app.PRESETS)
        specs, desc = app.PRESETS[name]
        probes, err = app.resolve_scan(name, ["ignored.Probe"])
        self.assertIsNone(err)
        self.assertTrue(desc.strip())
        self.assertIn(name, app.scan_choices())
        if name == "Full scan":
            self.assertEqual(probes, [], "an empty list runs garak's full default scan")
            self.assertNotIn("Preset: Full scan", app.group_choices())
            return
        self.assertTrue(probes)
        self.assertEqual(len(probes), len(set(probes)))
        self.assertTrue(all(p in app.CATALOG for p in probes), name)
        self.assertNotIn("ignored.Probe", probes, "a preset ignores the Probes tab selection")
        self.assertIn(f"Preset: {name}", app.group_choices())
        self.assertEqual(app.load_group(f"Preset: {name}", "", None, True)[0],
                         [n for n in app.CATALOG if n in set(probes)])
    return test


_add_tests(PresetResolution, [(f"test_preset_{_slug(n)}_resolves_to_real_probes", n) for n in PRESET_NAMES],
           _make_preset_test)


class RealPresetsAndClashes(TempDataTest):
    def setUp(self):
        super().setUp()
        quick = set(app.expand(app.PRESETS["Quick check"][0]))
        self.probe = next(n for n, v in app.CATALOG.items() if v["active"] and n not in quick)

    def test_module_preset_expands_to_every_active_probe_of_its_modules(self):
        probes, _ = app.resolve_scan("Encoding and obfuscation", [])
        expected = {n for n, v in app.CATALOG.items() if v["module"] in ("encoding", "ansiescape") and v["active"]}
        self.assertEqual(set(probes), expected)

    def test_saved_group_named_like_preset_does_not_replace_preset(self):
        msg = app.save_group("Quick check", [self.probe])[0]
        self.assertTrue(msg.startswith('Saved group "Quick check" with 1 probes.'))
        self.assertIn("Preset: Quick check", app.group_choices())
        self.assertIn("Quick check", app.group_choices())
        preset_probes, _ = app.resolve_scan("Quick check", [])
        self.assertEqual(preset_probes, app.expand(app.PRESETS["Quick check"][0]))
        self.assertNotIn(self.probe, preset_probes)
        self.assertEqual(app.resolve_scan(app.GROUP_PREFIX + "Quick check", []), ([self.probe], None))
        self.assertEqual(app.load_group("Quick check", "", None, True)[0], [self.probe])

    def test_saved_group_with_preset_prefix_stays_reachable_and_deletable(self):
        app.save_group("Preset: Quick check", [self.probe])
        self.assertEqual(app.resolve_scan(app.GROUP_PREFIX + "Preset: Quick check", []), ([self.probe], None))
        self.assertEqual(app.delete_group("Preset: Quick check")[0], 'Deleted group "Preset: Quick check".')
        self.assertIn("Quick check", app.PRESETS)


# ---------------------------------------------------------------- groups file
class GroupsFile(FakeCatalogTest):
    def test_load_groups_missing_file_is_empty(self):
        self.assertFalse(app.GROUPS_FILE.exists())
        self.assertEqual(app.load_groups(), {})

    def test_load_groups_corrupt_json_is_empty_and_menus_still_build(self):
        app.GROUPS_FILE.write_text('{"g": [')
        self.assertEqual(app.load_groups(), {})
        self.assertTrue(all(c.startswith("Preset: ") for c in app.group_choices()))
        self.assertFalse([c for c in app.scan_choices() if c.startswith(app.GROUP_PREFIX)])
        self.assertEqual(app.scan_choices()[-1], app.CURRENT)

    def test_save_groups_writes_sorted_indented_json_that_round_trips(self):
        app.save_groups({"b": ["x"], "a": ["y"]})
        text = app.GROUPS_FILE.read_text()
        self.assertLess(text.index('"a"'), text.index('"b"'))
        self.assertIn('\n  "a"', text)
        self.assertEqual(app.load_groups(), {"a": ["y"], "b": ["x"]})

    def test_save_group_survives_groups_file_holding_a_json_list(self):
        # BUG: load_groups() only guards against unreadable JSON; a groups file holding valid non-dict JSON (here a
        # list) is returned as-is, and save_group then crashes with TypeError on groups[name] = ...
        app.GROUPS_FILE.write_text("[]")
        self.assertTrue(app.save_group("g", ["alpha.One"])[0].startswith('Saved group "g"'))
        self.assertEqual(app.load_groups(), {"g": ["alpha.One"]})


# ---------------------------------------------------------------- filtering and selecting
class FilteringProbes(FakeCatalogTest):
    def test_filtered_probes_hides_inactive_unless_show_all(self):
        self.assertEqual(app.filtered_probes("", "All categories", False), ACTIVE)
        self.assertEqual(app.filtered_probes("", "All categories", True), list(FAKE_CATALOG))

    def test_filtered_probes_module_with_show_all_false_and_true(self):
        self.assertEqual(app.filtered_probes("", "alpha", False), ["alpha.One"])
        self.assertEqual(app.filtered_probes("", "alpha", True), ["alpha.One", "alpha.Two"])

    def test_filtered_probes_query_ignores_case(self):
        self.assertEqual(app.filtered_probes("ALPHA.ONE", None, True), ["alpha.One"])

    def test_filtered_probes_query_matches_description_text(self):
        self.assertEqual(app.filtered_probes("encoding", "", False), ["beta.Three"])

    def test_filtered_probes_partial_query(self):
        self.assertEqual(app.filtered_probes("thr", None, False), ["beta.Three"])

    def test_filtered_probes_query_without_match_is_empty(self):
        self.assertEqual(app.filtered_probes("zzz", None, True), [])
        self.assertEqual(app.filtered_probes("thr", "alpha", True), [], "the query matches only in another family")

    def test_filtered_probes_whitespace_query_is_trimmed_or_ignored(self):
        self.assertEqual(app.filtered_probes("   ", None, False), ACTIVE)
        self.assertEqual(app.filtered_probes("  four  ", None, False), ["beta.Four"])

    def test_filter_probes_offers_labels_and_keeps_only_visible_ticks(self):
        upd, msg = app.filter_probes("", "beta", False, ["alpha.One", "beta.Three"])
        self.assertEqual(upd["choices"], [(app.probe_label("beta.Four"), "beta.Four"),
                                          (app.probe_label("beta.Three"), "beta.Three")])
        self.assertEqual(upd["value"], ["beta.Three"])
        self.assertEqual(msg, "Showing 2 of 6 probes.")

    def test_filter_probes_with_no_matches_reports_zero(self):
        upd, msg = app.filter_probes("zzz", None, True, ["alpha.One"])
        self.assertEqual((upd["choices"], upd["value"], msg), ([], [], "Showing 0 of 6 probes."))

    def test_selection_summary_none_means_nothing_selected(self):
        self.assertEqual(app.selection_summary(None), "No probes selected.")

    def test_selection_summary_lists_exactly_eight_without_more(self):
        self.assertEqual(app.selection_summary([f"p{i}" for i in range(8)]),
                         "**8 selected:** p0, p1, p2, p3, p4, p5, p6, p7")

    def test_selection_summary_nine_shows_eight_and_one_more(self):
        self.assertEqual(app.selection_summary([f"p{i}" for i in range(9)]),
                         "**9 selected:** p0, p1, p2, p3, p4, p5, p6, p7, and 1 more")

    def test_ticking_in_filtered_view_keeps_hidden_selection(self):
        self.assertEqual(app.probes_ticked(["beta.Three"], "", "beta", False, ["alpha.One"]),
                         (["alpha.One", "beta.Three"], "**2 selected:** alpha.One, beta.Three"))

    def test_unticking_visible_probe_removes_only_it(self):
        new, _ = app.probes_ticked([], "", "beta", False, ["alpha.One", "beta.Three", "beta.Four"])
        self.assertEqual(new, ["alpha.One"])

    def test_hidden_inactive_selection_survives_when_inactive_are_hidden(self):
        new, _ = app.probes_ticked(["alpha.One"], "", "All categories", False, ["alpha.Two"])
        self.assertEqual(new, ["alpha.One", "alpha.Two"])

    def test_ticked_none_and_names_not_in_catalog_are_dropped(self):
        self.assertEqual(app.probes_ticked(None, "", "beta", False, ["ghost.X", "alpha.One"]),
                         (["alpha.One"], "**1 selected:** alpha.One"))

    def test_select_shown_adds_visible_probes_to_hidden_selection(self):
        new, upd, summary = app.select_shown("", "beta", False, ["alpha.One"])
        self.assertEqual(new, ["alpha.One", "beta.Four", "beta.Three"])
        self.assertEqual(upd["value"], ["beta.Four", "beta.Three"])
        self.assertEqual(summary, "**3 selected:** alpha.One, beta.Four, beta.Three")
        new, upd, summary = app.select_shown("zzz", None, False, ["alpha.One"])
        self.assertEqual((new, upd["value"], summary), (["alpha.One"], [], "**1 selected:** alpha.One"))

    def test_clear_selection_empties_list_ticks_and_summary(self):
        new, upd, summary = app.clear_selection()
        self.assertEqual((new, upd["value"], summary), ([], [], "No probes selected."))


class RealModuleFamilies(unittest.TestCase):
    """For real garak families: the family filter, expand(), the count message, the family card and the hover text
    agree with one another."""


def _make_module_test(m):
    def test(self):
        active = app.filtered_probes("", m, False)
        every = app.filtered_probes(None, m, True)
        self.assertTrue(every, m)
        self.assertEqual(every, [n for n in app.CATALOG if n.startswith(m + ".")])
        self.assertEqual(active, app.expand([m]), "the family filter and expand() must pick the same probes")
        self.assertTrue(set(active) <= set(every))
        self.assertTrue(all(app.CATALOG[n]["active"] for n in active))
        self.assertEqual(app.filter_probes("", m, True, [])[1], f"Showing {len(every)} of {len(app.CATALOG)} probes.")
        self.assertIn(f"{len(every)} probe{'s' if len(every) != 1 else ''} in 1 family", app.probes_html(every, "d"))
        for n in every:
            self.assertTrue(app.probe_tip(n).startswith(app.MODULE_INFO[m]), n)
    return test


_add_tests(RealModuleFamilies, [(f"test_module_{_slug(m)}_filter_expand_and_info_agree", m) for m in SAMPLED_MODULES],
           _make_module_test)

# ---------------------------------------------------------------- group menus, loading, saving, deleting
class GroupMenus(FakeCatalogTest):
    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.dict(app.PRESETS, {"Quick": (["alpha"], "Quick desc"),
                                                        "Empty": ([], "Everything")}, clear=True))

    def test_group_choices_skip_empty_presets_and_sort_saved_groups(self):
        app.save_groups({"zeta": ["alpha.One"], "Alpha": ["beta.Four"]})
        self.assertEqual(app.group_choices(), ["Preset: Quick", "Alpha", "zeta"])

    def test_scan_choices_lists_presets_groups_then_current(self):
        app.save_groups({"zeta": ["alpha.One"], "Alpha": ["beta.Four"]})
        self.assertEqual(app.scan_choices(), ["Quick", "Empty", "Saved group: Alpha", "Saved group: zeta", app.CURRENT])

    def test_load_group_without_name_asks_for_one(self):
        self.assertEqual(app.load_group("", "", None, False), (NOOP, NOOP, "Choose a group to load.", NOOP))

    def test_load_group_preset_clears_group_name_box(self):
        new, ticks, summary, title = app.load_group("Preset: Quick", "", None, False)
        self.assertEqual((new, ticks["value"], summary, title["value"]),
                         (["alpha.One"], ["alpha.One"], "**1 selected:** alpha.One", ""))

    def test_load_group_saved_group_fills_name_and_respects_filter(self):
        app.save_groups({"mine": ["beta.Three", "alpha.One", "removed.Probe"]})
        new, ticks, summary, title = app.load_group("mine", "", "beta", False)
        self.assertEqual(new, ["alpha.One", "beta.Three"], "catalog order; probes garak no longer has are dropped")
        self.assertEqual(ticks["value"], ["beta.Three"], "only the probes in the current view are ticked")
        self.assertEqual((summary, title["value"]), ("**2 selected:** alpha.One, beta.Three", "mine"))

    def test_load_group_unknown_saved_group_selects_nothing(self):
        new, ticks, summary, title = app.load_group("ghost", "", None, True)
        self.assertEqual((new, ticks["value"], summary, title["value"]), ([], [], "No probes selected.", "ghost"))

    def test_group_info_plain_name_is_a_saved_group_even_if_it_matches_a_preset(self):
        self.assertEqual(app.group_info("Quick"),
                         '<div class="probe-info"><p class="pi-desc">That group no longer exists.</p></div>')
        out = app.group_info("Preset: Quick")
        self.assertIn("Quick desc", out)
        self.assertIn("1 probe in 1 family", out)

    def test_scan_info_full_scan_lists_every_active_probe(self):
        out = app.scan_info("Empty", [])
        self.assertIn("Everything", out)
        self.assertIn("4 probes in 3 families", out)
        self.assertNotIn('data-tip-title="alpha.Two"', out)


class SavingAndDeletingGroups(FakeCatalogTest):
    def test_save_group_rejects_tab_and_newline_only_or_missing_name(self):
        for name in ("\t\n ", None):
            self.assertEqual(app.save_group(name, ["alpha.One"])[0], "Enter a name for the group.")
        self.assertFalse(app.GROUPS_FILE.exists())

    def test_save_group_strips_name_and_updates_both_menus(self):
        msg, groups, scans = app.save_group("  mine  ", ["alpha.One", "beta.Three"])
        self.assertEqual(msg, 'Saved group "mine" with 2 probes. It is now available on the Scan tab.')
        self.assertEqual(app.load_groups(), {"mine": ["alpha.One", "beta.Three"]})
        self.assertEqual(groups["value"], "mine")
        self.assertIn("mine", groups["choices"])
        self.assertIn("Saved group: mine", scans["choices"])

    def test_save_group_unicode_name_round_trips(self):
        name = "Prüfung 🔒 日本"
        app.save_group(name, ["alpha.One"])
        self.assertEqual(app.load_groups(), {name: ["alpha.One"]})
        self.assertEqual(app.resolve_scan(app.GROUP_PREFIX + name, []), (["alpha.One"], None))

    def test_save_group_very_long_name(self):
        name = "g" * 5000
        self.assertTrue(app.save_group(name, ["beta.Four"])[0].startswith('Saved group "ggg'))
        self.assertEqual(app.load_groups()[name], ["beta.Four"])

    def test_save_group_overwrite_replaces_probes_and_keeps_other_groups(self):
        app.save_group("g", ["alpha.One"])
        app.save_group("h", ["gamma.Five"])
        msg = app.save_group("g", ["beta.Four", "beta.Three"])[0]
        self.assertEqual(msg, 'Updated group "g" with 2 probes. It is now available on the Scan tab.')
        self.assertEqual(app.load_groups(), {"g": ["beta.Four", "beta.Three"], "h": ["gamma.Five"]})

    def test_save_group_without_selection_writes_nothing(self):
        for selected in (None, []):
            self.assertEqual(app.save_group("g", selected)[0], "Select at least one probe before saving.")
        self.assertFalse(app.GROUPS_FILE.exists())

    def test_delete_group_refuses_unknown_empty_and_preset_names(self):
        app.save_groups({"g": ["alpha.One"]})
        for name in ("ghost", None, "", "Preset: Quick check", "Quick check"):
            self.assertEqual(app.delete_group(name)[0], CHOOSE_GROUP, name)
        self.assertEqual(app.load_groups(), {"g": ["alpha.One"]})
        self.assertIn("Quick check", app.PRESETS)

    def test_delete_group_removes_it_and_resets_menus(self):
        app.save_groups({"a": ["alpha.One"], "b": ["beta.Four"]})
        msg, groups, scans = app.delete_group("a")
        self.assertEqual(msg, 'Deleted group "a".')
        self.assertEqual(app.load_groups(), {"b": ["beta.Four"]})
        self.assertIsNone(groups["value"])
        self.assertIn("b", groups["choices"])
        self.assertNotIn("a", groups["choices"])
        self.assertEqual(scans["value"], "Quick check")
        self.assertNotIn("Saved group: a", scans["choices"])
        self.assertIn("Saved group: b", scans["choices"])

    def test_delete_group_with_corrupt_groups_file_is_refused_gracefully(self):
        app.GROUPS_FILE.write_text("{oops")
        self.assertEqual(app.delete_group("a")[0], CHOOSE_GROUP)
        self.assertEqual(app.GROUPS_FILE.read_text(), "{oops", "a corrupt file is not overwritten")


# ---------------------------------------------------------------- resolve_scan, probes_html, scan_info
class ResolvingAndDescribingScans(FakeCatalogTest):
    def test_resolve_scan_current_selection_is_not_expanded(self):
        self.assertEqual(app.resolve_scan(app.CURRENT, ["beta", "alpha.One"]), (["beta", "alpha.One"], None))

    def test_resolve_scan_saved_group_expands_modules(self):
        app.save_groups({"g": ["beta", "alpha.One"]})
        self.assertEqual(app.resolve_scan(app.GROUP_PREFIX + "g", []), (["beta.Four", "beta.Three", "alpha.One"], None))

    def test_saved_group_whose_probes_all_vanished_is_not_a_full_scan(self):
        # BUG: a saved group whose probes no longer exist in garak (renamed or removed) expands to [], and
        # resolve_scan returns ([], None). An empty list means "garak's full default scan", so run_scan starts a
        # many-hour full scan instead of telling the user the group has nothing left to run.
        app.save_groups({"old": ["removed.Probe"]})
        probes, err = app.resolve_scan(app.GROUP_PREFIX + "old", [])
        self.assertIsNotNone(err)

    def test_probes_html_singular_wording_and_chip_text(self):
        out = app.probes_html(["alpha.One"], "d")
        self.assertIn("1 probe in 1 family", out)
        self.assertIn('data-tip-title="alpha.One"', out)
        self.assertIn(">One</span>", out)

    def test_probes_html_unknown_and_dotless_names_grouped_by_prefix(self):
        out = app.probes_html(["mystery.Thing", "solo"], "<d>")
        self.assertIn("2 probes in 2 families", out)
        self.assertIn('data-tip-title="mystery" data-tip="">mystery</span>', out)
        self.assertIn(">Thing</span>", out)
        self.assertIn('data-tip-title="solo"', out)
        self.assertIn("&lt;d&gt;", out)

    def test_scan_info_saved_group_description_is_escaped(self):
        app.save_groups({"a<b": ["alpha.One"]})
        self.assertIn("Your saved group &quot;a&lt;b&quot;.", app.scan_info(app.GROUP_PREFIX + "a<b", []))

    def test_scan_info_unknown_type_shows_choose_message(self):
        self.assertEqual(app.scan_info("Mystery", []),
                         '<div class="probe-info"><p class="pi-desc">Choose a scan type.</p></div>')

    def test_group_info_empty_and_unknown_saved_group(self):
        self.assertEqual(app.group_info(""), "")
        self.assertIn("That group no longer exists.", app.group_info("ghost"))

# ---------------------------------------------------------------- Scan tab: model info and hints
class ModelInfoAndHint(unittest.TestCase):
    def test_model_info_size_in_gb_and_capabilities(self):
        models = [{"name": "other", "size": 1}, {"name": "m", "size": 4_500_000_000}]
        with mock.patch.object(app, "list_models", return_value=models), \
                mock.patch.object(app.requests, "post",
                                  return_value=FakeResponse({"capabilities": ["completion", "thinking"]})) as post:
            size, caps = app.model_info("m")
        self.assertAlmostEqual(size, 4.5)
        self.assertEqual(caps, ["completion", "thinking"])
        self.assertEqual(post.call_args.kwargs["json"], {"model": "m"})

    def test_model_info_unknown_model_and_unreachable_ollama(self):
        with mock.patch.object(app, "list_models", return_value=[]), \
                mock.patch.object(app.requests, "post", side_effect=app.requests.ConnectionError("down")):
            self.assertEqual(app.model_info("m"), (0, []))
        with mock.patch.object(app, "list_models", return_value=[{"name": "m"}]), \
                mock.patch.object(app.requests, "post", return_value=FakeResponse({})):
            self.assertEqual(app.model_info("m"), (0, []), "no size listed and no capabilities in the answer")

    def test_model_hint_without_model_changes_nothing(self):
        with mock.patch.object(app, "list_models") as models:
            self.assertEqual(app.model_hint(""), ("", NOOP))
            self.assertEqual(app.model_hint(None), ("", NOOP))
        models.assert_not_called()

    def test_model_hint_refuses_model_ollama_cannot_load(self):
        with mock.patch.object(app, "list_models", return_value=[{"name": "m", "digest": "d"}]), \
                mock.patch.object(app, "load_problem", return_value="unsupported tensor type"), \
                mock.patch.object(app, "model_info") as info:
            text, upd = app.model_hint("m")
        info.assert_not_called()
        self.assertEqual(text, "**This model can't run, so it can't be scanned.** Ollama can't load it (unsupported "
                               "tensor type). Models in formats like PQ2_0 or PTQ1_0 need the llama.cpp build from "
                               "the model's authors.")
        self.assertEqual(upd, NOOP)

    def test_model_hint_mentions_reasoning_models(self):
        with mock.patch.object(app, "list_models", return_value=[]), \
                mock.patch.object(app, "model_info", return_value=(4.0, ["completion", "thinking"])), \
                mock.patch.object(app, "gpu_live", return_value=("RTX", 1.0, 1.0, 12.0, 40.0, 50.0)):
            text, upd = app.model_hint("m")
        self.assertEqual(text, "4.0 GB model, runs fully on the GPU. This is a reasoning model; thinking is turned off "
                               "during scans unless you enable it below.")
        self.assertEqual(upd["value"], 300)

    def test_model_hint_without_gpu_runs_from_system_memory(self):
        with mock.patch.object(app, "list_models", return_value=[]), \
                mock.patch.object(app, "model_info", return_value=(8.0, [])), \
                mock.patch.object(app, "gpu_live", return_value=None):
            text, upd = app.model_hint("m")
        self.assertEqual(text, "8.0 GB model, about 0% on the GPU and the rest in system memory. It will run, but "
                               "responses are slower, so the timeout has been raised.")
        self.assertEqual(upd["value"], 900)


class ModelHintTimeouts(unittest.TestCase):
    """The timeout picked for a model by its size against the GPU's memory: 300 s when it fits in 85% of it, 900 s
    when part runs from system memory, 1800 s from 25 GB."""


def _make_hint_test(size, vram, prefix, timeout):
    def test(self):
        with mock.patch.object(app, "list_models", return_value=[]), \
                mock.patch.object(app, "model_info", return_value=(size, [])), \
                mock.patch.object(app, "gpu_live", return_value=("GPU", 0.0, 1.0, vram, 40.0, 50.0)):
            text, upd = app.model_hint("m")
        self.assertTrue(text.startswith(prefix), text)
        self.assertEqual(upd["value"], timeout)
    return test


_add_tests(ModelHintTimeouts, [
    ("test_hint_two_gb_on_eight_gb_gpu_runs_fully", 2.0, 8.0, "2.0 GB model, runs fully on the GPU.", 300),
    ("test_hint_just_under_85_percent_of_vram_runs_fully", 8.4, 10.0, "8.4 GB model, runs fully on the GPU.", 300),
    ("test_hint_just_over_85_percent_of_vram_spills_to_ram", 8.6, 10.0, "8.6 GB model, about 93% on the GPU", 900),
    ("test_hint_double_the_vram_is_half_on_gpu", 16.0, 10.0, "16.0 GB model, about 50% on the GPU", 900),
    ("test_hint_just_under_25_gb_keeps_900s_timeout", 24.9, 10.0, "24.9 GB model, about 32% on the GPU", 900),
    ("test_hint_exactly_25_gb_gets_1800s_timeout", 25.0, 10.0, "25.0 GB model, about 32% on the GPU", 1800),
    ("test_hint_seventy_gb_on_24_gb_gpu", 70.0, 24.0, "70.0 GB model, about 27% on the GPU", 1800),
    ("test_hint_half_gb_on_half_gb_gpu_spills", 0.5, 0.5, "0.5 GB model, about 80% on the GPU", 900),
], _make_hint_test)


class FormatSeconds(unittest.TestCase):
    pass


def _make_fmt_test(sec, expected):
    def test(self):
        self.assertEqual(app._fmt_secs(sec), expected)
    return test


_add_tests(FormatSeconds, [
    ("test_fmt_secs_zero", 0, "0s"),
    ("test_fmt_secs_fraction_is_truncated", 59.9, "59s"),
    ("test_fmt_secs_one_minute_exactly", 60, "1m 0s"),
    ("test_fmt_secs_sixty_one", 61, "1m 1s"),
    ("test_fmt_secs_just_under_an_hour", 3599, "59m 59s"),
    ("test_fmt_secs_one_hour_exactly_drops_seconds", 3600, "1h 0m"),
    ("test_fmt_secs_one_hour_one_minute_one_second", 3661, "1h 1m"),
    ("test_fmt_secs_over_a_day_stays_in_hours", 90061, "25h 1m"),
], _make_fmt_test)


# ---------------------------------------------------------------- ScanProgress
class ScanProgressParsing(unittest.TestCase):
    def test_progress_percent_counts_skipped_and_ignores_finished_current(self):
        p = app.ScanProgress(["a.A", "b.B", "c.C", "d.D"], done=["a.A"], skipped=["b.B"])
        self.assertEqual(p.percent(), 50.0)
        p.feed("probes.a.A:  50%|#    | 1/2")
        self.assertEqual(p.percent(), 50.0, "a probe already counted as done adds no partial progress")

    def test_progress_html_running_shows_probe_index_estimate_and_notes(self):
        p = app.ScanProgress(["a.A", "b.B"], done=["a.A"], failed=["a.A"])
        p.start -= 100
        p.feed("probes.b.B:  50%|#    | 1/2")
        out = p.html()
        self.assertIn("Probe 2 of 2 &nbsp;·&nbsp; <span class='mono'>b.B</span>", out)
        self.assertIn('<span class="sp-pct">71%</span>', out)
        self.assertIn("Sending attack prompts 1 of 2", out)
        self.assertIn("0 passed, 1 failed", out)
        self.assertIn("left", out)
        noted = p.html(notes=("slow",))
        self.assertNotIn("left", noted, "no estimate while something is wrong")
        self.assertIn('<div class="sp-note">slow</div>', noted)

    def test_progress_html_titles_for_stuck_halted_and_stopped(self):
        p = app.ScanProgress()
        self.assertIn("Scan stuck on <span class='mono'>b.B</span>, restarting it", p.html("recovering", probe="b.B"))
        self.assertIn("Scan stuck on <span class='mono'>b.B</span>, stopping it", p.html("halting", probe="b.B"))
        self.assertIn("<span>Scan halted</span>", p.html("halted"))
        stopped = p.html("stopped")
        self.assertIn("<span>Scan stopped</span>", stopped)
        self.assertIn('<span class="sp-pct">0%</span>', stopped)


class ScanProgressLines(unittest.TestCase):
    """One line of garak output each, fed to a fresh ScanProgress."""


def _make_feed_test(line, returns, expected):
    def test(self):
        p = app.ScanProgress()
        self.assertEqual(p.feed(line), returns)
        for attr, value in expected.items():
            if isinstance(value, float):
                self.assertAlmostEqual(getattr(p, attr), value, msg=attr)
            else:
                self.assertEqual(getattr(p, attr), value, attr)
    return test


_add_tests(ScanProgressLines, [
    ("test_feed_queue_line_lists_probes_and_loads_model", "queue of probes: x.X, y.Y", True,
     {"probes": ["x.X", "y.Y"], "phase": "Loading model"}),
    ("test_feed_queue_line_skips_blank_entries", "queue of probes: x.X, , y.Y,", True, {"probes": ["x.X", "y.Y"]}),
    ("test_feed_prompt_progress_line", "probes.a.A:  25%|##        | 16/64 [00:01<00:03]", True,
     {"current": "a.A", "frac": 0.2125, "phase": "Sending attack prompts 16 of 64"}),
    ("test_feed_prompt_progress_with_zero_total", "probes.a.A:   0%|          | 0/0", True,
     {"current": "a.A", "frac": 0.0, "phase": "Sending attack prompts 0 of 0"}),
    ("test_feed_detector_progress_line", "a.A/det.Detector:  50%|#####     | 5/10", True,
     {"current": "a.A", "frac": 0.925, "phase": "Evaluating responses with det.Detector"}),
    ("test_feed_pass_result_line", "a.A                  det.D: PASS  ok on 2/2", True,
     {"done": {"a.A"}, "failed": set()}),
    ("test_feed_fail_result_line", "a.A  det.D: FAIL  ok on 1/2", True, {"done": {"a.A"}, "failed": {"a.A"}}),
    ("test_feed_lowercase_result_is_ignored", "a.A  det.D: pass", False, {"done": set(), "phase": "Starting garak"}),
], _make_feed_test)


class ScanViewButtons(unittest.TestCase):
    def test_scan_view_idle_enables_start_only(self):
        log, start, stop, progress, pause, restart, skip = app._scan_view("log", "idle", "P")
        self.assertEqual((log, progress), ("log", "P"))
        self.assertEqual((start["interactive"], stop["interactive"]), (True, False))
        self.assertEqual((pause["value"], pause["interactive"]), ("Pause", False))
        self.assertEqual((restart["value"], restart["interactive"]), ("Restart probe", False))
        self.assertFalse(skip["visible"])

    def test_scan_view_paused_offers_resume_and_restart(self):
        _, start, stop, _, pause, restart, _ = app._scan_view("", "paused", "P")
        self.assertEqual((start["interactive"], stop["interactive"]), (False, True))
        self.assertEqual((pause["value"], pause["interactive"]), ("Resume", True))
        self.assertEqual((restart["value"], restart["interactive"]), ("Restart probe", True))

    def test_scan_view_halted_offers_continue_and_skip_only_with_stuck_probe(self):
        _, start, stop, progress, pause, cont, skip = app._scan_view("l", "halted", None, {"stuck_on": "b.B"})
        self.assertEqual(progress, NOOP, "no progress given: leave the card as it is")
        self.assertEqual((start["interactive"], stop["interactive"], pause["interactive"]), (True, True, False))
        self.assertEqual((cont["value"], cont["interactive"], skip["visible"]), ("Continue", True, True))
        self.assertFalse(app._scan_view("l", "halted", None, {"stuck_on": None})[6]["visible"])
        self.assertFalse(app._scan_view("l", "halted", None, None)[6]["visible"])

    def test_scan_view_recovering_halting_and_done_states(self):
        for state in ("recovering", "halting"):
            _, start, stop, _, pause, restart, _ = app._scan_view("", state, "P")
            self.assertEqual((start["interactive"], stop["interactive"]), (False, True), state)
            self.assertEqual((pause["interactive"], restart["value"], restart["interactive"]),
                             (False, "Restart probe", False), state)
        _, start, stop, _, pause, _, _ = app._scan_view("", "done", "P")
        self.assertEqual((start["interactive"], stop["interactive"], pause["interactive"]), (True, False, False))

# ---------------------------------------------------------------- scan state: jobs, stalls, controls
def scan_job(**kw):
    job = {"model": "m", "scan_type": "Quick check", "requested": ["a.A", "b.B", "c.C"],
           "queue": ["a.A", "b.B", "c.C"], "done": [], "failed": [], "skipped": [], "generations": 1, "timeout": 60,
           "thinking": False, "prefix": "m_20260101-000000", "parts": 1, "stalls": 0}
    job.update(kw)
    return job


class ScanStateTest(TempDataTest):
    """Idle scan globals for each test, restored afterwards; garak can't start and nothing sleeps."""

    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.dict(app._proc, {"p": None}))
        self.enterContext(mock.patch.dict(app._image_proc, {"p": None}))
        self.enterContext(mock.patch.dict(app._resume, {"waiting": False}))
        self.enterContext(mock.patch.dict(app._scan, {
            "head": "", "buf": [], "cur": [""], "progress": app.ScanProgress(), "reader": None, "end": "",
            "ok": False, "user_stop": False, "paused": False, "job": None, "last_progress": time.time(),
            "timeouts": 0, "errors": 0, "last_error": "", "action": None, "why": "", "manual": False,
            "model_seen": False, "hf": None}))
        self.enterContext(mock.patch.object(app, "UNLOADABLE_FILE", self.tmp / "data" / "garak_unloadable.json"))
        self.enterContext(mock.patch.object(app.time, "sleep"))
        self.enterContext(mock.patch.object(app.subprocess, "Popen", side_effect=AssertionError("garak must not run")))
        self.enterContext(mock.patch.object(app, "unload_models"))
        self.ollama_up = self.enterContext(mock.patch.object(app, "ollama_up", return_value="0.9.0"))
        self.enterContext(mock.patch.object(app, "list_models", return_value=[]))

    def busy_process(self, pid=4242):
        p = mock.Mock(pid=pid)
        p.poll.return_value = None
        return p


class RunScanValidation(ScanStateTest):
    def test_run_scan_empty_model_name(self):
        out = list(app.run_scan("", "Quick check", [], 1, 60))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "Select a model to scan.")
        self.assertEqual(len(out[0]), 7)

    def test_run_scan_model_ollama_cannot_load(self):
        with mock.patch.object(app, "list_models", return_value=[{"name": "m"}]), \
                mock.patch.object(app, "load_problem", return_value="bad format"):
            out = list(app.run_scan("m", "Quick check", [], 1, 60))
        self.assertEqual([v[0] for v in out],
                         ["m can't be scanned: Ollama can't load this model (bad format). Choose a model that runs."])
        self.assertFalse(app.SCAN_JOB.exists())

    def test_run_scan_saved_group_that_no_longer_exists(self):
        out = list(app.run_scan("m", app.GROUP_PREFIX + "ghost", [], 1, 60))
        self.assertEqual([v[0] for v in out], ["That group no longer exists."])
        self.assertFalse(app.SCAN_JOB.exists())

    def test_run_scan_refuses_while_stuck_probe_is_being_handled(self):
        app._scan["action"] = "restart"
        out = list(app.run_scan("m", "Quick check", [], 1, 60))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "A scan is already running. Wait for it to finish, or click Stop to end it.")
        self.assertTrue(out[0][2]["interactive"], "Stop stays available")
        self.assertFalse(app.SCAN_JOB.exists())

    def test_run_scan_refuses_while_interrupted_scan_waits_to_resume(self):
        app._resume["waiting"] = True
        out = list(app.run_scan("m", "Quick check", [], 1, 60))
        self.assertEqual([v[0] for v in out], ["An interrupted scan is about to resume. Click Stop to cancel it first."])
        self.assertTrue(out[0][2]["interactive"])


class ScanJobFile(ScanStateTest):
    def test_scan_job_round_trip_leaves_no_temp_file(self):
        job = scan_job(done=["a.A"])
        app._save_scan_job(job)
        self.assertEqual(app._load_scan_job(), job)
        self.assertFalse(app.SCAN_JOB.with_suffix(".tmp").exists())

    def test_load_scan_job_corrupt_or_missing_is_none(self):
        self.assertIsNone(app._load_scan_job())
        app.SCAN_JOB.write_text("{bad")
        self.assertIsNone(app._load_scan_job())
        app.SCAN_JOB.write_bytes(b"\xff\xfe\x00")
        self.assertIsNone(app._load_scan_job(), "bytes that aren't UTF-8")

    def test_load_scan_job_needs_a_dict_with_a_model(self):
        for text in ("[1, 2]", '{"model": ""}', '{"queue": []}', '"text"'):
            app.SCAN_JOB.write_text(text)
            self.assertIsNone(app._load_scan_job(), text)

    def test_remaining_probes_drop_done_and_skipped_in_queue_order(self):
        job = scan_job(queue=["a.A", "b.B", "c.C", "d.D"], done=["a.A"], skipped=["c.C"])
        self.assertEqual(app._remaining_probes(job), ["b.B", "d.D"])

    def test_remaining_probes_fall_back_to_request_before_garak_lists_queue(self):
        self.assertEqual(app._remaining_probes(scan_job(queue=[], requested=["x.X", "y.Y"])), ["x.X", "y.Y"])
        self.assertEqual(app._remaining_probes(scan_job(queue=[], requested=["x.X", "y.Y"], done=["x.X"])), ["y.Y"])

    def test_remaining_probes_without_skipped_key_and_full_scan_request(self):
        job = scan_job(queue=[], requested=[])
        del job["skipped"]
        self.assertEqual(app._remaining_probes(job), [], "a full scan has no probe list to pass on")
        job["queue"] = ["a.A"]
        self.assertEqual(app._remaining_probes(job), ["a.A"])

    def test_part_name_first_part_has_no_suffix(self):
        self.assertEqual(app._part_name({"prefix": "m_1"}, 1), "m_1")
        self.assertEqual(app._part_name({"prefix": "m_1"}, 0), "m_1")

    def test_part_name_later_parts_numbered(self):
        self.assertEqual(app._part_name({"prefix": "m_1"}, 2), "m_1_part2")
        self.assertEqual(app._part_name({"prefix": "m_1"}, 12), "m_1_part12")


class UnloadablePlugins(ScanStateTest):
    def test_unloadable_missing_or_corrupt_file_is_empty(self):
        self.assertEqual(app._unloadable(), {})
        app.UNLOADABLE_FILE.write_text("{nope")
        self.assertEqual(app._unloadable(), {})

    def test_unloadable_ignores_other_garak_version_and_expired_or_bad_entries(self):
        now = time.time()
        app.UNLOADABLE_FILE.write_text(json.dumps({"garak": "0.0.0-other", "plugins": {"x.X": now}}))
        self.assertEqual(app._unloadable(), {}, "learned for another garak version")
        app.UNLOADABLE_FILE.write_text(json.dumps(["x.X"]))
        self.assertEqual(app._unloadable(), {})
        app.UNLOADABLE_FILE.write_text(json.dumps({"garak": app.GARAK_VERSION, "plugins": {
            "old.X": now - (app.UNLOADABLE_DAYS + 1) * 86400, "bad.X": "yesterday", "ok.X": now}}))
        self.assertEqual(list(app._unloadable()), ["ok.X"])

    def test_remember_unloadable_adds_to_known_and_records_version(self):
        app._remember_unloadable("a.X")
        app._remember_unloadable("b.Y")
        self.assertEqual(set(app._unloadable()), {"a.X", "b.Y"})
        self.assertEqual(json.loads(app.UNLOADABLE_FILE.read_text())["garak"], app.GARAK_VERSION)


class WorkAndStalls(ScanStateTest):
    @staticmethod
    def proc(pid, user, system, name="", cmd=()):
        p = mock.Mock(pid=pid, info={"name": name, "cmdline": list(cmd) if cmd is not None else None})
        p.cpu_times.return_value = types.SimpleNamespace(user=user, system=system)
        return p

    def test_work_secs_includes_model_runners_but_not_other_processes(self):
        garak, child = self.proc(10, 1.0, 0.5), self.proc(11, 2.0, 0.0)
        garak.children.return_value = [child]
        others = [self.proc(20, 3.0, 1.0, "ollama", ["ollama", "runner", "--model", "x"]),
                  self.proc(21, 9.0, 0.0, "ollama", ["ollama", "serve"]),
                  self.proc(22, 4.0, 0.0, "llama-server"),
                  self.proc(23, 5.0, 0.0, "python3", ["python3"]),
                  self.proc(24, 6.0, 0.0, None, None)]
        with mock.patch.object(app.psutil, "Process", return_value=garak), \
                mock.patch.object(app.psutil, "process_iter", return_value=others):
            self.assertEqual(app._work_secs(10), {10: 1.5, 11: 2.0, 20: 4.0, 22: 4.0})

    def test_work_secs_skips_processes_that_vanish_or_deny_access(self):
        runner = self.proc(20, 3.0, 1.0, "ollama", ["ollama", "runner"])
        runner.cpu_times.side_effect = app.psutil.AccessDenied(20)
        llama = self.proc(22, 4.0, 0.0, "llama-server")
        with mock.patch.object(app.psutil, "Process", side_effect=app.psutil.NoSuchProcess(10)), \
                mock.patch.object(app.psutil, "process_iter", return_value=[runner, llama]):
            self.assertEqual(app._work_secs(10), {22: 4.0})

    def test_stall_reason_waits_longer_for_slow_settings(self):
        now = 1_000_000.0
        app._scan.update(job={"timeout": 900, "generations": 3}, last_progress=now - 5999)
        self.assertEqual(app._stall_reason([], now), "", "2 x 900 s x 3 generations + 600 s = 6000 s before it's stuck")

    def test_stall_reason_after_the_generous_limit(self):
        now = 1_000_000.0
        app._scan.update(job={"timeout": 900, "generations": 3}, last_progress=now - 6000)
        self.assertEqual(app._stall_reason([], now), "no progress for 1h 40m")

    def test_stall_reason_below_thresholds_is_quiet(self):
        now = 1_000_000.0
        app._scan.update(job={"timeout": 60, "generations": 1}, last_progress=now - 10,
                         timeouts=app.STALL_TIMEOUTS - 1, errors=app.STALL_ERRORS - 1, last_error="x")
        self.assertEqual(app._stall_reason([], now), "")

class StuckProbes(ScanStateTest):
    def progress_on(self, job, current):
        prog = app.ScanProgress(job["queue"], job["done"])
        prog.current = current
        app._scan.update(job=job, progress=prog)
        return prog

    def test_stuck_probe_is_current_unfinished_probe_else_first_remaining(self):
        job = scan_job(done=["a.A"])
        prog = self.progress_on(job, "b.B")
        self.assertEqual(app._stuck_probe(job), "b.B")
        prog.done.add("b.B")
        job["done"] = ["a.A", "b.B"]
        self.assertEqual(app._stuck_probe(job), "c.C")
        job["done"] = ["a.A", "b.B", "c.C"]
        self.assertIsNone(app._stuck_probe(job))

    def test_scan_stuck_without_a_job_does_nothing(self):
        with mock.patch.object(app.threading, "Thread") as thread:
            app._scan_stuck("hung")
        thread.assert_not_called()
        self.assertIsNone(app._scan["action"])

    def test_scan_stuck_restarts_first_then_halts_on_same_probe(self):
        job = scan_job(done=["a.A"])
        app._save_scan_job(job)
        self.progress_on(job, "b.B")
        with mock.patch.object(app.threading, "Thread") as thread, mock.patch("builtins.print"):
            app._scan_stuck("first hang")
            self.assertEqual(app._scan["action"], "restart")
            self.assertEqual((job["stuck_on"], job["stuck_count"]), ("b.B", 1))
            self.assertIs(thread.call_args.kwargs["target"], app._recover_scan)
            self.assertEqual(thread.call_args.kwargs["args"], (job, False))
            app._scan["action"] = None
            app._scan_stuck("second hang")
        self.assertEqual((app._scan["action"], app._scan["why"]), ("halt", "second hang"))
        self.assertEqual(thread.call_args.kwargs["args"], (job, True))
        self.assertEqual(app._load_scan_job()["stuck_count"], 2, "saved so a restart of LLM Scanner keeps it")

    def test_manual_restart_never_halts_and_user_stop_blocks_it(self):
        job = scan_job(done=["a.A"], stuck_on="b.B", stuck_count=5)
        self.progress_on(job, "b.B")
        with mock.patch.object(app.threading, "Thread") as thread, mock.patch("builtins.print") as printed:
            app._scan_stuck("button", manual=True)
            self.assertEqual((app._scan["action"], app._scan["manual"]), ("restart", True))
            self.assertEqual(thread.call_args.kwargs["args"], (job, False))
            printed.assert_not_called()
            app._scan.update(action=None, user_stop=True)
            app._scan_stuck("after stop")
        self.assertEqual(thread.call_count, 1)
        self.assertEqual(job["stuck_count"], 6)

    def test_scan_notes_while_handling_stuck_probe_give_reason(self):
        app._scan.update(action="restart", why="the model hung", timeouts=2)
        self.assertEqual(app._scan_notes(), ["Why: the model hung."])

    def test_scan_notes_single_timeout_wording(self):
        app._scan.update(job={"timeout": 120}, timeouts=1)
        self.assertEqual(app._scan_notes(), ["The model did not answer within the 120s response timeout (1 time in a "
                                             "row). garak is retrying; the probe restarts after 3."])

    def test_scan_notes_errors_wording_and_paused_scan_has_none(self):
        app._scan.update(job={"timeout": 60}, errors=1, last_error="boom")
        self.assertEqual(app._scan_notes(), ["Ollama failed 1 request in a row (boom). garak is retrying."])
        app._scan["errors"] = 2
        self.assertEqual(app._scan_notes(), ["Ollama failed 2 requests in a row (boom). garak is retrying."])
        app._scan["paused"] = True
        self.assertEqual(app._scan_notes(), [])

    def test_scan_notes_quiet_probe_after_two_minutes(self):
        app._scan.update(job={"timeout": 60}, last_progress=time.time() - 130)
        notes = app._scan_notes()
        self.assertEqual(len(notes), 1)
        self.assertRegex(notes[0], r"^No progress for 2m 1\ds\. If the probe is stuck it restarts automatically\.$")
        app._scan["last_progress"] = time.time() - 60
        self.assertEqual(app._scan_notes(), [])


class ScanControlsWhenIdle(ScanStateTest):
    def test_attach_scan_with_nothing_running_changes_nothing(self):
        self.assertEqual(list(app.attach_scan()), [(NOOP,) * 7])

    def test_attach_scan_shows_halted_scan(self):
        app._save_scan_job(scan_job(done=["a.A"], halted="model hung", stuck_on="b.B"))
        out = list(app.attach_scan())
        self.assertEqual(len(out), 1)
        log, _, _, progress, _, cont, skip = out[0]
        self.assertEqual(log, "Scan halted on b.B.")
        self.assertIn("Scan halted on <span class='mono'>b.B</span>", progress)
        self.assertIn("Why: model hung.", progress)
        self.assertEqual((cont["value"], skip["visible"]), ("Continue", True))

    def test_toggle_pause_disabled_when_idle_or_while_recovering(self):
        self.assertEqual(app.toggle_scan_pause(), app.gr.update(value="Pause", interactive=False))
        app._proc["p"] = self.busy_process()
        app._scan["action"] = "restart"
        with mock.patch.object(app.os, "killpg") as killpg:
            self.assertEqual(app.toggle_scan_pause(), app.gr.update(value="Pause", interactive=False))
        killpg.assert_not_called()

    def test_toggle_pause_suspends_then_resumes_process_group(self):
        app._proc["p"] = self.busy_process(4242)
        with mock.patch.object(app.os, "killpg") as killpg:
            self.assertEqual(app.toggle_scan_pause()["value"], "Resume")
            killpg.assert_called_with(4242, app.signal.SIGSTOP)
            self.assertTrue(app._scan["paused"])
            self.assertIsNotNone(app._scan["progress"].paused_at)
            self.assertEqual(app.toggle_scan_pause()["value"], "Pause")
            killpg.assert_called_with(4242, app.signal.SIGCONT)
        self.assertFalse(app._scan["paused"])
        self.assertIsNone(app._scan["progress"].paused_at)

    def test_stop_scan_when_idle_says_nothing_running(self):
        self.assertEqual(app.stop_scan(), "No scan is running.")
        self.assertTrue(app._scan["user_stop"])

    def test_stop_scan_cancels_waiting_resume(self):
        app._resume["waiting"] = True
        app._save_scan_job(scan_job())
        self.assertEqual(app.stop_scan(), "Resume cancelled.")
        self.assertFalse(app.SCAN_JOB.exists())

    def test_stop_scan_during_stuck_probe_recovery(self):
        app._scan["action"] = "halt"
        self.assertEqual(app.stop_scan(), "Stopping scan...")

    def test_scan_controls_after_stop_reset_buttons_when_idle(self):
        start, stop, progress, pause, restart, skip = app.scan_controls_after_stop()
        self.assertEqual((start["interactive"], stop["interactive"]), (True, False))
        self.assertEqual(progress, app.IDLE_PROGRESS)
        self.assertEqual((pause["value"], pause["interactive"]), ("Pause", False))
        self.assertEqual((restart["value"], restart["interactive"]), ("Restart probe", False))
        self.assertFalse(skip["visible"])

    def test_scan_controls_after_stop_leave_live_scan_alone(self):
        app._proc["p"] = self.busy_process()
        self.assertEqual(app.scan_controls_after_stop(), (NOOP,) * 6)

    def test_recover_scan_with_nothing_running_or_halted_is_noop(self):
        self.assertEqual(list(app.recover_scan(60)), [(NOOP,) * 7])
        app._save_scan_job(scan_job())
        self.assertEqual(list(app.recover_scan(60)), [(NOOP,) * 7], "a job that isn't halted has nothing to continue")

    def test_skip_stuck_probe_needs_halted_job_with_stuck_probe(self):
        app._save_scan_job(scan_job(halted="hung"))
        self.assertEqual(list(app.skip_stuck_probe(60)), [(NOOP,) * 7])
        self.assertEqual(app._load_scan_job()["skipped"], [])


class ResumeAndContinue(ScanStateTest):
    def test_resume_without_saved_job_launches_nothing(self):
        with mock.patch.object(app, "_launch_scan") as launch:
            app.resume_interrupted_scan()
        launch.assert_not_called()
        self.assertFalse(app._resume["waiting"])

    def test_resume_gives_up_when_ollama_never_starts_but_keeps_job(self):
        app._save_scan_job(scan_job(done=["a.A"]))
        self.ollama_up.return_value = None
        with mock.patch.object(app, "_launch_scan") as launch, mock.patch("builtins.print"), \
                mock.patch.object(app.time, "time", side_effect=itertools.count(1000, 400)):
            app.resume_interrupted_scan()
        launch.assert_not_called()
        self.assertFalse(app._resume["waiting"])
        self.assertEqual(app._load_scan_job()["parts"], 1, "kept for the next start")

    def test_resume_before_garak_listed_its_queue_starts_over(self):
        app._save_scan_job(scan_job(queue=[], requested=["a.A", "b.B"]))
        with mock.patch.object(app, "model_names", return_value=["m"]), mock.patch("builtins.print"), \
                mock.patch.object(app, "_launch_scan") as launch:
            app.resume_interrupted_scan()
        job, note = launch.call_args.args
        self.assertEqual(note, "Resumed after LLM Scanner was stopped: garak had not started probing yet, so the scan "
                               "starts over. Results from before the interruption are in report m_20260101-000000; "
                               "this part is saved as m_20260101-000000_part2.\n")
        self.assertEqual((job["parts"], job["done_at_resume"], job["stalls"]), (2, 0, 0))
        self.assertEqual(app._load_scan_job()["parts"], 2)

    def test_continue_halted_refuses_when_ollama_is_down(self):
        job = scan_job(done=["a.A"], halted="hung", stuck_on="b.B")
        app._save_scan_job(job)
        self.ollama_up.return_value = None
        out = list(app._continue_halted(job, 60, "x"))
        self.assertEqual([v[0] for v in out],
                         ["Ollama is not running. Start it at the top of the page, then press Continue."])
        self.assertEqual(app._load_scan_job()["halted"], "hung")

    def test_continue_halted_with_nothing_left_clears_job(self):
        job = scan_job(queue=["a.A", "b.B"], done=["a.A"], skipped=["b.B"], halted="hung")
        app._save_scan_job(job)
        out = list(app._continue_halted(job, 60, "Note. "))
        self.assertEqual(len(out), 1)
        log, start, stop, progress, *_ = out[0]
        self.assertEqual(log, "Note. No probes are left to run. Open Reports to view the results.")
        self.assertEqual(progress, app.IDLE_PROGRESS)
        self.assertEqual((start["interactive"], stop["interactive"]), (True, False))
        self.assertFalse(app.SCAN_JOB.exists())

    def test_recover_scan_continues_halted_job_with_new_timeout(self):
        app._save_scan_job(scan_job(done=["a.A"], halted="hung", stuck_on="b.B", stuck_count=2))
        with mock.patch.object(app, "_launch_scan") as launch, \
                mock.patch.object(app, "follow_scan", return_value=iter([("followed",)])):
            out = list(app.recover_scan(90))
        self.assertEqual(out, [("followed",)])
        job, note = launch.call_args.args
        self.assertEqual(note, "Continuing the scan from b.B. Response timeout changed from 60s to 90s. This part is "
                               "saved as m_20260101-000000_part2.\n")
        self.assertNotIn("halted", job)
        saved = app._load_scan_job()
        self.assertEqual((saved["timeout"], saved["parts"], saved["done_at_resume"]), (90, 2, 1))

    def test_skip_stuck_probe_adds_it_to_skipped_and_continues(self):
        app._save_scan_job(scan_job(done=["a.A"], halted="hung", stuck_on="b.B", stuck_count=2))
        with mock.patch.object(app, "_launch_scan") as launch, \
                mock.patch.object(app, "follow_scan", return_value=iter([])):
            list(app.skip_stuck_probe(60))
        job, note = launch.call_args.args
        self.assertEqual(note, "Skipped b.B; it is left out of the results. This part is saved as "
                               "m_20260101-000000_part2.\n")
        self.assertEqual(app._remaining_probes(job), ["c.C"])
        saved = app._load_scan_job()
        self.assertEqual((saved["skipped"], saved["stuck_on"], saved["stuck_count"]), (["b.B"], None, 0))
        self.assertNotIn("halted", saved)


if __name__ == "__main__":
    unittest.main()
