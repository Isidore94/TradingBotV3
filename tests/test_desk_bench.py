"""Packet G0 - the desk workload bench and the layout-fit check.

Every test here failed before `scripts/ui/desk_bench.py` existed (the module
import raised `ModuleNotFoundError` at collection), which is the fail-before-fix
proof for a packet whose whole content is one new module.

The two that matter most are the ones about SAFETY rather than measurement:

* `test_the_module_imports_nothing_from_scripts_at_module_top` - the bench sets
  `TRADINGBOTV3_DATA_DIR` inside `main`, so a single first-party import at
  module top would resolve `project_paths` against the live home folder before
  the scratch directory was ever named. That is the exact shape of the
  2026-09-05 incident that overwrote the live 1.2 GB setup tracker.
* `test_stage_refuses_a_destination_under_the_live_store` and its sibling that
  checks the SOURCE mtimes - a staging tool that can write into the store it
  read is the same incident with extra steps.
"""

from __future__ import annotations

import ast
import io
import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

BENCH_SOURCE = SCRIPTS_DIR / "ui" / "desk_bench.py"

pytest.importorskip("PySide6", reason="the bench drives Qt widgets")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

from ui import desk_bench  # noqa: E402

_app = QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# G0.4 - the module must not reach the live store on import
# ---------------------------------------------------------------------------
def test_the_module_imports_nothing_from_scripts_at_module_top():
    """A first-party import at module top would resolve the live home folder.

    Checked against the repo rather than against a hard-coded name list, so a
    new module under `scripts/` is covered the day it is added.
    """
    tree = ast.parse(BENCH_SOURCE.read_text(encoding="utf-8"))
    top_level_modules: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import IS first-party
                top_level_modules.append(f"<relative:{node.module}>")
            elif node.module:
                top_level_modules.append(node.module)

    offenders = []
    for name in top_level_modules:
        head = name.split(".")[0]
        if name.startswith("<relative:"):
            offenders.append(name)
            continue
        if (SCRIPTS_DIR / f"{head}.py").exists() or (SCRIPTS_DIR / head).is_dir():
            offenders.append(name)
    assert offenders == [], (
        "desk_bench imports these first-party modules at module top, before "
        f"TRADINGBOTV3_DATA_DIR is set: {offenders}"
    )


def test_the_abort_fires_when_the_data_dir_resolves_under_the_live_home(monkeypatch, capsys):
    import project_paths

    monkeypatch.setattr(project_paths, "DATA_DIR", Path(r"C:\TradingBotData\data"))
    monkeypatch.setattr(project_paths, "SHARED_HOME_DIR", Path(r"C:\TradingBotData"))

    assert desk_bench._abort_if_live(sys.stdout) == 2
    printed = capsys.readouterr().out
    assert "REFUSED" in printed
    assert "TradingBotData" in printed


def test_the_abort_also_fires_for_the_das(monkeypatch, capsys):
    import project_paths

    monkeypatch.setattr(project_paths, "DATA_DIR", Path(r"\\MINI-PC\Trading Bot Data\data"))
    monkeypatch.setattr(project_paths, "SHARED_HOME_DIR", Path(r"\\MINI-PC\Trading Bot Data"))

    assert desk_bench._abort_if_live(sys.stdout) == 2
    assert "REFUSED" in capsys.readouterr().out


def test_a_scratch_data_dir_is_not_refused(monkeypatch, tmp_path):
    import project_paths

    monkeypatch.setattr(project_paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(project_paths, "SHARED_HOME_DIR", tmp_path)
    assert desk_bench._abort_if_live(sys.stdout) is None


@pytest.mark.parametrize(
    "path",
    [
        r"C:\TradingBotData",
        r"C:\TradingBotData\data\runtime",
        "c:/tradingbotdata/data",
        r"\\MINI-PC\Trading Bot Data\research_lake",
    ],
)
def test_is_under_live_store_recognises_every_shape_of_the_live_roots(path):
    assert desk_bench.is_under_live_store(path) is True


def test_is_under_live_store_does_not_flag_a_lookalike(tmp_path):
    assert desk_bench.is_under_live_store(tmp_path) is False
    assert desk_bench.is_under_live_store(r"C:\TradingBotDataScratch") is False


# ---------------------------------------------------------------------------
# G0.1 - the statistics helper
# ---------------------------------------------------------------------------
def test_the_statistics_helper_on_a_known_list():
    values = [10.0, 20.0, 30.0, 40.0, 100.0]
    summary = desk_bench.summarize(values)
    assert summary["n"] == 5
    # Nearest-rank: p50 is the 3rd of 5, p95 is the 5th. Nothing is invented
    # between two readings.
    assert summary["p50"] == 30.0
    assert summary["p95"] == 100.0
    assert summary["max"] == 100.0


def test_an_empty_sample_reports_none_and_never_zero():
    summary = desk_bench.summarize([])
    assert summary["n"] == 0
    assert summary["p50"] is None
    assert summary["p95"] is None
    assert summary["max"] is None


def test_one_reading_is_its_own_p50_p95_and_max():
    summary = desk_bench.summarize([7.5])
    assert (summary["p50"], summary["p95"], summary["max"]) == (7.5, 7.5, 7.5)


# ---------------------------------------------------------------------------
# G0.2 - the layout-fit check
# ---------------------------------------------------------------------------
def _stacked_widget(floors):
    """A page holding one child per floor, in one vertical layout."""
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    for floor in floors:
        child = QWidget()
        child.setMinimumHeight(floor)
        layout.addWidget(child)
    return page


def test_the_fit_check_flags_a_page_whose_minimum_exceeds_the_available_height():
    page = _stacked_widget([260] * 9)  # the Weekend Focus Review shape
    record = desk_bench.measure_fit(page, page="synthetic", size_label="3456x2160", available=2070)
    assert record["overflow"] is True
    assert record["required"] >= 2340
    assert record["overflow_px"] == record["required"] - 2070
    assert record["available"] == 2070


def test_the_fit_check_passes_a_page_that_fits():
    page = _stacked_widget([260, 260])
    record = desk_bench.measure_fit(page, page="synthetic", size_label="3456x2160", available=2070)
    assert record["overflow"] is False
    assert record["overflow_px"] == 0
    assert record["required"] <= 2070


def test_the_table_floor_sum_counts_a_table_on_a_hidden_tab():
    from PySide6.QtWidgets import QTableWidget, QTabWidget

    page = QWidget()
    layout = QVBoxLayout(page)
    tabs = QTabWidget()
    for _ in range(3):
        table = QTableWidget(0, 2)
        table.setMinimumHeight(260)
        tabs.addTab(table, "t")
    layout.addWidget(tabs)
    # Two of the three are behind a tab and never shown; their floor is what the
    # page needs the moment the trader clicks them.
    assert desk_bench.table_floor_sum(page) == 780


def test_fit_verdict_takes_the_larger_of_the_hint_and_the_explicit_floor():
    #: A page can set a minimum above what its children ask for, and below it.
    high_floor = desk_bench.fit_verdict(
        minimum_size_hint=100, minimum_size=3000, size_hint=500, available=2070
    )
    assert high_floor["required"] == 3000 and high_floor["overflow"] is True
    high_hint = desk_bench.fit_verdict(
        minimum_size_hint=3000, minimum_size=0, size_hint=500, available=2070
    )
    assert high_hint["required"] == 3000 and high_hint["overflow"] is True
    fits = desk_bench.fit_verdict(
        minimum_size_hint=100, minimum_size=0, size_hint=500, available=2070
    )
    assert fits["required"] == 100 and fits["overflow"] is False


# ---------------------------------------------------------------------------
# G0.3 - staging
# ---------------------------------------------------------------------------
def _make_source(root: Path) -> dict[str, float]:
    """A miniature home folder: some allowlisted files and some that are not."""
    files = {
        "data/runtime/master_avwap_setup_stats.csv": "a,b\n1,2\n",
        "data/runtime/master_avwap_tier_list.csv": "t\n",
        "data/runtime/veto_cohort_picks.csv": "v\n",
        "data/runtime/veto_cohort_outcomes.csv": "v\n",
        "data/runtime/evidence_ledgers/market_journal-202609.jsonl": "{}\n",
        "trader_annotations.jsonl": "{}\n",
        # NOT on the allowlist - the two the packet names by exclusion, plus a
        # 1.2 GB-shaped tracker stand-in.
        "data/runtime/master_avwap_setup_attributes.csv": "x\n",
        "data/runtime/master_avwap_setup_scenarios.csv": "x\n",
        "data/runtime/master_avwap_setup_tracker.json": "{}\n",
    }
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return {
        relative: (root / relative).stat().st_mtime_ns for relative in files
    }


def test_stage_copies_only_the_allowlist(tmp_path):
    source = tmp_path / "home"
    dest = tmp_path / "scratch"
    _make_source(source)

    result = desk_bench.stage(source, dest)
    copied = {name for name, _ in result.copied}

    assert "data/runtime/master_avwap_setup_stats.csv" in copied
    assert "data/runtime/veto_cohort_picks.csv" in copied
    assert "data/runtime/evidence_ledgers/market_journal-202609.jsonl" in copied
    assert "trader_annotations.jsonl" in copied

    assert not (dest / "data/runtime/master_avwap_setup_attributes.csv").exists()
    assert not (dest / "data/runtime/master_avwap_setup_scenarios.csv").exists()
    assert not (dest / "data/runtime/master_avwap_setup_tracker.json").exists()

    # Same relative layout, byte for byte.
    assert (dest / "data/runtime/master_avwap_setup_stats.csv").read_text(
        encoding="utf-8"
    ) == "a,b\n1,2\n"
    assert result.total_bytes == sum(size for _, size in result.copied)


def test_stage_lists_an_absent_allowlist_entry_instead_of_failing(tmp_path):
    source = tmp_path / "home"
    dest = tmp_path / "scratch"
    _make_source(source)

    result = desk_bench.stage(source, dest)
    # The control/study discovery exports do not exist until the next persisted
    # tracker write (live gate #74). A bench that refused to run without them
    # would be useless on the day it is needed.
    assert "data/runtime/master_avwap_control_discovery.csv" in result.absent
    assert result.copied, "the present files were still copied"


def test_stage_never_writes_to_the_source(tmp_path):
    source = tmp_path / "home"
    dest = tmp_path / "scratch"
    before = _make_source(source)

    desk_bench.stage(source, dest)

    after = {relative: (source / relative).stat().st_mtime_ns for relative in before}
    assert after == before
    # And nothing new appeared under the source either.
    assert sorted(p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()) == sorted(before)


@pytest.mark.parametrize(
    "destination",
    [r"C:\TradingBotData\scratch", r"\\MINI-PC\Trading Bot Data\scratch"],
)
def test_stage_refuses_a_destination_under_the_live_store(tmp_path, destination):
    source = tmp_path / "home"
    _make_source(source)
    with pytest.raises(desk_bench.StageRefused):
        desk_bench.stage(source, destination)


@pytest.mark.parametrize(
    "destination",
    [r"C:\TradingBotData\scratch", r"\\MINI-PC\Trading Bot Data\scratch"],
)
def test_a_second_independent_guard_refuses_the_same_destinations(destination):
    """Two guards, compared against their own literals, not one calling the other.

    This exists because of what happened while G0 was being built: proving the
    FIRST guard bites means breaking it and re-running, and that run staged six
    files into `C:\\TradingBotData\\scratch` and the same folder on the DAS. Both
    trees were removed and no live file was touched, but one guard was one too
    few - a reviewer repeating the proof must not be able to write into the
    store the guard exists to protect.
    """
    with pytest.raises(desk_bench.StageRefused):
        desk_bench._refuse_to_open_for_writing(Path(destination) / "any.csv")


def test_the_second_guard_lets_a_scratch_path_through(tmp_path):
    desk_bench._refuse_to_open_for_writing(tmp_path / "any.csv")


def test_the_stage_cli_refuses_a_live_destination_with_exit_code_2(tmp_path, capsys):
    source = tmp_path / "home"
    _make_source(source)
    code = desk_bench.main(
        ["stage", "--from", str(source), "--to", r"C:\TradingBotData\scratch"]
    )
    assert code == 2
    assert "REFUSED" in capsys.readouterr().out


def test_the_run_refuses_to_start_without_a_data_dir(capsys):
    with pytest.raises(SystemExit) as excinfo:
        desk_bench.main([])
    assert excinfo.value.code == 2
    assert "--data-dir is required" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# G0.1 - the settle wait and one real end-to-end op
# ---------------------------------------------------------------------------
def test_a_deadline_is_a_recorded_result_and_not_an_error():
    """A page that never settles reports `settled: False`; nothing raises."""
    page = QWidget()

    class _NeverIdle:
        def is_alive(self):
            return True

    page._pretend_worker = _NeverIdle()
    # `_widget_workers_running` looks for a live `threading.Thread`, so the
    # stand-in above is ignored and the page settles. The deadline itself is
    # what is under test, so it is set to zero.
    # G7.0 added the bench's own poll cost as a fourth value; the trigger moves,
    # the assertions do not.
    settle_ms, longest, settled, poll_cost_ms = desk_bench.settle(
        _app, page, deadline_s=0.0
    )
    assert settled is False
    assert settle_ms >= 0.0
    assert longest >= 0.0
    assert poll_cost_ms >= 0.0


def test_a_broken_op_is_recorded_as_an_error_and_does_not_stop_the_run():
    page = QWidget()

    def _boom():
        raise ValueError("no such widget")

    reading = desk_bench.time_op(_app, page, "synthetic.op", "1x1", _boom, deadline_s=0.5)
    assert reading.error.startswith("ValueError")
    assert reading.op == "synthetic.op"


@pytest.mark.qt
def test_the_bench_runs_end_to_end_over_one_cheap_page(tmp_path):
    """The `--smoke` shape: one page, one repeat, one size.

    Deliberately the AWAY Recap only. The full workload builds the Research tab,
    which constructs eight children and reads a 294 MB outcome CSV on a worker -
    real work that belongs in a bench run and not in the suite. This proves the
    wiring (construct -> show -> op -> settle -> fit -> aggregate) on the real
    path in well under a second.
    """
    payload = desk_bench.run_bench(
        sizes=[(3456, 2160)],
        repeat=1,
        deadline_s=5.0,
        panel_names=("away_recap",),
        record_screens=False,
    )
    assert payload["schema"] == "desk_bench_v1"
    ops = {row["op"] for row in payload["ops"]}
    assert {"away_recap.construct", "away_recap.show", "away_recap.reload"} <= ops
    for row in payload["ops"]:
        assert row["samples"] == 1
        assert row["sync_ms"]["p95"] is not None
        assert row["longest_iteration_ms"]["max"] is not None

    # The chrome is measured from widgets, so the available height is the
    # window height minus something real and non-zero.
    assert 0 < payload["chrome"]["total"] < 2160
    fit = [row for row in payload["fit"] if row["page"] == "away_recap"]
    assert fit and fit[0]["available"] == 2160 - payload["chrome"]["total"]


def test_the_smoke_panel_set_is_a_subset_of_the_full_one():
    assert set(desk_bench.SMOKE_PANEL_NAMES) <= set(desk_bench.PANEL_NAMES)


def test_compare_names_a_new_op_and_a_missing_one():
    current = [
        {"op": "a", "size": "1x1", "sync_ms": {"p95": 10.0}, "settle_ms": {"p95": 20.0}},
        {"op": "new", "size": "1x1", "sync_ms": {"p95": 1.0}, "settle_ms": {"p95": 2.0}},
    ]
    baseline = [
        {"op": "a", "size": "1x1", "sync_ms": {"p95": 4.0}, "settle_ms": {"p95": 5.0}},
        {"op": "gone", "size": "1x1", "sync_ms": {"p95": 4.0}, "settle_ms": {"p95": 5.0}},
    ]
    text = desk_bench.format_compare(current, baseline)
    assert "NEW" in text
    assert "GONE" in text
    assert "6.0" in text  # 10.0 - 4.0, positive is slower


def test_parse_size_reads_the_trader_window_and_rejects_nonsense():
    assert desk_bench.parse_size("3456x2160") == (3456, 2160)
    assert desk_bench.parse_size(" 2560X1440 ") == (2560, 1440)
    for bad in ("3456", "0x100", "axb"):
        with pytest.raises(ValueError):
            desk_bench.parse_size(bad)


# ---------------------------------------------------------------------------
# The fix round, 2026-09-06: the guards run BEFORE anything is created
#
# The reviewer's blocker: `main()` called `_prepare_environment` first, which
# mkdirs the data directory and writes
# `_localappdata/TradingBotV3/local_settings.json` inside it, and only then
# asked whether that directory was the live store. So `--data-dir
# C:\TradingBotData` exited 2 with three new directories and a file already
# sitting in the live home folder - the packet's own invariant, broken by the
# code that states it.
#
# These tests drive the ORDERING with BOTH guards pointed at a FAKE root under
# `tmp_path`. A sabotaged guard is never aimed at the real live paths as a
# destination - doing exactly that is how the builder created
# `C:\TradingBotData\scratch` on 2026-09-06 while proving the first guard
# worked - and `_prepare_environment` is replaced by a raiser, so on the unfixed
# code the test stops at that assertion instead of running on against whatever
# store the environment happened to resolve.
# ---------------------------------------------------------------------------
def _point_both_guards_at(monkeypatch, fake_root: Path) -> None:
    marker = str(fake_root).replace("/", "\\").casefold()
    monkeypatch.setattr(desk_bench, "LIVE_STORE_ROOTS", (marker,))
    monkeypatch.setattr(desk_bench, "WRITE_REFUSAL_PREFIXES", (marker + "\\",))


def _explode_if_reached(*_args, **_kwargs):
    raise AssertionError(
        "_prepare_environment ran before the live-store guards: it mkdirs the "
        "data directory and writes a settings file into it, so by the time the "
        "refusal is printed the files already exist"
    )


def test_main_refuses_a_live_data_dir_before_anything_is_created(tmp_path, monkeypatch):
    fake_root = Path(str(tmp_path)).resolve() / "fakelive"
    _point_both_guards_at(monkeypatch, fake_root)
    monkeypatch.setattr(desk_bench, "_prepare_environment", _explode_if_reached)

    stream = io.StringIO()
    code = desk_bench.main(["--data-dir", str(fake_root / "home")], stream=stream)

    assert code == 2
    assert "REFUSED" in stream.getvalue()
    assert not fake_root.exists(), (
        "the refusal created something under the (fake) live root: "
        f"{sorted(str(p) for p in fake_root.rglob('*'))}"
    )


def test_main_refuses_a_live_out_path_before_anything_is_created(tmp_path, monkeypatch):
    fake_root = Path(str(tmp_path)).resolve() / "fakelive"
    scratch = Path(str(tmp_path)).resolve() / "scratch"
    _point_both_guards_at(monkeypatch, fake_root)
    monkeypatch.setattr(desk_bench, "_prepare_environment", _explode_if_reached)

    stream = io.StringIO()
    code = desk_bench.main(
        ["--data-dir", str(scratch), "--out", str(fake_root / "bench.json")],
        stream=stream,
    )

    assert code == 2
    assert "--out" in stream.getvalue()
    assert not fake_root.exists()
    assert not scratch.exists(), "a refused --out still created the scratch data directory"


def test_both_guards_run_and_either_one_alone_refuses(tmp_path, monkeypatch):
    """Independence, checked on a fake root so nothing points at the live store."""
    fake_root = Path(str(tmp_path)).resolve() / "fakelive"
    marker = str(fake_root).replace("/", "\\").casefold()
    target = fake_root / "home"

    monkeypatch.setattr(desk_bench, "LIVE_STORE_ROOTS", (marker,))
    monkeypatch.setattr(desk_bench, "WRITE_REFUSAL_PREFIXES", ("z:\\nothing\\",))
    assert desk_bench.refuse_live_destination("--data-dir", target, io.StringIO()) == 2

    monkeypatch.setattr(desk_bench, "LIVE_STORE_ROOTS", ("z:\\nothing",))
    monkeypatch.setattr(desk_bench, "WRITE_REFUSAL_PREFIXES", (marker + "\\",))
    assert desk_bench.refuse_live_destination("--data-dir", target, io.StringIO()) == 2

    monkeypatch.setattr(desk_bench, "LIVE_STORE_ROOTS", ("z:\\nothing",))
    monkeypatch.setattr(desk_bench, "WRITE_REFUSAL_PREFIXES", ("z:\\nothing\\",))
    assert desk_bench.refuse_live_destination("--data-dir", target, io.StringIO()) is None
    assert not fake_root.exists()


# ---------------------------------------------------------------------------
# The machine-local settings seed (advisory 2)
# ---------------------------------------------------------------------------
def test_the_settings_seed_carries_the_display_keys_and_no_credential_or_path():
    real = {
        "qt_ui_scale": 1.25,
        "qt_theme": "dark",
        "qt_compact_density": True,
        "daily_bars_source": "yahoo",
        "market_prep_openai_api_key": "sk-live-secret",
        "journal_questrade_refresh_token": "refresh-secret",
        "push_ntfy_token": "ntfy-secret",
        "desk_link_token": "link-secret",
        "journal_ibkr_flex_token": "flex-secret",
        "shared_data_dir": r"C:\TradingBotData",
        "research_store_dir": r"\\MINI-PC\Trading Bot Data\research_lake",
        "ai_store_dir": r"\\MINI-PC\Trading Bot Data\ai_store",
        "qt_autopilot_auto_arm": True,
    }
    seed = desk_bench.machine_settings_seed(real)

    assert seed["qt_ui_scale"] == 1.25
    assert seed["qt_theme"] == "dark"
    assert seed["daily_bars_source"] == "yahoo", (
        "the trader's pin is what makes the bench's settings real rather than synthetic"
    )
    assert seed["qt_autopilot_auto_arm"] is False, "a bench never arms, whatever the desk saved"
    for forbidden in (
        "market_prep_openai_api_key",
        "journal_questrade_refresh_token",
        "push_ntfy_token",
        "desk_link_token",
        "journal_ibkr_flex_token",
        "shared_data_dir",
        "research_store_dir",
        "ai_store_dir",
    ):
        assert forbidden not in seed, f"{forbidden} reached a scratch directory"


def test_the_settings_seed_is_defaults_when_the_real_file_is_missing():
    assert desk_bench.machine_settings_seed({}) == {"qt_autopilot_auto_arm": False}


def test_the_allowlist_no_longer_names_a_settings_file_that_is_never_there():
    assert "local_settings.json" not in desk_bench.STAGE_ALLOWLIST, (
        "the home-folder root holds no local_settings.json; the real one is "
        "machine-local and is carried by machine_settings_seed"
    )
