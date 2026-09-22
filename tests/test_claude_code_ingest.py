# -*- coding: utf-8 -*-
"""Coverage for the cw fork's additions to ingest.py:

  - recursive traversal of Claude Code's two-level transcript layout
    (<raw_log_dir>/<project-key>/<session>.jsonl, plus subagent
    transcripts one level deeper)
  - recovering a cw unit name from a project-key directory name
  - honest skip accounting through convert_incremental (the path the
    daemon actually calls, not just convert_all)
  - surviving source-file deletion (converted rows must not vanish)
  - ingestion heartbeat / staleness detection

None of this touches recall.py / index_exact.py / index_vector.py /
query_rules.py / state_index.py / topic.py -- those are out of scope
here (see issue #105).
"""
import json
import time
from pathlib import Path

import pytest

from lossless_memory import ingest
from lossless_memory.config import config


def _claude_line(ts, role, text, session, tool_use=None):
    content = text
    if role == "assistant":
        blocks = [{"type": "text", "text": text}] if text else []
        if tool_use:
            blocks.append({"type": "tool_use", "name": tool_use[0],
                            "input": {"command": tool_use[1]}})
        content = blocks
    return json.dumps({
        "type": role,
        "timestamp": ts,
        "sessionId": session,
        "message": {"role": role, "content": content},
    }, ensure_ascii=False)


def _harness_meta_line(session):
    """A record type Claude Code writes that isn't a conversation turn
    at all (e.g. a mode/permission notice) -- _parse_claude_line must
    reject it, and that rejection must show up as "skipped", not be
    silently absorbed."""
    return json.dumps({"type": "summary", "sessionId": session, "summary": "n/a"})


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config(force_reload=True)
    yield tmp_path


# ---------------------------------------------------------------------
# 1. project-key <-> cw unit name mapping (issue #105 requirement 3)
# ---------------------------------------------------------------------
def test_escape_like_claude_code_matches_observed_real_mapping():
    """Ground-truth check against the exact example in issue #105: the
    real project-key directory Claude Code created for the unit
    2026-09-22_160959 was measured to be this literal string."""
    got = ingest._escape_like_claude_code(
        "/home/takuya/claude-workspaces/units/2026-09-22_160959")
    assert got == "-home-takuya-claude-workspaces-units-2026-09-22-160959"


def test_unit_name_for_project_key_recovers_unit_that_exists_on_disk(tmp_path):
    units_dir = tmp_path / "claude-workspaces" / "units"
    unit_dir = units_dir / "2026-09-22_160959"
    unit_dir.mkdir(parents=True)
    project_key = ingest._escape_like_claude_code(str(unit_dir))

    unit = ingest._unit_name_for_project_key(project_key, str(units_dir))
    assert unit == "2026-09-22_160959"


def test_unit_name_for_project_key_rejects_unit_that_does_not_exist_on_disk(tmp_path):
    """A project-key can decode to a syntactically valid unit name
    without that unit actually existing (a coincidence, or the unit was
    since deleted) -- it must not be trusted just because it parses."""
    units_dir = tmp_path / "claude-workspaces" / "units"
    units_dir.mkdir(parents=True)  # units_dir exists, but no "2026-09-22_160959" inside it
    project_key = ingest._escape_like_claude_code(str(units_dir / "2026-09-22_160959"))

    assert ingest._unit_name_for_project_key(project_key, str(units_dir)) is None


def test_unit_name_for_project_key_rejects_colliding_sibling_path(tmp_path):
    """Reproduces the review's exact false-positive: Claude Code's
    escaping ('/', '.', '_' -> '-') is not injective, so a sibling
    directory that was never under units_dir at all can escape to the
    identical project-key as a real "<units_dir>/<unit>" project."""
    root = tmp_path / "claude-workspaces"
    units_dir = root / "units"
    units_dir.mkdir(parents=True)
    sibling = root / "units_2026-09-22_160959"  # NOT nested under units_dir
    sibling.mkdir()

    fake_key = ingest._escape_like_claude_code(str(sibling))
    real_key = ingest._escape_like_claude_code(str(units_dir / "2026-09-22_160959"))
    assert fake_key == real_key  # the collision itself -- escaping really did lose information

    # the sibling exists, but the *decoded* unit dir ("units/2026-09-22_160959") does not
    assert ingest._unit_name_for_project_key(fake_key, str(units_dir)) is None


def test_unit_name_for_project_key_rejects_non_matching_project(tmp_path):
    units_dir = tmp_path / "claude-workspaces" / "units"
    units_dir.mkdir(parents=True)
    # a project outside units_dir must not resolve to a unit
    assert ingest._unit_name_for_project_key(
        "-home-takuya--claude-plugins-everything-claude-code",
        str(units_dir)) is None
    # units_dir unset -> always None
    assert ingest._unit_name_for_project_key(
        "-home-takuya-claude-workspaces-units-2026-09-22-160959", None) is None
    # malformed suffix (not a YYYY-MM-DD-HHMMSS unit name) -> None
    assert ingest._unit_name_for_project_key(
        "-home-takuya-claude-workspaces-units-not-a-unit",
        str(units_dir)) is None


# ---------------------------------------------------------------------
# 2. recursive traversal + per-row unit stamping (requirements 1 & 3)
# ---------------------------------------------------------------------
@pytest.fixture()
def claude_code_tree(tmp_path):
    """<raw>/<cw-unit-project>/<session>.jsonl              (a cw unit)
       <raw>/<cw-unit-project>/<session>/subagents/a.jsonl  (its subagent)
       <raw>/<other-project>/<session>.jsonl                (not a cw unit)
    """
    raw = tmp_path / "raw"
    units_dir = tmp_path / "claude-workspaces" / "units"
    unit_name = "2026-09-22_160959"
    (units_dir / unit_name).mkdir(parents=True)  # the unit must exist on disk (issue #105 review 2)

    cw_project_key = ingest._escape_like_claude_code(str(units_dir / unit_name))
    other_project_key = ingest._escape_like_claude_code(str(tmp_path / "some-other-repo"))

    cw_proj = raw / cw_project_key
    cw_proj.mkdir(parents=True)
    session_a = "11111111-1111-1111-1111-111111111111"
    (cw_proj / f"{session_a}.jsonl").write_text(
        _claude_line("2026-09-22T07:00:00Z", "user", "hello from the cw unit", session_a) + "\n"
        + _claude_line("2026-09-22T07:00:05Z", "assistant", "hi", session_a,
                        tool_use=("Bash", "ls")) + "\n",
        encoding="utf-8",
    )
    subagent_dir = cw_proj / session_a / "subagents"
    subagent_dir.mkdir(parents=True)
    session_b = "agent-deadbeef"
    (subagent_dir / f"{session_b}.jsonl").write_text(
        _claude_line("2026-09-22T07:05:00Z", "user", "subagent task", session_b) + "\n",
        encoding="utf-8",
    )

    other_proj = raw / other_project_key
    other_proj.mkdir(parents=True)
    session_c = "22222222-2222-2222-2222-222222222222"
    (other_proj / f"{session_c}.jsonl").write_text(
        _claude_line("2026-09-22T08:00:00Z", "user", "hello from elsewhere", session_c) + "\n"
        + _harness_meta_line(session_c) + "\n",
        encoding="utf-8",
    )

    return {"raw": raw, "units_dir": units_dir, "unit_name": unit_name}


def test_flat_glob_misses_nested_transcripts(claude_code_tree, workspace):
    """Baseline: without raw_log_recursive, the upstream flat glob finds
    nothing under the two-level Claude Code layout -- this is exactly
    the gap issue #105 exists to close."""
    cfg = {"data_dir": "./logs", "raw_log_dir": str(claude_code_tree["raw"]),
           "ingest_format": "claude_code", "raw_log_recursive": False}
    Path("config.json").write_text(json.dumps(cfg), encoding="utf-8")
    config(force_reload=True)

    total, skipped, days = ingest.convert_all(fmt="claude_code")
    assert total == 0
    assert skipped == 0
    assert days == 0


def test_recursive_ingest_reads_every_level_and_stamps_unit(claude_code_tree, workspace):
    cfg = {
        "data_dir": "./logs",
        "raw_log_dir": str(claude_code_tree["raw"]),
        "ingest_format": "claude_code",
        "raw_log_recursive": True,
        "cw_units_dir": str(claude_code_tree["units_dir"]),
    }
    Path("config.json").write_text(json.dumps(cfg), encoding="utf-8")
    config(force_reload=True)

    total, skipped, days = ingest.convert_all(fmt="claude_code")
    # session_a: 1 user text + 1 assistant text + 1 assistant action = 3
    # session_b (subagent): 1 user text = 1
    # session_c: 1 user text = 1 (the harness meta line is skipped, not dropped silently)
    assert total == 5
    assert skipped == 1  # requirement: excluded records are counted, not silently discarded
    assert days == 1

    day_file = next(Path("logs/main").glob("*.jsonl"))
    rows = [json.loads(l) for l in day_file.read_text(encoding="utf-8").splitlines()]
    by_text = {r["text"]: r for r in rows}

    assert by_text["hello from the cw unit"]["unit"] == "2026-09-22_160959"
    assert by_text["hi"]["unit"] == "2026-09-22_160959"
    action_row = next(r for r in rows if r["type"] == "action")
    assert action_row["unit"] == "2026-09-22_160959"
    assert by_text["subagent task"]["unit"] == "2026-09-22_160959"  # nested one level deeper still
    assert by_text["hello from elsewhere"]["unit"] is None  # not under cw_units_dir


def test_unit_stays_null_when_cw_units_dir_unset(claude_code_tree, workspace):
    cfg = {
        "data_dir": "./logs",
        "raw_log_dir": str(claude_code_tree["raw"]),
        "ingest_format": "claude_code",
        "raw_log_recursive": True,
        # cw_units_dir intentionally omitted -- feature must default off
    }
    Path("config.json").write_text(json.dumps(cfg), encoding="utf-8")
    config(force_reload=True)

    ingest.convert_all(fmt="claude_code")
    day_file = next(Path("logs/main").glob("*.jsonl"))
    rows = [json.loads(l) for l in day_file.read_text(encoding="utf-8").splitlines()]
    assert all(r["unit"] is None for r in rows)


# ---------------------------------------------------------------------
# 3. convert_incremental: skip accounting + survives source deletion
#    (requirements: "除外したレコードの件数が実行結果に出る" and
#    "取り込み済みの transcript が元ファイル側で削除されても...失われない")
# ---------------------------------------------------------------------
@pytest.fixture()
def plain_workspace(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg = {"data_dir": "./logs", "raw_log_dir": str(raw), "ingest_format": "plain"}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config(force_reload=True)
    return raw


def _plain_line(ts, role, text):
    return json.dumps({"ts": ts, "role": role, "text": text})


def test_convert_incremental_reports_skipped_rows(plain_workspace):
    f = plain_workspace / "log.jsonl"
    f.write_text(
        _plain_line("2026-09-01T00:00:00Z", "user", "hello") + "\n"
        + "not json at all\n"
        + _plain_line("2026-09-01T00:01:00Z", "assistant", "hi") + "\n",
        encoding="utf-8",
    )
    new_rows, did_full, skipped = ingest.convert_incremental(fmt="plain")
    assert new_rows == 2
    assert did_full is True  # first call always rebuilds from scratch
    assert skipped == 1

    # append one more good line and one more bad line
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(_plain_line("2026-09-01T00:02:00Z", "user", "again") + "\n")
        fh.write("still not json\n")
    new_rows, did_full, skipped = ingest.convert_incremental(fmt="plain")
    assert new_rows == 1
    assert did_full is False
    assert skipped == 1

    # a third call with nothing appended must be a true no-op
    new_rows, did_full, skipped = ingest.convert_incremental(fmt="plain")
    assert (new_rows, skipped) == (0, 0)


def test_ingested_rows_survive_source_file_deletion(plain_workspace):
    f = plain_workspace / "log.jsonl"
    f.write_text(_plain_line("2026-09-01T00:00:00Z", "user", "keep me") + "\n",
                 encoding="utf-8")
    ingest.convert_incremental(fmt="plain")
    day_file = Path("logs/main/2026-09-01.jsonl")
    before = day_file.read_text(encoding="utf-8")
    assert "keep me" in before

    f.unlink()
    new_rows, did_full, skipped = ingest.convert_incremental(fmt="plain")
    assert new_rows == 0
    after = day_file.read_text(encoding="utf-8")
    assert after == before  # untouched -- the source file disappearing must not lose it


# ---------------------------------------------------------------------
# 4. heartbeat / staleness detection (requirement 2). Two independent
# signals matter here and must not be conflated:
#   - last_run_at(): did the process itself execute successfully?
#   - last_sources_found_at(): did raw_log_dir actually resolve to any
#     source file? Zero new rows is normal (a quiet day); zero source
#     files found is never normal (raw_log_dir is missing/typo'd/
#     unmounted) even though the run still "succeeds" and returns 0.
# ---------------------------------------------------------------------
def test_never_run_is_reported_as_stale(plain_workspace):
    assert ingest.last_run_at() is None
    assert ingest.staleness_seconds() is None
    assert ingest.last_sources_found_at() is None
    assert ingest.source_staleness_seconds() is None
    assert ingest.is_stale(10 ** 9) is True  # never having run is always stale


def test_heartbeat_recorded_when_sources_are_found(plain_workspace):
    f = plain_workspace / "log.jsonl"
    f.write_text(_plain_line("2026-09-01T00:00:00Z", "user", "hi") + "\n", encoding="utf-8")

    before = time.time()
    ingest.convert_all(fmt="plain")
    after = time.time()

    last = ingest.last_run_at()
    assert last is not None
    assert before - 1 <= last.timestamp() <= after + 1
    assert ingest.last_sources_found_at() is not None
    assert ingest.last_sources_found_count() == 1

    assert ingest.is_stale(10 ** 9) is False
    assert ingest.is_stale(-1) is True  # any positive elapsed time exceeds a negative budget


def test_missing_raw_log_dir_is_detected_as_stale(tmp_path, monkeypatch):
    """Reproduces the review's exact repro for issue #105: raw_log_dir
    points at a path that doesn't exist. Each convert_incremental()
    call still completes "successfully" (a glob over a missing
    directory just returns nothing, not an error), so last_run_at()
    keeps advancing every time -- but last_sources_found_at() must
    never be set, and is_stale() must report that, not "ok"."""
    missing = tmp_path / "does-not-exist"
    cfg = {"data_dir": "./logs", "raw_log_dir": str(missing), "ingest_format": "plain"}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config(force_reload=True)

    for _ in range(3):
        new_rows, did_full, skipped = ingest.convert_incremental(fmt="plain")
        assert (new_rows, skipped) == (0, 0)

    assert ingest.last_run_at() is not None        # the process itself keeps "succeeding"
    assert ingest.last_sources_found_at() is None  # but it has never actually found anything
    assert ingest.is_stale(3600) is True           # so is_stale() must catch it, not say "ok"


def test_source_staleness_freezes_when_sources_disappear(plain_workspace):
    """A healthy day (sources present, quiet or not) must not look
    stale; the same raw_log_dir later resolving to nothing must, even
    though last_run_at() keeps advancing because the process is still
    alive and completing runs on schedule."""
    f = plain_workspace / "log.jsonl"
    f.write_text(_plain_line("2026-09-01T00:00:00Z", "user", "hi") + "\n", encoding="utf-8")
    ingest.convert_incremental(fmt="plain")
    run_at_1 = ingest.last_run_at()
    found_at_1 = ingest.last_sources_found_at()
    assert found_at_1 is not None
    assert ingest.last_sources_found_count() == 1

    time.sleep(0.01)
    ingest.convert_incremental(fmt="plain")  # no new bytes, but the source is still there
    run_at_1b = ingest.last_run_at()
    found_at_1b = ingest.last_sources_found_at()
    assert run_at_1b > run_at_1
    assert found_at_1b > found_at_1  # still finding it every run -- keeps refreshing, not frozen yet

    f.unlink()
    plain_workspace.rmdir()  # raw_log_dir itself is now gone, not just the one file
    time.sleep(0.01)
    new_rows, did_full, skipped = ingest.convert_incremental(fmt="plain")
    assert (new_rows, skipped) == (0, 0)

    run_at_2 = ingest.last_run_at()
    found_at_2 = ingest.last_sources_found_at()
    assert run_at_2 > run_at_1b              # the process itself is still alive and ticking
    assert found_at_2 == found_at_1b         # but "found sources" is now frozen at the last time
                                              # it actually found something, not overwritten
    assert ingest.last_sources_found_count() == 1  # last known count, not overwritten with 0
