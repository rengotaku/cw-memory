# -*- coding: utf-8 -*-
"""Regression tests for the 5 fixes in claude-workspaces#106 (recall
precision / unbounded output / type filter / echo detection / 0-hit
diagnostics). Each test class below maps to one point in the issue.

These are separate from test_roundtrip.py's happy-path fixture because
several of these points need control that fixture doesn't give:
inserting a claude_code-format tool_use block (for the type filter),
crafting near-"now" timestamps (for the echo window), and generating
enough rows to exceed a cap.
"""
import datetime
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from lossless_memory import ingest, index_exact, recall
from lossless_memory.config import config

SAMPLE_LOG = Path(__file__).resolve().parent.parent / "examples" / "sample_log.jsonl"


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """A throwaway working directory with its own config.json (plain
    format), mirroring test_roundtrip.py's fixture."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy(SAMPLE_LOG, raw_dir / "sample_log.jsonl")

    cfg = {
        "user_name": "Sam",
        "ai_name": "Nova",
        "data_dir": "./logs",
        "raw_log_dir": str(raw_dir),
        "ingest_format": "plain",
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    config(force_reload=True)
    yield tmp_path


def _write_plain_rows(raw_dir, name, rows):
    (raw_dir / name).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Point 1: no unbounded path -- a date-scoped (or --compact, same code
# path) query used to return literally every row in range. It must now
# be capped, and the drop must be counted, never silent.
# ---------------------------------------------------------------------
def test_date_scope_caps_rows_and_reports_omission(workspace):
    raw_dir = workspace / "raw"
    cap = index_exact.DATE_SCOPE_ROW_CAP
    n = cap + 15
    base = datetime.datetime(2026, 9, 10, 0, 5, 0, tzinfo=datetime.timezone.utc)  # 09:05 JST
    rows = [{"ts": (base + datetime.timedelta(minutes=i)).isoformat(),
             "role": "user", "text": "note number %d" % i} for i in range(n)]
    _write_plain_rows(raw_dir, "bulk.jsonl", rows)

    ingest.convert_all(fmt="plain")
    index_exact.build_index()

    date_range = index_exact._jst_day_range_utc(datetime.date(2026, 9, 10))
    kept, meta = index_exact.search_time(date_range, None, [])

    assert len(kept) == cap
    assert meta["omitted"] == n - cap
    # the *most recent* rows are kept, not the oldest
    assert kept[-1]["text"] == "note number %d" % (n - 1)
    assert kept[0]["text"] == "note number %d" % (n - cap)


def test_around_row_cap_reports_omission(workspace, monkeypatch):
    """search_with_context's context pull (--around) is capped too
    (AROUND_ROW_CAP). The cap is shrunk for this test rather than
    generating >120 real rows -- the capping logic itself (sort by ts,
    keep the most recent N, count the rest) is exactly what the
    date-scope test above already exercises at full scale; this test
    is about search_with_context wiring that logic in, specifically."""
    raw_dir = workspace / "raw"
    base = datetime.datetime(2026, 9, 12, 1, 0, 0, tzinfo=datetime.timezone.utc)
    rows = [{"ts": (base + datetime.timedelta(minutes=i)).isoformat(),
             "role": "user" if i % 2 == 0 else "assistant",
             "text": "budget talk item %d" % i} for i in range(30)]
    _write_plain_rows(raw_dir, "many.jsonl", rows)
    ingest.convert_all(fmt="plain")
    index_exact.build_index()

    monkeypatch.setattr(index_exact, "AROUND_ROW_CAP", 5)
    kept, omitted = index_exact.search_with_context("budget", hits=4, around=5)

    assert len(kept) == 5
    assert omitted > 0
    assert len(kept) + omitted >= 5  # never silently smaller than what was actually found


# ---------------------------------------------------------------------
# Point 2: type='action' rows (tool-call bodies) are excluded by
# default, included only when asked.
# ---------------------------------------------------------------------
CC_TRANSCRIPT = [
    {"type": "user", "timestamp": "2026-09-11T01:00:00Z",
     "message": {"role": "user", "content": "please check on widgetapp deployment status"}},
    {"type": "assistant", "timestamp": "2026-09-11T01:00:05Z",
     "message": {"role": "assistant", "content": [
         {"type": "text", "text": "Checking now."},
         {"type": "tool_use", "name": "Bash", "input": {"command": "widgetapp status --verbose"}},
     ]}},
]


@pytest.fixture()
def cc_workspace(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sess1.jsonl").write_text(
        "\n".join(json.dumps(o) for o in CC_TRANSCRIPT) + "\n", encoding="utf-8")
    cfg = {"user_name": "Sam", "ai_name": "Nova", "data_dir": "./logs",
           "raw_log_dir": str(raw_dir), "ingest_format": "claude_code"}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config(force_reload=True)
    yield tmp_path


def test_action_rows_excluded_by_default_and_included_with_flag(cc_workspace):
    ingest.convert_all(fmt="claude_code")
    index_exact.build_index()

    default_rows = index_exact.search("widgetapp", limit=10)
    assert default_rows, "the user's own message should still be findable"
    assert all(r["type"] != "action" for r in default_rows)

    with_action = index_exact.search("widgetapp", limit=10, include_action=True)
    assert any(r["type"] == "action" for r in with_action)


def test_recall_fetch_excludes_action_by_default_and_includes_with_flag(cc_workspace):
    ingest.convert_all(fmt="claude_code")
    index_exact.build_index()

    default_hits = recall.fetch("widgetapp", log_limit=10, tail=False)
    assert all(r.get("type") != "action" for r in default_hits)

    with_action_hits = recall.fetch("widgetapp", log_limit=10, tail=False, include_action=True)
    assert any(r.get("type") == "action" for r in with_action_hits)


# ---------------------------------------------------------------------
# Point 3: echo detection only applies to the AI's own very-recent
# reply. A genuine hit (any actor, or an older AI row) must never be
# dropped just because its length happens to be >= some threshold.
# ---------------------------------------------------------------------
def test_is_echo_no_longer_length_gated():
    # 11 chars: used to require len(q) >= 10 to even be considered an
    # echo; that's no longer the deciding factor, just containment.
    assert recall._is_echo("feed-digest", "notes about feed-digest here")
    # 8 chars: used to be exempt from echo detection purely for being
    # short; containment now drives it regardless of length.
    assert recall._is_echo("cw-topic", "notes about cw-topic here")
    assert not recall._is_echo("feed-digest", "totally unrelated text")


def test_fuse_drops_only_ai_actor_recent_echo_not_genuine_hits():
    query = "feed-digest deployment notes"
    now = datetime.datetime.now(datetime.timezone.utc)

    ai_recent_echo = {  # AI, 20 min ago (past ECHO_EXCLUDE_MIN, inside ECHO_WINDOW_MINUTES)
        "ts": (now - datetime.timedelta(minutes=20)).isoformat(),
        "actor": recall.ACTOR_AI, "text": "as I said, feed-digest deployment notes are ready"}
    user_old_same_text = {  # user, 5 days ago -- a genuine hit, not an echo of anything
        "ts": (now - datetime.timedelta(days=5)).isoformat(),
        "actor": recall.ACTOR_USER, "text": "feed-digest deployment notes were finalized last week"}
    ai_old_same_text = {  # AI, 5 days ago -- old enough that it's a real memory, not an echo
        "ts": (now - datetime.timedelta(days=5)).isoformat(),
        "actor": recall.ACTOR_AI, "text": "feed-digest deployment notes were finalized last week"}

    out = recall._fuse([ai_recent_echo, user_old_same_text, ai_old_same_text], [],
                       limit=10, recency=False, _fuse_query=query)
    kept_texts = {r["text"] for r in out}

    assert ai_recent_echo["text"] not in kept_texts, "AI's own recent echo should be dropped"
    assert user_old_same_text["text"] in kept_texts, "a genuine hit must not be dropped for its length"
    assert ai_old_same_text["text"] in kept_texts, "an older AI row is a memory, not an echo"


def test_fuse_does_not_drop_recent_user_actor_as_echo():
    """#106 review point 3: removing the `r.get("actor") == ACTOR_AI`
    condition from _drop() left all 9 then-existing tests green,
    because every prior echo-shaped test case also happened to be
    outside ECHO_WINDOW_MINUTES or otherwise not content-echo-shaped.
    This is the case that condition actually exists for: a *user*
    row, recent enough to be inside the echo window, whose text
    happens to contain the query verbatim -- a genuine utterance, not
    the AI parroting the query back. Only the actor check protects it;
    delete that check and this test fails."""
    query = "feed-digest deployment notes"
    now = datetime.datetime.now(datetime.timezone.utc)
    user_recent_same_text = {  # user, 20 min ago -- past ECHO_EXCLUDE_MIN, inside ECHO_WINDOW_MINUTES
        "ts": (now - datetime.timedelta(minutes=20)).isoformat(),
        "actor": recall.ACTOR_USER,
        "text": "as I said, feed-digest deployment notes are ready",
    }
    out = recall._fuse([user_recent_same_text], [], limit=10, recency=False, _fuse_query=query)
    kept_texts = {r["text"] for r in out}
    assert user_recent_same_text["text"] in kept_texts, \
        "a recent *user* row must not be dropped as an echo -- only the AI's own recent reply can be"


# ---------------------------------------------------------------------
# Point 4: a literal, contiguous phrase in the query should out-rank
# (and, when it's found, exclude) a row that only shares scattered
# bigram fragments with it -- the "structured"/precise tier tried
# before the loose OR-of-any-bigram fallback. It must not, however,
# *replace* a genuinely relevant AND-tier row (same keywords, different
# order / something in between) -- the two tiers are merged, not an
# either-or (#106 review point 1).
# ---------------------------------------------------------------------
def test_phrase_query_matches_contiguous_substring_only(workspace):
    """Isolates the phrase tier's own FTS query from the rest of
    search()'s cascade. This has to be tested below the full search()
    call now: since tiers 1 and 2 are merged (review point 1), a row
    that satisfies the AND tier is a legitimate part of search()'s
    result even if the phrase tier alone wouldn't have matched it (see
    test_phrase_tier_merges_with_and_tier_relevant_row) -- so the
    "scattered, non-contiguous text must not match" property belongs
    to the phrase query specifically, not to search()'s overall output."""
    raw_dir = workspace / "raw"
    phrase = "七から十まで完了"
    # every individual bigram of `phrase` appears in `scattered`, so
    # the loose OR-of-any-bigram tier (and, since both of the AND
    # tier's extracted keywords also happen to be among those bigrams,
    # the AND tier too) would match this row -- only a *phrase* query
    # requires the bigrams to appear consecutively, telling them apart.
    scattered = "七かXからXら十X十まXまでXで完X完了、という無関係な羅列"
    rows = [
        {"ts": "2026-09-05T01:00:00+00:00", "role": "user", "text": "進捗は" + phrase + "した"},
        {"ts": "2026-09-05T02:00:00+00:00", "role": "assistant", "text": scattered},
    ]
    _write_plain_rows(raw_dir, "phrase.jsonl", rows)
    ingest.convert_all(fmt="plain")
    index_exact.build_index()

    con = sqlite3.connect(index_exact._index_db())
    try:
        matched = con.execute(
            "SELECT text FROM recall WHERE bigram MATCH ?",
            [index_exact._bigram_phrase_query(phrase)],
        ).fetchall()
    finally:
        con.close()
    texts = [r[0] for r in matched]
    assert texts, "the literal phrase must be found"
    assert all(phrase in t for t in texts), "a scattered, non-contiguous text must not match the phrase query"


def test_phrase_tier_merges_with_and_tier_relevant_row(workspace):
    """#106 review point 1: search() used to `return` the moment the
    phrase tier found anything, so a query whose words appear
    contiguously in one (possibly minor) row would hide a more
    relevant row where the same words appear in a different order or
    with something between them -- the ordinary case in Japanese. Both
    rows below match "七から十まで完了" via different tiers; both must
    come back."""
    raw_dir = workspace / "raw"
    phrase_row_text = "進捗は七から十まで完了した"          # matches tier 1 (literal phrase)
    and_only_row_text = "完了報告：作業は十まで進んだ"       # matches tier 2 only (keywords out of order)
    rows = [
        {"ts": "2026-09-05T01:00:00+00:00", "role": "user", "text": phrase_row_text},
        {"ts": "2026-09-05T02:00:00+00:00", "role": "assistant", "text": and_only_row_text},
    ]
    _write_plain_rows(raw_dir, "merge.jsonl", rows)
    ingest.convert_all(fmt="plain")
    index_exact.build_index()

    hits = index_exact.search("七から十まで完了", limit=10)
    texts = [r["text"] for r in hits]
    assert phrase_row_text in texts, "the literal phrase match must still be found"
    assert and_only_row_text in texts, \
        "an AND-tier-only relevant row must not be lost just because the phrase tier found something else"
    # phrase evidence is stronger -- it should rank above the AND-only supplement
    assert texts.index(phrase_row_text) < texts.index(and_only_row_text)


# ---------------------------------------------------------------------
# Point 5: on 0 hits, tell "genuinely nothing on record" apart from
# "something's there but got filtered/ranked out" -- e.g. every match
# is type=action and got excluded by the point-2 default.
# ---------------------------------------------------------------------
ACTION_ONLY_TRANSCRIPT = [
    # a single assistant turn with *only* a tool_use block (no text
    # block, and no preceding user turn) -- the resulting index has
    # exactly one row, and it's type=action. This makes "everything
    # matched is type=action" deterministic instead of depending on
    # incidental bigram overlap with other rows in a bigger fixture.
    {"type": "assistant", "timestamp": "2026-09-11T01:00:05Z",
     "message": {"role": "assistant", "content": [
         {"type": "tool_use", "name": "Bash", "input": {"command": "widgetapp status --verbose"}},
     ]}},
]


@pytest.fixture()
def action_only_workspace(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sess1.jsonl").write_text(
        "\n".join(json.dumps(o) for o in ACTION_ONLY_TRANSCRIPT) + "\n", encoding="utf-8")
    cfg = {"user_name": "Sam", "ai_name": "Nova", "data_dir": "./logs",
           "raw_log_dir": str(raw_dir), "ingest_format": "claude_code"}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config(force_reload=True)
    yield tmp_path


def test_zero_hit_reports_index_count_when_everything_was_type_action(action_only_workspace):
    ingest.convert_all(fmt="claude_code")
    index_exact.build_index()

    assert index_exact.raw_hit_count("verbose", include_action=True) > 0
    assert index_exact.raw_hit_count("verbose", include_action=False) == 0

    lines = recall.recall("verbose")
    joined = "\n".join(lines)
    assert "No hits" in joined
    assert "type=action" in joined
    assert "--action" in joined


def test_zero_hit_reports_genuinely_absent_when_index_has_nothing(workspace):
    """A filter that structurally can't match anything (a nonexistent
    actor) exercises the "genuinely nothing, not just filtered out"
    message deterministically -- a content-based nonsense query can't
    be used here, since bigram matching is deliberately loose enough
    that some 2-char fragment of almost any string coincides with
    *something* in a nontrivial corpus (that looseness is exactly what
    point 4's phrase tier exists to work around for real queries)."""
    ingest.convert_all(fmt="plain")
    index_exact.build_index()

    assert index_exact.raw_hit_count("budget", actor="NoSuchActor", include_action=True) == 0

    lines = recall.recall("budget", actor="NoSuchActor")
    joined = "\n".join(lines)
    assert "No hits" in joined
    assert "isn't just a filtering artifact" in joined


def test_zero_hit_reports_unknown_not_absent_when_index_unreadable(workspace):
    """#106 review point 2: raw_hit_count() used to collapse "index
    read failed" into the same 0 it returns for "checked, found
    nothing", and recall() then asserted "the index has 0 matches ...
    isn't just a filtering artifact" -- a false claim, since the index
    was never actually read. A corrupt index must be reported as
    unknown, never as confirmed absence."""
    ingest.convert_all(fmt="plain")
    index_exact.build_index()

    db_path = index_exact._index_db()
    with open(db_path, "w", encoding="utf-8") as f:
        f.write("not a sqlite database")

    assert index_exact.raw_hit_count("budget") is None, \
        "an unreadable index must report None (unknown), never 0 (confirmed absent)"

    lines = recall.recall("budget")
    joined = "\n".join(lines)
    assert "No hits" not in joined or "couldn't be checked" in joined
    assert "isn't just a filtering artifact" not in joined, \
        "must not claim confirmed absence when the index couldn't actually be read"
    assert "couldn't be checked" in joined
    assert "NOT confirmed absence" in joined
