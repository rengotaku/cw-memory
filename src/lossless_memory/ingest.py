# -*- coding: utf-8 -*-
"""Convert raw conversation logs into the lossless 7-field record format.

Two input formats are supported:

  "claude_code" -- Claude Code's own per-session JSONL transcripts (the
      files it writes under its own projects directory). Reads
      message.content, keeps user/assistant text, and also emits a
      type="action" row for every tool_use block so tool calls stay in
      the timeline. Lines that look like a context-compaction boundary
      are tagged type="meta" (see query_rules.compact_range).

  "plain" -- a generic {"ts": ISO8601, "role": "user"|"assistant",
      "text": "..."} JSON-lines file. This is the format most other
      integrations should produce; it has no notion of tool calls.

Both produce the same seven core fields per row (ts, actor, role, type,
text, model, session), grouped into daily files under
<data_dir>/main/YYYY-MM-DD.jsonl, plus an eighth "unit" field (see
"cw unit recovery" below; null when not applicable).

Claude Code writes transcripts two levels deep --
<raw_log_dir>/<project-key>/<session>.jsonl, plus subagent transcripts
one level deeper still (<project-key>/<session>/subagents/*.jsonl) --
not the single flat directory this module's source glob originally
assumed. Set raw_log_recursive=true in config.json to walk the whole
tree instead of only <raw_log_dir>'s top level.

cw unit recovery: when cw_units_dir is set in config.json, each
"claude_code" row is stamped with the cw unit (e.g. "2026-09-22_160959")
its transcript belongs to, recovered from the project-key directory
name Claude Code derives from the session's working directory (see
_unit_name_for_project_key).

The source files are never modified -- this module only reads them.
Re-running convert_all() is idempotent (it rewrites the daily files
from scratch); convert_incremental() only reads the bytes appended
since the last run and is safe to call often (e.g. from daemon.py).

Ingestion staleness: every successful convert_all() / convert_incremental()
call records a heartbeat (data_dir()/_ingest_heartbeat.json), tracking two
separate things: whether this process is still running at all
(last_run_at()), and whether raw_log_dir still actually resolves to any
source file (last_sources_found_at()). They're tracked separately on
purpose -- a quiet day with zero new rows is normal and must not look
stale, but zero *source files found* never is, even though a run that
finds nothing still "succeeds" and returns cleanly. A typo'd path, an
unmounted drive, or a permission change all look exactly like that: the
daemon keeps completing runs on schedule while silently converting
nothing, which is indistinguishable from healthy operation unless
source-discovery is checked on its own. Claude Code deletes transcripts
after cleanupPeriodDays (default 30), so that gap going unnoticed means
losing conversations for good. is_stale() checks both; `python -m
lossless_memory.ingest --check-stale` (or a monitoring cron job calling
it) is what's meant to notice from outside this process before it's too
late.
"""
import os
import re
import json
import glob
import argparse
from datetime import datetime, timedelta, timezone

from .config import config, data_dir

# Rows whose text matches these are tagged type="meta" instead of "text"
# (e.g. a context-compaction boundary, or a harness-injected command
# echo) so search can exclude them by default.
META_HEADS = ("This session is being continued",)
META_MARKS = ("<command-name>", "<local-command-stdout>", "<task-notification>")

# Day bucketing uses the same +9h (JST) offset as index_exact.py's date
# vocabulary, so "which file a row lands in" and "which day a search
# for that row resolves to" agree with each other.
_DAY_OFFSET_HOURS = 9


def _out_main_dir():
    d = os.path.join(data_dir(), "main")
    os.makedirs(d, exist_ok=True)
    return d


def _state_file():
    return os.path.join(data_dir(), "_ingest_state.json")


def _heartbeat_file():
    return os.path.join(data_dir(), "_ingest_heartbeat.json")


# ---------------------------------------------------------------------
# Source file discovery (flat or recursive) and cw unit recovery
# ---------------------------------------------------------------------
def list_source_files(src_dir, recursive=False):
    """List raw *.jsonl session files under src_dir, sorted. Claude Code
    writes them two levels deep (<src_dir>/<project-key>/<session>.jsonl,
    plus subagent transcripts one level deeper still); recursive=True
    walks the whole tree instead of only src_dir's top level."""
    if recursive:
        return sorted(glob.glob(os.path.join(src_dir, "**", "*.jsonl"), recursive=True))
    return sorted(glob.glob(os.path.join(src_dir, "*.jsonl")))


# A Claude Code project-key directory name is the project's absolute
# working-directory path with every '/', '.', and '_' replaced by '-'
# (e.g. "/home/takuya/claude-workspaces/units/2026-09-22_160959" ->
# "-home-takuya-claude-workspaces-units-2026-09-22-160959").
_UNIT_SUFFIX_RE = re.compile(r"(\d{4}-\d{2}-\d{2})-(\d{6})")


def _escape_like_claude_code(path):
    return re.sub(r"[/_.]", "-", path)


def _unit_name_for_project_key(project_key, units_dir):
    """Recover a cw unit name (e.g. "2026-09-22_160959") from a Claude
    Code project-key directory name, if that project's working directory
    was <units_dir>/<unit>. Returns None if project_key doesn't match a
    unit that actually exists on disk under units_dir.

    The escaping Claude Code applies ('/', '.', '_' -> '-') is not
    injective, so a sibling directory that isn't under units_dir at all
    (e.g. ".../units_2026-09-22_160959" next to ".../units/") can
    escape to the exact same project-key as a real
    ".../units/2026-09-22_160959" project. Checking that the decoded
    unit directory actually exists rejects that false positive; it
    can't fully disambiguate the two source paths (the escaping has
    already thrown that information away), but it does guarantee this
    function never claims a unit that isn't really there."""
    if not project_key or not units_dir:
        return None
    units_dir = os.path.expanduser(units_dir).rstrip("/")
    prefix = _escape_like_claude_code(units_dir) + "-"
    if not project_key.startswith(prefix):
        return None
    m = _UNIT_SUFFIX_RE.fullmatch(project_key[len(prefix):])
    if not m:
        return None
    unit = f"{m.group(1)}_{m.group(2)}"
    if not os.path.isdir(os.path.join(units_dir, unit)):
        return None
    return unit


def _unit_for_file(fp, src_dir, units_dir):
    """The cw unit a source file belongs to, or None (flat sources, or
    a file directly under src_dir, never have one)."""
    if not units_dir:
        return None
    rel_dir = os.path.dirname(os.path.relpath(fp, src_dir))
    if not rel_dir:
        return None
    project_key = rel_dir.split(os.sep)[0]
    return _unit_name_for_project_key(project_key, units_dir)


# ---------------------------------------------------------------------
# Ingestion heartbeat / staleness detection. Claude Code deletes session
# transcripts after cleanupPeriodDays (default 30) -- if something stops
# converting them (the daemon process dying, but just as easily
# raw_log_dir becoming a typo'd/unmounted/permission-denied path while
# the daemon keeps running and "succeeding" at converting nothing), the
# unconverted tail is lost for good once that window passes.
#
# A run that finds zero source files is indistinguishable, from the
# row count alone, from a quiet day with no new conversation -- both
# convert 0 rows. So new_rows=0 must never by itself count as stale
# (every day with no chatting would trip it), but source_count=0 must
# always be tracked, because it's the one signal that only means "the
# source is gone", never "nothing to say today". last_run_at() tracks
# whether this process is still executing at all; last_sources_found_at()
# tracks whether raw_log_dir still resolves to anything. is_stale()
# checks both, so something outside this process can notice either kind
# of gap before it's too late.
# ---------------------------------------------------------------------
def _read_heartbeat():
    try:
        with open(_heartbeat_file(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _record_heartbeat(new_rows, source_count):
    now = datetime.now(timezone.utc).isoformat()
    prev = _read_heartbeat()
    hb = {"last_run_at": now, "last_new_rows": int(new_rows),
          "last_source_count": int(source_count)}

    if int(new_rows) > 0:
        hb["last_new_rows_at"] = now
    elif prev.get("last_new_rows_at"):
        hb["last_new_rows_at"] = prev["last_new_rows_at"]

    if int(source_count) > 0:
        hb["last_sources_found_at"] = now
        hb["last_sources_found_count"] = int(source_count)
    elif prev.get("last_sources_found_at"):
        hb["last_sources_found_at"] = prev["last_sources_found_at"]
        hb["last_sources_found_count"] = prev.get("last_sources_found_count")

    try:
        os.makedirs(os.path.dirname(_heartbeat_file()), exist_ok=True)
        with open(_heartbeat_file(), "w", encoding="utf-8") as f:
            json.dump(hb, f)
    except OSError:
        pass  # a failed heartbeat write must never fail the ingest run itself


def _parse_iso(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def last_run_at():
    """UTC datetime of the most recent successful ingest run (regardless
    of whether it found any source files), or None if it has never run
    (no heartbeat recorded yet)."""
    return _parse_iso(_read_heartbeat().get("last_run_at"))


def last_sources_found_at():
    """UTC datetime of the most recent ingest run that found at least
    one source file under raw_log_dir, or None if that has never
    happened. Stays frozen (unlike last_run_at()) while raw_log_dir is
    empty, missing, unmounted, or misconfigured, even though the run
    itself keeps completing "successfully" with zero rows."""
    return _parse_iso(_read_heartbeat().get("last_sources_found_at"))


def last_sources_found_count():
    """How many source files were found as of last_sources_found_at(),
    or None if that has never happened."""
    return _read_heartbeat().get("last_sources_found_count")


def staleness_seconds(now=None):
    """Seconds since the last successful ingest run, or None if it has
    never run."""
    last = last_run_at()
    if last is None:
        return None
    now = now or datetime.now(timezone.utc)
    return (now - last).total_seconds()


def source_staleness_seconds(now=None):
    """Seconds since ingest last found at least one source file under
    raw_log_dir, or None if it never has."""
    last = last_sources_found_at()
    if last is None:
        return None
    now = now or datetime.now(timezone.utc)
    return (now - last).total_seconds()


def is_stale(max_age_seconds):
    """True if ingest has never run, hasn't run successfully within
    max_age_seconds, or -- just as importantly -- hasn't found any
    source file under raw_log_dir within max_age_seconds. That last
    check is what catches a misconfigured/typo'd/unmounted raw_log_dir:
    without it, a daemon that keeps "succeeding" at converting zero
    rows from a path that resolves to nothing looks identical to a
    perfectly healthy quiet day."""
    s = staleness_seconds()
    if s is None or s > max_age_seconds:
        return True
    src_s = source_staleness_seconds()
    return src_s is None or src_s > max_age_seconds


def _is_meta(text):
    if any(text.startswith(h) for h in META_HEADS):
        return True
    if any(m in text for m in META_MARKS):
        return True
    return False


def _day_of_ts(ts):
    try:
        d = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S") + timedelta(hours=_DAY_OFFSET_HOURS)
        return d.strftime("%Y-%m-%d")
    except Exception:
        return ts[:10]


def _clean(text):
    """Strip harness-injected noise (e.g. a caveat block) without
    touching the meaning of the actual message."""
    if not text:
        return ""
    text = re.sub(r"<local-command-caveat>.*?</local-command-caveat>", "", text, flags=re.S)
    return text.strip()


# ---------------------------------------------------------------------
# "claude_code" format
# ---------------------------------------------------------------------
def _extract_text_claude(content):
    """Pull the text body out of a Claude Code message.content value.
    User content is a plain string; assistant content is a list of
    blocks ({"type": "text", ...} / {"type": "tool_use", ...} / ...).
    Only the text blocks are the message body."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                t = c.get("text", "")
                if t:
                    parts.append(t)
        return "\n".join(parts).strip()
    return ""


def _action_rows_claude(content, ts, session, day, actor_ai, model_ai, unit=None):
    """One type="action" row per tool_use block (tool_result blocks are
    dropped -- only the call itself is kept)."""
    rows = []
    if not isinstance(content, list):
        return rows
    for c in content:
        if not isinstance(c, dict) or c.get("type") != "tool_use":
            continue
        name = c.get("name") or "tool"
        val = ""
        inp = c.get("input")
        if isinstance(inp, dict):
            for v in inp.values():
                if isinstance(v, str) and v.strip():
                    val = v.strip()
                    break
        text = f"{name}: {val[:200]}"
        rec = {"ts": ts, "actor": actor_ai, "role": "ai", "type": "action",
               "text": text, "model": model_ai, "session": session, "unit": unit}
        rows.append((day, ts, json.dumps(rec, ensure_ascii=False) + "\n"))
    return rows


def _parse_claude_line(line, session, actor_user, actor_ai, model_ai, unit=None):
    line = line.strip()
    if not line:
        return None
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return None
    if d.get("type") not in ("user", "assistant"):
        return None  # skip harness-internal record types
    msg = d.get("message", {})
    if not isinstance(msg, dict):
        return None
    ts = d.get("timestamp")
    if not ts:
        return None
    role_raw = msg.get("role")
    day = _day_of_ts(ts)
    rows = []

    raw_text = _clean(_extract_text_claude(msg.get("content")))
    if raw_text:
        if role_raw == "user":
            actor, role, model = actor_user, "user", None
        else:
            actor, role, model = actor_ai, "ai", model_ai
        typ = "meta" if _is_meta(raw_text) else "text"
        rec = {"ts": ts, "actor": actor, "role": role, "type": typ,
               "text": raw_text, "model": model, "session": session, "unit": unit}
        rows.append((day, ts, json.dumps(rec, ensure_ascii=False) + "\n"))

    if role_raw == "assistant":
        rows.extend(_action_rows_claude(msg.get("content"), ts, session, day, actor_ai, model_ai, unit=unit))

    return rows or None


# ---------------------------------------------------------------------
# "plain" format: {"ts": ..., "role": "user"|"assistant", "text": ...}
# ---------------------------------------------------------------------
def _parse_plain_line(line, session, actor_user, actor_ai, model_ai, unit=None):
    line = line.strip()
    if not line:
        return None
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return None
    ts = d.get("ts")
    text = _clean(str(d.get("text") or ""))
    if not ts or not text:
        return None
    role_raw = d.get("role")
    day = _day_of_ts(ts)
    if role_raw == "user":
        actor, role, model = actor_user, "user", None
    else:
        actor, role, model = actor_ai, "ai", model_ai
    typ = "meta" if _is_meta(text) else "text"
    rec = {"ts": ts, "actor": actor, "role": role, "type": typ,
           "text": text, "model": model, "session": session, "unit": unit}
    return [(day, ts, json.dumps(rec, ensure_ascii=False) + "\n")]


_PARSERS = {"claude_code": _parse_claude_line, "plain": _parse_plain_line}


def _names():
    cfg = config()
    return (cfg.get("user_name") or "user", cfg.get("ai_name") or "assistant",
            cfg.get("model_name") or "unknown")


def convert_all(fmt="plain", source=None, recursive=None):
    """Rebuild every daily file in <data_dir>/main from scratch.
    Idempotent: running it again produces the same output. recursive=None
    reads raw_log_recursive from config; pass True/False to override it.
    Returns (total_rows, skipped_rows, day_file_count)."""
    parser = _PARSERS.get(fmt)
    if parser is None:
        raise ValueError("unknown ingest format: %r (use 'claude_code' or 'plain')" % (fmt,))
    cfg = config()
    src_dir = source or cfg.get("raw_log_dir")
    if not src_dir:
        raise ValueError("no source directory given (pass source=... or set raw_log_dir in config)")
    src_dir = os.path.expanduser(src_dir)
    if recursive is None:
        recursive = bool(cfg.get("raw_log_recursive"))
    units_dir = cfg.get("cw_units_dir")
    out_main = _out_main_dir()
    actor_user, actor_ai, model_ai = _names()

    files = list_source_files(src_dir, recursive=recursive)
    by_date = {}
    total = 0
    skipped = 0
    for fp in files:
        session = os.path.splitext(os.path.basename(fp))[0]
        unit = _unit_for_file(fp, src_dir, units_dir)
        with open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                parsed = parser(line, session, actor_user, actor_ai, model_ai, unit=unit)
                if parsed:
                    for day, ts, out in parsed:
                        by_date.setdefault(day, []).append((ts, out))
                        total += 1
                else:
                    skipped += 1

    # Sort by timestamp before writing: new lines always land at the end
    # of a day file, so a later incremental pass can trust that the
    # bytes before its last-seen position never change.
    for day, lines in by_date.items():
        lines.sort(key=lambda x: x[0])
        with open(os.path.join(out_main, day + ".jsonl"), "w", encoding="utf-8") as f:
            f.writelines(l for _, l in lines)

    _record_heartbeat(total, len(files))
    return total, skipped, len(by_date)


def _save_positions(fmt, source, recursive=False):
    state = {"fmt": fmt, "files": {}}
    for fp in list_source_files(source, recursive=recursive):
        try:
            state["files"][fp] = os.path.getsize(fp)
        except OSError:
            pass
    with open(_state_file(), "w", encoding="utf-8") as f:
        json.dump(state, f)


def convert_incremental(fmt="plain", source=None, recursive=None):
    """Convert only the bytes appended to each source file since the
    last call. Falls back to a full convert_all() on the very first
    call, or whenever a source file has shrunk (rotated). recursive=None
    reads raw_log_recursive from config; pass True/False to override it.
    Returns (new_row_count, did_full_rebuild, skipped_row_count)."""
    cfg = config()
    src_dir = source or cfg.get("raw_log_dir")
    if not src_dir:
        raise ValueError("no source directory given (pass source=... or set raw_log_dir in config)")
    src_dir = os.path.expanduser(src_dir)
    if recursive is None:
        recursive = bool(cfg.get("raw_log_recursive"))
    parser = _PARSERS.get(fmt)
    if parser is None:
        raise ValueError("unknown ingest format: %r (use 'claude_code' or 'plain')" % (fmt,))
    out_main = _out_main_dir()

    try:
        with open(_state_file(), encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        state = {}
    if state.get("fmt") != fmt:
        state = {"fmt": fmt, "files": {}}
    if not state.get("files") or not glob.glob(os.path.join(out_main, "*.jsonl")):
        total, skipped, _days = convert_all(fmt=fmt, source=src_dir, recursive=recursive)
        _save_positions(fmt, src_dir, recursive=recursive)
        return total, True, skipped

    units_dir = cfg.get("cw_units_dir")
    actor_user, actor_ai, model_ai = _names()
    positions = state["files"]
    new_by_date = {}
    total_new = 0
    skipped_new = 0

    files = list_source_files(src_dir, recursive=recursive)
    for fp in files:
        session = os.path.splitext(os.path.basename(fp))[0]
        unit = _unit_for_file(fp, src_dir, units_dir)
        pos = int(positions.get(fp, 0))
        try:
            size = os.path.getsize(fp)
        except OSError:
            continue
        if size < pos:
            total, skipped, _d = convert_all(fmt=fmt, source=src_dir, recursive=recursive)
            _save_positions(fmt, src_dir, recursive=recursive)
            return total, True, skipped
        if size == pos:
            continue

        skip_first = False
        if pos > 0:
            try:
                with open(fp, "rb") as fb:
                    fb.seek(pos - 1)
                    skip_first = fb.read(1) != b"\n"
            except OSError:
                skip_first = False

        with open(fp, encoding="utf-8", errors="replace") as f:
            f.seek(pos)
            if skip_first:
                f.readline()
            for line in f:
                parsed = parser(line, session, actor_user, actor_ai, model_ai, unit=unit)
                if parsed:
                    for day, ts, out in parsed:
                        new_by_date.setdefault(day, []).append((ts, out))
                        total_new += 1
                else:
                    skipped_new += 1
            positions[fp] = f.tell()

    for day, lines in new_by_date.items():
        lines.sort(key=lambda x: x[0])
        day_path = os.path.join(out_main, day + ".jsonl")
        last_ts = ""
        day_exists = os.path.exists(day_path)
        if day_exists:
            with open(day_path, "rb") as f:
                f.seek(max(0, os.path.getsize(day_path) - 65536))
                tail = f.read().decode("utf-8", errors="ignore").strip().splitlines()
            for tl in reversed(tail):
                try:
                    last_ts = json.loads(tl).get("ts") or ""
                    break
                except Exception:
                    continue
        if day_exists and not last_ts:
            last_ts = "9999"  # forces the merge branch below (safer than a bad append)
        if not last_ts or lines[0][0] >= last_ts:
            with open(day_path, "a", encoding="utf-8") as f:
                f.writelines(l for _, l in lines)
        else:
            merged = []
            if os.path.exists(day_path):
                with open(day_path, encoding="utf-8") as f:
                    for l in f:
                        try:
                            merged.append((json.loads(l).get("ts") or "", l))
                        except Exception:
                            continue
            merged += lines
            merged.sort(key=lambda x: x[0])
            with open(day_path, "w", encoding="utf-8") as f:
                f.writelines(l for _, l in merged)

    state["files"] = positions
    with open(_state_file(), "w", encoding="utf-8") as f:
        json.dump(state, f)
    _record_heartbeat(total_new, len(files))
    return total_new, False, skipped_new


def _build_arg_parser():
    p = argparse.ArgumentParser(
        prog="python -m lossless_memory.ingest",
        description="Convert raw conversation logs into the lossless daily JSONL format.",
    )
    p.add_argument("--format", dest="fmt", default=None,
                    help="parser to use: 'claude_code' or 'plain' "
                         "(default: config.json's ingest_format, or 'plain')")
    p.add_argument("--source", dest="source", default=None,
                    help="override raw_log_dir from config.json")
    p.add_argument("--recursive", dest="recursive", action="store_true",
                    help="walk --source recursively for *.jsonl "
                         "(default: config.json's raw_log_recursive)")
    p.add_argument("--no-recursive", dest="recursive", action="store_false",
                    help="disable recursive traversal even if config.json enables it")
    p.set_defaults(recursive=None)
    p.add_argument("--check-stale", dest="check_stale", action="store_true",
                    help="report whether ingest has run recently instead of converting; "
                         "exits 1 if it hasn't (or has never run)")
    p.add_argument("--max-stale-hours", dest="max_stale_hours", type=float, default=None,
                    help="override config.json's stale_after_hours for --check-stale")
    return p


def _check_stale(max_stale_hours):
    """Print what ingest's own staleness verdict is based on -- both the
    last time it ran at all, and (separately) the last time raw_log_dir
    actually resolved to any source file -- and return the process exit
    code (0 ok, 1 stale)."""
    max_age_hours = max_stale_hours
    if max_age_hours is None:
        max_age_hours = float(config().get("stale_after_hours") or 24)

    last = last_run_at()
    if last is None:
        print("ingest: STALE -- never run (no heartbeat recorded yet)")
        return 1
    run_age_hours = staleness_seconds() / 3600.0
    run_stale = run_age_hours > max_age_hours
    print(f"ingest: last run {last.isoformat()} ({run_age_hours:.1f}h ago) -- "
          f"{'STALE' if run_stale else 'ok'} (threshold {max_age_hours:.1f}h)")

    last_src = last_sources_found_at()
    if last_src is None:
        print("ingest: STALE -- raw_log_dir has never resolved to any source file "
              "(check raw_log_dir / mount / permissions in config.json)")
        return 1
    src_age_hours = source_staleness_seconds() / 3600.0
    src_stale = src_age_hours > max_age_hours
    print(f"ingest: last found source files {last_src.isoformat()} "
          f"({src_age_hours:.1f}h ago, {last_sources_found_count()} file(s)) -- "
          f"{'STALE' if src_stale else 'ok'} (threshold {max_age_hours:.1f}h)")

    return 1 if (run_stale or src_stale) else 0


def main():
    args = _build_arg_parser().parse_args()

    if args.check_stale:
        raise SystemExit(_check_stale(args.max_stale_hours))

    fmt = args.fmt or config().get("ingest_format") or "plain"
    total, skipped, days = convert_all(fmt=fmt, source=args.source, recursive=args.recursive)
    print(f"converted: {total} rows / skipped (no text): {skipped} / day files: {days}")
    print(f"output: {_out_main_dir()}")


if __name__ == "__main__":
    main()
