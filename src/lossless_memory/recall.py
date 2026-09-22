# -*- coding: utf-8 -*-
"""recall -- the one entry point for pulling memory back out by hand.

    python -m lossless_memory.recall "query"

This is the only door: it fuses index_exact (exact match) and
index_vector (semantic match) results, removes duplicates, and prints
them in chronological order with a timestamp on every line -- never
summarized. Don't call index_exact / index_vector separately; call
this.

The strongest way to query is "date + word(s)": naming a date (an
explicit YYYY-MM-DD, or one of the Japanese relative words index_exact
understands) switches to time-scoped mode and returns only what's in
that range, without mixing in semantic matches from elsewhere.

The output always starts with [NOW] -- knowing the current time first
keeps you from drowning in a wave of search results whose own
timestamps you haven't oriented yourself against yet.
"""
import sys
import os
import io
import re
import json
import glob
import datetime

from .config import config, data_dir
from . import index_exact
# index_vector pulls in sentence-transformers, which is heavy and not
# needed for exact-only use -- imported lazily at each call site below
# instead of at module load, so recall still works (exact search only)
# in an environment that never installed it.

try:
    from . import query_rules as _qr
except Exception:
    _qr = None

try:
    from .ingest import META_HEADS, META_MARKS
except Exception:
    META_HEADS = ("This session is being continued",)
    META_MARKS = ("<command-name>", "<local-command-stdout>", "<task-notification>")


def _silence():
    """Swallow whatever a model-loading library prints to stdout on
    import, so it doesn't clutter the search output."""
    real = sys.stdout
    sys.stdout = io.StringIO()
    return real


# Failures are recorded here instead of being indistinguishable from a
# genuine "nothing found" -- a broken search and an empty result must
# never look the same.
_ERR_LOG = os.path.join(data_dir(), "recall_errors.log")
LAST_ERRORS = []
LAST_MODE = []       # which path served this query: "time" / "L3" (time -> semantic fallback) / "exact-only" / []
LAST_TIME_TOTAL = []
LAST_TIME_META = []
LAST_TAIL = []
LAST_OMITTED = []    # rows dropped by a row cap (date-scope / --around / --compact), never silent


def _note_error(part, ex):
    msg = "[%s] %s: %s %s" % (datetime.datetime.now().strftime("%m-%d %H:%M:%S"),
                              part, type(ex).__name__, str(ex)[:120])
    LAST_ERRORS.append(msg)
    try:
        with open(_ERR_LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


# Optional query-rewriting: a growable {word: [alternatives]} dictionary
# (data_dir/query_synonyms.json) can supply up to _MAX_VARIANTS reworded
# copies of a query to also search, on top of the original. Empty/absent
# by default.
_MAX_VARIANTS = 2


def _synonyms_path():
    return os.path.join(data_dir(), "query_synonyms.json")


def _expand_query(query):
    try:
        with open(_synonyms_path(), encoding="utf-8") as f:
            syn = json.load(f)
    except Exception:
        return []
    out = []
    for word, alts in syn.items():
        if word.startswith("_") or word not in query:
            continue
        for alt in alts:
            if alt in query:
                continue
            v = query.replace(word, alt)
            if v != query and v not in out:
                out.append(v)
            if len(out) >= _MAX_VARIANTS:
                return out
    return out


def _ts_jst(ts):
    """Render a stored UTC ts as a JST "YYYY-MM-DD HH:MM" string."""
    s = str(ts or "")
    try:
        t = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.timezone.utc)
        jst = datetime.timezone(datetime.timedelta(hours=9))
        return t.astimezone(jst).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return s[:16].replace("T", " ")


def _dedupe_rows(rows):
    """Collapse the same message into one row (ts+actor+text+type as
    the key, so a text row and an action row at the same timestamp
    stay separate)."""
    seen, out = set(), []
    for r in rows or []:
        key = ((r.get("ts") or ""), (r.get("actor") or ""), (r.get("text") or ""),
               (r.get("type") or "text"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


_cfg0 = config()
ACTOR_USER = _cfg0.get("user_name") or "user"
ACTOR_AI = _cfg0.get("ai_name") or "assistant"


def _split_time(query):
    try:
        return index_exact._split_time_query(query)
    except Exception:
        return (None, None, [])


def _vector_search(query, limit):
    """Lazily import index_vector (sentence-transformers is heavy and
    optional) and run a semantic search. [] if the dependency isn't
    installed or the search itself fails."""
    try:
        from . import index_vector
    except Exception as ex:
        _note_error("index_vector(import)", ex)
        return []
    try:
        return list(index_vector.search(query, limit=limit) or [])
    except Exception as ex:
        _note_error("index_vector", ex)
        return []


def fetch(query, log_limit=4, recency=True, tail=True, actor=None, around=0, exact_only=False,
          include_action=False):
    """The shared core of recall. Returns a deduped list of raw-row
    dicts. Not meant to be called directly by most users -- see
    recall() below, which formats the result for display.

    type='action' rows (tool-call bodies -- 61% of stored rows, mostly
    low-signal) are excluded unless include_action=True. index_exact
    already applies this at the SQL level (so it doesn't crowd out real
    hits within a LIMIT); this function re-applies it defensively to
    every source (including index_vector, which this module doesn't
    own and can't guarantee filters the same way) so the contract holds
    regardless of where a row came from."""
    del LAST_ERRORS[:]
    del LAST_MODE[:]
    del LAST_TIME_TOTAL[:]
    del LAST_TIME_META[:]
    del LAST_TAIL[:]
    del LAST_OMITTED[:]
    ses_hits, vec_hits = [], []
    real = _silence()
    try:
        _dr, _tr, _kws = _split_time(query)
        _time_mode = _dr is not None
        if exact_only:
            LAST_MODE.append("exact-only")
        if _time_mode:
            # Time-scoped mode: index_exact only. Semantic search is
            # not mixed in here so a date range stays a hard boundary.
            try:
                _hits = max(log_limit, 8) if _kws else 2
                ses_hits, _meta = index_exact.search_time(
                    _dr, _tr, _kws, actor=actor, hits=_hits, around=int(around),
                    include_action=include_action)
                ses_hits = list(ses_hits or [])
                LAST_MODE.append("time")
                LAST_TIME_TOTAL.append(int(_meta.get("total", 0)))
                LAST_TIME_META.append(_meta)
                LAST_OMITTED.append(int(_meta.get("omitted", 0)))
                # If the date range turned up almost nothing, fall back
                # to a semantic search across all time rather than
                # concluding "nothing" too quickly.
                if len(ses_hits) < 2 and not exact_only:
                    _vec_l3 = _vector_search(query, max(log_limit, 8))
                    if _vec_l3:
                        ses_hits = _dedupe_rows(list(ses_hits) + _vec_l3)
                        LAST_MODE.append("L3")
            except Exception as ex:
                _note_error("index_exact(time)", ex)
                _time_mode = False
        if not _time_mode:
            try:
                if around:
                    _ctx, _ctx_omitted = index_exact.search_with_context(
                        query, actor=actor, hits=max(log_limit, 4), around=int(around),
                        include_action=include_action)
                    ses_hits = list(_ctx or [])
                    LAST_OMITTED.append(int(_ctx_omitted or 0))
                else:
                    ses_hits = list(index_exact.search(
                        query, actor, limit=max(log_limit, 8), include_action=include_action) or [])
            except Exception as ex:
                _note_error("index_exact", ex)
            if not actor and not exact_only:
                vec_hits = _vector_search(query, max(log_limit, 8))
                _ds0 = [r.get("distance") for r in vec_hits if r.get("distance") is not None]
                if (not _ds0) or min(_ds0) > VEC_FAR_CUT_NORMAL:
                    for q in _expand_query(query):
                        try:
                            ses_hits += list(index_exact.search(
                                q, actor, limit=max(log_limit, 8), include_action=include_action) or [])
                        except Exception as ex:
                            _note_error("index_exact", ex)
                        vec_hits += _vector_search(q, max(log_limit, 8))
    finally:
        sys.stdout = real

    if not include_action:
        ses_hits = [r for r in ses_hits if (r.get("type") or "text") != "action"]
        vec_hits = [r for r in vec_hits if (r.get("type") or "text") != "action"]

    if _time_mode:
        uniq = _dedupe_rows(ses_hits)
    elif actor or exact_only:
        uniq = _dedupe_rows(ses_hits)[: log_limit * 2]
    else:
        uniq = _fuse(ses_hits, vec_hits, log_limit * 2, recency=recency, _fuse_query=query)
    if tail and not _time_mode and not actor:
        exclude = {((r.get("ts") or "")[:16], (r.get("text") or "")[:40]) for r in uniq}
        _tail_search(query, exclude)
    return uniq


# RRF fusion with recency weighting and a staged relevance cutoff.
RRF_K = 60
RRF_REL_CUT = 0.35
# Two cutoffs instead of one hard accept/reject line: 0.51-0.60 is a
# "maybe" band that's kept but flagged [maybe], not dropped outright.
VEC_FAR_CUT_NORMAL = 0.51
VEC_FAR_CUT_CONFIRM = 0.60
ECHO_EXCLUDE_MIN = 15


def _is_echo(query, text):
    """Whether `text` looks like it's just repeating `query` back
    verbatim. Used only for the AI's own very recent reply (see
    ECHO_WINDOW_MINUTES / _drop below) -- there the length of the
    overlap isn't what makes something "just an echo"; a short
    identifier (e.g. an 8-char one like "cw-topic") repeated back is
    exactly as much an echo as a long one. A hard length floor here
    used to be the deciding factor and wrongly swallowed genuine hits
    whose *content* (not an echo at all) happened to be a 10+ char
    identifier (#106 point 3)."""
    try:
        q = re.sub(r"\s", "", query)
        t = re.sub(r"\s", "", text or "")
        if not q:
            return False
        if len(q) < 20:
            return q in t
        step = 10
        for i in range(0, len(q) - 20 + 1, step):
            if q[i:i + 20] in t:
                return True
        if q[-20:] in t:
            return True
        return False
    except Exception:
        return False


RECENCY_HALF_LIFE_DAYS = 30.0


def _recency_factor(ts):
    try:
        d = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        age_days = max(0.0, (datetime.datetime.now(d.tzinfo) - d).total_seconds() / 86400)
        decay = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
        return 0.5 + 0.5 * decay
    except Exception:
        return 1.0


def _heat_hit(text):
    try:
        from . import state_index
        return state_index._heat(text or "")
    except Exception:
        return False


def _too_recent(ts):
    try:
        d = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        now = datetime.datetime.now(d.tzinfo)
        return (now - d).total_seconds() < ECHO_EXCLUDE_MIN * 60
    except Exception:
        return False


def _recent_minutes(ts, minutes):
    try:
        d = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        now = datetime.datetime.now(d.tzinfo)
        return (now - d).total_seconds() < minutes * 60
    except Exception:
        return False


# How far back a row still counts as "the AI still echoing the query
# back" (see _is_echo / _drop). Provisional -- not yet tuned against
# real sessions; the point of this window is just to be well past
# ECHO_EXCLUDE_MIN (which already drops *anything* in the last 15
# minutes regardless of actor/content) but not so wide that a genuine
# older hit whose actor happens to be the AI gets swept up.
ECHO_WINDOW_MINUTES = 60


def _fuse(ses_hits, vec_hits, limit, recency=True, _fuse_query=""):
    """Reciprocal-rank fusion of the exact and semantic hit lists, plus
    a recency weighting and the two-stage distance cutoff. Returns the
    fused list, highest score first, deduplicated."""
    _ds = [r.get("distance") for r in vec_hits if r.get("distance") is not None]
    pre_best_d = min(_ds) if _ds else None

    def _drop(r):
        if _too_recent(r.get("ts")):
            return True
        # Only the AI's *own* very recent reply can be "just echoing
        # the query" -- a matching row from the user, or an older AI
        # row, is a genuine memory, not an echo, regardless of its
        # length (#106 point 3: length alone used to be the test, and
        # dropped real hits like "feed-digest").
        if (r.get("actor") == ACTOR_AI and _recent_minutes(r.get("ts"), ECHO_WINDOW_MINUTES)
                and _is_echo(_fuse_query, r.get("text"))):
            return True
        return False
    ses_hits = [r for r in ses_hits if not _drop(r)]
    vec_hits = [r for r in vec_hits if not _drop(r)]

    def key(r):
        return (r.get("ts"), (r.get("text") or "")[:40])
    scores, items = {}, {}
    for rank, r in enumerate(ses_hits, start=1):
        k = key(r)
        scores[k] = scores.get(k, 0.0) + 1.0 / (RRF_K + rank)
        items.setdefault(k, r)
    for rank, r in enumerate(vec_hits, start=1):
        k = key(r)
        scores[k] = scores.get(k, 0.0) + 1.0 / (RRF_K + rank)
        if k in items:
            if items[k].get("distance") is None and r.get("distance") is not None:
                items[k]["distance"] = r.get("distance")
        else:
            items[k] = r
    if not scores:
        return []
    if pre_best_d is not None and pre_best_d > VEC_FAR_CUT_CONFIRM:
        return []
    for k in scores:
        d = items[k].get("distance")
        if d is not None and VEC_FAR_CUT_NORMAL < d <= VEC_FAR_CUT_CONFIRM:
            items[k]["confirm"] = True
    if recency:
        for k in scores:
            f = _recency_factor(items[k].get("ts"))
            if f < 0.9 and _heat_hit(items[k].get("text")):
                f = 0.9
            scores[k] *= f
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    top = ordered[0][1]
    out = []
    for k, s in ordered:
        if s < top * RRF_REL_CUT:
            break
        out.append(items[k])
        if len(out) >= limit:
            break
    return out


_TAIL_BYTES = 512 * 1024


def _is_meta_text(text):
    text = text or ""
    if any(text.startswith(h) for h in META_HEADS):
        return True
    if any(m in text for m in META_MARKS):
        return True
    return False


def _tail_search(query, exclude_keys, limit=3):
    """Best-effort: catch a message so recent the index hasn't picked
    it up yet, by scanning the tail of the raw source files directly.
    Only runs for the "claude_code" ingest format, since it parses
    that transcript shape specifically; a no-op otherwise.

    Matching requires *every* extracted keyword to be present (an AND,
    same precision philosophy as index_exact's phrase/AND tiers), not
    a fraction of them. The old "at least 1/3 of the words" bar meant
    a single common short word (a bare "10", say) alone was enough to
    surface *any* recent message containing it, unconditionally shown
    ahead of the real, better-ranked results -- which is what made
    "today's unrelated chatter" drown out an older real match
    (#106 point 4). Keyword extraction is delegated to
    index_exact._extract_keywords so filler/stopwords are excluded the
    same way the main search does, instead of a separate, looser regex."""
    del LAST_TAIL[:]
    cfg = config()
    if cfg.get("ingest_format") != "claude_code" or not cfg.get("raw_log_dir"):
        return
    raw_dir = cfg["raw_log_dir"]
    try:
        words = index_exact._extract_keywords(index_exact._strip_query(query))[:12]
        if not words:
            return
        files = sorted(glob.glob(os.path.join(raw_dir, "*.jsonl")),
                       key=os.path.getmtime, reverse=True)[:2]
        hits = []
        for fp in files:
            with open(fp, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - _TAIL_BYTES))
                blob = f.read().decode("utf-8", errors="ignore")
            for line in blob.splitlines():
                if '"type"' not in line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                typ = o.get("type")
                if typ not in ("user", "assistant"):
                    continue
                msg = o.get("message") or {}
                content = msg.get("content")
                text = ""
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = " ".join(c.get("text", "") for c in content
                                    if isinstance(c, dict) and c.get("type") == "text")
                if len(text) < 10 or "task-notification" in text:
                    continue
                if _is_meta_text(text):
                    continue
                n_match = sum(1 for w in words if w in text)
                if n_match == len(words):
                    ts = o.get("timestamp") or ""
                    key = (ts[:16], text[:40])
                    if key in exclude_keys:
                        continue
                    hits.append({"ts": ts, "actor": ACTOR_USER if typ == "user" else ACTOR_AI,
                                 "text": text, "n_match": n_match})
        hits.sort(key=lambda h: h["ts"], reverse=True)
        hits.sort(key=lambda h: -h["n_match"])
        LAST_TAIL.extend(hits[:limit])
    except Exception as ex:
        _note_error("tail", ex)


def _tag_for(r):
    typ = r.get("type") or "text"
    actor = r.get("actor") or ""
    if typ == "action":
        tag = "%s (action)" % actor
    elif typ == "thinking":
        tag = "%s (thinking)" % actor
    else:
        tag = actor
    if r.get("confirm"):
        tag = "[maybe] " + tag
    return tag


# ============================================================
# Optional "compaction" feature: for a caller whose ingest format
# ("claude_code") marks context-compaction boundaries, --compact[=N]
# returns everything said since the N-th-to-last boundary. Depends on
# query_rules.compact_range; a no-op (with a clear message) if
# query_rules isn't available or the session can't be identified.
# ============================================================
def _detect_session():
    """Guess the current session id from the newest file in
    data_dir/main, when --session isn't given."""
    try:
        main_dir = os.path.join(data_dir(), "main")
        files = sorted(glob.glob(os.path.join(main_dir, "*.jsonl")),
                       key=os.path.getmtime, reverse=True)
        for fp in files:
            try:
                with open(fp, encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except Exception:
                continue
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("session"):
                    return d.get("session")
    except Exception:
        pass
    return None


def _compact_lines(session, n, include_action=False):
    lines = []
    if _qr is None:
        lines.append("query_rules is unavailable, so compaction boundaries can't be read.")
        return lines
    sess = session or _detect_session()
    if not sess:
        lines.append("Couldn't identify a session (pass --session=ID to specify one).")
        return lines
    try:
        start, end = _qr.compact_range(sess, int(n))
    except Exception as ex:
        _note_error("compact_range", ex)
        lines.append("Couldn't read compaction boundaries.")
        return lines
    if start is None:
        lines.append("This session has %d compaction boundary(ies)." % end)
        return lines
    omitted = 0
    try:
        s_dt = datetime.datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        e_dt = datetime.datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        if s_dt.tzinfo is None:
            s_dt = s_dt.replace(tzinfo=datetime.timezone.utc)
        if e_dt.tzinfo is None:
            e_dt = e_dt.replace(tzinfo=datetime.timezone.utc)
        # --compact used to be an unbounded "everything since the
        # boundary" pull -- search_time's own row cap (see
        # DATE_SCOPE_ROW_CAP) now applies here too, since this is the
        # same date-scoped-with-no-keywords path (#106 point 1).
        rows, _meta = index_exact.search_time((s_dt, e_dt), None, [], include_action=include_action)
        omitted = int(_meta.get("omitted", 0))
    except Exception as ex:
        _note_error("compact(search_time)", ex)
        rows = []
    lines.append("-- before compaction: %d row(s) [%s, %s) --" % (len(rows), _ts_jst(start), _ts_jst(end)))
    if omitted:
        lines.append("-- %d earlier row(s) omitted (oldest first) -- pass --action or narrow the range for more" % omitted)
    for h in rows:
        ts = _ts_jst(h.get("ts"))
        tag = _tag_for(h)
        text = (h.get("text") or "").replace("\n", " ")
        lines.append("[%s] %s: %s" % (ts, tag, text))
    return lines


# --full used to mean "no cap at all" on a row's character count. This
# is the cap that replaces that -- provisional, like the other caps in
# this module: not yet tuned against how long a genuinely-needed row
# actually gets, just high enough that --full is still useful for the
# common case (a long-but-not-huge message) without letting one giant
# tool-output row make the response unbounded again.
DEFAULT_CHAR_CAP = 300
FULL_CHAR_CAP = 4000


def recall(query, limit=6, recency=True, actor=None, around=0, full=False,
          compact_n=0, session=None, include_action=False):
    """The manual entry point. Calls fetch() and formats the result
    for display. Returns a list of display lines, [NOW] first."""
    del LAST_MODE[:]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = ["[NOW] " + now + " -- anchor on the current time before reading anything below."]

    if compact_n:
        lines.extend(_compact_lines(session, compact_n, include_action=include_action))
        return lines

    _dr, _tr, _kws = _split_time(query)
    if _dr is None and _tr is not None:
        lines.append("A time-of-day alone can't narrow the range (searching all time). "
                     "Add a date to narrow it (e.g. \"2026-09-02 evening budget\").")

    uniq = fetch(query, actor=actor, around=around, log_limit=limit, recency=recency,
                include_action=include_action)
    if "exact-only" in LAST_MODE:
        lines.append("Semantic search is unavailable right now; showing exact date/keyword matches only.")

    if LAST_TAIL and "time" not in LAST_MODE:
        lines.append("-- most recent (not yet indexed) --")
        for h in LAST_TAIL:
            ts = _ts_jst(h.get("ts"))
            text = (h.get("text") or "").replace("\n", " ")[:200]
            lines.append("[%s] %s: %s" % (ts, h.get("actor", ""), text))

    if uniq:
        char_cap = FULL_CHAR_CAP if full else DEFAULT_CHAR_CAP
        ordered = sorted(uniq, key=lambda r: (r.get("ts") or ""))
        texts_show = []
        n_truncated = 0
        for h in ordered:
            t = (h.get("text") or "").replace("\n", " ")
            if len(t) > char_cap:
                t = t[:char_cap] + "..."
                n_truncated += 1
            texts_show.append(t)
        total_chars = sum(len(t) for t in texts_show)
        # Only claim a truncation happened if one actually did -- the
        # old note was unconditional ("(truncated to 300 chars)") even
        # when every row was already under the cap, which is the kind
        # of small mismatch #106's "verbatim, and say so if trimmed"
        # criterion is about.
        cut_note = " (%d row(s) truncated to %d chars)" % (n_truncated, char_cap) if n_truncated else ""

        omitted = sum(LAST_OMITTED) if LAST_OMITTED else 0

        if "time" in LAST_MODE:
            meta0 = LAST_TIME_META[0] if LAST_TIME_META else {}
            types = meta0.get("types") or {}
            _label = {"text": "message", "action": "action", "thinking": "thinking", "meta": "meta"}
            _type_parts = ", ".join("%s %d" % (_label.get(k, k), v) for k, v in types.items())
            _head = "-- date-scoped: %d row(s), ~%d chars%s (chronological)" % (len(ordered), total_chars, cut_note)
            if _type_parts:
                _head += " -- " + _type_parts
            lines.append(_head + " --")
            if omitted:
                lines.append("-- %d earlier row(s) omitted (oldest first, kept the most recent) -- "
                             "narrow the range or add a keyword for the rest" % omitted)
            if meta0.get("level") == "L2":
                lines.append("Few keyword matches, so showing the whole range chronologically instead.")
            if meta0.get("fallback_day"):
                lines.append("No date was given, so showing the most recent %s that matches." % meta0["fallback_day"])
        else:
            lines.append("-- %d row(s), ~%d chars%s (verbatim, timestamped, chronological) --"
                        % (len(ordered), total_chars, cut_note))
            if omitted:
                lines.append("-- %d row(s) omitted by --around's cap (oldest first) --" % omitted)
        if "L3" in LAST_MODE:
            lines.append("Nothing in that date range, so searching all time by meaning instead.")

        for h, t in zip(ordered, texts_show):
            ts = _ts_jst(h.get("ts"))
            tag = _tag_for(h)
            lines.append("[%s] %s: %s" % (ts, tag, t))

        _days = {}
        _kw_total = None
        if "time" in LAST_MODE and LAST_TIME_META:
            _days = dict(LAST_TIME_META[0].get("days") or {})
            _kw_total = LAST_TIME_META[0].get("kw_total")
        else:
            for h in ordered:
                d = _ts_jst(h.get("ts"))[:10]
                _days[d] = _days.get(d, 0) + 1
        if len(_days) > 1:
            _pre = ""
            if _kw_total and _kw_total > len(ordered):
                _pre = "keyword matches: %d total -- " % _kw_total
            lines.append("-- " + _pre + "days matched: "
                         + " / ".join("%s (%d)" % (d, n) for d, n in sorted(_days.items()))
                         + " -- spread across multiple days; naming one would narrow this")

    if "time" in LAST_MODE and not uniq:
        _now_utc = datetime.datetime.now(datetime.timezone.utc)
        if _dr is not None and _dr[0] > _now_utc:
            lines.append("That range is in the future -- there's nothing recorded yet.")
        else:
            _total_in_range = LAST_TIME_TOTAL[0] if LAST_TIME_TOTAL else 0
            if _total_in_range:
                lines.append("Nothing matched the keyword(s), but the index has %d row(s) in that date range "
                             "(not shown, or excluded because they're type=action -- pass --action to include those)."
                             % _total_in_range)
            else:
                lines.append("Nothing on record in that range (the range itself resolved correctly). "
                             "Try different wording or double-check the date.")
    if LAST_ERRORS:
        lines.append("Note: part of the search failed (the index may be mid-rebuild). Results below are incomplete:")
        for e in LAST_ERRORS:
            lines.append("  " + e)
    # "nothing was found" used to be detected as len(lines) == 1 (only
    # the [NOW] line present), which silently broke the moment any
    # other note got appended -- most commonly LAST_ERRORS' own
    # index_vector(import) ModuleNotFoundError, which fires on *every*
    # query whenever the optional sentence-transformers/sqlite-vec
    # dependency isn't installed (the expected, supported state per
    # #106's own setup instructions). That made point 5's diagnostic
    # unreachable in exactly the environment it was written for.
    # Check the actual "did anything get shown" condition instead.
    if not uniq and not LAST_TAIL and "time" not in LAST_MODE:
        # Absence should be told apart from "filtered out" *and* from
        # "couldn't check": a raw, type-inclusive bigram count against
        # the index tells whether this is truly nothing on record, or
        # something's there but didn't survive ranking/recency/the
        # type filter (#106 point 5; e.g. "feed-digest" used to report
        # 0 here while the index had 5 -- all type=action, silently
        # excluded by default).
        #
        # raw_hit_count() returns None (never 0) when the index itself
        # couldn't be read (corrupt/locked/mid-rebuild). That must NOT
        # be reported as "confirmed 0" -- #106 review point 2 found
        # this module claiming "the index has 0 matches ... isn't just
        # a filtering artifact" immediately after logging that the
        # index read had actually failed. "Couldn't check" and
        # "checked, found nothing" are different claims and must read
        # differently here.
        try:
            _idx_all = index_exact.raw_hit_count(query, actor=actor, include_action=True)
        except Exception:
            _idx_all = None
        if _idx_all is None:
            lines.append("No hits above the relevance/recency cutoff, and the index itself couldn't be checked "
                         "just now (see the error note above if one was logged) -- this is NOT confirmed absence, "
                         "just unknown. Try again once the index is readable.")
        elif _idx_all:
            if include_action:
                lines.append("No hits above the relevance/recency cutoff, but the index has %d loose match(es) "
                             "for this text -- try --around or different wording." % _idx_all)
            else:
                try:
                    _idx_visible = index_exact.raw_hit_count(query, actor=actor, include_action=False)
                except Exception:
                    _idx_visible = None
                if _idx_visible == 0:
                    lines.append("No hits: the index has %d loose match(es) for this text, but all of them are "
                                 "type=action (tool-call bodies, excluded by default) -- pass --action to include them."
                                 % _idx_all)
                elif _idx_visible is None:
                    lines.append("No hits above the relevance/recency cutoff; the index has %d loose match(es) "
                                 "overall, but whether any of those are visible (non-action) couldn't be checked "
                                 "just now -- try --around, different wording, or --action." % _idx_all)
                else:
                    lines.append("No hits above the relevance/recency cutoff, but the index has %d loose match(es) "
                                 "for this text -- try --around or different wording." % _idx_all)
        else:
            lines.append("No hits. Absence is reported as absence, never guessed around -- "
                         "but try at least one more phrasing (a different name for the same thing) before concluding there's nothing. "
                         "The index was checked and has 0 matches for this text either, so this isn't just a filtering artifact.")
    return lines


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("--")]
    actor = None
    if "--user" in argv:
        actor = ACTOR_USER
    elif "--ai" in argv:
        actor = ACTOR_AI
    around = 0
    for a in argv:
        if a.startswith("--around"):
            try:
                around = int(a.split("=", 1)[1]) if "=" in a else 1
            except ValueError:
                around = 1
            around = max(0, around)
    if around > index_exact.AROUND_MAX_N:
        around = index_exact.AROUND_MAX_N
    full = "--full" in argv
    include_action = "--action" in argv
    compact_n = 0
    for i, a in enumerate(argv):
        if a.startswith("--compact"):
            try:
                if "=" in a:
                    compact_n = int(a.split("=", 1)[1])
                elif i + 1 < len(argv) and argv[i + 1].isdigit():
                    compact_n = int(argv[i + 1])
                else:
                    compact_n = 1
            except ValueError:
                compact_n = 1
    session = None
    for a in argv:
        if a.startswith("--session="):
            session = a.split("=", 1)[1]

    if compact_n:
        for line in recall("", compact_n=compact_n, session=session, include_action=include_action):
            print(line)
        return

    if not args or not args[0].strip():
        print('usage: python -m lossless_memory.recall "query" [options]')
        print("The one entry point for pulling memory back out; don't call index_exact / index_vector directly.")
        print("")
        print('The strongest query is "date + word(s)": naming a date scopes the search to that range only.')
        print('  example: python -m lossless_memory.recall "2026-09-02 budget"')
        print('           python -m lossless_memory.recall "2026-09-02"   <- no words = show the whole day')
        print("  dates understood: an explicit YYYY-MM-DD / M/D, or the Japanese relative words")
        print("  (today/yesterday/N days ago/last week/...). English relative dates are not supported yet.")
        print("")
        print("--around[=N]  = also show N rows before/after each hit (default 1, capped at %d; a cap is"
              % index_exact.AROUND_MAX_N)
        print("                always noted if it drops any rows, never silent)")
        print("--user        = only rows from the configured user_name")
        print("--ai          = only rows from the configured ai_name")
        print("--action      = include type=action rows (tool-call bodies; excluded by default)")
        print("--all-time    = no recency decay (weigh old and new equally)")
        print("--full        = don't truncate rows to %d chars (still capped at %d, not unbounded)"
              % (DEFAULT_CHAR_CAP, FULL_CHAR_CAP))
        print("--compact[=N] = everything since the N-th-to-last compaction boundary (claude_code format only,")
        print("                row count capped like a date-scoped query; a drop is always noted)")
        print("--session=ID  = session id to use with --compact (default: auto-detected)")
        sys.exit(1)

    query = " ".join(args)
    lines = recall(query, recency=("--all-time" not in sys.argv),
                   actor=actor, around=around, full=full, include_action=include_action)
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
