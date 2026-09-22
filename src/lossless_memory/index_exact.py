# -*- coding: utf-8 -*-
"""index_exact -- exact-match recall: full-text search plus a temporal
ruler.

Role:
  Reads the converted logs (ingest.py's output, under
  <data_dir>/main/*.jsonl, plus any extra corpora under
  <data_dir>/corpus/<name>/*.jsonl) and searches them "exactly" with
  SQLite FTS5.

Philosophy:
  - This is the exact-match path: keyword / date / time-of-day filters
    return the matching raw rows verbatim, never summarized. No
    hallucination is possible here -- it can only return what is
    literally on record.
  - The converted logs are the source of truth; this module only
    reads them. The search index (index_exact.db) is a derived
    artifact -- delete it and build_index() rebuilds it from the logs.

Japanese support:
  Japanese has no word boundaries, so text is split into bigrams
  (2-character windows) before being stored in FTS5, and queries are
  bigram-split the same way before matching.

Date / time-of-day vocabulary:
  The date and time-of-day words this module understands ("today",
  "yesterday", "last week", "at night", ...) are Japanese, since that
  is the vocabulary this project was built and tested against.
  Explicit ISO dates (YYYY-MM-DD) and bare M/D dates are language
  neutral and always work. See the project README for the exact
  limitation.

  Row timestamps are stored as UTC ISO8601. The date/time-of-day words
  a user speaks are interpreted in JST (UTC+9) and converted before
  comparison, so "today" means "today in Japan time", not UTC.
"""
import os
import re
import json
import glob
import sqlite3
import hashlib
from datetime import datetime, timedelta, timezone, time as _time

try:
    from . import query_rules as _qr
except Exception:
    _qr = None

from .config import data_dir

_LOG_MAIN = None  # resolved lazily so tests can point data_dir() elsewhere
_CORE_FIELDS = ("ts", "actor", "role", "type", "text", "model", "session")

_UTC = timezone.utc
_JST = timezone(timedelta(hours=9))


def _log_main():
    return os.path.join(data_dir(), "main")


def _log_corpus():
    return os.path.join(data_dir(), "corpus")


def _index_db():
    return os.path.join(data_dir(), "index_exact.db")


# ============================================================
# Query pre-processing dictionaries (filler words, stopwords).
# Defaults live in query_rules.py; a settings file can extend them.
# Time words themselves (today/yesterday/at night/...) are handled
# separately below and are *not* filtered here -- _extract_date and
# _extract_time_range read them before they're stripped.
# ============================================================
_FILLERS = [
    "えっと", "えーっと", "えっとー", "えーと", "え〜と", "え〜っと",
    "あの", "あのー", "あのねー", "あー", "あーっと",
    "うー", "うーん", "んー", "んーっと", "んーと",
    "なんていうか", "なんていうの", "なんかさ", "なんかね",
    "なんだろう", "何だろう", "なんていうんだろう", "何て言うんだろう",
    "だから", "だからー", "だからさ", "だからね",
    "ねえ", "ねー", "ほら", "ほらね",
    "まあ", "まあね", "まあさ",
    "そうそう", "そういや", "そうそうそう",
    "いやー", "いやいや", "いやちょっと", "いや別に", "いやだから",
    "やっぱ", "やっぱり", "やっぱね", "やっぱさ",
    "じゃあ", "ちょっと", "多分", "ありがとう",
]

_TAIL_FILLERS = [
    "じゃん", "じゃんね", "じゃない", "じゃないかな", "じゃないか",
    "だよね", "だよ", "だね", "なんだよね", "なんだよ",
    "かな", "かなあ", "かなぁ", "かしら",
    "だっけ", "だっけな", "だっけかな", "っけ",
    "とか", "みたいな", "って感じ", "って感じかな",
    "いいかな", "からね", "んだけど", "んだけどもね",
]

# Stopwords: low-signal words plus the trigger words themselves
# ("remember", "recall", ...) so a "do you remember X" query doesn't
# turn "remember" into a search keyword. Time words are also stopped
# here since _extract_date / _extract_time_range read them separately.
_STOPWORDS = {
    "これ", "それ", "あれ", "どれ", "この", "その", "どの",
    "ここ", "そこ", "あそこ", "どこ",
    "ちょっと", "さっき", "今度", "今日", "昨日", "一昨日", "明日",
    "先週", "先月", "来週", "来月", "最近", "前回", "おととい",
    "たぶん", "きっと", "まあ", "なんか", "なんだか",
    "覚えてる", "覚えて", "覚えてるかい", "覚えていない",
    "憶えてる", "憶えて", "憶えてるかい",
    "言った", "言ってた", "言ってたじゃん", "話した", "話してた",
    "思い出", "思い出して", "記憶", "記録",
    "行った", "行ってた", "行きました", "やった", "やってた", "した", "してた",
    "一緒に", "について", "に関して",
    "一緒", "本当", "最初", "最後", "大体", "結局", "結果",
    "全部", "本気", "実際", "全然", "絶対", "普通", "感じ", "様子",
    "場合", "時間", "気分", "今回", "状態", "状況",
    "大丈夫", "ちゃんと", "一応", "まず", "とりあえず",
    "早速", "元々", "全く", "別に",
    "参照", "インスタンス", "形式", "テスト",
}


def _filters():
    """Return (fillers, tail_fillers, stopwords), from query_rules if
    available, else the built-in defaults above."""
    if _qr is not None:
        try:
            f = _qr.load_filters()
            return (f.get("fillers") or _FILLERS, f.get("tail_fillers") or _TAIL_FILLERS,
                    f.get("stopwords") or _STOPWORDS)
        except Exception:
            pass
    return _FILLERS, _TAIL_FILLERS, _STOPWORDS


def _strip_query(text):
    """Strip filler words and stopwords from a query, leaving the
    substance. Falls back to the original text if stripping empties
    it out."""
    if not text:
        return ""
    s = str(text)
    fillers, tail_fillers, stopwords = _filters()
    for f in sorted(fillers, key=len, reverse=True):
        s = s.replace(f, " ")
    for w in sorted(stopwords, key=len, reverse=True):
        s = s.replace(w, " ")
    changed = True
    while changed:
        changed = False
        stripped = s.rstrip("?？!！。、 　")
        for tf in tail_fillers:
            if stripped.endswith(tf):
                s = stripped[:-len(tf)].rstrip()
                changed = True
                break
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return "".join(str(text).split())
    return s


def _extract_keywords(text):
    """Pull out up to 5 meaningful chunks from a search query: runs of
    2+ katakana, 2+ kanji, kanji+okurigana, or alphanumeric tokens.
    Stopwords are excluded. If query_rules is available, also expands
    each keyword through its correction dictionary (e.g. mishearings)."""
    if not text:
        return []
    keywords = []
    seen = set()
    _, _tf, stopwords = _filters()
    _stop_lower = {s.lower() for s in stopwords}
    pattern = r"[ァ-ヶー]{2,}|[一-龥々]{2,}|[一-龥々][ぁ-ん]|[A-Za-z0-9][A-Za-z0-9_\-]+"
    # A single kanji followed by a particle ("日に", "何の") isn't a
    # useful keyword; a single kanji followed by an inflection ending
    # ("眠い", "見た") is, so only the particle case is dropped.
    _particles = set("にのをはがでともへやかねよさわ")
    for m in re.finditer(pattern, str(text)):
        kw = m.group(0)
        if len(kw) < 2:
            continue
        if len(kw) == 2 and "一" <= kw[0] <= "鿿" and kw[1] in _particles:
            continue
        if kw in stopwords or kw.lower() in _stop_lower:
            continue
        if kw in seen:
            continue
        keywords.append(kw)
        seen.add(kw)
        if len(keywords) >= 5:
            break
    if _qr is not None:
        try:
            keywords = _qr.expand_keywords(keywords)
        except Exception:
            pass
    return keywords


# Markers of an "I don't have that" response. These are pushed to the
# back of the result order (never dropped) so a real hit outranks a
# hedge that happens to contain the same words.
_NEGATIVE_MARKERS = (
    "残っておりません", "残っていません", "残っておらず",
    "覚えておりません", "覚えていません", "覚えてはおりません",
    "ございません",
    "思い出せません", "思い出せず",
    "持っておりません", "持っていません",
    "現在のわたし", "今のわたし",
    "情報だけだと", "わかりませんでした", "分かりませんでした",
)


def _is_negative(text):
    if not text:
        return False
    return any(m in text for m in _NEGATIVE_MARKERS)


def _bigram(text):
    """Split whitespace-stripped text into 2-character windows, for
    FTS5 storage (Japanese has no word boundaries to tokenize on)."""
    if not text:
        return ""
    s = "".join(str(text).split())
    if len(s) <= 1:
        return s
    return " ".join(s[i:i + 2] for i in range(len(s) - 1))


def _bigram_query(text):
    """Same bigram split, joined with OR so any overlapping bigram
    counts as a partial match."""
    if not text:
        return ""
    s = "".join(str(text).split())
    if len(s) <= 1:
        return f'"{s}"' if s else ""
    grams = [s[i:i + 2] for i in range(len(s) - 1)]
    return " OR ".join(f'"{g}"' for g in grams)


def _bigram_phrase_query(text):
    """A *quoted* FTS5 phrase built the same way the bigram column is
    populated (see _bigram): the stored column is a space-joined
    sequence of overlapping 2-char windows, so a quoted phrase built
    from that same sequence only matches when the windows appear
    *consecutively* -- which is exactly the condition for the original
    (whitespace-stripped) text to appear as a literal substring.

    This is a much stronger match than _bigram_query's OR-of-any-window
    (which matches if any single 2-char fragment appears anywhere), and
    is used as the first, most precise attempt before falling back to
    the looser OR query. See "structured index" note in search()."""
    if not text:
        return ""
    s = "".join(str(text).split())
    if len(s) <= 1:
        return f'"{s}"' if s else ""
    return f'"{_bigram(s)}"'


def _keyword_and_query(keywords):
    """Multiple keywords joined with AND, e.g. ["work", "sleepy"] ->
    ("work") AND ("sleepy")."""
    groups = []
    for kw in keywords:
        bq = _bigram_query(kw)
        if bq:
            groups.append(f"({bq})")
    return " AND ".join(groups)


# ============================================================
# Temporal ruler: turns a spoken date/time-of-day into a UTC range
# comparable against the stored ts. Row ts is UTC; the words a user
# speaks ("today", "3 days ago", "at night") are JST.
# ============================================================
def _parse_ts(ts_str):
    """Parse a stored ts string into an aware UTC datetime, or None."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(str(ts_str))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_UTC)
        return dt.astimezone(_UTC)
    except (ValueError, TypeError):
        return None


def _jst_day_range_utc(d):
    """The [00:00, 23:59:59.999999] window of JST date d, as a (start,
    end) pair of aware UTC datetimes."""
    start_jst = datetime.combine(d, _time(0, 0, 0, 0), tzinfo=_JST)
    end_jst = datetime.combine(d, _time(23, 59, 59, 999999), tzinfo=_JST)
    return (start_jst.astimezone(_UTC), end_jst.astimezone(_UTC))


_RE_DATE_YMD = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")
_RE_DATE_MD_JP = re.compile(r"(\d{1,2})月(\d{1,2})日")
_RE_MONTH_JP = re.compile(r"(?<![\d年])(\d{1,2})月(?![\d\s]*日)")
_RE_DAYS_AGO = re.compile(r"(?<!\d)(\d{1,4})日前")
_RE_WEEKS_AGO = re.compile(r"(?<!\d)(\d{1,4})週間前")
_RE_YEARS_AGO = re.compile(r"(?<!\d)(\d{1,2})年前")
_RE_MONTHS_AGO = re.compile(r"(?<!\d)(\d{1,2})(?:ヶ|か|カ)月前")
_RE_DATE_MD = re.compile(r"(?<![\d:])(\d{1,2})/(\d{1,2})(?!\d)")
_DATE_REL_WORDS = ("一昨日", "おととい", "昨夜", "昨日", "今朝", "今夜", "今晩", "今日", "先々月", "先週", "先月", "一昨年", "去年")


def _year_range(y):
    return (_jst_day_range_utc(datetime(y, 1, 1).date())[0],
            _jst_day_range_utc(datetime(y, 12, 31).date())[1])


def _extract_date(text):
    """Read a date out of the user's words (JST) and return a UTC
    (start, end) datetime pair, or None. Recognizes explicit
    YYYY-MM-DD / YYYY/MM/DD / M/D dates (language neutral), Japanese
    month/day forms, and Japanese relative words (today, yesterday,
    N days/weeks/months/years ago, last week/month, ...)."""
    if not text:
        return None
    s = str(text)
    today = datetime.now(_JST).date()

    m = _RE_DATE_YMD.search(s)
    if m:
        try:
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
            return _jst_day_range_utc(d)
        except ValueError:
            pass

    m = _RE_DATE_MD_JP.search(s)
    if m:
        try:
            d = datetime(today.year, int(m.group(1)), int(m.group(2))).date()
            return _jst_day_range_utc(d)
        except ValueError:
            pass

    # "July" alone -> that whole month, this year (or last year if the
    # named month hasn't happened yet this year).
    m = _RE_MONTH_JP.search(s)
    if m:
        mo = int(m.group(1))
        if 1 <= mo <= 12:
            y = today.year if mo <= today.month else today.year - 1
            first = datetime(y, mo, 1).date()
            nxt = datetime(y + 1, 1, 1).date() if mo == 12 else datetime(y, mo + 1, 1).date()
            return (_jst_day_range_utc(first)[0], _jst_day_range_utc(nxt - timedelta(days=1))[1])

    m = _RE_DAYS_AGO.search(s)
    if m:
        d = today - timedelta(days=int(m.group(1)))
        return _jst_day_range_utc(d)

    m = _RE_WEEKS_AGO.search(s)
    if m:
        base = today - timedelta(weeks=int(m.group(1)))
        start = _jst_day_range_utc(base - timedelta(days=3))[0]
        end = _jst_day_range_utc(base + timedelta(days=3))[1]
        return (start, end)

    m = _RE_YEARS_AGO.search(s)
    if m:
        try:
            return _year_range(today.year - int(m.group(1)))
        except ValueError:
            pass

    m = _RE_MONTHS_AGO.search(s)
    if m:
        y, mo = today.year, today.month - int(m.group(1))
        while mo <= 0:
            mo += 12
            y -= 1
        try:
            first = datetime(y, mo, 1).date()
            nxt = datetime(y + 1, 1, 1).date() if mo == 12 else datetime(y, mo + 1, 1).date()
            return (_jst_day_range_utc(first)[0], _jst_day_range_utc(nxt - timedelta(days=1))[1])
        except ValueError:
            pass

    if "一昨日" in s or "おととい" in s:
        return _jst_day_range_utc(today - timedelta(days=2))
    if "昨夜" in s:  # last night = yesterday (time-of-day handled by _extract_time_range)
        return _jst_day_range_utc(today - timedelta(days=1))
    if "今朝" in s or "今夜" in s or "今晩" in s:  # this morning/tonight = today
        return _jst_day_range_utc(today)
    if "昨日" in s:
        return _jst_day_range_utc(today - timedelta(days=1))
    if "今日" in s:
        return _jst_day_range_utc(today)
    if "先週" in s:
        start = _jst_day_range_utc(today - timedelta(days=7))[0]
        end = _jst_day_range_utc(today - timedelta(days=1))[1]
        return (start, end)
    if "先々月" in s:
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        first_prev = last_prev.replace(day=1)
        last_prev2 = first_prev - timedelta(days=1)
        return (_jst_day_range_utc(last_prev2.replace(day=1))[0], _jst_day_range_utc(last_prev2)[1])
    if "先月" in s:
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        first_prev = last_prev.replace(day=1)
        start = _jst_day_range_utc(first_prev)[0]
        end = _jst_day_range_utc(last_prev)[1]
        return (start, end)
    if "一昨年" in s:
        return _year_range(today.year - 2)
    if "去年" in s:
        return _year_range(today.year - 1)

    m = _RE_DATE_MD.search(s)
    if m:
        try:
            d = datetime(today.year, int(m.group(1)), int(m.group(2))).date()
            return _jst_day_range_utc(d)
        except ValueError:
            pass

    return None


# Named time-of-day windows (JST). End is exclusive.
_TIME_WORDS = [
    ("深夜", (0, 5)),
    ("早朝", (4, 7)),
    ("明け方", (3, 6)),
    ("朝", (5, 11)),
    ("午前", (5, 12)),
    ("昼間", (9, 17)),
    ("昼", (11, 14)),
    ("正午", (11, 13)),
    ("午後", (12, 18)),
    ("夕方", (16, 19)),
    ("夕", (16, 19)),
    ("夜中", (22, 24)),
    ("夜", (18, 24)),
    ("晩", (18, 23)),
]

# Words that look like a time-of-day mention but aren't (a duration,
# or a fixed compound like "dinner"/"all-nighter"). Stripped before
# matching so they don't get misread as a time filter.
_TIME_FALSE_FRIENDS = re.compile(
    r"(夕食|夕飯|夕張|朝食|朝飯|朝礼|昼食|昼飯|夜食|夜勤|徹夜|一昼夜)")


def _extract_time_range(text):
    """Read a time-of-day out of the user's words (JST) and return
    (start_hour, end_hour), or None."""
    if not text:
        return None
    s = _TIME_FALSE_FRIENDS.sub(" ", str(text))

    m = re.search(r"(\d{1,2})時頃", s)
    if m:
        h = int(m.group(1)) % 24
        return (max(0, h - 1), min(24, h + 2))

    m = re.search(r"(\d{1,2})時(?!間)", s)  # "N時間" is a duration, not a time
    if m:
        h = int(m.group(1)) % 24
        return (h, min(24, h + 1))

    for word, rng in sorted(_TIME_WORDS, key=lambda x: len(x[0]), reverse=True):
        if word in s:
            return rng
    return None


def _in_date_range(ts_dt, date_range):
    if date_range is None:
        return True
    if ts_dt is None:
        return False
    start, end = date_range
    return start <= ts_dt <= end


def _in_time_range(ts_dt, time_jst):
    if time_jst is None:
        return True
    if ts_dt is None:
        return False
    h = ts_dt.astimezone(_JST).hour
    start, end = time_jst
    return start <= h < end


def _src_key(fp):
    """Tag rows with where they came from, e.g. "main/2026-08-29" or
    "corpus/notes/2026-01-26"."""
    base = os.path.dirname(_log_main())
    rel = os.path.relpath(fp, base).replace("\\", "/")
    return rel[:-6] if rel.endswith(".jsonl") else rel


def _insert_rows(con, table, src, lines):
    n = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a corrupted line is skipped, not fixed -- source logs are read-only
        con.execute(
            "INSERT INTO %s"
            "(bigram, ts, actor, role, type, text, model, session, src)"
            " VALUES (?,?,?,?,?,?,?,?,?)" % table,
            (
                _bigram(r.get("text", "")),
                r.get("ts"), r.get("actor"), r.get("role"),
                r.get("type"), r.get("text"), r.get("model"), r.get("session"),
                src,
            ),
        )
        n += 1
    return n


def _tail_sig(lines, k):
    """Fingerprint of line k (used to detect a rewritten file before
    trusting an append)."""
    if k <= 0 or k > len(lines):
        return ""
    return hashlib.sha1(lines[k - 1].encode("utf-8", "replace")).hexdigest()[:16]


def build_index(force=False):
    """(Re)build the FTS5 index from <data_dir>/main and
    <data_dir>/corpus/*/*.jsonl.

    Normally does an incremental append: each source file's previous
    line count is tracked in src_progress, and only new lines are
    inserted. A file whose line count shrank, or whose last-seen line
    changed, is fully re-indexed. force=True (or no usable existing
    index) triggers a full rebuild, built into a side table and then
    swapped in atomically so readers never see an empty index while
    it's being rebuilt.

    Returns the index's total row count."""
    index_db = _index_db()
    os.makedirs(os.path.dirname(index_db), exist_ok=True)
    con = sqlite3.connect(index_db)
    try:
        con.execute("PRAGMA journal_mode=WAL")  # don't block readers while writing
        files = sorted(glob.glob(os.path.join(_log_main(), "*.jsonl")))
        files += sorted(glob.glob(os.path.join(_log_corpus(), "*", "*.jsonl")))
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info(recall)")]
        except sqlite3.Error:
            cols = []
        full = force or ("src" not in cols)
        ddl = ("CREATE VIRTUAL TABLE %s USING fts5("
               "  bigram,"
               "  ts UNINDEXED, actor UNINDEXED, role UNINDEXED,"
               "  type UNINDEXED, text UNINDEXED, model UNINDEXED, session UNINDEXED,"
               "  src UNINDEXED,"
               "  tokenize='unicode61'"
               ")")
        con.execute("CREATE TABLE IF NOT EXISTS src_progress"
                    "(src TEXT PRIMARY KEY, nlines INTEGER, sig TEXT)")
        if full:
            con.execute("DROP TABLE IF EXISTS recall_new")
            con.execute(ddl % "recall_new")
            prog = {}
            for fp in files:
                src = _src_key(fp)
                with open(fp, encoding="utf-8") as f:
                    lines = f.readlines()
                _insert_rows(con, "recall_new", src, lines)
                prog[src] = (len(lines), _tail_sig(lines, len(lines)))
            con.commit()
            con.isolation_level = None
            con.execute("BEGIN IMMEDIATE")
            con.execute("DROP TABLE IF EXISTS recall")
            con.execute("ALTER TABLE recall_new RENAME TO recall")
            con.execute("DELETE FROM src_progress")
            con.executemany("INSERT INTO src_progress(src, nlines, sig) VALUES (?,?,?)",
                            [(k, v[0], v[1]) for k, v in prog.items()])
            con.execute("COMMIT")
            con.isolation_level = ""
        else:
            prog = {r[0]: (r[1], r[2]) for r in con.execute("SELECT src, nlines, sig FROM src_progress")}
            seen = set()
            for fp in files:
                src = _src_key(fp)
                seen.add(src)
                with open(fp, encoding="utf-8") as f:
                    lines = f.readlines()
                cur = len(lines)
                done, sig = prog.get(src, (0, ""))
                if cur == done and _tail_sig(lines, cur) == sig:
                    continue  # unchanged
                if cur < done or _tail_sig(lines, done) != sig:
                    con.execute("DELETE FROM recall WHERE src = ?", (src,))  # file was rewritten
                    done = 0
                _insert_rows(con, "recall", src, lines[done:])
                con.execute("INSERT OR REPLACE INTO src_progress(src, nlines, sig) VALUES (?,?,?)",
                            (src, cur, _tail_sig(lines, cur)))
            for src in list(prog):
                if src not in seen:  # source file disappeared -- drop its rows, not the logs
                    con.execute("DELETE FROM recall WHERE src = ?", (src,))
                    con.execute("DELETE FROM src_progress WHERE src = ?", (src,))
            con.commit()
        return con.execute("SELECT count(*) FROM recall").fetchone()[0]
    finally:
        con.close()


def _type_clause(include_action):
    """type != 'meta' is always excluded (compaction boundaries / harness
    noise, never a real message). type='action' (a tool call's own body)
    is excluded too unless include_action is True -- it's 61% of stored
    rows and, being high-volume/low-signal, otherwise crowds out real
    messages both in FTS rank order and in the final row count."""
    if include_action:
        return "type != 'meta'"
    return "type != 'meta' AND type != 'action'"


def _fetch_rows(con, match_query, actor, hard_limit, include_action=False):
    type_clause = _type_clause(include_action)
    if match_query:
        sql = ("SELECT rowid, ts, actor, role, type, text, model, session"
               " FROM recall WHERE bigram MATCH ? AND " + type_clause)
        params = [match_query]
        if actor:
            sql += " AND actor = ?"
            params.append(actor)
        sql += " ORDER BY rank LIMIT ?"
        params.append(int(hard_limit))
    else:
        # no keyword (date/time-of-day only): take everything, newest
        # first, and let the time filter below narrow it down.
        sql = ("SELECT rowid, ts, actor, role, type, text, model, session"
               " FROM recall WHERE " + type_clause)
        params = []
        if actor:
            sql += " AND actor = ?"
            params.append(actor)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(int(hard_limit))
    return [dict(r) for r in con.execute(sql, params)]


def _apply_time_filters(rows, date_range, time_jst):
    """Filter rows by date and/or time-of-day (JST). Whichever filter
    is None passes everything through."""
    if date_range is None and time_jst is None:
        return rows
    out = []
    for r in rows:
        ts_dt = _parse_ts(r.get("ts"))
        if _in_date_range(ts_dt, date_range) and _in_time_range(ts_dt, time_jst):
            out.append(r)
    return out


def _fetch_day(con, date_range, time_jst, actor, limit, include_action=False):
    """Date mode: return the whole day (or time-of-day window within
    it) in chronological order, for a "what did we talk about on day
    X" query -- the whole flow, not a scattering of fragments. If too
    many rows match, keep the most recent `limit` (chronological order
    is preserved). Returns (rows, omitted_count)."""
    sql = "SELECT ts, actor, role, type, text, model, session FROM recall WHERE " + _type_clause(include_action)
    params = []
    if actor:
        sql += " AND actor = ?"
        params.append(actor)
    sql += " ORDER BY ts ASC"
    rows = [dict(r) for r in con.execute(sql, params)]
    rows = _apply_time_filters(rows, date_range, time_jst)
    omitted = 0
    if len(rows) > limit:
        omitted = len(rows) - limit
        rows = rows[-limit:]
    return rows, omitted


def search(keyword, actor=None, limit=5, include_action=False):
    """Search by keyword and/or date and/or time-of-day, newest first.
    All three are optional and combine freely -- with nothing at all
    it just returns the most recent rows. Returns a list of raw-row
    dicts (the 7 core fields). Builds the index automatically if it
    doesn't exist yet.

    Two precise tiers are tried first and *merged* (not early-returned
    from individually -- see _MERGE note below); two looser tiers are
    each tried in turn only if that merge found nothing:
      1. a literal-substring phrase match (_bigram_phrase_query) --
         the "structured" half of "full-text + structure" (#106): the
         bigram column is a sequence, so a quoted phrase built the same
         way requires the windows to appear *consecutively*, which is
         equivalent to the original text appearing verbatim
      2. an AND of each extracted keyword's own bigram match (already
         fairly precise when 2+ keywords are found) -- also catches a
         row where the same words appear in a different order, or with
         other words between them, which the phrase tier cannot match
         by construction and which is the ordinary case in Japanese
         (word order isn't fixed the way it is in English)
      3. an OR of every bigram in the cleaned query (loosest -- may
         match on any 2-char fragment shared with unrelated text)
      4. no filter at all (most recent rows)
    type='action' rows (tool-call bodies) are excluded unless
    include_action=True.

    _MERGE (#106 review point 1): tier 1 used to return immediately on
    any hit, without ever trying tier 2 -- so a query whose words
    happen to appear contiguously *somewhere*, even in an unimportant
    row, would silently hide a more relevant row where the same words
    appear out of order or with something between them (real repro:
    "topic 注入 打ち切り" returned only a minor correction-note aside
    once the phrase tier matched it, burying the actual "注入打ち切り"
    status row that the AND tier alone finds top-ranked). Tiers 1 and
    2 are now always merged whenever either finds anything -- phrase
    hits first (stronger evidence: the exact text appears verbatim),
    AND hits as a supplement below, deduplicated -- rather than
    switching behavior based on how many phrase hits there happen to
    be, which would make the result set change discontinuously across
    similar queries."""
    if not os.path.exists(_index_db()):
        build_index()

    con = sqlite3.connect(_index_db())
    con.row_factory = sqlite3.Row
    try:
        date_range = _extract_date(keyword)        # read from the raw text, before stripping
        time_jst = _extract_time_range(keyword)
        cleaned = _strip_query(keyword)             # strip filler words
        keywords = _extract_keywords(cleaned)

        # date/time filters can remove rows, so over-fetch before trimming to limit
        hard_limit = int(limit) * 8 + 20

        def _run(match_query):
            rows = _fetch_rows(con, match_query, actor, hard_limit, include_action=include_action)
            rows = _apply_time_filters(rows, date_range, time_jst)
            rows.sort(key=lambda r: _is_negative(r.get("text", "")))  # negatives last (stable sort)
            for r in rows:
                r.pop("rowid", None)
            return rows[:int(limit)]

        def _row_key(r):
            return (r.get("ts"), r.get("actor"), r.get("text"), r.get("type"))

        phrase_rows = _run(_bigram_phrase_query(cleaned)) if cleaned.strip() else []
        and_rows = _run(_keyword_and_query(keywords)) if len(keywords) >= 2 else []
        if phrase_rows or and_rows:
            seen = set()
            merged = []
            for r in phrase_rows + and_rows:
                k = _row_key(r)
                if k in seen:
                    continue
                seen.add(k)
                merged.append(r)
                if len(merged) >= int(limit):
                    break
            if merged:
                return merged

        if cleaned.strip():
            rows = _run(_bigram_query(cleaned))
            if rows:
                return rows
        return _run("")
    finally:
        con.close()


def raw_hit_count(keyword, actor=None, include_action=True):
    """Diagnostic-only: how many rows match this query by the same
    tiered strategy search() uses (phrase, then AND-of-keywords, then
    OR-of-every-bigram -- see search()'s docstring), ignoring date/time
    scoping and any relevance/recency cutoff. Stops at the first tier
    that finds anything, same as search() does, rather than always
    using the loosest tier -- an OR-of-bigrams count alone is so loose
    it's rarely truly 0 for any real corpus, which would defeat the
    point of this function (telling "genuinely nothing on record"
    apart from "something's there but got filtered/ranked out",
    #106 point 5, e.g. when every match is type=action and the caller
    didn't ask to include it).

    Returns None -- never 0 -- if the index couldn't actually be read
    (corrupt file, mid-rebuild, locked, ...). 0 and "couldn't tell"
    used to be collapsed together here, which let a broken index be
    reported as confirmed proof of absence (#106 review point 2: a
    corrupt-index repro produced "index has 0 matches ... isn't just a
    filtering artifact" -- a false claim, since the index was never
    actually read). Callers must not treat None as 0."""
    try:
        if not os.path.exists(_index_db()):
            build_index()
        con = sqlite3.connect(_index_db())
        try:
            cleaned = _strip_query(keyword)
            if not cleaned.strip():
                return 0
            keywords = _extract_keywords(cleaned)

            def _count(match_query):
                if not match_query:
                    return 0
                sql = "SELECT count(*) FROM recall WHERE bigram MATCH ? AND " + _type_clause(include_action)
                params = [match_query]
                if actor:
                    sql += " AND actor = ?"
                    params.append(actor)
                return con.execute(sql, params).fetchone()[0]

            for mq in (_bigram_phrase_query(cleaned),
                       _keyword_and_query(keywords) if len(keywords) >= 2 else None,
                       _bigram_query(cleaned)):
                if not mq:
                    continue
                n = _count(mq)
                if n:
                    return n
            return 0
        finally:
            con.close()
    except (sqlite3.Error, OSError):
        return None


# --around's help text used to advertise "no cap" -- these are the caps
# that replace that. Provisional: nobody has yet measured how many
# rows/queries a session actually needs, so these are a first guess
# to close the "returns literally everything" hole, not a tuned value.
AROUND_MAX_N = 20        # clamp on the N in --around=N itself
AROUND_ROW_CAP = 120     # clamp on the total row count a context pull returns


def search_with_context(keyword, actor=None, hits=2, around=1, day_limit=40, include_action=False):
    """Like search(), but also returns `around` rows before and after
    each hit, from the same session, in chronological order with no
    duplicates. Returns (rows, omitted_count) -- omitted_count is the
    number of rows dropped by AROUND_ROW_CAP / day_limit, never silent."""
    around = max(0, min(int(around), AROUND_MAX_N))
    if not os.path.exists(_index_db()):
        build_index()
    con = sqlite3.connect(_index_db())
    con.row_factory = sqlite3.Row
    try:
        date_range = _extract_date(keyword)
        time_jst = _extract_time_range(keyword)
        cleaned = _strip_query(keyword)
        keywords = _extract_keywords(cleaned)

        # Date mode with no other keywords ("what did we talk about on
        # July 4th") means "show me the whole day", not "find a
        # fragment" -- use _fetch_day instead of a hit + context pull.
        if date_range is not None and not keywords:
            return _fetch_day(con, date_range, time_jst, actor, day_limit, include_action=include_action)

        hard_limit = int(hits) * 8 + 20

        def _candidates(match_query):
            rows = _fetch_rows(con, match_query, actor, hard_limit, include_action=include_action)
            rows = _apply_time_filters(rows, date_range, time_jst)
            return rows

        # Same merge as search() (#106 review point 1): a phrase hit
        # and an AND-of-keywords hit are combined, not early-returned
        # from individually, so a relevant row whose words appear out
        # of order (or with something between them) isn't hidden just
        # because some other row happens to contain the query as a
        # contiguous substring.
        phrase_cand = _candidates(_bigram_phrase_query(cleaned)) if cleaned.strip() else []
        and_cand = _candidates(_keyword_and_query(keywords)) if len(keywords) >= 2 else []
        cand = []
        if phrase_cand or and_cand:
            seen_rowids = set()
            for r in phrase_cand + and_cand:
                if r["rowid"] in seen_rowids:
                    continue
                seen_rowids.add(r["rowid"])
                cand.append(r)
        if not cand and cleaned.strip():
            cand = _candidates(_bigram_query(cleaned))
        if not cand:
            cand = _candidates("")
        if not cand:
            return [], 0

        cand.sort(key=lambda r: _is_negative(r.get("text", "")))
        hit_sel = cand[:int(hits)]

        rowid_session = {r["rowid"]: r["session"]
                         for r in con.execute("SELECT rowid, session FROM recall")}
        wanted = set()
        for h in hit_sel:
            rid = h["rowid"]
            sess = h.get("session")
            wanted.add(rid)
            for i in range(rid - around, rid + around + 1):
                if i >= 1 and rowid_session.get(i) == sess:
                    wanted.add(i)
        if not wanted:
            return [], 0
        omitted = 0
        if len(wanted) > AROUND_ROW_CAP:
            # keep the highest rowids (== most recent rows) -- same
            # "trim the oldest, say so" rule as the date-scoped cap.
            kept = sorted(wanted, reverse=True)[:AROUND_ROW_CAP]
            omitted = len(wanted) - len(kept)
            wanted = set(kept)
        placeholders = ",".join("?" * len(wanted))
        rows = con.execute(
            "SELECT rowid, ts, actor, role, type, text, model, session"
            f" FROM recall WHERE rowid IN ({placeholders}) ORDER BY rowid ASC",
            sorted(wanted),
        )
        result = []
        for r in rows:
            d = dict(r)
            d.pop("rowid", None)
            result.append(d)
        # Context rows are pulled by adjacent rowid within the same
        # session, so a hit near a date boundary can pull in the
        # previous/next day. If a date/time filter was given, re-apply
        # it to the context rows too.
        if date_range is not None or time_jst is not None:
            result = _apply_time_filters(result, date_range, time_jst)
        return result, omitted
    finally:
        con.close()


def _split_time_query(query):
    """The single source of truth for splitting a query into
    (date_range, time_jst, keywords). Callers must not re-parse dates
    themselves -- two independent regexes drifting apart is how a
    keyword like the day-of-month in "July 19th" ends up re-matched as
    a search term.

    - date_range: from _extract_date (UTC start/end pair) or None
    - time_jst:   from _extract_time_range (JST hour range) or None
    - keywords:   the remaining search terms with all time words
                  removed (max 5, per _extract_keywords)
    """
    s = str(query or "")
    date_range = _extract_date(s)
    time_jst = _extract_time_range(s)

    masked = s
    for rx in (_RE_DATE_YMD, _RE_DATE_MD_JP, _RE_DAYS_AGO, _RE_WEEKS_AGO,
               _RE_YEARS_AGO, _RE_MONTHS_AGO, _RE_DATE_MD, _RE_MONTH_JP):
        masked = rx.sub(" ", masked)
    for w in _DATE_REL_WORDS:
        masked = masked.replace(w, " ")
    if time_jst is not None:
        # protect fixed compounds (e.g. "dinner") before stripping time words
        prot = {}
        for i, m in enumerate(_TIME_FALSE_FRIENDS.finditer(masked)):
            tok = "%d" % i
            prot[tok] = m.group(0)
        for tok, w in prot.items():
            masked = masked.replace(w, tok, 1)
        masked = re.sub(r"(\d{1,2})時(頃|台|ごろ)?(?!間)", " ", masked)
        for word, _rng in sorted(_TIME_WORDS, key=lambda x: len(x[0]), reverse=True):
            masked = masked.replace(word, " ")
        for tok, w in prot.items():
            masked = masked.replace(tok, w)
    keywords = _extract_keywords(_strip_query(masked)) if masked.strip() else []
    # A temporal modifier ("the first time", "back then") means "which
    # occurrence", not "which date" -- widen date_range to everything.
    if _qr is not None:
        try:
            if _qr.has_temporal_modifier(s):
                date_range = None
        except Exception:
            pass
    return date_range, time_jst, keywords


def _latest_day_with_hours(time_jst):
    """When only a time-of-day was given (no date), find the most
    recent JST day that has rows in that window and return its
    (start, end) in UTC. None if nothing matches."""
    if not os.path.exists(_index_db()):
        build_index()
    con = sqlite3.connect(_index_db())
    try:
        for (ts,) in con.execute("SELECT ts FROM recall WHERE type != 'meta' ORDER BY ts DESC"):
            t = _parse_ts(ts)
            if t is not None and _in_time_range(t, time_jst):
                return _jst_day_range_utc(t.astimezone(_JST).date())
        return None
    finally:
        con.close()


# Date-scoped mode (and --compact, which is date-scoped under the hood)
# used to return literally every row in range -- 64,165 bytes / 166
# rows for one real query. This is the cap that replaces "everything".
# Provisional, same caveat as AROUND_MAX_N above: not yet tuned against
# how many rows a session actually needs; the oldest rows in range are
# dropped first and the drop count is always reported (never silent).
DATE_SCOPE_ROW_CAP = 60


def search_time(date_range, time_jst, keywords, actor=None,
                hits=8, around=0, day_limit=40, include_action=False,
                row_cap=DATE_SCOPE_ROW_CAP):
    """The formal entry point for time-scoped search: takes an
    already-parsed (date_range, time_jst, keywords) -- typically from
    _split_time_query -- and returns only what's inside that range.
    Never re-parses the original text.

    Returns (rows, meta) where rows is chronological (oldest first)
    and meta carries: total (row count in range before keyword
    filtering, for an honest "how much is really there"), kw_total
    (row count after keyword filtering, or None in date-only mode),
    days (per-day counts), types (per-type counts), level ("L1" =
    keyword filter applied and kept, "L2" = keyword filter dropped
    for being too narrow), fallback_day (set when no date was given
    and a time-of-day fell back to the most recent matching day), and
    omitted (how many in-range rows were dropped by row_cap -- the
    oldest ones, always counted, never silently truncated).

    type='action' rows are excluded unless include_action=True."""
    around = max(0, min(int(around), AROUND_MAX_N))
    fallback_day = None
    if date_range is None and time_jst is not None:
        date_range = _latest_day_with_hours(time_jst)
        if date_range is not None:
            fallback_day = date_range[0].astimezone(_JST).strftime("%Y-%m-%d")
    if date_range is None:
        return [], {"total": 0, "kw_total": None, "days": {}, "types": {},
                     "level": None, "fallback_day": fallback_day, "omitted": 0}
    if not os.path.exists(_index_db()):
        build_index()
    con = sqlite3.connect(_index_db())
    con.row_factory = sqlite3.Row
    try:
        # Coarse filter in SQL first (ts is stored as a sortable ISO
        # string), then apply the exact time-of-day filter in Python.
        # A pure-Python scan of every row was measurably too slow once
        # the log grew large.
        _lo = (date_range[0] - timedelta(seconds=1)).isoformat()
        _hi = (date_range[1] + timedelta(seconds=1)).isoformat()
        _type_extra = "" if include_action else " AND type != 'action'"
        _sql = ("SELECT rowid, ts, actor, role, type, text, model, session"
                " FROM recall WHERE ts >= ? AND ts <= ? AND type != 'meta' AND type != 'doc'"
                + _type_extra)
        _params = [_lo, _hi]
        if actor:
            _sql += " AND actor = ?"
            _params.append(actor)
        all_rows = [dict(r) for r in con.execute(_sql, _params)]
        in_range = _apply_time_filters(all_rows, date_range, time_jst)
        total = len(in_range)

        def _day_counts(rows):
            d = {}
            for r in rows:
                t = _parse_ts(r.get("ts"))
                if t is not None:
                    k = t.astimezone(_JST).strftime("%m-%d")
                    d[k] = d.get(k, 0) + 1
            return d

        def _type_counts(rows):
            d = {}
            for r in rows:
                k = r.get("type") or "text"
                d[k] = d.get(k, 0) + 1
            return d

        def _cap(rows_chrono_asc, cap):
            """Keep the most recent `cap` rows of an ascending-by-ts
            list; return (kept, omitted_count)."""
            if len(rows_chrono_asc) <= cap:
                return rows_chrono_asc, 0
            return rows_chrono_asc[-cap:], len(rows_chrono_asc) - cap

        if not keywords:
            rows = sorted(in_range, key=lambda r: (r.get("ts") or ""))
            rows, omitted = _cap(rows, row_cap)
            meta = {"total": total, "kw_total": None, "days": _day_counts(in_range),
                    "types": _type_counts(in_range), "level": None,
                    "fallback_day": fallback_day, "omitted": omitted}
            for r in rows:
                r.pop("rowid", None)
            return rows, meta

        # With keywords: narrow the range first, then search inside it.
        # Ranking the whole table first and *then* clipping to the
        # range can drop a real match that just didn't rank highly
        # globally; the in-range set is small enough to scan directly.
        cand = [r for r in in_range
                if all(k in (r.get("text") or "") for k in keywords)]
        level = "L1"
        if len(cand) < 2:
            cand = list(in_range)
            level = "L2"
        if not cand:
            return [], {"total": total, "kw_total": 0, "days": {}, "types": {},
                         "level": level, "fallback_day": fallback_day, "omitted": 0}

        cand_kw_total = len(cand)
        cand_chrono = sorted(cand, key=lambda r: (r.get("ts") or ""))
        cand_chrono, omitted = _cap(cand_chrono, row_cap)
        cand = cand_chrono
        cand.sort(key=lambda r: _is_negative(r.get("text", "")))
        hit_sel = cand

        if around > 0:
            rowid_session = {r["rowid"]: r["session"]
                             for r in con.execute("SELECT rowid, session FROM recall")}
            wanted = set()
            for h in hit_sel:
                rid = h["rowid"]
                sess = h.get("session")
                wanted.add(rid)
                for i in range(rid - around, rid + around + 1):
                    if i >= 1 and rowid_session.get(i) == sess:
                        wanted.add(i)
            if len(wanted) > AROUND_ROW_CAP:
                kept = sorted(wanted, reverse=True)[:AROUND_ROW_CAP]
                omitted += len(wanted) - len(kept)
                wanted = set(kept)
            placeholders = ",".join("?" * len(wanted))
            rows = [dict(r) for r in con.execute(
                "SELECT rowid, ts, actor, role, type, text, model, session"
                f" FROM recall WHERE rowid IN ({placeholders}) ORDER BY rowid ASC",
                sorted(wanted))]
            rows = _apply_time_filters(rows, date_range, time_jst)
        else:
            rows = hit_sel

        rows = sorted(rows, key=lambda r: (r.get("ts") or ""))
        meta = {"total": total, "kw_total": cand_kw_total, "days": _day_counts(cand),
                "types": _type_counts(cand), "level": level, "fallback_day": fallback_day,
                "omitted": omitted}
        for r in rows:
            r.pop("rowid", None)
        return rows, meta
    finally:
        con.close()


if __name__ == "__main__":
    import sys
    print("=== index_exact self-test ===")
    n = build_index(force=("--force" in sys.argv))
    print(f"index built: {n} rows")

    print("\n--- keyword search ---")
    for kw in ("memory", "budget"):
        hits = search(kw, limit=3)
        print(f"search '{kw}' -> {len(hits)} hit(s)")

    print("\n--- date-scoped search demo ---")
    for kw in ("昨日のログ", "今日の話"):
        hits = search(kw, limit=3)
        print(f"search '{kw}' -> {len(hits)} hit(s)")

    print("\n--- with context ---")
    ctx, ctx_omitted = search_with_context("memory", hits=1, around=1)
    print(f"hit + context -> {len(ctx)} row(s), {ctx_omitted} omitted")
