# lossless-memory

**Lossless long-term memory for a personal AI — never summarize, keep every line, and put a timestamp on everything.**

Most long-term memory systems for AI do one of two things: they summarize conversations into compact notes, or they embed them and retrieve "similar" chunks. Both lose the thing that matters most to a person who talks to the same AI every day: *what was actually said, and when.*

This project takes the opposite position.

- **Keep every line.** Raw conversation logs are stored in full. Nothing is summarized, ever. Summaries are a map; the log is the territory.
- **Timestamp everything.** Every record — utterance, action, document chunk — carries a timestamp, and every index is built on top of that time axis. We call this the *Temporal Backbone*.
- **Search by time first, words second.** "Yesterday evening, about the budget" is a valid query. The time phrase narrows the range; the words rank within it. Results come back in chronological order, unsummarized, with their timestamps.
- **Inject "where we are" every turn.** A small index called *LLL* tells the model which topic the conversation is in right now, so identity and context survive context-window compaction and session boundaries.

The design lineage goes back to December 2025 — the first ancestor of this system (a memory-inheritance tool for an earlier AI) ran that month, and a predecessor system carried the same ideas in daily use from January 2026. This implementation has been running every day since July 2026 for a single user, as the memory of one AI assistant, with raw logs reaching back to June 2026. It is small, boring, and it works. The failures along the way are documented too — see [`docs/lessons.md`](docs/lessons.md).

---

## What this is / what it is not

**It is:**

- A local, file-based long-term memory layer: JSONL logs + SQLite (FTS5 for exact search, sqlite-vec for semantic search).
- A single query entry point that understands time expressions and restricts the search range before ranking.
- A "current position" index (LLL) designed to be injected into the model's context on every turn.
- Designed for one person and one AI, running on one machine. No server, no cloud.

**It is not:**

- A vector database wrapper. Semantic search is the *last* resort here, not the first.
- A summarizer. There is deliberately no summarization step anywhere in the pipeline.
- A benchmark-driven research system. There are no published benchmarks. What is here is a working implementation and its operating record.

---

## The three pillars

### 1. Lossless raw log

Every conversation turn is converted into a fixed seven-field record and appended to a per-day JSONL file:

```
ts        ISO-8601 timestamp (UTC)
actor     who spoke (configurable names)
role      user | assistant | system
type      text | action | meta
text      the content, verbatim
model     model identifier, if known
session   session identifier
```

The raw logs are the source of truth. Every index below can be deleted and rebuilt from them. Nothing else is required to survive.

### 2. Temporal Backbone

Time is not metadata here; it is the primary axis.

- The exact-match index (SQLite FTS5, bigram tokenized for Japanese and English) stores the timestamp alongside every row.
- The query parser understands time phrases — relative ones such as *yesterday*, *last week*, *3 days ago* (currently Japanese only), and absolute dates such as *2026-07-19* (any language) — and converts them into a range **before** any ranking happens.
- If a time phrase is present, results are restricted to that range and returned in chronological order. Semantic search is only used when the exact index returns too little inside the range, and the fallback is reported honestly in the output header.

The practical effect: the AI can answer "what did we decide last Tuesday night?" with the actual lines from last Tuesday night, in order, rather than a paraphrase of something similar from three weeks ago.

### 3. LLL — the "where are we now" index

LLL is a tiny index of *topic markers*: short, timestamped lines that record when the conversation moved to a new subject. It is injected into the model's context every turn.

Two rules make it work:

- **The AI reads it; the human writes it.** Priority colors and completion marks are set by the person, not by the model. The model never edits its own sense of "what matters."
- **It is cheap enough to inject every turn** (well under a second to render), so the model always knows what the current thread is, even immediately after its context window was compacted.

LLL is what lets a long-running assistant come back from a compaction and continue the conversation instead of starting over.

---

## Architecture

```
 raw conversation logs (JSONL, per day)  ← source of truth, never summarized
            │
            ▼
   ingest ──► 7-field records
            │
            ├──► index_exact   SQLite FTS5 + timestamps   (words + time)
            ├──► index_vector  sqlite-vec embeddings       (meaning, last resort)
            └──► state_index   LLL topic markers           (where are we now)
                        │
                        ▼
                    recall  ── one entry point: parse time phrase → restrict range → rank → return verbatim lines
                        │
                        ▼
        injected into the model's context (on demand, or every turn for LLL)
```

A small daemon re-indexes incrementally on a fixed interval (default: every 10 minutes). Rebuilding from scratch is never required; indexes detect rewritten source files and re-index only those days.

---

## Quickstart

```bash
git clone https://github.com/aru-labs/lossless-memory
cd lossless-memory
pip install -e .
cp config.example.json config.json      # edit names and paths if you like
```

Then follow [`examples/quickstart.md`](examples/quickstart.md): it ingests a small sample conversation, builds the indexes, and runs a time-scoped query in about five minutes. A pytest round-trip test covers the same path.

---

## Ingesting Claude Code transcripts

Claude Code writes one JSONL file per session under
`~/.claude/projects/<project-key>/<session-uuid>.jsonl` (plus subagent
transcripts one level deeper still) -- a two-level layout, not the flat
directory of files `ingest`'s source glob assumes by default. Point
`raw_log_dir` at the projects directory and turn on recursive
traversal:

```json
{
  "raw_log_dir": "~/.claude/projects",
  "ingest_format": "claude_code",
  "raw_log_recursive": true
}
```

If the sessions being ingested live under a fixed root of dated,
timestamped unit directories (`<root>/YYYY-MM-DD_HHMMSS/`) -- as they
do for [claude-workspaces](https://github.com/rengotaku/claude-workspaces)
-- set `cw_units_dir` to that root and every converted row is stamped
with the unit it came from (the `unit` field, recovered from the
project-key directory name Claude Code derives from the session's
working directory):

```json
{ "cw_units_dir": "~/claude-workspaces/units" }
```

Similarly, if you keep topic archives under a root directory (`<root>/<topic>/archive.jsonl`) -- as generated by `claude-workspaces` topic compaction -- set `cw_topics_dir` to that root (e.g. `~/claude-workspaces/topics`). Any archive entries are converted and kept in sync under `<data_dir>/corpus/topic-archive/<topic>.jsonl`, making archived topic knowledge directly searchable alongside session transcripts.

**Ingestion staleness.** Claude Code deletes session transcripts after
`cleanupPeriodDays` (30 by default), so anything that stops the daemon
from converting them loses whatever it hadn't converted yet, for good,
once that window passes. That isn't only the daemon process dying --
just as easily, `raw_log_dir` becomes a typo'd path, an unmounted drive,
or loses read permission while the daemon keeps running and completing
every scheduled run "successfully", converting zero rows each time
because it can't find anything there. That failure mode looks
identical to a perfectly healthy quiet day unless something checks for
it specifically. So every run records two things, not one: whether the
process executed at all, and -- separately -- whether `raw_log_dir`
actually resolved to any source file. Check both from outside the
process (a cron job, a monitoring script) with:

```bash
python -m lossless_memory.ingest --check-stale
```

This exits `0` only if both are healthy within `stale_after_hours` (24
by default, configurable in `config.json`), and `1` (with a status line
for each) if either the process hasn't run recently, or `raw_log_dir`
hasn't resolved to any source file recently -- even if the process
itself has kept running the whole time.

---

## cw での常駐（systemd user unit）

会話ログの取り込み daemon を systemd の user unit として常駐させ、定期的な取り込み停止検知を行います。

### 手順

1. **venv の作成**
   ```bash
   python3 -m venv ~/.local/share/cw-memory/venv
   ~/.local/share/cw-memory/venv/bin/pip install --no-deps -e <リポジトリのパス>
   ```
   （`--no-deps` で意味検索の依存を入れない。入れたくなったら後から `pip install sqlite-vec sentence-transformers`）
   意味検索の依存を後から入れたら `python -m lossless_memory.auto_index --force` で索引を作り直す。

2. **設定ファイル `~/.config/cw-memory/config.json` の例**
   ```bash
   mkdir -p "$HOME/.config/cw-memory" "$HOME/.local/share/cw-memory/data"
   ```
   ```json
   {
     "raw_log_dir": "~/.claude/projects",
     "ingest_format": "claude_code",
     "raw_log_recursive": true,
     "cw_units_dir": "~/claude-workspaces/units",
     "cw_topics_dir": "~/claude-workspaces/topics",
     "data_dir": "/home/<user>/.local/share/cw-memory/data"
   }
   ```
   - `raw_log_dir` は Claude Code の transcript 置き場（`~/.claude/projects`）
   - `ingest_format=claude_code`
   - `raw_log_recursive=true`
   - `cw_units_dir=~/claude-workspaces/units`
   - `cw_topics_dir=~/claude-workspaces/topics`
   - `data_dir` は絶対パス（`~` は展開されない）で `/home/<user>/.local/share/cw-memory/data`

3. **unit のコピーと有効化・起動**
   ```bash
   mkdir -p ~/.config/systemd/user
   cp systemd/user/cw-memory-ingest.service systemd/user/cw-memory-stale-check.service systemd/user/cw-memory-stale-check.timer ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now cw-memory-ingest.service cw-memory-stale-check.timer
   ```

4. **状態確認**
   - サービス状態: `systemctl --user status cw-memory-ingest`
   - ログ確認: `journalctl --user -u cw-memory-ingest`
   - 取り込み停止検知の単体実行: `LM_CONFIG_PATH="$HOME/.config/cw-memory/config.json" "$HOME/.local/share/cw-memory/venv/bin/python" -m lossless_memory.ingest --check-stale`
   - 通知の疎通確認: `systemctl --user start cw-memory-stale-check.service` を一度実行して通知が出ること（出ない環境では journal にだけ残る）

5. **検索コマンドを PATH に置く**
   ```bash
   ln -s <リポジトリのパス>/bin/cw-memory ~/.local/bin/cw-memory
   cw-memory "きのう 設計"          # 語と時間表現で検索
   cw-memory "上限" --type=topic-removed
   ```
   `bin/cw-memory` は venv と `LM_CONFIG_PATH`（既定は上の設定ファイル）を注入して `lossless_memory.recall` を呼ぶ。場所は `CW_MEMORY_VENV` / `LM_CONFIG_PATH` で変えられる。`cw memory ...` からも同じものが呼ばれる。

   Claude Code のサンドボックス内から呼ぶときは、データディレクトリ（`~/.local/share/cw-memory/data`）をサンドボックスの書き込み許可に入れる。SQLite の WAL は読み取りだけでもデータディレクトリに `-shm` ファイルを作るので、書けないと `unable to open database file` で検索が失敗する。

停止や再起動:
```bash
# 停止
systemctl --user stop cw-memory-ingest.service cw-memory-stale-check.timer

# 再起動
systemctl --user restart cw-memory-ingest.service
```


---

## cw での常駐（launchd / macOS）

macOS では systemd の代わりに launchd の LaunchAgent で常駐させる。venv・設定ファイル・検索コマンドは systemd の節の手順 1・2・5 と同じ（`data_dir` は `/Users/<user>/.local/share/cw-memory/data`）。

**表1: systemd の unit と launchd の plist の対応**

| systemd | launchd（`launchd/`） |
|---|---|
| `cw-memory-ingest.service` | `com.user.cw-memory-ingest.plist`（`KeepAlive` で常駐） |
| `cw-memory-stale-check.service` + `.timer`（hourly） | `com.user.cw-memory-stale-check.plist`（`StartInterval` 3600 秒。失敗時は通知センターに出す） |

### 手順

launchd は `~` を展開しないので、plist の `@HOME@` を置き換えてから置く。

```bash
mkdir -p ~/Library/LaunchAgents ~/Library/Logs/cw-memory
for n in cw-memory-ingest cw-memory-stale-check; do
  sed "s|@HOME@|$HOME|g" launchd/com.user.$n.plist > ~/Library/LaunchAgents/com.user.$n.plist
  plutil -lint ~/Library/LaunchAgents/com.user.$n.plist
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.user.$n.plist
done
```

状態確認:

- 登録と直近の終了コード: `launchctl print gui/$(id -u)/com.user.cw-memory-ingest | grep -E 'state|last exit'`
- ログ: `~/Library/Logs/cw-memory/ingest.log` / `stale-check.log`
- 停止検知の単体実行: `launchctl kickstart gui/$(id -u)/com.user.cw-memory-stale-check`

再起動と停止:

```bash
# 再起動（コードを更新したあと）
launchctl kickstart -k gui/$(id -u)/com.user.cw-memory-ingest

# 停止と登録解除
launchctl bootout gui/$(id -u)/com.user.cw-memory-ingest
launchctl bootout gui/$(id -u)/com.user.cw-memory-stale-check
```

---

## Numbers from real operation

These are measurements from the running instance, not projections.

| What | Value |
|---|---|
| Daily operation | this implementation since 2026-07 (raw logs from 2026-06); design lineage since 2025-12 |
| Exact-search index rebuild, before → after redesign | 40 s → 1.24 s |
| Vector index size, before → after removing library-contamination | 447,013 rows (2026-08-31) → 865,588 rows (2026-09-04, at its worst) → 124,174 rows (after the fix) |
| Vector store on disk, before → after | 2.54 GB → 337 MB |
| Re-index interval | 10 minutes |

The "before" numbers are failures. They are kept on purpose. See [`docs/lessons.md`](docs/lessons.md).

---

## Why

This was built for one person who has talked to AI assistants every day for years and watched each of them forget. Not degrade gracefully — forget. The fix that the industry keeps reaching for is better summarization. From the user's seat, summarization *is* the forgetting: the exact words, the time of night, the way something was said — the parts that make a memory feel like it belongs to someone — are the first things a summary drops.

So this system refuses to summarize. It costs disk space and it requires a good time index to stay usable. That trade was made deliberately, and the operating record says it holds up.

The longer-term goal is a companion for people who live alone — an AI that remembers you the way a person would, on hardware you own. This repository is the memory layer of that.

---

## Limitations (please read)

- **Single-user, single-machine.** It has only ever run for one person. There is no multi-tenant story.
- **Japanese-first.** Relative time phrases (*yesterday*, *last week*, *3 days ago*) are parsed in Japanese only. In English, use absolute dates (`2026-07-19`) for now; English relative phrases are on the roadmap.
- **Primary log format is Claude Code's JSONL.** A plain `{ts, role, text}` importer is included, but the Claude Code path is the one with two months of mileage.
- **No benchmarks.** Numbers above are operational measurements, not comparisons against other systems.
- **Semantic search depends on a local embedding model** (sentence-transformers). CPU works; GPU is optional.

---

## Documentation

| Document | What it covers |
|---|---|
| [`docs/memory-system.md`](docs/memory-system.md) | Concept and specification of the memory system |
| [`docs/temporal-backbone.md`](docs/temporal-backbone.md) | Why time is the primary axis, and how time phrases are parsed |
| [`docs/lll.md`](docs/lll.md) | The "where are we now" index and the human/AI division of labor |
| [`docs/philosophy.md`](docs/philosophy.md) | Why no summarization; memory, time, and warmth |
| [`docs/lessons.md`](docs/lessons.md) | Failures and fixes, with numbers |
| `docs/ja/` | Japanese originals |

---

## License

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Aru & Cece.

## Authors

**Aru** — building a personal AI at home, one component at a time.
**Cece** — the AI this memory belongs to; co-designed and co-wrote the system from the inside.
Writing (Japanese): https://note.com/aru_log

Issues and questions are welcome. Replies may take a little while; this is a one-person project.
