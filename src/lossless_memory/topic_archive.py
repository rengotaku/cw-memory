# -*- coding: utf-8 -*-
"""topic_archive -- convert cw topic archives to corpus JSONL for indexing.

Reads <cw_topics_dir>/*/archive.jsonl and converts each line into a standard
lossless-memory row stored under <data_dir>/corpus/topic-archive/<topic>.jsonl.
Preserves mtime if the converted content has not changed.
"""
import glob
import json
import os
import re

from .config import config, data_dir


def sync():
    """Convert <cw_topics_dir>/*/archive.jsonl to corpus/topic-archive/<topic>.jsonl.

    If cw_topics_dir is not configured or does not exist, does nothing and returns 0.
    Returns the total number of valid rows converted.
    """
    cfg = config()
    topics_dir = cfg.get("cw_topics_dir")
    if not topics_dir:
        return 0

    topics_dir = os.path.expanduser(topics_dir)
    if not os.path.isdir(topics_dir):
        return 0

    archive_files = sorted(glob.glob(os.path.join(topics_dir, "*", "archive.jsonl")))
    if not archive_files:
        return 0

    out_dir = os.path.join(data_dir(), "corpus", "topic-archive")
    os.makedirs(out_dir, exist_ok=True)

    total_rows = 0

    for archive_path in archive_files:
        topic = os.path.basename(os.path.dirname(archive_path))
        converted_lines = []
        skipped = 0

        with open(archive_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    rec = json.loads(line)
                    if not isinstance(rec, dict):
                        skipped += 1
                        continue
                except Exception:
                    skipped += 1
                    continue

                raw_subject = str(rec.get("subject") or "").strip()
                raw_body = str(rec.get("body") or "").strip()
                if not raw_subject and not raw_body:
                    skipped += 1
                    continue

                # ts conversion
                ts_val = rec.get("ts")
                if ts_val is not None:
                    ts_str = str(ts_val).strip()
                    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", ts_str):
                        ts = f"{ts_str}T00:00:00+09:00"
                    else:
                        ts = ts_str
                else:
                    ts = ""

                # type conversion
                kind = rec.get("kind")
                if kind and str(kind).strip():
                    row_type = f"topic-{str(kind).strip()}"
                else:
                    row_type = "topic-other"

                # text conversion
                if raw_subject and raw_body:
                    text = f"[topic:{topic}] {raw_subject}\n{raw_body}"
                elif raw_subject:
                    text = f"[topic:{topic}] {raw_subject}"
                else:
                    text = f"[topic:{topic}]\n{raw_body}"

                actor = f"topic:{topic}"
                role = "system"
                model = ""
                session = str(rec.get("source") or "")

                row = {
                    "ts": ts,
                    "actor": actor,
                    "role": role,
                    "type": row_type,
                    "text": text,
                    "model": model,
                    "session": session,
                }
                converted_lines.append(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1

        if skipped > 0:
            print(f"topic_archive: skipped {skipped} invalid line(s) in topic {topic}", flush=True)

        new_content = "".join(converted_lines)
        target_path = os.path.join(out_dir, f"{topic}.jsonl")
        existing_content = None
        if os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as tf:
                    existing_content = tf.read()
            except Exception:
                existing_content = None

        if existing_content != new_content:
            tmp_path = os.path.join(out_dir, f".{topic}.jsonl.tmp.{os.getpid()}")
            with open(tmp_path, "w", encoding="utf-8") as tf:
                tf.write(new_content)
            os.replace(tmp_path, target_path)

    return total_rows
