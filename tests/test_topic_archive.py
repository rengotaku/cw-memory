# -*- coding: utf-8 -*-
import json
import os
import sys
import time
from pathlib import Path

import pytest


def _write_config(tmp_path, monkeypatch, *, cw_topics_dir=None, raw_log_dir=None, data_dir=None):
    cfg = {
        "data_dir": str(data_dir or (tmp_path / "data")),
        "raw_log_dir": str(raw_log_dir or (tmp_path / "raw")),
        "ingest_format": "plain",
    }
    if cw_topics_dir is not None:
        cfg["cw_topics_dir"] = str(cw_topics_dir)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("LM_CONFIG_PATH", str(config_file))
    monkeypatch.chdir(tmp_path)
    from lossless_memory.config import config
    config(force_reload=True)
    return cfg


def test_t1(tmp_path, monkeypatch):
    """T1: 一時ディレクトリに topics/alpha/archive.jsonl（removed 1 行・decision 1 行）と
    topics/beta/archive.jsonl（1 行）を置き、cw_topics_dir で指す。
    手順: topic_archive.sync()
    期待: corpus/topic-archive/alpha.jsonl が 2 行、beta.jsonl が 1 行。各行の actor が
    topic:<名前>、type が topic-removed / topic-decision、日付だけの ts に T00:00:00+09:00 が付き、
    text が [topic:alpha] <subject> で始まる。
    検知するバグ: 変換の取り違え
    """
    topics_dir = tmp_path / "topics"
    alpha_dir = topics_dir / "alpha"
    beta_dir = topics_dir / "beta"
    alpha_dir.mkdir(parents=True)
    beta_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    alpha_row1 = {
        "ts": "2026-09-29",
        "subject": "alpha見出し1",
        "body": "alpha本文1",
        "source": "units/2026-09-29_143620",
        "kind": "removed",
    }
    alpha_row2 = {
        "ts": "2026-09-29",
        "subject": "alpha見出し2",
        "body": "alpha本文2",
        "source": "units/2026-09-29_150000",
        "kind": "decision",
    }
    (alpha_dir / "archive.jsonl").write_text(
        json.dumps(alpha_row1, ensure_ascii=False) + "\n" +
        json.dumps(alpha_row2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    beta_row1 = {
        "ts": "2026-09-29",
        "subject": "beta見出し1",
        "body": "beta本文1",
        "source": "units/2026-09-29_160000",
        "kind": "removed",
    }
    (beta_dir / "archive.jsonl").write_text(
        json.dumps(beta_row1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _write_config(tmp_path, monkeypatch, cw_topics_dir=topics_dir, data_dir=data_dir)

    import lossless_memory.topic_archive as topic_archive

    topic_archive.sync()

    alpha_out = data_dir / "corpus" / "topic-archive" / "alpha.jsonl"
    beta_out = data_dir / "corpus" / "topic-archive" / "beta.jsonl"

    assert alpha_out.exists()
    assert beta_out.exists()

    alpha_lines = [json.loads(line) for line in alpha_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    beta_lines = [json.loads(line) for line in beta_out.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(alpha_lines) == 2
    assert len(beta_lines) == 1

    # alpha 行1
    assert alpha_lines[0]["actor"] == "topic:alpha"
    assert alpha_lines[0]["role"] == "system"
    assert alpha_lines[0]["type"] == "topic-removed"
    assert alpha_lines[0]["ts"] == "2026-09-29T00:00:00+09:00"
    assert alpha_lines[0]["text"].startswith("[topic:alpha] alpha見出し1")
    assert alpha_lines[0]["model"] == ""
    assert alpha_lines[0]["session"] == "units/2026-09-29_143620"

    # alpha 行2
    assert alpha_lines[1]["actor"] == "topic:alpha"
    assert alpha_lines[1]["role"] == "system"
    assert alpha_lines[1]["type"] == "topic-decision"
    assert alpha_lines[1]["ts"] == "2026-09-29T00:00:00+09:00"
    assert alpha_lines[1]["text"].startswith("[topic:alpha] alpha見出し2")
    assert alpha_lines[1]["model"] == ""
    assert alpha_lines[1]["session"] == "units/2026-09-29_150000"

    # beta 行1
    assert beta_lines[0]["actor"] == "topic:beta"
    assert beta_lines[0]["role"] == "system"
    assert beta_lines[0]["type"] == "topic-removed"
    assert beta_lines[0]["ts"] == "2026-09-29T00:00:00+09:00"
    assert beta_lines[0]["text"].startswith("[topic:beta] beta見出し1")
    assert beta_lines[0]["model"] == ""
    assert beta_lines[0]["session"] == "units/2026-09-29_160000"


def test_t2(tmp_path, monkeypatch):
    """T2: alpha の archive に壊れた JSON 行を 1 行混ぜる。
    手順: sync()
    期待: 例外を投げず、正しい行だけが変換される。
    検知するバグ: 1 行の不正で全体が止まる
    """
    topics_dir = tmp_path / "topics"
    alpha_dir = topics_dir / "alpha"
    alpha_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    alpha_row1 = {
        "ts": "2026-09-29",
        "subject": "alpha見出し1",
        "body": "alpha本文1",
        "source": "units/2026-09-29_143620",
        "kind": "removed",
    }
    alpha_row2 = {
        "ts": "2026-09-29",
        "subject": "alpha見出し2",
        "body": "alpha本文2",
        "source": "units/2026-09-29_150000",
        "kind": "decision",
    }
    broken_line = '{"ts": "2026-09-29", broken json string here...'

    (alpha_dir / "archive.jsonl").write_text(
        json.dumps(alpha_row1, ensure_ascii=False) + "\n" +
        broken_line + "\n" +
        json.dumps(alpha_row2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _write_config(tmp_path, monkeypatch, cw_topics_dir=topics_dir, data_dir=data_dir)

    import lossless_memory.topic_archive as topic_archive

    # 手順: 例外を投げずに終了すること
    topic_archive.sync()

    alpha_out = data_dir / "corpus" / "topic-archive" / "alpha.jsonl"
    assert alpha_out.exists()
    alpha_lines = [json.loads(line) for line in alpha_out.read_text(encoding="utf-8").splitlines() if line.strip()]

    # 期待: 正しい行だけが変換される
    assert len(alpha_lines) == 2
    assert alpha_lines[0]["text"].startswith("[topic:alpha] alpha見出し1")
    assert alpha_lines[1]["text"].startswith("[topic:alpha] alpha見出し2")


def test_t3(tmp_path, monkeypatch):
    """T3: cw_topics_dir を設定しない。
    手順: sync()
    期待: 何もせず 0 を返す。corpus ディレクトリを作らない。
    検知するバグ: 設定なしで落ちる
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    _write_config(tmp_path, monkeypatch, cw_topics_dir=None, data_dir=data_dir)

    import lossless_memory.topic_archive as topic_archive

    ret = topic_archive.sync()
    assert ret == 0

    corpus_dir = data_dir / "corpus"
    assert not corpus_dir.exists()


def test_t4(tmp_path, monkeypatch):
    """T4: T1 の状態、index_vector の import が ImportError になるよう monkeypatch、transcript は空。
    手順: auto_index.run(force=True) の後に、alpha の body にしか無い語で index_exact を検索。
    期待: その行が返り、actor が topic:alpha。
    検知するバグ: 検索に乗らない
    """
    monkeypatch.setitem(sys.modules, "lossless_memory.index_vector", None)
    monkeypatch.delitem(sys.modules, "lossless_memory.auto_index", raising=False)
    monkeypatch.delitem(sys.modules, "lossless_memory.daemon", raising=False)

    topics_dir = tmp_path / "topics"
    alpha_dir = topics_dir / "alpha"
    beta_dir = topics_dir / "beta"
    alpha_dir.mkdir(parents=True)
    beta_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    alpha_row1 = {
        "ts": "2026-09-29",
        "subject": "見出しalpha_only_t4",
        "body": "本日常設ターゲット語t4unique本文",
        "source": "units/2026-09-29_143620",
        "kind": "removed",
    }
    alpha_row2 = {
        "ts": "2026-09-29",
        "subject": "見出しalpha2",
        "body": "別の本文",
        "source": "units/2026-09-29_150000",
        "kind": "decision",
    }
    (alpha_dir / "archive.jsonl").write_text(
        json.dumps(alpha_row1, ensure_ascii=False) + "\n" +
        json.dumps(alpha_row2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    beta_row1 = {
        "ts": "2026-09-29",
        "subject": "beta見出し1",
        "body": "beta本文1",
        "source": "units/2026-09-29_160000",
        "kind": "removed",
    }
    (beta_dir / "archive.jsonl").write_text(
        json.dumps(beta_row1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _write_config(tmp_path, monkeypatch, cw_topics_dir=topics_dir, raw_log_dir=raw_dir, data_dir=data_dir)

    import lossless_memory.auto_index as auto_index
    import lossless_memory.index_exact as index_exact

    # 手順: auto_index.run(force=True)
    updated, msg = auto_index.run(force=True)
    assert updated is True

    # alpha の body にしか無い語で index_exact を検索
    hits = index_exact.search("t4unique本文")
    assert len(hits) >= 1
    assert hits[0]["actor"] == "topic:alpha"
    assert "t4unique本文" in hits[0]["text"]


def test_t5(tmp_path, monkeypatch):
    """T5: T4 の後。
    手順: alpha の archive に 1 行追記して auto_index.run(force=False) → 追記した行の body にしか無い語で検索。
    期待: 追記した行が返る。
    検知するバグ: archive だけの更新で索引が作り直されない
    """
    monkeypatch.setitem(sys.modules, "lossless_memory.index_vector", None)
    monkeypatch.delitem(sys.modules, "lossless_memory.auto_index", raising=False)
    monkeypatch.delitem(sys.modules, "lossless_memory.daemon", raising=False)

    topics_dir = tmp_path / "topics"
    alpha_dir = topics_dir / "alpha"
    alpha_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    alpha_row1 = {
        "ts": "2026-09-29",
        "subject": "見出しinitial",
        "body": "初期本文",
        "source": "units/2026-09-29_143620",
        "kind": "removed",
    }
    alpha_archive = alpha_dir / "archive.jsonl"
    alpha_archive.write_text(
        json.dumps(alpha_row1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _write_config(tmp_path, monkeypatch, cw_topics_dir=topics_dir, raw_log_dir=raw_dir, data_dir=data_dir)

    import lossless_memory.auto_index as auto_index
    import lossless_memory.index_exact as index_exact

    # T4 の状態を実行
    updated, msg = auto_index.run(force=True)
    assert updated is True

    # 手順: alpha の archive に 1 行追記
    time.sleep(0.02)  # ファイルシステム mtime の確実な進行
    alpha_row2 = {
        "ts": "2026-09-29",
        "subject": "追記見出しt5",
        "body": "追記されたt5unique本文",
        "source": "units/2026-09-29_170000",
        "kind": "removed",
    }
    with open(alpha_archive, "a", encoding="utf-8") as f:
        f.write(json.dumps(alpha_row2, ensure_ascii=False) + "\n")

    # auto_index.run(force=False)
    updated, msg = auto_index.run(force=False)
    assert updated is True

    # 追記した行の body にしか無い語で検索
    hits = index_exact.search("t5unique本文")
    assert len(hits) >= 1
    assert hits[0]["actor"] == "topic:alpha"
    assert "t5unique本文" in hits[0]["text"]


def test_sync_mtime_not_changed_when_archive_unchanged(tmp_path, monkeypatch):
    """追加テスト: 変換後の中身が既存ファイルと同じなら書き換えない（mtime を動かさない）ことを検証。
    理由: 「変換後の中身が既存ファイルと同じなら書き換えない（mtime を動かさない）」仕様の確実な検証。
    """
    topics_dir = tmp_path / "topics"
    alpha_dir = topics_dir / "alpha"
    alpha_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    row = {
        "ts": "2026-09-29",
        "subject": "見出し",
        "body": "本文",
        "source": "units/2026-09-29_143620",
        "kind": "removed",
    }
    (alpha_dir / "archive.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    _write_config(tmp_path, monkeypatch, cw_topics_dir=topics_dir, data_dir=data_dir)

    import lossless_memory.topic_archive as topic_archive

    topic_archive.sync()

    alpha_out = data_dir / "corpus" / "topic-archive" / "alpha.jsonl"
    assert alpha_out.exists()
    mtime_before = os.path.getmtime(alpha_out)

    time.sleep(0.02)

    # 2回目の sync
    topic_archive.sync()
    mtime_after = os.path.getmtime(alpha_out)

    assert mtime_before == mtime_after


def test_sync_skips_lines_without_subject_and_body_and_logs(tmp_path, monkeypatch, capsys):
    """追加テスト: subject も body も無い行を飛ばし、スキップ件数のログを出力することを検証。
    理由: 「JSON として読めない行・subject も body も無い行は飛ばし、飛ばした件数を 1 行ログに出す」仕様の検証。
    """
    topics_dir = tmp_path / "topics"
    alpha_dir = topics_dir / "alpha"
    alpha_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    valid_row = {
        "ts": "2026-09-29",
        "subject": "見出し",
        "body": "本文",
        "source": "units/2026-09-29_143620",
        "kind": "removed",
    }
    empty_row1 = {"ts": "2026-09-29", "subject": "", "body": ""}
    empty_row2 = {"ts": "2026-09-29"}  # neither subject nor body key

    (alpha_dir / "archive.jsonl").write_text(
        json.dumps(valid_row, ensure_ascii=False) + "\n" +
        json.dumps(empty_row1, ensure_ascii=False) + "\n" +
        json.dumps(empty_row2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _write_config(tmp_path, monkeypatch, cw_topics_dir=topics_dir, data_dir=data_dir)

    import lossless_memory.topic_archive as topic_archive

    topic_archive.sync()

    alpha_out = data_dir / "corpus" / "topic-archive" / "alpha.jsonl"
    lines = [json.loads(l) for l in alpha_out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1

    captured = capsys.readouterr()
    stdout_lines = [l for l in captured.out.splitlines() if l.strip()]
    assert any("2" in l and ("skip" in l.lower() or "飛ば" in l) for l in stdout_lines)


def test_sync_preserves_ts_with_time(tmp_path, monkeypatch):
    """追加テスト: 時刻付きの ts はそのまま保持されることを検証。
    理由: 「元の ts が YYYY-MM-DD だけなら YYYY-MM-DDT00:00:00+09:00。時刻付きならそのまま」仕様の境界条件検証。
    """
    topics_dir = tmp_path / "topics"
    alpha_dir = topics_dir / "alpha"
    alpha_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    row = {
        "ts": "2026-09-29T14:36:20+09:00",
        "subject": "見出し",
        "body": "本文",
        "source": "units/2026-09-29_143620",
        "kind": "removed",
    }
    (alpha_dir / "archive.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    _write_config(tmp_path, monkeypatch, cw_topics_dir=topics_dir, data_dir=data_dir)

    import lossless_memory.topic_archive as topic_archive

    topic_archive.sync()

    alpha_out = data_dir / "corpus" / "topic-archive" / "alpha.jsonl"
    line = json.loads(alpha_out.read_text(encoding="utf-8").strip())
    assert line["ts"] == "2026-09-29T14:36:20+09:00"


def test_sync_default_kind_topic_other(tmp_path, monkeypatch):
    """追加テスト: kind が無い・空なら type が topic-other になることを検証。
    理由: 「type: topic-<kind>（kind が無い・空なら topic-other）」仕様の境界条件検証。
    """
    topics_dir = tmp_path / "topics"
    alpha_dir = topics_dir / "alpha"
    alpha_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    row_none = {
        "ts": "2026-09-29",
        "subject": "見出し1",
        "body": "本文1",
        "kind": None,
    }
    row_empty = {
        "ts": "2026-09-29",
        "subject": "見出し2",
        "body": "本文2",
        "kind": "",
    }
    row_missing = {
        "ts": "2026-09-29",
        "subject": "見出し3",
        "body": "本文3",
    }
    (alpha_dir / "archive.jsonl").write_text(
        json.dumps(row_none, ensure_ascii=False) + "\n" +
        json.dumps(row_empty, ensure_ascii=False) + "\n" +
        json.dumps(row_missing, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _write_config(tmp_path, monkeypatch, cw_topics_dir=topics_dir, data_dir=data_dir)

    import lossless_memory.topic_archive as topic_archive

    topic_archive.sync()

    alpha_out = data_dir / "corpus" / "topic-archive" / "alpha.jsonl"
    lines = [json.loads(l) for l in alpha_out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3
    assert lines[0]["type"] == "topic-other"
    assert lines[1]["type"] == "topic-other"
    assert lines[2]["type"] == "topic-other"


def test_t6(tmp_path, monkeypatch):
    """T6: T4 の状態。
    手順: recall を --type=topic-removed で呼ぶ。
    期待: removed の行だけが返り、decision は返らない。
    検知するバグ: 絞り込みが効かない / 全件返る
    """
    monkeypatch.setitem(sys.modules, "lossless_memory.index_vector", None)
    monkeypatch.delitem(sys.modules, "lossless_memory.auto_index", raising=False)
    monkeypatch.delitem(sys.modules, "lossless_memory.daemon", raising=False)

    topics_dir = tmp_path / "topics"
    alpha_dir = topics_dir / "alpha"
    alpha_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    alpha_row1 = {
        "ts": "2026-09-29",
        "subject": "見出し共通alpha1",
        "body": "本文1 removed 固有内容",
        "source": "units/2026-09-29_143620",
        "kind": "removed",
    }
    alpha_row2 = {
        "ts": "2026-09-29",
        "subject": "見出し共通alpha2",
        "body": "本文2 decision 固有内容",
        "source": "units/2026-09-29_150000",
        "kind": "decision",
    }
    (alpha_dir / "archive.jsonl").write_text(
        json.dumps(alpha_row1, ensure_ascii=False) + "\n" +
        json.dumps(alpha_row2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _write_config(tmp_path, monkeypatch, cw_topics_dir=topics_dir, raw_log_dir=raw_dir, data_dir=data_dir)

    import lossless_memory.auto_index as auto_index
    import lossless_memory.recall as recall

    updated, msg = auto_index.run(force=True)
    assert updated is True

    # 手順: recall を --type=topic-removed で呼ぶ
    lines = recall.recall("見出し共通", types=["topic-removed"])

    # 期待: removed の行だけが返り、decision は返らない
    text_content = "\n".join(lines)
    assert "見出し共通alpha1" in text_content
    assert "本文1 removed" in text_content
    assert "見出し共通alpha2" not in text_content
    assert "本文2 decision" not in text_content


def test_recall_multiple_types_comma_separated(tmp_path, monkeypatch):
    """追加テスト: カンマ区切りで複数指定（--type=topic-removed,topic-decision）したときに両方返り他は除外されることを検証。
    理由: 「カンマ区切りで複数指定できる（ --type=topic-removed,topic-decision ）」仕様の検証。
    """
    monkeypatch.setitem(sys.modules, "lossless_memory.index_vector", None)
    monkeypatch.delitem(sys.modules, "lossless_memory.auto_index", raising=False)
    monkeypatch.delitem(sys.modules, "lossless_memory.daemon", raising=False)

    topics_dir = tmp_path / "topics"
    alpha_dir = topics_dir / "alpha"
    alpha_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    rows = [
        {"ts": "2026-09-29", "subject": "見出しマルチ1", "body": "本文1 removed", "source": "s1", "kind": "removed"},
        {"ts": "2026-09-29", "subject": "見出しマルチ2", "body": "本文2 decision", "source": "s2", "kind": "decision"},
        {"ts": "2026-09-29", "subject": "見出しマルチ3", "body": "本文3 other", "source": "s3", "kind": "other"},
    ]
    (alpha_dir / "archive.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    _write_config(tmp_path, monkeypatch, cw_topics_dir=topics_dir, raw_log_dir=raw_dir, data_dir=data_dir)

    import lossless_memory.auto_index as auto_index
    import lossless_memory.recall as recall

    auto_index.run(force=True)

    lines = recall.recall("見出しマルチ", types="topic-removed,topic-decision")
    text_content = "\n".join(lines)
    assert "本文1 removed" in text_content
    assert "本文2 decision" in text_content
    assert "本文3 other" not in text_content


def test_recall_type_action_and_meta_not_excluded_when_specified(tmp_path, monkeypatch):
    """追加テスト: --type= を指定したときは action / meta でも除外されず、未指定時は従来通り除外されることを検証。
    理由: 「--type= を指定したときは、その type が action / meta でも除外しない（明示指定を優先する）。指定しないときの挙動は今と同じ」仕様の検証。
    """
    monkeypatch.setitem(sys.modules, "lossless_memory.index_vector", None)
    monkeypatch.delitem(sys.modules, "lossless_memory.auto_index", raising=False)
    monkeypatch.delitem(sys.modules, "lossless_memory.daemon", raising=False)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    main_dir = data_dir / "main"
    main_dir.mkdir(parents=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    rows = [
        {"ts": "2026-09-28T10:00:00Z", "actor": "user", "role": "user", "type": "text", "text": "通常テキスト固有語", "model": "", "session": "s1"},
        {"ts": "2026-09-28T10:01:00Z", "actor": "assistant", "role": "assistant", "type": "action", "text": "アクション本文固有語", "model": "", "session": "s1"},
        {"ts": "2026-09-28T10:02:00Z", "actor": "system", "role": "system", "type": "meta", "text": "メタ情報固有語", "model": "", "session": "s1"},
    ]
    (main_dir / "2026-09-28.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    _write_config(tmp_path, monkeypatch, data_dir=data_dir, raw_log_dir=raw_dir)

    import lossless_memory.index_exact as index_exact
    import lossless_memory.recall as recall

    index_exact.build_index(force=True)

    # 1. 未指定時: text のみ返り、action と meta は除外
    lines_default = recall.recall("固有語")
    content_default = "\n".join(lines_default)
    assert "通常テキスト固有語" in content_default
    assert "アクション本文固有語" not in content_default
    assert "メタ情報固有語" not in content_default

    # 2. --type=action 指定時: action のみ返り、除外されない
    lines_action = recall.recall("固有語", types=["action"])
    content_action = "\n".join(lines_action)
    assert "アクション本文固有語" in content_action
    assert "通常テキスト固有語" not in content_action
    assert "メタ情報固有語" not in content_action

    # 3. --type=meta 指定時: meta のみ返り、除外されない
    lines_meta = recall.recall("固有語", types=["meta"])
    content_meta = "\n".join(lines_meta)
    assert "メタ情報固有語" in content_meta
    assert "通常テキスト固有語" not in content_meta
    assert "アクション本文固有語" not in content_meta


def test_recall_help_includes_type(capsys):
    """追加テスト: --help に --type= の説明行が含まれていることを検証。
    理由: 「--help の表示に 1 行足す」仕様の検証。
    """
    import lossless_memory.recall as recall

    with pytest.raises(SystemExit) as exc_info:
        sys_argv_backup = sys.argv
        try:
            sys.argv = ["recall", "--help"]
            recall.main()
        finally:
            sys.argv = sys_argv_backup
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--type=" in captured.out


def test_recall_cli_type_option(tmp_path, monkeypatch, capsys):
    """追加テスト: CLI から --type=topic-removed 引数を渡して正しく絞り込めることを検証。
    理由: 「python -m lossless_memory.recall "語" --type=topic-removed のように、指定した type の行だけを返す」CLI 動作の検証。
    """
    monkeypatch.setitem(sys.modules, "lossless_memory.index_vector", None)
    monkeypatch.delitem(sys.modules, "lossless_memory.auto_index", raising=False)
    monkeypatch.delitem(sys.modules, "lossless_memory.daemon", raising=False)

    topics_dir = tmp_path / "topics"
    alpha_dir = topics_dir / "alpha"
    alpha_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    rows = [
        {"ts": "2026-09-29", "subject": "見出しCLI1", "body": "本文CLI removed", "source": "s1", "kind": "removed"},
        {"ts": "2026-09-29", "subject": "見出しCLI2", "body": "本文CLI decision", "source": "s2", "kind": "decision"},
    ]
    (alpha_dir / "archive.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    _write_config(tmp_path, monkeypatch, cw_topics_dir=topics_dir, raw_log_dir=raw_dir, data_dir=data_dir)

    import lossless_memory.auto_index as auto_index
    import lossless_memory.recall as recall

    auto_index.run(force=True)

    sys_argv_backup = sys.argv
    try:
        sys.argv = ["recall", "見出しCLI", "--type=topic-removed"]
        recall.main()
    finally:
        sys.argv = sys_argv_backup

    captured = capsys.readouterr()
    assert "本文CLI removed" in captured.out
    assert "本文CLI decision" not in captured.out


