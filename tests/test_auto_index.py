# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

import pytest


def _claude_line(ts, role, text, session):
    return json.dumps({
        "type": role,
        "timestamp": ts,
        "sessionId": session,
        "message": {"role": role, "content": text},
    }, ensure_ascii=False)


def test_t1(tmp_path, monkeypatch):
    """T1: sys.modules["lossless_memory.index_vector"] = None を monkeypatch で入れ、
    import が ImportError になる状態を作る。raw_log_dir に claude_code 形式の transcript を
    1 本置いた一時ディレクトリを config で指す。auto_index.run(force=True) を実行し、
    例外を投げずに終わり、index_exact の索引 DB が作られ、その transcript の本文が
    index_exact の検索で 1 件以上返ることを検証。
    """
    monkeypatch.setitem(sys.modules, "lossless_memory.index_vector", None)
    monkeypatch.delitem(sys.modules, "lossless_memory.auto_index", raising=False)
    monkeypatch.delitem(sys.modules, "lossless_memory.daemon", raising=False)

    data_dir_path = tmp_path / "data"
    raw_dir_path = tmp_path / "raw"
    data_dir_path.mkdir()
    raw_dir_path.mkdir()

    proj_dir = raw_dir_path / "test-proj"
    proj_dir.mkdir()
    session_id = "11111111-1111-1111-1111-111111111111"
    line = _claude_line("2026-09-29T10:00:00Z", "user", "hello claude code search target text", session_id)
    (proj_dir / f"{session_id}.jsonl").write_text(line + "\n", encoding="utf-8")

    config_file = tmp_path / "config.json"
    cfg = {
        "data_dir": str(data_dir_path),
        "raw_log_dir": str(raw_dir_path),
        "ingest_format": "claude_code",
        "raw_log_recursive": True,
    }
    config_file.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("LM_CONFIG_PATH", str(config_file))
    monkeypatch.chdir(tmp_path)

    from lossless_memory.config import config
    config(force_reload=True)

    import lossless_memory.auto_index as auto_index
    import lossless_memory.index_exact as index_exact

    updated, msg = auto_index.run(force=True)
    assert updated is True

    db_path = data_dir_path / "index_exact.db"
    assert db_path.exists()

    hits = index_exact.search("hello claude code search target text")
    assert len(hits) >= 1
    assert any("hello claude code search target text" in h["text"] for h in hits)


def test_t2(tmp_path, monkeypatch):
    """T2: T1 と同じ前提で、
    import lossless_memory.auto_index と import lossless_memory.daemon
    どちらも import で落ちないことを検証。
    """
    monkeypatch.setitem(sys.modules, "lossless_memory.index_vector", None)
    monkeypatch.delitem(sys.modules, "lossless_memory.auto_index", raising=False)
    monkeypatch.delitem(sys.modules, "lossless_memory.daemon", raising=False)

    data_dir_path = tmp_path / "data"
    raw_dir_path = tmp_path / "raw"
    data_dir_path.mkdir()
    raw_dir_path.mkdir()

    config_file = tmp_path / "config.json"
    cfg = {
        "data_dir": str(data_dir_path),
        "raw_log_dir": str(raw_dir_path),
        "ingest_format": "claude_code",
        "raw_log_recursive": True,
    }
    config_file.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("LM_CONFIG_PATH", str(config_file))
    monkeypatch.chdir(tmp_path)

    from lossless_memory.config import config
    config(force_reload=True)

    # 手順: import lossless_memory.auto_index と import lossless_memory.daemon
    # 期待: どちらも import で落ちない
    import lossless_memory.auto_index  # noqa: F401
    import lossless_memory.daemon      # noqa: F401


def test_vector_index_skipped_log_emitted(tmp_path, monkeypatch, capsys):
    """追加テスト: index_vector の ImportError 時にベクトル索引スキップのログが1行出力されることを検証。
    理由: 「飛ばしたことを 1 行ログに出す」仕様の確実な検証。
    """
    monkeypatch.setitem(sys.modules, "lossless_memory.index_vector", None)
    monkeypatch.delitem(sys.modules, "lossless_memory.auto_index", raising=False)
    monkeypatch.delitem(sys.modules, "lossless_memory.daemon", raising=False)

    data_dir_path = tmp_path / "data"
    raw_dir_path = tmp_path / "raw"
    data_dir_path.mkdir()
    raw_dir_path.mkdir()

    proj_dir = raw_dir_path / "test-proj"
    proj_dir.mkdir()
    session_id = "11111111-1111-1111-1111-111111111111"
    line = _claude_line("2026-09-29T10:00:00Z", "user", "log check message", session_id)
    (proj_dir / f"{session_id}.jsonl").write_text(line + "\n", encoding="utf-8")

    config_file = tmp_path / "config.json"
    cfg = {
        "data_dir": str(data_dir_path),
        "raw_log_dir": str(raw_dir_path),
        "ingest_format": "claude_code",
        "raw_log_recursive": True,
    }
    config_file.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("LM_CONFIG_PATH", str(config_file))
    monkeypatch.chdir(tmp_path)

    from lossless_memory.config import config
    config(force_reload=True)

    import lossless_memory.auto_index as auto_index

    updated, msg = auto_index.run(force=True)
    assert updated is True
    assert "semantic 0" in msg

    captured = capsys.readouterr()
    stdout_lines = [l for l in captured.out.splitlines() if l.strip()]
    assert any("vector" in l.lower() and "skip" in l.lower() for l in stdout_lines)


def test_auto_index_incremental_without_vector(tmp_path, monkeypatch):
    """追加テスト: daemon が呼ぶ増分モード（force=False）で index_vector 不在時も動作することを検証。
    理由: daemon.tick() の通常実行パスにおける回帰防止。
    """
    monkeypatch.setitem(sys.modules, "lossless_memory.index_vector", None)
    monkeypatch.delitem(sys.modules, "lossless_memory.auto_index", raising=False)
    monkeypatch.delitem(sys.modules, "lossless_memory.daemon", raising=False)

    data_dir_path = tmp_path / "data"
    raw_dir_path = tmp_path / "raw"
    data_dir_path.mkdir()
    raw_dir_path.mkdir()

    proj_dir = raw_dir_path / "test-proj"
    proj_dir.mkdir()
    session_id = "11111111-1111-1111-1111-111111111111"
    line = _claude_line("2026-09-29T10:00:00Z", "user", "incremental message", session_id)
    (proj_dir / f"{session_id}.jsonl").write_text(line + "\n", encoding="utf-8")

    config_file = tmp_path / "config.json"
    cfg = {
        "data_dir": str(data_dir_path),
        "raw_log_dir": str(raw_dir_path),
        "ingest_format": "claude_code",
        "raw_log_recursive": True,
    }
    config_file.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("LM_CONFIG_PATH", str(config_file))
    monkeypatch.chdir(tmp_path)

    from lossless_memory.config import config
    config(force_reload=True)

    import lossless_memory.auto_index as auto_index

    # First incremental run indexes the new file
    updated, msg = auto_index.run(force=False)
    assert updated is True
    assert "semantic 0" in msg

    # Second incremental run detects no change
    updated, msg = auto_index.run(force=False)
    assert updated is False
    assert "skipped" in msg

