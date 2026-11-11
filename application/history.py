#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
会话历史：MySQL 存储。

一个会话一行 chat_sessions，会话里的每条消息一行 chat_messages：
    chat_sessions  id(conv_id) / owner_id / title / created_at / updated_at
    chat_messages  conv_id / seq / role / content / redline

保存时机不变：每轮回答结束后整份覆盖（先删旧消息再按 seq 批量插入），
删除 + 插入放在同一个事务里，中途失败会回滚，不会留下半份对话。

账号隔离与家庭组：
    - 列表 / 打开 / 删除都带 owner_id，不能跨账号操作；
    - 家庭组主账号按成员的 owner_id 只读查看；
    - 升级前 data/history/*.json 里的旧记录由 migrate_legacy() 一次性导入，
      导入成功后把 JSON 移到 data/history/migrated/ 归档，重复调用不会重复导入。
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import auth

logger = logging.getLogger(__name__)

# 只用于旧 JSON 归档；新数据全部在 MySQL 里
HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "history"
MAX_ITEMS = 100

_VALID_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

_SCHEMA_SESSIONS = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id VARCHAR(80) NOT NULL,
    owner_id BIGINT UNSIGNED NOT NULL,
    title VARCHAR(64) NOT NULL DEFAULT '',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_chat_sessions_owner_updated (owner_id, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_SCHEMA_MESSAGES = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    conv_id VARCHAR(80) NOT NULL,
    seq INT UNSIGNED NOT NULL,
    role VARCHAR(16) NOT NULL,
    content MEDIUMTEXT NOT NULL,
    redline TINYINT(1) NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_chat_message_seq (conv_id, seq)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_schema_ready = False


def _ensure_tables() -> None:
    global _schema_ready
    if _schema_ready:
        return
    try:
        with auth.db_cursor() as cur:
            cur.execute(_SCHEMA_SESSIONS)
            cur.execute(_SCHEMA_MESSAGES)
        _schema_ready = True
    except auth.DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure history schema failed: %s", exc)
        raise


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _title(messages: List[Dict]) -> str:
    for message in messages:
        if message.get("role") == "user":
            text = (message.get("content") or "").strip().replace("\n", " ")
            if text:
                return text[:24] + ("…" if len(text) > 24 else "")
    return "空对话"


def _message_rows(conv_id: str, messages: List[Dict]) -> List[tuple]:
    rows = []
    for seq, message in enumerate(messages, start=1):
        role = str(message.get("role") or "assistant")
        if role not in ("user", "assistant", "system"):
            role = "assistant"
        rows.append(
            (
                conv_id,
                seq,
                role,
                str(message.get("content") or ""),
                1 if message.get("redline") else 0,
            )
        )
    return rows


def save(conv_id: str, messages: List[Dict], owner_id: Optional[int] = None) -> bool:
    """整份覆盖写入 MySQL。失败返回 False，调用方不用管。"""
    if not conv_id or not messages or owner_id is None:
        return False
    if not _VALID_ID.fullmatch(str(conv_id)):
        return False

    title = _title(messages)
    rows = _message_rows(conv_id, messages)
    now = datetime.now()
    try:
        _ensure_tables()
        with auth.db_transaction() as cur:
            cur.execute(
                "SELECT owner_id FROM chat_sessions WHERE id = %s FOR UPDATE",
                (conv_id,),
            )
            existing = cur.fetchone()
            if existing is not None and int(existing["owner_id"]) != int(owner_id):
                # 会话 id 撞到别人的记录：拒绝覆盖
                return False

            if existing is None:
                cur.execute(
                    "INSERT INTO chat_sessions"
                    " (id, owner_id, title, created_at, updated_at)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    (conv_id, int(owner_id), title, now, now),
                )
            else:
                cur.execute(
                    "UPDATE chat_sessions SET title = %s, updated_at = %s"
                    " WHERE id = %s",
                    (title, now, conv_id),
                )

            cur.execute("DELETE FROM chat_messages WHERE conv_id = %s", (conv_id,))
            if rows:
                cur.executemany(
                    "INSERT INTO chat_messages"
                    " (conv_id, seq, role, content, redline)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    rows,
                )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("save history failed: %s", exc)
        return False


def list_all(owner_id: Optional[int] = None, limit: int = MAX_ITEMS) -> List[Dict]:
    """按更新时间倒序返回会话摘要；给了 owner_id 就只返回该账号的记录。"""
    try:
        _ensure_tables()
        sql = (
            "SELECT s.id, s.owner_id, s.title,"
            " UNIX_TIMESTAMP(s.updated_at) AS updated_at,"
            " (SELECT COUNT(*) FROM chat_messages m WHERE m.conv_id = s.id) AS count"
            " FROM chat_sessions s"
        )
        params: List[object] = []
        if owner_id is not None:
            sql += " WHERE s.owner_id = %s"
            params.append(int(owner_id))
        sql += " ORDER BY s.updated_at DESC, s.id DESC LIMIT %s"
        params.append(int(limit))

        with auth.db_cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall() or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("list history failed: %s", exc)
        return []

    return [
        {
            "id": str(row["id"]),
            "owner_id": int(row["owner_id"]),
            "title": str(row.get("title") or "未命名"),
            "updated_at": float(row.get("updated_at") or 0),
            "count": int(row.get("count") or 0),
        }
        for row in rows
    ]


def load(conv_id: str, owner_id: Optional[int] = None) -> Optional[Dict]:
    if not _VALID_ID.fullmatch(str(conv_id or "")):
        return None
    try:
        _ensure_tables()
        with auth.db_cursor() as cur:
            cur.execute(
                "SELECT id, owner_id, title, UNIX_TIMESTAMP(updated_at) AS updated_at"
                " FROM chat_sessions WHERE id = %s",
                (conv_id,),
            )
            session = cur.fetchone()
            if not session:
                return None
            if owner_id is not None and int(session["owner_id"]) != int(owner_id):
                return None
            cur.execute(
                "SELECT role, content, redline FROM chat_messages"
                " WHERE conv_id = %s ORDER BY seq ASC",
                (conv_id,),
            )
            rows = cur.fetchall() or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("load history failed: %s", exc)
        return None

    return {
        "id": str(session["id"]),
        "owner_id": int(session["owner_id"]),
        "title": str(session.get("title") or "未命名"),
        "updated_at": float(session.get("updated_at") or 0),
        "messages": [
            {
                "role": str(row.get("role") or "assistant"),
                "content": str(row.get("content") or ""),
                "redline": bool(row.get("redline")),
            }
            for row in rows
        ],
    }


def delete(conv_id: str, owner_id: Optional[int] = None) -> bool:
    if not _VALID_ID.fullmatch(str(conv_id or "")):
        return False
    try:
        _ensure_tables()
        with auth.db_transaction() as cur:
            cur.execute(
                "SELECT owner_id FROM chat_sessions WHERE id = %s FOR UPDATE",
                (conv_id,),
            )
            row = cur.fetchone()
            if not row:
                return False
            if owner_id is not None and int(row["owner_id"]) != int(owner_id):
                return False
            cur.execute("DELETE FROM chat_messages WHERE conv_id = %s", (conv_id,))
            cur.execute("DELETE FROM chat_sessions WHERE id = %s", (conv_id,))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("delete history failed: %s", exc)
        return False


def _archive_path(path: Path) -> Path:
    archive_dir = HISTORY_DIR / "migrated"
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / path.name
    if target.exists():
        target = archive_dir / f"{path.stem}-{int(time.time())}{path.suffix}"
    return target


def migrate_legacy(default_owner_id: Optional[int]) -> int:
    """把 data/history/*.json 一次性导入 MySQL，成功的文件移到 migrated/ 归档。

    已有 owner_id 的按原账号导入；没有归属的归到最早注册的账号
    （default_owner_id）。重复调用安全：MySQL 已存在的会话不会重复插入。
    返回本次真正导入的条数。
    """
    if not HISTORY_DIR.exists():
        return 0
    paths = sorted(HISTORY_DIR.glob("*.json"))
    if not paths:
        return 0
    try:
        _ensure_tables()
    except Exception as exc:  # noqa: BLE001
        logger.warning("migrate history skipped: %s", exc)
        return 0

    migrated = 0
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skip broken history file %s: %s", path.name, exc)
            continue

        conv_id = str(data.get("id") or path.stem)
        messages = data.get("messages") or []
        if not _VALID_ID.fullmatch(conv_id) or not messages:
            continue

        stored_owner = data.get("owner_id")
        if stored_owner is not None:
            owner_id = int(stored_owner)
        elif default_owner_id:
            owner_id = int(default_owner_id)
        else:
            continue

        try:
            updated_at = float(data.get("updated_at") or 0)
        except (TypeError, ValueError):
            updated_at = 0
        stamp = (
            datetime.fromtimestamp(updated_at) if updated_at > 0 else datetime.now()
        )
        title = str(data.get("title") or _title(messages))[:64]
        rows = _message_rows(conv_id, messages)

        try:
            with auth.db_transaction() as cur:
                cur.execute(
                    "SELECT id FROM chat_sessions WHERE id = %s FOR UPDATE",
                    (conv_id,),
                )
                if cur.fetchone() is None:
                    cur.execute(
                        "INSERT INTO chat_sessions"
                        " (id, owner_id, title, created_at, updated_at)"
                        " VALUES (%s, %s, %s, %s, %s)",
                        (conv_id, owner_id, title, stamp, stamp),
                    )
                    cur.executemany(
                        "INSERT INTO chat_messages"
                        " (conv_id, seq, role, content, redline)"
                        " VALUES (%s, %s, %s, %s, %s)",
                        rows,
                    )
                    migrated += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("migrate history file %s failed: %s", path.name, exc)
            continue

        try:
            path.replace(_archive_path(path))
        except OSError as exc:
            logger.warning("archive history file %s failed: %s", path.name, exc)

    if migrated:
        logger.info("migrated %s history files to MySQL", migrated)
    return migrated
