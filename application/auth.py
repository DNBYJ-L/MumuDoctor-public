#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
账号体系：注册、登录、模式偏好（MySQL）。

数据库连接参数全部从环境变量读取（线上由 systemd 的 EnvironmentFile 注入）：

    MYSQL_HOST       默认 127.0.0.1
    MYSQL_PORT       默认 3306
    MYSQL_USER       默认 mumudoctor
    MYSQL_PASSWORD   必填，没有就报「数据库未配置」，不会连到别的库
    MYSQL_DATABASE   默认 mumudoctor

表结构由 ensure_schema() 在第一次访问时自动创建（CREATE TABLE IF NOT EXISTS）：

    users
      id              BIGINT UNSIGNED 自增主键
      account         账号，唯一
      password_hash   口令摘要，格式 pbkdf2_sha256$<轮数>$<盐>$<摘要>
      gender          male / female / other
      age             1–120
      default_mode    normal / elder，登录后默认使用的界面模式
      prompt_on_login 1 = 登录时询问 60 岁以上用户用哪种模式；0 = 直接按 default_mode
      created_at / updated_at / last_login_at

密码只以摘要形式落库：PBKDF2-HMAC-SHA256，每账号随机盐，比对用 compare_digest。
本模块返回的用户字典不含 password_hash，可以直接放进 st.session_state。
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Iterator, Optional

logger = logging.getLogger(__name__)

MODE_NORMAL = "normal"
MODE_ELDER = "elder"
MODES = (MODE_NORMAL, MODE_ELDER)

GENDERS = ("male", "female", "other")

ACCOUNT_MIN = 2
ACCOUNT_MAX = 32
PASSWORD_MIN = 6
PASSWORD_MAX = 64
AGE_MIN = 1
AGE_MAX = 120

# 账号只允许中英文、数字和少量连接符；不允许空格，避免「看起来一样、其实不同」的账号
_ACCOUNT_RE = re.compile(r"^[\w\u4e00-\u9fa5.@+-]{2,32}$", re.UNICODE)

_HASH_ALGO = "pbkdf2_sha256"
_HASH_ITERATIONS = 260_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    account VARCHAR(32) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    gender ENUM('male', 'female', 'other') NOT NULL,
    age TINYINT UNSIGNED NOT NULL,
    default_mode ENUM('normal', 'elder') NOT NULL DEFAULT 'normal',
    prompt_on_login TINYINT(1) NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_login_at DATETIME NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uk_users_account (account)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_schema_ready = False


class AuthError(Exception):
    """可以直接展示给用户的账号错误。"""


class ValidationError(AuthError):
    """注册信息不合法。"""


class DatabaseError(AuthError):
    """数据库连不上或读写失败；细节只写进服务端日志。"""


# ===================== 数据库 =====================

def _driver():
    try:
        import pymysql  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出可操作的提示
        raise DatabaseError(
            "缺少 MySQL 驱动 PyMySQL，请在运行环境执行：pip install PyMySQL"
        ) from exc
    return pymysql


def _settings() -> Dict[str, object]:
    """每次连接时读环境变量，改 .env 重启服务即可生效，不需要改代码。"""
    if "MYSQL_PASSWORD" not in os.environ:
        raise DatabaseError(
            "数据库未配置：请设置 MYSQL_PASSWORD 环境变量后重启服务。"
        )
    try:
        port = int(os.environ.get("MYSQL_PORT", "3306"))
    except ValueError as exc:
        raise DatabaseError("MYSQL_PORT 不是合法端口号。") from exc
    return {
        "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
        "port": port,
        "user": os.environ.get("MYSQL_USER", "mumudoctor"),
        "password": os.environ["MYSQL_PASSWORD"],
        "database": os.environ.get("MYSQL_DATABASE", "mumudoctor"),
    }


def _connect(autocommit: bool = True):
    pymysql = _driver()
    try:
        return pymysql.connect(
            **_settings(),
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=autocommit,
            connect_timeout=5,
            read_timeout=10,
            write_timeout=10,
        )
    except pymysql.MySQLError as exc:
        logger.warning("MySQL connect failed: %s", exc)
        raise DatabaseError("数据库暂时连不上，请稍后再试。") from exc


def _ensure_schema(conn) -> None:
    global _schema_ready
    if _schema_ready:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA)
        _schema_ready = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure schema failed: %s", exc)
        raise DatabaseError("数据库表初始化失败，请检查数据库权限。") from exc


@contextmanager
def _cursor() -> Iterator:
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            yield cur
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def db_cursor() -> Iterator:
    """给同库模块（family.py 等）复用的游标：同一个 .env、同样的异常口径。

    autocommit 已开启，单条 INSERT/UPDATE 直接生效；需要多条一起成功时，
    用下面的 db_transaction()。
    """
    with _cursor() as cur:
        yield cur


@contextmanager
def db_transaction() -> Iterator:
    """多条语句一起成功 / 一起回滚的事务游标。

    历史记录保存要先删旧消息再插新消息，中途失败不能留下半份对话；
    这类跨表操作走这里。DDL（首次建表）不放在事务里，避免 MySQL 隐式提交。
    """
    conn = _connect(autocommit=False)
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


# ===================== 口令 =====================

def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("ascii"), _HASH_ITERATIONS
    )
    return f"{_HASH_ALGO}${_HASH_ITERATIONS}${salt}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt, expected = stored.split("$", 3)
        if algo != _HASH_ALGO:
            return False
        rounds = int(iterations)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("ascii"), rounds
        )
        return hmac.compare_digest(digest.hex(), expected)
    except (ValueError, AttributeError):
        return False


# ===================== 校验 =====================

def normalize_account(account: str) -> str:
    return (account or "").strip()


def validate_account(account: str) -> str:
    account = normalize_account(account)
    if not account:
        raise ValidationError("请填写账号。")
    if not (ACCOUNT_MIN <= len(account) <= ACCOUNT_MAX):
        raise ValidationError(f"账号长度要在 {ACCOUNT_MIN}–{ACCOUNT_MAX} 个字符之间。")
    if not _ACCOUNT_RE.fullmatch(account):
        raise ValidationError("账号只能用中文、字母、数字和 . @ + - _，不能有空格。")
    return account


def validate_password(password: str) -> None:
    if not password:
        raise ValidationError("请填写密码。")
    if not (PASSWORD_MIN <= len(password) <= PASSWORD_MAX):
        raise ValidationError(
            f"密码长度要在 {PASSWORD_MIN}–{PASSWORD_MAX} 个字符之间。"
        )


def validate_gender(gender: str) -> str:
    if gender not in GENDERS:
        raise ValidationError("请选择性别。")
    return gender


def validate_age(age: int) -> int:
    try:
        value = int(age)
    except (TypeError, ValueError) as exc:
        raise ValidationError("年龄要填数字。") from exc
    if not (AGE_MIN <= value <= AGE_MAX):
        raise ValidationError(f"年龄要在 {AGE_MIN}–{AGE_MAX} 岁之间。")
    return value


# ===================== 用户读写 =====================

def _public_user(row: Dict) -> Dict:
    """只保留能进 session_state / 前端的信息，口令摘要不出这个模块。"""
    return {
        "id": int(row["id"]),
        "account": str(row["account"]),
        "gender": str(row["gender"]),
        "age": int(row["age"]),
        "default_mode": str(row["default_mode"]),
        "prompt_on_login": bool(row["prompt_on_login"]),
    }


def get_user(user_id: int) -> Optional[Dict]:
    try:
        with _cursor() as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (int(user_id),))
            row = cur.fetchone()
        return _public_user(row) if row else None
    except DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_user failed: %s", exc)
        raise DatabaseError("读取账号信息失败，请稍后再试。") from exc


def oldest_user_id() -> Optional[int]:
    """最早注册的账号 id。

    账号体系上线前，对话历史没有 owner_id；升级后把旧记录归到最早注册的
    账号（也就是原来单人使用这个应用的人），避免历史突然消失。
    """
    try:
        with _cursor() as cur:
            cur.execute("SELECT id FROM users ORDER BY id ASC LIMIT 1")
            row = cur.fetchone()
        return int(row["id"]) if row else None
    except DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("oldest_user_id failed: %s", exc)
        raise DatabaseError("读取账号列表失败，请稍后再试。") from exc


def register(account: str, password: str, gender: str, age: int) -> Dict:
    """创建账号并返回用户信息；账号重复、格式错误都抛 AuthError。"""
    account = validate_account(account)
    validate_password(password)
    gender = validate_gender(gender)
    age = validate_age(age)

    pymysql = _driver()
    try:
        with _cursor() as cur:
            cur.execute(
                "INSERT INTO users (account, password_hash, gender, age,"
                " default_mode, prompt_on_login)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (account, _hash_password(password), gender, age, MODE_NORMAL, 1),
            )
            cur.execute("SELECT * FROM users WHERE account = %s", (account,))
            row = cur.fetchone()
    except pymysql.err.IntegrityError as exc:
        logger.info("register duplicate account: %s", account)
        raise ValidationError("这个账号已经被注册了，换一个试试。") from exc
    except DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("register failed: %s", exc)
        raise DatabaseError("注册失败，数据库暂时不可用。") from exc

    if not row:
        raise DatabaseError("注册后没有读到账号信息，请稍后再试。")
    return _public_user(row)


def authenticate(account: str, password: str) -> Optional[Dict]:
    """账号密码正确返回用户信息；账号不存在或密码错误返回 None。"""
    account = normalize_account(account)
    if not account or not password:
        return None

    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT * FROM users WHERE account = %s",
                (account,),
            )
            row = cur.fetchone()
            if not row or not _verify_password(password, row["password_hash"]):
                return None
            cur.execute(
                "UPDATE users SET last_login_at = %s WHERE id = %s",
                (datetime.now(), int(row["id"])),
            )
        return _public_user(row)
    except DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("authenticate failed: %s", exc)
        raise DatabaseError("登录失败，数据库暂时不可用。") from exc


def update_mode_preferences(
    user_id: int,
    default_mode: Optional[str] = None,
    prompt_on_login: Optional[bool] = None,
) -> Dict:
    """改默认模式 / 登录时是否询问；返回更新后的用户信息。"""
    fields = []
    values: list = []
    if default_mode is not None:
        if default_mode not in MODES:
            raise ValidationError("未知的界面模式。")
        fields.append("default_mode = %s")
        values.append(default_mode)
    if prompt_on_login is not None:
        fields.append("prompt_on_login = %s")
        values.append(1 if prompt_on_login else 0)
    if not fields:
        user = get_user(user_id)
        if not user:
            raise DatabaseError("账号不存在。")
        return user

    values.append(int(user_id))
    try:
        with _cursor() as cur:
            cur.execute(
                f"UPDATE users SET {', '.join(fields)} WHERE id = %s",
                tuple(values),
            )
            cur.execute("SELECT * FROM users WHERE id = %s", (int(user_id),))
            row = cur.fetchone()
    except DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("update_mode_preferences failed: %s", exc)
        raise DatabaseError("保存模式设置失败，请稍后再试。") from exc

    if not row:
        raise DatabaseError("账号不存在。")
    return _public_user(row)
