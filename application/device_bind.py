#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""站内绑定设备账号：把小米运动健康的登录态落到 device_bridge/accounts/<站点账号>/。

登录态从哪来
    本机跑过 8567 那个桥接工具（mihealth-api）之后，会生成一份 config.json，
    里面是 ssecurity / service_token / cuser_id / pass_token。用户把这份文件传上来，
    或者手填 passToken + cUserId，本站就替他把文件写进对应账号的目录，并立刻起一次同步。

本模块只做三件事：校验字段、写文件（600）、拉起 daily.sh 的同步；不解析健康数据。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List

BRIDGE_DIR = os.environ.get(
    "DEVICE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "device_bridge"),
)
ACCOUNTS_DIR = os.environ.get("DEVICE_ACCOUNTS_DIR", os.path.join(BRIDGE_DIR, "accounts"))
PYTHON = os.environ.get("DEVICE_PYTHON", sys.executable)
LOGS_DIR = os.path.join(BRIDGE_DIR, "logs")
REQUIRED = ("ssecurity", "service_token", "cuser_id")


class BindError(Exception):
    """可以直接展示给用户的绑定错误。"""


def account_dir(account: str) -> str:
    return os.path.join(ACCOUNTS_DIR, str(account))


def config_path(account: str) -> str:
    return os.path.join(account_dir(account), "config.json")


def _clean(payload: Dict) -> Dict:
    """只留白名单字段，顺手去掉空值；不认识的字段一律丢掉。"""
    allowed = (
        "ssecurity",
        "service_token",
        "cuser_id",
        "user_id",
        "pass_token",
        "sid",
        "region",
        "host",
        "proxies",
        "db",
        "min_interval",
        "device_id",
        "session_at",
    )
    out: Dict[str, object] = {}
    for key in allowed:
        value = payload.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value in (None, ""):
            continue
        out[key] = value
    out.setdefault("sid", "miothealth")
    out.setdefault("region", "cn")
    return out


_PASS_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:passToken|pass_token)\s*[\"']?\s*[:=]\s*[\"']?([^\"'&;,\s}]+)",
    re.I,
)
_CUSER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:cUserId|cuser_id)\s*[\"']?\s*[:=]\s*[\"']?([^\"'&;,\s}]+)",
    re.I,
)
_USER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:userId|user_id)\s*[\"']?\s*[:=]\s*[\"']?(\d{1,20})", re.I
)


def extract_credentials(text: str) -> Dict:
    """从 Cookie 串 / DevTools 响应 JSON / config.json 片段里抠出 passToken 和 cUserId。

    浏览器登录小米账号后，passToken 通常是 HttpOnly 的，document.cookie 读不到；
    所以最常见的来源是 DevTools 里 serviceLoginAuth2 的响应 JSON，或者
    Application → Cookies 里复制出来的一整串。这里对两种格式都做兼容。
    """
    from urllib.parse import unquote

    text = text or ""
    found: Dict[str, str] = {}
    for key, pattern in (
        ("pass_token", _PASS_RE),
        ("cuser_id", _CUSER_RE),
        ("user_id", _USER_RE),
    ):
        match = pattern.search(text)
        if not match:
            continue
        value = match.group(1).strip().strip("\"'")
        if "%" in value:
            value = unquote(value)
        found[key] = value
    return found


def parse_pasted_credentials(text: str) -> Dict:
    """把用户粘贴的内容变成可换票的字段；优先用粘贴里的值。"""
    found = extract_credentials(text)
    if not found.get("pass_token"):
        raise BindError(
            "没从粘贴内容里找到 passToken；请复制 passToken 那一整段，"
            "或直接在上面的输入框手填。"
        )
    if not found.get("cuser_id"):
        raise BindError("找到了 passToken，但没找到 cUserId；请把两者一起粘贴，或手填 cUserId。")
    return parse_manual(found["pass_token"], found["cuser_id"], found.get("user_id", ""))


def parse_config_text(text: str) -> Dict:
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise BindError("文件不是合法的 JSON，请传 8567 生成的 config.json") from exc
    if not isinstance(data, dict):
        raise BindError("config.json 应该是一个对象（键值对）")
    clean = _clean(data)
    missing = [key for key in REQUIRED if not clean.get(key)]
    if missing:
        raise BindError("配置里缺少：" + "、".join(missing))
    return clean


def parse_manual(pass_token: str, cuser_id: str, user_id: str = "") -> Dict:
    """手填 passToken 时，先用它换一次票，补齐 ssecurity / service_token。"""
    pass_token = (pass_token or "").strip()
    cuser_id = (cuser_id or "").strip()
    if not pass_token or not cuser_id:
        raise BindError("passToken 和 cUserId 都要填")
    exchanged = exchange_pass_token(pass_token, cuser_id, user_id.strip())
    return _clean(exchanged)


def exchange_pass_token(pass_token: str, cuser_id: str, user_id: str = "") -> Dict:
    """调用桥接工具的登录模块，用 passToken 换 serviceToken（顺带验证凭据）。"""
    script = r"""
import json, sys
sys.path.insert(0, %r)
from login import exchange_pass_token
token, cuser, uid = sys.argv[1], sys.argv[2], sys.argv[3]
res = exchange_pass_token(token, cuser, uid)
print(json.dumps(res, ensure_ascii=False))
""" % BRIDGE_DIR
    try:
        proc = subprocess.run(
            [PYTHON, "-c", script, pass_token, cuser_id, user_id],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired as exc:
        raise BindError("连小米服务器超时，请稍后重试") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise BindError(
            "passToken 验证失败：" + (detail[-1] if detail else "未知错误")
        )
    try:
        data = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (TypeError, ValueError, IndexError) as exc:
        raise BindError("登录返回格式异常，请重试") from exc
    data["pass_token"] = pass_token
    data["cuser_id"] = data.get("cuser_id") or cuser_id
    return data


def verify(config: Dict, account: str) -> None:
    """写盘前再验一次：有 passToken 就换票确认，没有就只做字段检查。"""
    if not config.get("pass_token"):
        return
    data = exchange_pass_token(
        str(config["pass_token"]),
        str(config.get("cuser_id") or ""),
        str(config.get("user_id") or ""),
    )
    config.update({k: v for k, v in data.items() if v})


def save(account: str, config: Dict) -> str:
    """把配置写到 accounts/<account>/config.json（目录 700、文件 600）。"""
    if not account:
        raise BindError("缺少站点账号")
    directory = account_dir(account)
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
        path = config_path(account)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        os.chmod(path, 0o600)
    except OSError as exc:
        raise BindError(f"写入配置失败：{exc}") from exc
    return path


def start_sync(account: str) -> None:
    """后台跑一次该账号的同步（不阻塞页面）。日志落在 device_bridge/logs/<账号>.log。"""
    try:
        os.makedirs(LOGS_DIR, mode=0o755, exist_ok=True)
        log_path = os.path.join(LOGS_DIR, f"{account}.log")
        handle = open(log_path, "a", encoding="utf-8")
        handle.write(
            "\n===== %s 手动触发同步 =====\n" % time.strftime("%Y-%m-%d %H:%M:%S")
        )
        handle.flush()
        subprocess.Popen(
            ["/bin/bash", os.path.join(BRIDGE_DIR, "daily.sh"), account],
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=BRIDGE_DIR,
            start_new_session=True,
        )
    except OSError as exc:
        raise BindError(f"拉起同步失败：{exc}") from exc


def status(account: str) -> List[str]:
    """最近一次同步的日志尾巴，给页面显示用。"""
    path = os.path.join(LOGS_DIR, f"{account}.log")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.read().strip().splitlines()
    except OSError:
        return []
    return lines[-6:]


def unbind(account: str) -> None:
    """解绑：删掉登录态和本地库，看板里就不会再出现这个设备账号。"""
    directory = account_dir(account)
    for name in ("config.json", "mihealth.db", "mihealth.db-wal", "mihealth.db-shm"):
        path = os.path.join(directory, name)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            continue
    try:
        os.rmdir(directory)
    except OSError:
        pass
