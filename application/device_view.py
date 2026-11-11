#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""设备看板的数据层：只读 device_bridge 的 SQLite，供「数据看板」的设备区渲染。

多账号
    每个绑定了小米账号的站点账号，在 device_bridge/accounts/<站点账号>/ 下有一份
    config.json（登录态）和 mihealth.db（同步下来的数据）；没迁移的多账号结构时，
    退回到 device_bridge/ 根目录下的单份 mihealth.db（对应 DEVICE_ACCOUNT）。
    页面的「设备」区只读当前「数据对象」自己的账号，不再提供跨账号的切换。

本模块不联网、不写库；文件不存在时返回空列表，由页面决定提示什么。
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence

BRIDGE_DIR = os.environ.get(
    "DEVICE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "device_bridge"),
)
ACCOUNTS_DIR = os.environ.get("DEVICE_ACCOUNTS_DIR", os.path.join(BRIDGE_DIR, "accounts"))
# 迁移前的单账号布局：device_bridge/mihealth.db
LEGACY_DB = os.environ.get("DEVICE_DB", os.path.join(BRIDGE_DIR, "mihealth.db"))
LEGACY_ACCOUNT = os.environ.get("DEVICE_ACCOUNT", "")

# 睡眠分期的状态码（小米云端返回）：2=深睡 3=浅睡 4=REM
STAGE_LABELS = {2: "深睡", 3: "浅睡", 4: "REM"}


# ===================== 账号与库路径 =====================


def db_path_for(account: Optional[str]) -> Optional[str]:
    """这个站点账号的设备库路径；没有绑定就返回 None。"""
    if not account:
        return None
    candidate = os.path.join(ACCOUNTS_DIR, str(account), "mihealth.db")
    if os.path.exists(candidate):
        return candidate
    if LEGACY_ACCOUNT and str(account) == LEGACY_ACCOUNT and os.path.exists(LEGACY_DB):
        return LEGACY_DB
    return None


def accounts() -> List[str]:
    """所有有设备数据的站点账号（按名字排序）。

    供排查 / 运维使用；看板只按当前「数据对象」的账号取库，不拿这份列表
    去做跨账号切换。
    """
    found = set()
    if os.path.isdir(ACCOUNTS_DIR):
        for name in os.listdir(ACCOUNTS_DIR):
            if os.path.exists(os.path.join(ACCOUNTS_DIR, name, "mihealth.db")):
                found.add(name)
    if LEGACY_ACCOUNT and os.path.exists(LEGACY_DB):
        found.add(LEGACY_ACCOUNT)
    return sorted(found)


def available(account: Optional[str] = None) -> bool:
    return db_path_for(account or LEGACY_ACCOUNT) is not None


# ===================== 读库 =====================


def _records(keys: Sequence[str], since: datetime, account: Optional[str]) -> List[Dict]:
    path = db_path_for(account)
    if not path:
        return []
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        con.row_factory = sqlite3.Row
        marks = ",".join(["?"] * len(keys))
        rows = con.execute(
            f"SELECT key, ts, value FROM records WHERE key IN ({marks}) AND ts >= ?"
            " ORDER BY ts ASC",
            (*keys, int(since.timestamp())),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()

    out: List[Dict] = []
    for row in rows:
        try:
            value = json.loads(row["value"] or "{}")
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            out.append({"key": row["key"], "ts": float(row["ts"]), "value": value})
    return out


def _local(ts: float) -> datetime:
    return datetime.fromtimestamp(float(ts))


# ===================== 各类数据 =====================


def steps_by_day(
    days: int = 7, today: Optional[date] = None, account: Optional[str] = None
) -> List[Dict]:
    """每天总步数（原始记录是每分钟增量，这里按天求和）。"""
    today = today or date.today()
    since = datetime.combine(today - timedelta(days=days - 1), datetime.min.time())
    totals: Dict[str, int] = {}
    for item in _records(("steps",), since, account):
        try:
            count = int(float(item["value"].get("steps", 0)))
        except (TypeError, ValueError):
            continue
        day = _local(item["ts"]).date().isoformat()
        totals[day] = totals.get(day, 0) + max(count, 0)
    return [
        {"day": day, "steps": totals.get(day, 0)}
        for day in sorted(
            (today - timedelta(days=offset)).isoformat()
            for offset in range(days - 1, -1, -1)
        )
    ]


def heart_rate_points(hours: int = 168, account: Optional[str] = None) -> List[Dict]:
    """心率抽查点（原始样本，不聚合）。"""
    since = datetime.now() - timedelta(hours=hours)
    points: List[Dict] = []
    for item in _records(("heart_rate",), since, account):
        try:
            bpm = float(item["value"].get("bpm"))
            ts = float(item["value"].get("time") or item["ts"])
        except (TypeError, ValueError):
            continue
        if 20 <= bpm <= 240:
            points.append({"when": _local(ts), "value": bpm})
    points.sort(key=lambda item: item["when"])
    return points


def spo2_points(hours: int = 168, account: Optional[str] = None) -> List[Dict]:
    since = datetime.now() - timedelta(hours=hours)
    points: List[Dict] = []
    for item in _records(("spo2",), since, account):
        try:
            value = float(item["value"].get("spo2"))
            ts = float(item["value"].get("time") or item["ts"])
        except (TypeError, ValueError):
            continue
        if 70 <= value <= 100:
            points.append({"when": _local(ts), "value": value})
    points.sort(key=lambda item: item["when"])
    return points


def sleep_nights(days: int = 7, account: Optional[str] = None) -> List[Dict]:
    """最近若干晚的睡眠：起止时间、时长、分期片段（按醒来时间倒序）。"""
    since = datetime.now() - timedelta(days=days + 1)
    nights: Dict[str, Dict] = {}
    for item in _records(("sleep",), since, account):
        data = item["value"]
        try:
            bedtime = float(data.get("bedtime") or data.get("device_bedtime") or 0)
            wake = float(
                data.get("device_wake_up_time") or data.get("wake_up_time") or 0
            )
            minutes = float(data.get("duration") or 0)
        except (TypeError, ValueError):
            continue
        if not wake:
            continue
        if not minutes and bedtime:
            minutes = (wake - bedtime) / 60.0
        hours = round(minutes / 60.0, 1)
        if hours <= 0:
            continue
        key = _local(wake).date().isoformat()
        stages: List[Dict] = []
        for segment in data.get("items") or []:
            try:
                state = int(segment.get("state"))
                start = float(segment.get("start_time"))
                end = float(segment.get("end_time"))
            except (TypeError, ValueError):
                continue
            label = STAGE_LABELS.get(state)
            if not label:
                continue
            stages.append(
                {
                    "label": label,
                    "start": _local(start),
                    "end": _local(end),
                    "hours": max((end - start) / 3600.0, 0.01),
                }
            )
        night = {
            "day": key,
            "bedtime": _local(bedtime) if bedtime else None,
            "wake": _local(wake),
            "hours": hours,
            "stages": stages,
        }
        hit = nights.get(key)
        if hit is None or night["hours"] > hit["hours"]:
            nights[key] = night
    return [nights[key] for key in sorted(nights, reverse=True)]


def today_summary(account: Optional[str] = None) -> Dict:
    """今日概览：步数、心率、血氧、昨夜睡眠。"""
    today = date.today()
    steps = steps_by_day(1, account=account)
    today_steps = steps[-1]["steps"] if steps else 0

    hr_today = [
        point
        for point in heart_rate_points(48, account=account)
        if point["when"].date() == today
    ]
    spo2_today = [
        point
        for point in spo2_points(48, account=account)
        if point["when"].date() == today
    ]
    nights = sleep_nights(3, account=account)
    last_night = nights[0] if nights else None

    hr_values = [point["value"] for point in hr_today]
    spo2_values = [point["value"] for point in spo2_today]
    return {
        "date": today,
        "steps": today_steps,
        "hr_avg": round(sum(hr_values) / len(hr_values)) if hr_values else None,
        "hr_min": round(min(hr_values)) if hr_values else None,
        "hr_max": round(max(hr_values)) if hr_values else None,
        "hr_count": len(hr_values),
        "spo2_min": round(min(spo2_values)) if spo2_values else None,
        "spo2_count": len(spo2_values),
        "sleep_hours": last_night["hours"] if last_night else None,
        "sleep_day": last_night["day"] if last_night else None,
    }
