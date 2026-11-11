#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""设备每日同步：把 device_bridge 拉到的小米运动健康数据按天聚合，写进打卡。

为什么按天
    手表的心率是抽查（一天几十次），原样落库会在看板上画成噪声；按约定只保留
    每天一条中位数，明显异常的抽查单独记一条。睡眠、血氧本来就是每天一条：
    睡眠按「醒来日期」记当晚总时长与睡眠效率，血氧记当晚最低值。

数据来源
    device_bridge（mihealth-api）的本地 SQLite：records(family, key, sid, ts, value, num, ...)，
    value 是原始 JSON。本模块只读这个库，不联网、不碰小米接口。

写库
    checkin.existing_pairs 去重（同一个人同一分钟同一指标不重复写），
    checkin.save_many 一个事务整批写入，和手动打卡共用同一套范围校验。

用法
    python device_daily.py --db device_bridge/accounts/<站点账号>/mihealth.db \
        --account <站点账号> --days 7 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import auth  # noqa: E402
import checkin  # noqa: E402

# 每日心率：抽查低于/高于这两个数才算"异常"，单独记一条（不是诊断阈值）
HR_LOW = 50
HR_HIGH = 130
# 一天至少这么多次抽查才写中位数：样本太少时中位数容易只代表运动时段
HR_MIN_SAMPLES = 20
# 每日中位数落在几点：固定 08:00，图表上按天排列更整齐
HR_MEDIAN_HOUR = 8
# 血氧低于这个值多半是佩戴松动的噪声，不写进库
SPO2_FLOOR = 70


def _local(ts: float) -> datetime:
    return datetime.fromtimestamp(float(ts))


def _load_records(db_path: str, since: datetime) -> List[Dict]:
    since_ts = int(since.timestamp())
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise SystemExit(f"打不开 device_bridge 数据库：{exc}")
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT key, ts, value FROM records"
            " WHERE key IN ('heart_rate', 'sleep', 'spo2') AND ts >= ?"
            " ORDER BY ts ASC",
            (since_ts,),
        ).fetchall()
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


def _rows_heart_rate(records: Sequence[Dict], now: datetime) -> List[Dict]:
    """每天一条中位数；当天偏低/偏高的抽查各单独一条。"""
    by_day: Dict[str, List[Dict]] = {}
    for item in records:
        if item["key"] != "heart_rate":
            continue
        try:
            bpm = float(item["value"].get("bpm"))
            ts = float(item["value"].get("time") or item["ts"])
        except (TypeError, ValueError):
            continue
        if not 20 <= bpm <= 240:
            continue
        by_day.setdefault(_local(ts).date().isoformat(), []).append({"bpm": bpm, "ts": ts})

    rows: List[Dict] = []
    for day, samples in sorted(by_day.items()):
        values = [item["bpm"] for item in samples]
        if len(values) < HR_MIN_SAMPLES:
            continue
        median = int(round(statistics.median(values)))
        stamp = datetime.fromisoformat(day).replace(hour=HR_MEDIAN_HOUR, minute=0)
        if stamp > now - timedelta(minutes=1):
            stamp = (now - timedelta(minutes=1)).replace(second=0, microsecond=0)
        rows.append(
            {
                "recorded_at": stamp,
                "values": {"heart_rate": median},
                "note": f"日均中位数 · {len(values)} 次抽查",
            }
        )

        low = min(samples, key=lambda item: item["bpm"])
        if low["bpm"] < HR_LOW:
            rows.append(
                {
                    "recorded_at": _local(low["ts"]),
                    "values": {"heart_rate": int(round(low["bpm"]))},
                    "note": "异常心率 · 偏低（手表抽查）",
                }
            )
        high = max(samples, key=lambda item: item["bpm"])
        if high["bpm"] > HR_HIGH:
            rows.append(
                {
                    "recorded_at": _local(high["ts"]),
                    "values": {"heart_rate": int(round(high["bpm"]))},
                    "note": "异常心率 · 偏高（手表抽查）",
                }
            )
    return rows


def _rows_sleep(records: Sequence[Dict]) -> List[Dict]:
    """每晚一条：总时长（小时）+ 睡眠效率（%），按醒来日期记。"""
    rows: List[Dict] = []
    seen: set = set()
    for item in records:
        if item["key"] != "sleep":
            continue
        data = item["value"]
        if data.get("is_uncomplete"):
            continue
        try:
            wake = float(
                data.get("device_wake_up_time") or data.get("wake_up_time") or 0
            )
            bedtime = float(data.get("bedtime") or 0)
            minutes = float(data.get("duration") or 0)
        except (TypeError, ValueError):
            continue
        if not minutes and wake and bedtime:
            minutes = (wake - bedtime) / 60.0
        hours = round(minutes / 60.0, 1)
        # 不设时长下限：白天小睡也照实记（之前按 0.5 小时过滤，用户明确要求取消）
        if not 0 < hours <= 16.0:
            continue
        stamp = _local(wake or (bedtime + minutes * 60)).replace(second=0, microsecond=0)
        day = stamp.date().isoformat()
        if day in seen:  # 同一晚可能有睡眠分段，只留最长的一段
            continue
        seen.add(day)
        values: Dict[str, float] = {"sleep_hours": hours}
        try:
            efficiency = float(data.get("sleep_efficiency") or 0)
        except (TypeError, ValueError):
            efficiency = 0.0
        if 0 < efficiency <= 100:
            values["sleep_efficiency"] = round(efficiency, 0)
        rows.append({"recorded_at": stamp, "values": values, "note": "睡眠同步"})
    return rows


def _rows_spo2(records: Sequence[Dict]) -> List[Dict]:
    """每天一条：当日最低血氧，记在那次测量的时间。"""
    by_day: Dict[str, Dict] = {}
    for item in records:
        if item["key"] != "spo2":
            continue
        try:
            value = float(item["value"].get("spo2"))
            ts = float(item["value"].get("time") or item["ts"])
        except (TypeError, ValueError):
            continue
        if not SPO2_FLOOR <= value <= 100:
            continue
        day = _local(ts).date().isoformat()
        hit = by_day.get(day)
        if hit is None or value < hit["value"]:
            by_day[day] = {"value": value, "ts": ts}
    return [
        {
            "recorded_at": _local(hit["ts"]),
            "values": {"spo2": int(round(hit["value"]))},
            "note": "当日最低血氧",
        }
        for _, hit in sorted(by_day.items())
    ]


def build_rows(records: Sequence[Dict], now: Optional[datetime] = None) -> List[Dict]:
    now = now or datetime.now()
    rows = (
        _rows_heart_rate(records, now)
        + _rows_sleep(records)
        + _rows_spo2(records)
    )
    rows.sort(key=lambda item: item["recorded_at"])
    return rows


def user_id_for(account: str) -> Optional[int]:
    with auth.db_cursor() as cur:
        cur.execute("SELECT id FROM users WHERE account = %s", (account,))
        row = cur.fetchone()
    return int(row["id"]) if row else None


def save(
    db_path: str,
    user_id: int,
    days: int = 7,
    now: Optional[datetime] = None,
    dry_run: bool = False,
) -> Dict:
    now = now or datetime.now()
    since = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    records = _load_records(db_path, since)
    rows = build_rows(records, now)

    start = min((item["recorded_at"] for item in rows), default=since)
    end = max((item["recorded_at"] for item in rows), default=now)
    existing = checkin.existing_pairs(user_id, start, end) if rows else set()

    fresh: List[Dict] = []
    skipped = 0
    for row in rows:
        stamp = row["recorded_at"].replace(second=0, microsecond=0)
        keep = {
            key: value
            for key, value in row["values"].items()
            if (key, stamp) not in existing
        }
        skipped += len(row["values"]) - len(keep)
        if keep:
            fresh.append(
                {
                    "recorded_at": stamp,
                    "values": keep,
                    "glucose_type": "随机",
                    "note": row["note"][:200],
                }
            )

    stats = {
        "records": len(records),
        "rows": len(rows),
        "fresh_rows": len(fresh),
        "values": sum(len(item["values"]) for item in fresh),
        "skipped": skipped,
        "days": days,
        "samples": [
            {
                "when": item["recorded_at"].strftime("%Y-%m-%d %H:%M"),
                "values": item["values"],
                "note": item["note"],
            }
            for item in fresh[:8]
        ],
    }
    if dry_run or not fresh:
        return stats
    stats["written"] = checkin.save_many(user_id, fresh)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="设备数据按天聚合后写入打卡")
    parser.add_argument("--db", required=True, help="device_bridge 的 mihealth.db 路径")
    parser.add_argument("--account", required=True, help="站点账号")
    parser.add_argument("--days", type=int, default=7, help="回溯天数")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写库")
    args = parser.parse_args()

    user_id = user_id_for(args.account)
    if not user_id:
        raise SystemExit(f"站点上找不到账号 {args.account}")
    stats = save(args.db, user_id, days=args.days, dry_run=args.dry_run)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
