#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打卡与数据看板：指标定义、MySQL 读写、趋势统计与分析文案。

数据模型（见 _SCHEMA）
    checkins        一条打卡 = 一行（user_id / recorded_at / note）
    checkin_values  这条打卡里的每个指标各一行（metric_key / value / extra / recorded_at）

为什么不是"一个指标一列"：指标会继续加（腰围、血氧、尿酸……），
拆成行以后加指标只改本文件顶部的 SERIES，不用改表结构，也不用迁移数据；
画图时按 metric_key 取一条曲线即可。recorded_at / user_id 在 values 表里冗余一份，
是为了让"某个用户的某个指标最近 N 天"走一个索引就能取到。

数据库连接参数与 auth.py 完全一致（同一个 .env），失败时抛 CheckinError 子类，
页面层直接把 str(exc) 显示给用户，细节只写服务端日志。

本模块除 MySQL 外不依赖 Streamlit，可以单独测试；分析部分只做趋势汇总，
单次数值的判定口径沿用 selfcheck.py，不在这里重复写一套阈值。
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from selfcheck import (
    BP_ALARM_DBP,
    BP_ALARM_SBP,
    GLUCOSE_ALARM_HIGH,
    GLUCOSE_ALARM_LOW,
    TEMP_ALARM_HIGH,
    TEMP_ALARM_LOW,
)

logger = logging.getLogger(__name__)

# ===================== 指标定义 =====================


@dataclass(frozen=True)
class Series:
    """一条曲线（一个可测量的量）。"""

    key: str
    label: str
    unit: str
    group: str
    reference: str
    low: Optional[float] = None
    high: Optional[float] = None
    decimals: int = 1
    step: float = 0.1
    vmin: float = 0.0
    vmax: float = 100.0
    input: bool = True  # False 表示由系统计算，不接受用户直接填写（目前只有 BMI）


@dataclass(frozen=True)
class Group:
    """看板上的一张图：同一张图里的曲线单位一致。"""

    key: str
    label: str
    unit: str
    series: Tuple[str, ...]
    hint: str = ""


SERIES: Dict[str, Series] = {
    "sbp": Series("sbp", "收缩压", "mmHg", "bp", "90–139 mmHg", 90, 139, 0, 1, 70, 260),
    "dbp": Series("dbp", "舒张压", "mmHg", "bp", "60–89 mmHg", 60, 89, 0, 1, 40, 160),
    "glucose": Series(
        "glucose",
        "血糖",
        "mmol/L",
        "glucose",
        "空腹 3.9–6.1 · 餐后 2 小时 ＜7.8 mmol/L",
        3.9,
        7.8,
        1,
        0.1,
        2.0,
        35.0,
    ),
    "tc": Series("tc", "总胆固醇", "mmol/L", "lipid", "＜5.2 mmol/L", None, 5.2, 2, 0.01, 1.0, 20.0),
    "tg": Series("tg", "甘油三酯", "mmol/L", "lipid", "＜1.7 mmol/L", None, 1.7, 2, 0.01, 0.3, 30.0),
    "ldl": Series(
        "ldl", "低密度脂蛋白", "mmol/L", "lipid", "＜3.4 mmol/L", None, 3.4, 2, 0.01, 0.3, 15.0
    ),
    "hdl": Series(
        "hdl", "高密度脂蛋白", "mmol/L", "lipid", "≥1.0 mmol/L", 1.0, None, 2, 0.01, 0.2, 5.0
    ),
    "weight": Series("weight", "体重", "kg", "weight", "结合身高看 BMI", None, None, 1, 0.1, 15.0, 300.0),
    "bmi": Series("bmi", "BMI", "BMI", "bmi", "18.5–23.9", 18.5, 23.9, 1, 0.1, 8.0, 80.0, input=False),
    "height": Series("height", "身高", "cm", "height", "—", None, None, 1, 0.5, 80.0, 250.0),
    "heart_rate": Series(
        "heart_rate", "心率", "次/分", "hr", "60–100 次/分", 60, 100, 0, 1, 30, 220
    ),
    "temp": Series("temp", "体温", "°C", "temp", "36.0–37.2 °C", 36.0, 37.2, 1, 0.1, 34.0, 43.0),
    # 设备（手环 / 手表）每日同步的指标：不接受手填，由 device_daily 按天写入
    "sleep_hours": Series(
        "sleep_hours", "睡眠时长", "小时", "sleep", "成人 7–9 小时", 7.0, 9.0, 1, 0.1, 0.0, 16.0
    ),
    "sleep_efficiency": Series(
        "sleep_efficiency", "睡眠效率", "%", "sleep_eff", "≥85%", 85.0, None, 0, 1, 0.0, 100.0
    ),
    "spo2": Series("spo2", "血氧", "%", "spo2", "≥95%", 95.0, None, 0, 1, 70.0, 100.0),
}

# 看板上图的顺序：先血压血糖，再血脂体重，最后是辅助指标
GROUPS: Tuple[Group, ...] = (
    Group("bp", "血压", "mmHg", ("sbp", "dbp"), "静坐 5 分钟后测，早晚各一次更准"),
    Group("glucose", "血糖", "mmol/L", ("glucose",), "不同测量时点的参考范围不一样"),
    Group("lipid", "血脂", "mmol/L", ("tc", "tg", "ldl", "hdl"), "建议空腹 10 小时以上抽血"),
    Group("weight", "体重", "kg", ("weight",), "固定时间、同一台秤称重"),
    Group("bmi", "BMI", "BMI", ("bmi",), "由身高体重自动计算"),
    Group("height", "身高", "cm", ("height",), "成人身高基本稳定"),
    Group("hr", "心率", "次/分", ("heart_rate",), "静息状态下测量"),
    Group("temp", "体温", "°C", ("temp",), "腋温夹紧测量 5 分钟"),
    Group("sleep", "睡眠时长", "小时", ("sleep_hours",), "按醒来日期记，跨夜算一晚"),
    Group("spo2", "血氧", "%", ("spo2",), "每晚最低值，低于 95% 需要留意"),
)
GROUP_MAP: Dict[str, Group] = {group.key: group for group in GROUPS}

GLUCOSE_TYPES = ("空腹", "餐后2小时", "随机")
GLUCOSE_RANGES = {
    "空腹": (3.9, 6.1),
    "餐后2小时": (3.9, 7.8),
    "随机": (3.9, 11.1),
}

# 录入范围之外的值多半是打错了（比如身高填成 17），直接拒绝，不静默截断


class CheckinError(Exception):
    """可以直接展示给用户的打卡错误。"""


class ValidationError(CheckinError):
    """录入内容不合法。"""


class DatabaseError(CheckinError):
    """数据库连不上或读写失败；细节只写进服务端日志。"""


# ===================== 数据库 =====================

_SCHEMA_CHECKINS = """
CREATE TABLE IF NOT EXISTS checkins (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    user_id BIGINT UNSIGNED NOT NULL,
    recorded_at DATETIME NOT NULL,
    note VARCHAR(200) NOT NULL DEFAULT '',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_checkins_user_time (user_id, recorded_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_SCHEMA_VALUES = """
CREATE TABLE IF NOT EXISTS checkin_values (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    checkin_id BIGINT UNSIGNED NOT NULL,
    user_id BIGINT UNSIGNED NOT NULL,
    metric_key VARCHAR(24) NOT NULL,
    value DECIMAL(8,2) NOT NULL,
    extra VARCHAR(32) NOT NULL DEFAULT '',
    recorded_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    KEY idx_values_user_metric_time (user_id, metric_key, recorded_at),
    KEY idx_values_checkin (checkin_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_schema_ready = False


def _driver():
    try:
        import pymysql  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出可操作的提示
        raise DatabaseError(
            "缺少 MySQL 驱动 PyMySQL，请在运行环境执行：pip install PyMySQL"
        ) from exc
    return pymysql


def _settings() -> Dict[str, object]:
    """每次连接时读环境变量，改 .env 重启服务即可生效。"""
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


def _connect():
    pymysql = _driver()
    try:
        return pymysql.connect(
            **_settings(),
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False,  # 一条打卡 + 它的若干数值必须一起成功
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
            cur.execute(_SCHEMA_CHECKINS)
            cur.execute(_SCHEMA_VALUES)
        conn.commit()
        _schema_ready = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure checkin schema failed: %s", exc)
        raise DatabaseError("打卡数据表初始化失败，请检查数据库权限。") from exc


@contextmanager
def _cursor(commit: bool = False) -> Iterator:
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            yield cur
        if commit:
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


# ===================== 校验 =====================


def _clean_value(series: Series, raw) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{series.label} 要填数字。") from exc
    if not (series.vmin <= value <= series.vmax):
        raise ValidationError(
            f"{series.label} 的录入范围是 {series.vmin:g}–{series.vmax:g} {series.unit}，"
            "请检查是否填错。"
        )
    return round(value, series.decimals)


def bmi_of(height_cm: float, weight_kg: float) -> float:
    """BMI = 体重(kg) / 身高(m)²，保留一位小数。"""
    return round(float(weight_kg) / ((float(height_cm) / 100) ** 2), 1)


def bmi_status(bmi: float) -> Tuple[str, str]:
    """返回 (status, 文案)，与自查页的口径一致。"""
    if bmi < 18.5:
        return "low", "偏瘦"
    if bmi < 24:
        return "ok", "正常"
    if bmi < 28:
        return "watch", "超重"
    return "high", "肥胖"


def normalize_values(
    values: Dict[str, object], glucose_type: str = "空腹"
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """把表单里的一堆 Optional 值整理成「要落库的指标」；留空的一律丢掉。"""
    clean: Dict[str, float] = {}
    for key, raw in (values or {}).items():
        if raw is None or raw == "":
            continue
        series = SERIES.get(key)
        if series is None or not series.input:
            continue  # BMI 等派生指标不接受手填
        clean[key] = _clean_value(series, raw)

    if ("sbp" in clean) != ("dbp" in clean):
        raise ValidationError("血压要同时填收缩压和舒张压，只填一个没法画图。")
    if not clean:
        raise ValidationError("至少填写一项数据再保存。")

    if "height" in clean and "weight" in clean:
        clean["bmi"] = bmi_of(clean["height"], clean["weight"])

    extras: Dict[str, str] = {}
    if "glucose" in clean:
        extras["glucose"] = glucose_type if glucose_type in GLUCOSE_TYPES else "随机"
    return clean, extras


def validate_recorded_at(recorded_at: datetime, now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now()
    if recorded_at is None:
        raise ValidationError("请选择打卡时间。")
    if recorded_at > now + timedelta(minutes=10):
        raise ValidationError("打卡时间不能晚于当前时间。")
    if recorded_at < now - timedelta(days=3650):
        raise ValidationError("打卡时间太早了，请检查年份。")
    return recorded_at.replace(second=0, microsecond=0)


# ===================== 写入 =====================


def save(
    user_id: int,
    recorded_at: datetime,
    values: Dict[str, object],
    note: str = "",
    glucose_type: str = "空腹",
) -> int:
    """保存一条打卡，返回 checkin id。没有任何指标时抛 ValidationError。"""
    if not user_id:
        raise ValidationError("请先登录。")
    recorded_at = validate_recorded_at(recorded_at)
    clean, extras = normalize_values(values, glucose_type)
    note = (note or "").strip()[:200]

    pymysql = _driver()
    try:
        with _cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO checkins (user_id, recorded_at, note) VALUES (%s, %s, %s)",
                (int(user_id), recorded_at, note),
            )
            checkin_id = int(cur.lastrowid)
            rows = [
                (
                    checkin_id,
                    int(user_id),
                    key,
                    value,
                    extras.get(key, ""),
                    recorded_at,
                )
                for key, value in clean.items()
            ]
            cur.executemany(
                "INSERT INTO checkin_values"
                " (checkin_id, user_id, metric_key, value, extra, recorded_at)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                rows,
            )
    except CheckinError:
        raise
    except pymysql.MySQLError as exc:
        logger.warning("save checkin failed: %s", exc)
        raise DatabaseError("打卡保存失败，请稍后再试。") from exc
    except Exception as exc:  # noqa: BLE001
        logger.warning("save checkin failed: %s", exc)
        raise DatabaseError("打卡保存失败，请稍后再试。") from exc

    return checkin_id


def save_many(user_id: int, items: Sequence[Dict], limit: int = 5000) -> int:
    """一次写入多条打卡（设备导入用）：整批一个事务，要么全成功，要么全回滚。

    同一 recorded_at 的多条会并成一条打卡，避免看板上出现两个重叠的点。
    返回写入的打卡（时间点）条数。
    """
    if not user_id:
        raise ValidationError("请先登录。")
    if not items:
        raise ValidationError("没有可导入的记录。")
    if len(items) > limit:
        raise ValidationError(f"一次最多写入 {limit} 条，请分批导入。")

    grouped: Dict[datetime, Dict[str, object]] = {}
    for index, item in enumerate(items, start=1):
        try:
            recorded_at = validate_recorded_at(item.get("recorded_at"))
            clean, extras = normalize_values(
                item.get("values") or {}, item.get("glucose_type") or "空腹"
            )
        except ValidationError as exc:
            raise ValidationError(f"第 {index} 条：{exc}") from exc
        entry = grouped.setdefault(recorded_at, {"values": {}, "extras": {}, "note": ""})
        entry["values"].update(clean)  # type: ignore[union-attr]
        entry["extras"].update(extras)  # type: ignore[union-attr]
        note = str(item.get("note") or "").strip()[:200]
        if note:
            entry["note"] = note

    pymysql = _driver()
    try:
        with _cursor(commit=True) as cur:
            for recorded_at, entry in grouped.items():
                cur.execute(
                    "INSERT INTO checkins (user_id, recorded_at, note) VALUES (%s, %s, %s)",
                    (int(user_id), recorded_at, entry["note"]),
                )
                checkin_id = int(cur.lastrowid)
                values = entry["values"]  # type: ignore[assignment]
                extras = entry["extras"]  # type: ignore[assignment]
                rows = [
                    (
                        checkin_id,
                        int(user_id),
                        key,
                        value,
                        extras.get(key, ""),
                        recorded_at,
                    )
                    for key, value in values.items()
                ]
                cur.executemany(
                    "INSERT INTO checkin_values"
                    " (checkin_id, user_id, metric_key, value, extra, recorded_at)"
                    " VALUES (%s, %s, %s, %s, %s, %s)",
                    rows,
                )
    except CheckinError:
        raise
    except pymysql.MySQLError as exc:
        logger.warning("save many checkins failed: %s", exc)
        raise DatabaseError("批量写入失败，数据没有保存，请稍后再试。") from exc
    except Exception as exc:  # noqa: BLE001
        logger.warning("save many checkins failed: %s", exc)
        raise DatabaseError("批量写入失败，数据没有保存，请稍后再试。") from exc

    return len(grouped)


def delete(user_id: int, checkin_id: int) -> None:
    """删掉一条打卡及其数值；只能删自己的。"""
    try:
        with _cursor(commit=True) as cur:
            cur.execute(
                "DELETE FROM checkin_values WHERE checkin_id = %s AND user_id = %s",
                (int(checkin_id), int(user_id)),
            )
            cur.execute(
                "DELETE FROM checkins WHERE id = %s AND user_id = %s",
                (int(checkin_id), int(user_id)),
            )
    except DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("delete checkin failed: %s", exc)
        raise DatabaseError("删除失败，请稍后再试。") from exc


# ===================== 读取 =====================


def list_points(user_id: int, since: Optional[datetime] = None) -> List[Dict]:
    """按时间升序返回某个用户的全部指标点。"""
    sql = (
        "SELECT checkin_id, metric_key, value, extra, recorded_at"
        " FROM checkin_values WHERE user_id = %s"
    )
    params: List[object] = [int(user_id)]
    if since is not None:
        sql += " AND recorded_at >= %s"
        params.append(since)
    sql += " ORDER BY recorded_at ASC, id ASC"

    try:
        with _cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall() or []
    except DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("list points failed: %s", exc)
        raise DatabaseError("读取打卡数据失败，请稍后再试。") from exc

    return [
        {
            "checkin_id": int(row["checkin_id"]),
            "metric_key": str(row["metric_key"]),
            "value": float(row["value"]),
            "extra": str(row.get("extra") or ""),
            "recorded_at": row["recorded_at"],
        }
        for row in rows
        if row.get("metric_key") in SERIES
    ]


def existing_pairs(user_id: int, start: datetime, end: datetime) -> set:
    """某个时间窗口里已经存在的 (metric_key, recorded_at)，导入去重用。"""
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT metric_key, recorded_at FROM checkin_values"
                " WHERE user_id = %s AND recorded_at BETWEEN %s AND %s",
                (int(user_id), start, end),
            )
            rows = cur.fetchall() or []
    except DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("existing pairs failed: %s", exc)
        raise DatabaseError("读取已有打卡失败，请稍后再试。") from exc

    return {
        (str(row["metric_key"]), row["recorded_at"])
        for row in rows
        if row.get("metric_key") in SERIES
    }


def stats(user_id: int) -> Dict:
    """看板顶部用的三个数：累计打卡次数、打卡天数、最近一次时间。"""
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS records,"
                " COUNT(DISTINCT DATE(recorded_at)) AS days,"
                " MAX(recorded_at) AS latest"
                " FROM checkins WHERE user_id = %s",
                (int(user_id),),
            )
            row = cur.fetchone() or {}
    except DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("checkin stats failed: %s", exc)
        raise DatabaseError("读取打卡统计失败，请稍后再试。") from exc

    return {
        "records": int(row.get("records") or 0),
        "days": int(row.get("days") or 0),
        "latest": row.get("latest"),
    }


def recent(user_id: int, limit: int = 8) -> List[Dict]:
    """最近几条打卡，给"最近记录"列表用。"""
    try:
        with _cursor() as cur:
            cur.execute(
                "SELECT id, recorded_at, note FROM checkins"
                " WHERE user_id = %s ORDER BY recorded_at DESC, id DESC LIMIT %s",
                (int(user_id), int(limit)),
            )
            rows = cur.fetchall() or []
            if not rows:
                return []
            ids = [int(row["id"]) for row in rows]
            placeholders = ", ".join(["%s"] * len(ids))
            cur.execute(
                "SELECT checkin_id, metric_key, value, extra FROM checkin_values"
                f" WHERE checkin_id IN ({placeholders})",
                tuple(ids),
            )
            values = cur.fetchall() or []
    except DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("recent checkins failed: %s", exc)
        raise DatabaseError("读取最近打卡失败，请稍后再试。") from exc

    by_checkin: Dict[int, List[Dict]] = {}
    for row in values:
        key = str(row["metric_key"])
        series = SERIES.get(key)
        if series is None:
            continue
        by_checkin.setdefault(int(row["checkin_id"]), []).append(
            {
                "key": key,
                "label": series.label,
                "unit": series.unit,
                "value": float(row["value"]),
                "extra": str(row.get("extra") or ""),
            }
        )

    return [
        {
            "id": int(row["id"]),
            "recorded_at": row["recorded_at"],
            "note": str(row.get("note") or ""),
            "items": by_checkin.get(int(row["id"]), []),
        }
        for row in rows
    ]


def latest_values(points: Sequence[Dict]) -> Dict[str, Dict]:
    """每个指标最近一次的值，看板卡片上显示"最近一次"。"""
    latest: Dict[str, Dict] = {}
    for point in points:
        current = latest.get(point["metric_key"])
        if current is None or point["recorded_at"] >= current["recorded_at"]:
            latest[point["metric_key"]] = point
    return latest


# ===================== 趋势统计 =====================


def in_range(series: Series, value: float, extra: str = "") -> bool:
    """单点是否落在参考范围内；血糖按测量时点分别判断。

    浮点误差（5.2 + 0.9 = 6.1000000000000005）会让边界值被误判成超标，
    比较时留 1e-9 的余量。
    """
    if series.key == "glucose":
        low, high = GLUCOSE_RANGES.get(extra or "", GLUCOSE_RANGES["随机"])
    else:
        low, high = series.low, series.high
    if low is not None and value < low - 1e-9:
        return False
    if high is not None and value > high + 1e-9:
        return False
    return True


def alarm_for(series: Series, value: float, extra: str = "") -> bool:
    """是否越过 selfcheck.py 里那条"建议尽快就医"的警报线。"""
    if series.key == "sbp":
        return value >= BP_ALARM_SBP
    if series.key == "dbp":
        return value >= BP_ALARM_DBP
    if series.key == "glucose":
        return value <= GLUCOSE_ALARM_LOW or value >= GLUCOSE_ALARM_HIGH
    if series.key == "temp":
        return value >= TEMP_ALARM_HIGH or value <= TEMP_ALARM_LOW
    return False


def _slope_per_day(points: Sequence[Tuple[datetime, float]]) -> float:
    """最小二乘斜率（单位 / 天）。点太少时返回 0。"""
    if len(points) < 2:
        return 0.0
    start = points[0][0]
    xs = [(moment - start).total_seconds() / 86400 for moment, _ in points]
    ys = [value for _, value in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


@dataclass
class SeriesStats:
    key: str
    label: str
    unit: str
    group: str
    group_label: str
    reference: str
    count: int
    mean: float
    minimum: float
    maximum: float
    first: float
    last: float
    latest_at: Optional[datetime]
    change_pct: float
    slope_per_day: float
    trend: str
    abnormal: int
    points: List[Tuple[datetime, float]] = field(default_factory=list)

    @property
    def decimals(self) -> int:
        return SERIES[self.key].decimals

    def fmt(self, value: float) -> str:
        return f"{value:.{self.decimals}f}"


@dataclass
class CheckinReport:
    """一次"指标趋势分析"的全部素材，简要版和 PDF 详细版共用。"""

    series: List[SeriesStats]
    group_keys: List[str]
    window_days: Optional[int]
    start: Optional[datetime]
    end: Optional[datetime]
    records: int
    days: int
    conclusion: str = ""
    suggestions: List[str] = field(default_factory=list)
    urgent: List[str] = field(default_factory=list)

    @property
    def groups_label(self) -> str:
        labels = [GROUP_MAP[key].label for key in self.group_keys if key in GROUP_MAP]
        return "、".join(labels) if labels else "—"

    @property
    def window_label(self) -> str:
        return f"近 {self.window_days} 天" if self.window_days else "全部记录"


def _trend_label(rel_change: float, count: int) -> str:
    if count < 3:
        return "记录较少，暂不判断趋势"
    if abs(rel_change) < 0.05:
        return "整体平稳"
    return "呈上升趋势" if rel_change > 0 else "呈下降趋势"


def analyze(
    points: Sequence[Dict],
    group_keys: Sequence[str],
    window_days: Optional[int] = None,
) -> CheckinReport:
    """把一堆指标点汇总成报告素材。points 应已按时间窗口过滤。"""
    selected = [key for key in group_keys if key in GROUP_MAP]
    allowed = set()
    for key in selected:
        allowed.update(GROUP_MAP[key].series)

    buckets: Dict[str, List[Tuple[datetime, float, str]]] = {}
    for point in points:
        if point["metric_key"] not in allowed:
            continue
        buckets.setdefault(point["metric_key"], []).append(
            (point["recorded_at"], point["value"], point.get("extra") or "")
        )

    stats_list: List[SeriesStats] = []
    for group in GROUPS:
        if group.key not in selected:
            continue
        for series_key in group.series:
            rows = sorted(buckets.get(series_key, []), key=lambda item: item[0])
            if not rows:
                continue
            series = SERIES[series_key]
            values = [value for _, value, _ in rows]
            moments = [moment for moment, _, _ in rows]
            mean = sum(values) / len(values)
            first, last = values[0], values[-1]
            change_pct = ((last - first) / first * 100) if first else 0.0
            slope = _slope_per_day([(moment, value) for moment, value, _ in rows])
            span_days = max((moments[-1] - moments[0]).total_seconds() / 86400, 1.0) if len(rows) > 1 else 1.0
            relative = (slope * span_days / mean) if mean else 0.0
            abnormal = sum(
                1 for _, value, extra in rows if not in_range(series, value, extra)
            )
            stats_list.append(
                SeriesStats(
                    key=series_key,
                    label=series.label,
                    unit=series.unit,
                    group=group.key,
                    group_label=group.label,
                    reference=series.reference,
                    count=len(values),
                    mean=mean,
                    minimum=min(values),
                    maximum=max(values),
                    first=first,
                    last=last,
                    latest_at=moments[-1],
                    change_pct=change_pct,
                    slope_per_day=slope,
                    trend=_trend_label(relative, len(values)),
                    abnormal=abnormal,
                    points=[(moment, value) for moment, value, _ in rows],
                )
            )

    checkin_ids = {
        point["checkin_id"] for point in points if point["metric_key"] in allowed
    }
    moments = [
        point["recorded_at"] for point in points if point["metric_key"] in allowed
    ]
    report = CheckinReport(
        series=stats_list,
        group_keys=selected,
        window_days=window_days,
        start=min(moments) if moments else None,
        end=max(moments) if moments else None,
        records=len(checkin_ids),
        days=len({moment.date() for moment in moments}),
    )

    # 最新一次越过警报线的指标：简报里要单独提醒（与自查页同一条线）
    urgent: List[str] = []
    for item in stats_list:
        latest_value = item.points[-1][1]
        if alarm_for(SERIES[item.key], latest_value):
            urgent.append(item.label)
    report.urgent = urgent
    return report


# ===================== 分析文案 =====================

_SUGGESTION_BY_KEY = {
    "sbp": "每天固定时间测血压并记录，食盐控制在每日 5 克以内",
    "dbp": "每天固定时间测血压并记录，避免熬夜与连续久坐",
    "glucose": "减少含糖饮料与精制主食，按测量类型固定时点复测",
    "tc": "减少动物油脂与油炸食品，两周后复查血脂",
    "tg": "限酒、控糖，抽血前保证空腹 10 小时以上",
    "ldl": "少吃肥肉与加工肉制品，必要时到心内科评估",
    "hdl": "每周累计 150 分钟中等强度有氧运动，有助于抬高高密度脂蛋白",
    "weight": "控制总热量，每周固定时间称重一次看趋势",
    "bmi": "先定一个小目标：3 个月减重 5%，同时增加日常活动量",
    "heart_rate": "避免咖啡因与熬夜，静息状态下复测心率",
    "temp": "多饮水、保证休息；反复发热请及时就医",
    "height": "身高变化不大，保持每季度测量一次即可",
}

_GENERIC_SUGGESTIONS = (
    "尽量在相同时间、相同条件下测量，趋势才有可比性",
    "单次数值只作参考，连续两周的记录比一次测量更有意义",
    "把这次报告里的异常项记下来，2–4 周后按同一条件复测一次",
)


def _conclusion_from_rules(report: CheckinReport) -> str:
    if not report.series:
        return "所选指标在统计范围内没有记录，先完成几次打卡再来生成趋势分析。"

    head = (
        f"{report.window_label}共打卡 {report.records} 次（覆盖 {report.days} 天），"
        f"分析了 {len(report.series)} 项指标的记录。"
    )

    abnormal = [item for item in report.series if item.abnormal]
    if abnormal:
        parts = "、".join(
            f"{item.label}（{item.abnormal}/{item.count} 次超出参考范围）"
            for item in abnormal[:4]
        )
        body = f"其中 {parts} 值得留意。"
    else:
        body = "所选指标在记录期间都没有超出参考范围。"

    trending = [item for item in report.series if item.trend in ("呈上升趋势", "呈下降趋势")]
    if trending:
        trend_parts = "、".join(
            f"{item.label}{item.trend.replace('呈', '').replace('趋势', '')}"
            for item in trending[:3]
        )
        body += f"从走势看，{trend_parts}；其余指标基本平稳。"
    else:
        body += "从走势看，各项指标整体平稳，没有明显持续上升或下降的迹象。"

    latest = [item for item in report.series if item.latest_at]
    if latest:
        newest = max(latest, key=lambda item: item.latest_at)
        gap = (datetime.now() - newest.latest_at).days
        gap_text = "就在今天" if gap <= 0 else f"距离现在已经 {gap} 天"
        body += (
            f"最近一次记录在 {newest.latest_at.strftime('%Y-%m-%d %H:%M')}，{gap_text}，"
            "如果间隔较久，建议再补一次打卡做对比。"
        )

    if report.urgent:
        head = (
            f"最近一次记录中，{'、'.join(report.urgent)} 已经越过建议尽快就医的警报线，"
            "请今天内复测并联系医生。" + head
        )

    return head + body + "单次测量会受情绪、饮食、睡眠和测量方式影响，趋势比单个数值更可靠。"


def fallback_analysis(report: CheckinReport) -> Tuple[str, List[str]]:
    """模型不可用时的兜底：结论和建议都由统计规则拼出来。"""
    conclusion = _conclusion_from_rules(report)

    suggestions: List[str] = []
    ordered = sorted(report.series, key=lambda item: (-item.abnormal, item.key not in ("sbp", "dbp", "glucose")))
    for item in ordered:
        if item.abnormal == 0 and item.trend == "整体平稳":
            continue
        tip = _SUGGESTION_BY_KEY.get(item.key)
        if tip and tip not in suggestions:
            suggestions.append(tip)
        if len(suggestions) >= 4:
            break

    for tip in _GENERIC_SUGGESTIONS:
        if len(suggestions) >= 3:
            break
        if tip not in suggestions:
            suggestions.append(tip)

    return conclusion, suggestions[:5]


def build_prompt(report: CheckinReport, elder: bool = False) -> str:
    rows = "\n".join(
        f"- {item.label}（{item.unit}，参考 {item.reference}）：记录 {item.count} 次，"
        f"平均 {item.fmt(item.mean)}，最低 {item.fmt(item.minimum)}，最高 {item.fmt(item.maximum)}，"
        f"最近一次 {item.fmt(item.last)}，变化 {item.change_pct:+.1f}%，走势{item.trend}，"
        f"超出参考范围 {item.abnormal} 次"
        for item in report.series
    )
    urgent = "、".join(report.urgent) if report.urgent else "无"

    if elder:
        return f"""你是全科医生助手。下面是这位老人最近 {report.window_label} 的打卡趋势统计（不是单次测量）。

{rows}

最近一次越过警报线的指标：{urgent}

要求：
1. 先写「结论：」，80–120 字，最多 3 句。第一句说整体要不要紧，第二句点出最需要留意的一项，
   用"可能与……有关"这类措辞，不下诊断。
2. 再写「建议：」，只写 3 条，每条单独一行，动词开头，不超过 15 字，要具体可做。
3. 不用专业名词，非用不可就在括号里换成大白话。
4. 不出现药名、剂量，不出现"确诊""你得了"这类判断。
5. 只依据上面的统计，不要编造没有记录的指标。"""

    return f"""你是全科医生助手。下面是用户{report.window_label}的打卡趋势统计（不是单次测量）。

{rows}

最近一次越过警报线的指标：{urgent}

请据此写一段趋势解读，要求：
1. 先写「结论：」，150–220 字。第一句总述整体趋势（哪些平稳、哪些上升或下降、有没有超出参考范围），
   再点出最值得注意的 1–3 项，说明可能相关的原因（用"可能与……有关"这类措辞），不要下诊断。
2. 再写「建议：」，3–5 条，每条单独一行，动词开头，不超过 30 字，要具体可执行，优先针对异常项和上升趋势。
3. 不出现药名和剂量，不出现"确诊""你得了"这类判断。
4. 若最近一次记录越过警报线，第一句就要提示尽快复测或就医。
5. 只依据上面的统计，不要编造没有记录的指标，也不要逐条复述数字。"""


# ===================== 报告里的建议 =====================


def followup_plan(report: CheckinReport) -> List[Tuple[str, str]]:
    """详细版报告的"复测与就诊"：按指标给下一步。"""
    plan: List[Tuple[str, str]] = []

    def has(key: str) -> bool:
        return any(item.key == key for item in report.series)

    def item(key: str) -> Optional[SeriesStats]:
        for candidate in report.series:
            if candidate.key == key:
                return candidate
        return None

    bp = item("sbp") or item("dbp")
    if bp and (bp.abnormal or bp.trend == "呈上升趋势"):
        plan.append(("血压", "连续 7 天早晚各测一次并记录，复诊时带上这份记录。"))
    glucose = item("glucose")
    if glucose and (glucose.abnormal or glucose.trend != "整体平稳"):
        plan.append(("血糖", "按同一测量时点连续记录两周；条件允许时加查糖化血红蛋白。"))
    if any(item(key) for key in ("tc", "tg", "ldl", "hdl")):
        plan.append(("血脂", "建议空腹 10 小时后复查一次血脂四项，由医生判断是否需要干预。"))
    weight = item("weight") or item("bmi")
    if weight and weight.abnormal:
        plan.append(("体重", "每周固定时间称重一次看趋势，3 个月减重 5% 是稳妥的目标。"))
    if item("heart_rate"):
        plan.append(("心率", "静息状态下复测；持续异常建议做一次心电图。"))
    if item("temp"):
        plan.append(("体温", "体温反复超过 38.5 ℃ 或持续三天，请及时就医。"))

    if report.urgent:
        plan.insert(0, ("就医", f"最近一次 {'、'.join(report.urgent)} 越过警报线，建议今天内复测并就医。"))

    if not plan:
        plan.append(("保持", "维持现有作息与测量习惯，每 3–6 个月复查一次基础指标即可。"))

    return plan
