#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""设备数据导入：把手表 / 手环导出的 CSV、JSON 写进打卡数据。

为什么不是"直连"
    Redmi Watch 6 这类小米手表走私有 BLE 协议，健康数据只落在手机上的
    Mi Fitness（小米运动健康）里；服务端是云上的 Streamlit，既没有蓝牙，
    厂商也没有公开的健康数据 API。所以系统接的是"手机侧已经导出的那份
    数据"，而不是手表本身。手机怎么导出见页面上的三步说明。

支持的输入
    CSV / JSON，两种表结构都能认：
        宽表：一行一个时间点，一列一个指标（时间, 心率, 体重, 体温 …）
        长表：时间 + 指标名 + 数值 三列（时间, 指标, 数值）
    列名中英文都行，带单位（"心率(bpm)"）也行；血压可以写成一格 "120/80"，
    也可以分成「收缩压 / 舒张压」两列。同一个时间点的多行会并成一条打卡。

写库复用打卡模块
    checkin.normalize_values 负责范围校验、血压成对、BMI 派生；
    checkin.save_many 负责一个事务整批写入。本模块不直接写 SQL，也不改表结构。
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import checkin

# 一次导入的上限：再多就先分批，免得页面点一下就把数据库写爆
MAX_ROWS = 5000
MAX_BYTES = 8 * 1024 * 1024

TEMPLATE_CSV = """时间,心率,体重,体温,收缩压,舒张压,血糖
2026-11-01 07:30,62,68.4,36.5,,,
2026-11-01 21:40,,68.2,36.6,118,76,
2026-11-02 07:20,64,,,120,78,5.4
2026-11-02 20:10,71,,,126,82,
"""

TEMPLATE_NAME = "设备导入模板.csv"


class DeviceImportError(Exception):
    """可以直接展示给用户的导入错误。"""


@dataclass
class Row:
    """一条待导入的记录：一个时间点 + 这时的若干指标。"""

    recorded_at: datetime
    values: Dict[str, float] = field(default_factory=dict)
    glucose_type: str = "随机"


@dataclass
class ParseResult:
    rows: List[Row] = field(default_factory=list)
    invalid: List[str] = field(default_factory=list)  # 逐行原因，最多留 20 条
    skipped_empty: int = 0  # 这一行一个能用的指标都没有
    columns: Dict[str, int] = field(default_factory=dict)  # metric_key -> 条数
    format_label: str = ""

    @property
    def value_count(self) -> int:
        return sum(self.columns.values())


@dataclass
class Plan:
    rows: List[Row]
    duplicates: int  # 库里已经有、这次会跳过的指标条数


# ===================== 列名识别 =====================

_BRACKETS = re.compile(r"[（(\[【][^）)\]】]*[）)\]】]")
_PUNCT = re.compile(r"[\s_\-—–·•.,，。:：;；/\\|%‰]+")


def _norm(text: object) -> str:
    """列名归一化：去掉括号里的单位、标点和大小写差异。"""
    raw = _BRACKETS.sub("", str(text or ""))
    return _PUNCT.sub("", raw).lower()


_TIME_HEADERS = (
    "时间",
    "日期",
    "记录时间",
    "测量时间",
    "采集时间",
    "时间戳",
    "日期时间",
    "记录日期",
    "date",
    "time",
    "datetime",
    "timestamp",
    "recorded_at",
    "recordedat",
    "date_time",
    "start_time",
    "starttime",
)
_TIME_NORMS = {_norm(word) for word in _TIME_HEADERS}
_TIME_CONTAINS = sorted(
    ((word, _norm(word)) for word in _TIME_HEADERS if len(_norm(word)) >= 2),
    key=lambda item: -len(item[1]),
)

# 指标别名：key 与 checkin.SERIES 对应；"血压"单独处理，一格 "120/80" 拆成两个数
_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "sbp": ("收缩压", "高压", "收缩压值", "sbp", "sys", "systolic", "systolic pressure"),
    "dbp": ("舒张压", "低压", "舒张压值", "dbp", "dia", "diastolic", "diastolic pressure"),
    "glucose": (
        "血糖",
        "血糖值",
        "血液葡萄糖",
        "glucose",
        "blood glucose",
        "bloodglucose",
        "bloodsugar",
        "bg",
        "glu",
    ),
    "tc": ("总胆固醇", "胆固醇", "tc", "cholesterol", "total cholesterol"),
    "tg": ("甘油三酯", "甘油三脂", "tg", "triglyceride", "triglycerides"),
    "ldl": ("低密度脂蛋白", "低密度脂蛋白胆固醇", "ldl", "ldl-c", "ldlc"),
    "hdl": ("高密度脂蛋白", "高密度脂蛋白胆固醇", "hdl", "hdl-c", "hdlc"),
    "weight": ("体重", "重量", "weight", "体重值"),
    "height": ("身高", "height"),
    "heart_rate": (
        "心率",
        "心跳",
        "静息心率",
        "心率值",
        "heart rate",
        "heartrate",
        "pulse",
        "hr",
        "bpm",
    ),
    "temp": ("体温", "体表温度", "温度", "temp", "temperature", "body temperature"),
    "sleep_hours": (
        "睡眠时长",
        "睡眠小时",
        "睡眠时间",
        "sleep",
        "sleep duration",
        "sleephours",
    ),
    "sleep_efficiency": (
        "睡眠效率",
        "睡眠评分",
        "睡眠质量",
        "sleep efficiency",
        "sleepefficiency",
    ),
    "spo2": ("血氧", "血氧饱和度", "spo2", "sp02", "blood oxygen", "oxygen"),
}

_BP_ALIASES = ("血压", "血压值", "blood pressure", "bloodpressure", "bp")

_EXACT: Dict[str, str] = {}
_CONTAINS: List[Tuple[str, str]] = []
for _key, _words in _METRIC_ALIASES.items():
    for _word in _words:
        _normed = _norm(_word)
        if _normed:
            _EXACT.setdefault(_normed, _key)
            if len(_normed) >= 2:
                _CONTAINS.append((_normed, _key))
for _word in _BP_ALIASES:
    _normed = _norm(_word)
    if _normed:
        _EXACT.setdefault(_normed, "_bp")
        if len(_normed) >= 2:
            _CONTAINS.append((_normed, "_bp"))
_CONTAINS.sort(key=lambda item: -len(item[0]))

_LONG_METRIC_NORMS = {
    _norm(word)
    for word in ("指标", "指标名称", "项目", "项目名称", "类型", "测量项", "metric", "type", "item")
}
_LONG_VALUE_NORMS = {
    _norm(word)
    for word in ("数值", "测量值", "值", "结果", "value", "result", "reading")
}
_GLUCOSE_TYPE_NORMS = {
    _norm(word)
    for word in ("时点", "血糖时点", "测量时点", "状态", "glucose type", "context")
}

_EMPTY_TOKENS = {"", "-", "--", "—", "n/a", "na", "null", "none", "nan", "无"}
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _is_time_header(header: object) -> bool:
    normed = _norm(header)
    if not normed:
        return False
    if normed in _TIME_NORMS:
        return True
    return any(word in normed for word, _ in _TIME_CONTAINS)


def _key_for_header(header: object) -> Optional[str]:
    """表头 -> 指标 key；认不出来返回 None。"""
    normed = _norm(header)
    if not normed:
        return None
    if normed in _EXACT:
        return _EXACT[normed]
    for word, key in _CONTAINS:
        if word in normed:
            return key
    return None


def _key_for_label(label: object) -> Optional[str]:
    """长表里的指标名 -> key：先按列名别名，再试 checkin 里的中文名。"""
    key = _key_for_header(label)
    if key:
        return key
    normed = _norm(label)
    for series_key, series in checkin.SERIES.items():
        if normed and normed == _norm(series.label):
            return series_key
    return None


def _to_number(raw: object) -> Optional[float]:
    """从单元格里抠出一个数：'62 bpm'、'68.4kg'、'36.5°C' 都能读。"""
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if math.isfinite(value) else None
    text = str(raw).strip().replace("，", ",")
    if text.lower() in _EMPTY_TOKENS:
        return None
    match = _NUMBER.search(text.replace(",", ""))
    if not match:
        return None
    value = float(match.group())
    return value if math.isfinite(value) else None


def _to_bp(raw: object) -> Optional[Tuple[float, float]]:
    """'120/80'、'120 / 80 mmHg' -> (120.0, 80.0)。"""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return None
    text = str(raw or "").strip()
    if not text:
        return None
    numbers = [float(item) for item in _NUMBER.findall(text.replace(",", ""))]
    if len(numbers) < 2:
        return None
    return numbers[0], numbers[1]


# ===================== 时间解析 =====================

_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y年%m月%d日 %H:%M:%S",
    "%Y年%m月%d日 %H:%M",
    "%Y年%m月%d日",
    "%Y%m%d%H%M%S",
)


def _to_local_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone().replace(tzinfo=None)


def _parse_time(raw: object) -> Optional[datetime]:
    if isinstance(raw, datetime):
        return _to_local_naive(raw)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        stamp = float(raw)
        if stamp > 1e12:  # 毫秒时间戳
            stamp /= 1000.0
        if 946684800 <= stamp <= 4102444800:  # 2000-01-01 ~ 2100-01-01
            return datetime.fromtimestamp(stamp)
        return None
    text = str(raw or "").strip()
    if not text:
        return None
    if text.isdigit():
        return _parse_time(float(text))
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
    if parsed is not None:
        return _to_local_naive(parsed)
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


# ===================== 读文件 =====================


def _decode(data: bytes) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DeviceImportError("文件编码认不出来，请另存为 UTF-8 的 CSV 再上传。")


def _sniff_delimiter(text: str) -> str:
    sample = "\n".join(line for line in text.splitlines()[:6] if line.strip())
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return "\t" if sample.count("\t") > sample.count(",") else ","


def _looks_like_header(row: Sequence[str], filename: str = "") -> bool:
    """表头要同时有「时间列」和至少一个能认出来的指标（或长表的指标 / 数值列）。

    导出文件常在正式表头前加一行「导出时间,2026-11-05」之类的前言，
    只看"有没有时间"会把它误认成表头，所以指标列也一起看。
    单指标文件的指标名写在文件名上（heart_rate.csv），也认。
    """
    if len(row) < 2:
        return False
    if not any(_is_time_header(cell) for cell in row):
        return False
    if any(_key_for_header(cell) for cell in row):
        return True
    if _long_columns(row) is not None:
        return True
    return bool(filename) and _key_for_label(Path(filename).stem) is not None


def _csv_table(
    text: str, name: str = ""
) -> Tuple[List[str], List[Tuple[int, List[str]]]]:
    delimiter = _sniff_delimiter(text)
    lines = [row for row in csv.reader(io.StringIO(text), delimiter=delimiter)]
    if not lines:
        raise DeviceImportError("文件里没有内容。")

    header_at: Optional[int] = None
    for index, row in enumerate(lines[:10]):
        if _looks_like_header(row, name):
            header_at = index
            break
    if header_at is None:
        raise DeviceImportError(
            "没找到表头：需要一列时间（「时间 / 日期 / date / time」）"
            "和至少一列指标（心率 / 体重 / 体温 / 血压 / 血糖…）。"
            "如果这份文件只有一个指标，把文件名取成指标名（如 heart_rate.csv）也可以。"
        )

    headers = [str(cell).strip() for cell in lines[header_at]]
    rows: List[Tuple[int, List[str]]] = []
    for offset, row in enumerate(lines[header_at + 1 :], start=header_at + 2):
        if not any(str(cell).strip() for cell in row):
            continue
        cells = [str(cell).strip() for cell in row][: len(headers)]
        cells += [""] * (len(headers) - len(cells))
        rows.append((offset, cells))
    return headers, rows


def _json_records(text: str) -> List[object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DeviceImportError(f"JSON 格式不对：第 {exc.lineno} 行附近。") from exc

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "items", "records", "list", "values", "rows", "数据", "记录"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    raise DeviceImportError("JSON 里没有可以导入的记录列表。")


def _json_table(text: str) -> Tuple[List[str], List[Tuple[int, List[str]]]]:
    records = _json_records(text)
    if not records:
        raise DeviceImportError("JSON 里没有记录。")

    if all(isinstance(item, dict) for item in records):
        headers: List[str] = []
        for item in records:
            for key in item:
                if key not in headers:
                    headers.append(str(key))
        rows = []
        for index, item in enumerate(records, start=1):
            cells = [str(item.get(header, "") or "") for header in headers]
            rows.append((index, cells))
        return headers, rows

    if all(isinstance(item, (list, tuple)) for item in records):
        headers = [str(cell) for cell in records[0]]
        rows = []
        for index, item in enumerate(records[1:], start=2):
            cells = [str(cell) for cell in item][: len(headers)]
            cells += [""] * (len(headers) - len(cells))
            rows.append((index, cells))
        return headers, rows

    raise DeviceImportError("JSON 的结构不支持：需要对象数组，或二维数组。")


def _mapping_from_filename(
    name: str,
    headers: Sequence[str],
    table: Sequence[Tuple[int, List[str]]],
    time_index: int,
) -> Dict[int, str]:
    """单指标文件：文件名就是指标名（heart_rate.csv、体重.csv）。

    导出工具常按数据键一列一个文件，表头只有 time / value，认不出指标列时
    按文件名补；值列取第一个能读成数字的列。
    """
    key = _key_for_label(Path(name).stem)
    if key is None:
        return {}
    for index in range(len(headers)):
        if index == time_index:
            continue
        cells = [row[index] for _, row in table[:50]]
        if not cells:
            continue
        numeric = sum(1 for cell in cells if _to_number(cell) is not None)
        if numeric * 2 >= len(cells):
            return {index: key}
    return {}


def _long_columns(headers: Sequence[str]) -> Optional[Tuple[int, int]]:
    """长表：返回 (指标列下标, 数值列下标)。"""
    metric_index: Optional[int] = None
    value_index: Optional[int] = None
    for index, header in enumerate(headers):
        normed = _norm(header)
        if metric_index is None and normed in _LONG_METRIC_NORMS:
            metric_index = index
        elif value_index is None and normed in _LONG_VALUE_NORMS:
            value_index = index
    if metric_index is not None and value_index is not None:
        return metric_index, value_index
    return None


def _glucose_type_of(value: object) -> str:
    text = str(value or "").strip()
    return text if text in checkin.GLUCOSE_TYPES else "随机"


def _merge_rows(rows: Sequence[Row]) -> List[Row]:
    """同一时间点的多行并成一条打卡，避免看板上出现两个重叠的点。"""
    merged: Dict[datetime, Row] = {}
    for row in rows:
        hit = merged.get(row.recorded_at)
        if hit is None:
            merged[row.recorded_at] = Row(
                recorded_at=row.recorded_at,
                values=dict(row.values),
                glucose_type=row.glucose_type,
            )
            continue
        hit.values.update(row.values)
        if row.glucose_type != "随机":
            hit.glucose_type = row.glucose_type
    return [merged[key] for key in sorted(merged)]


def _note_invalid(result: ParseResult, line: int, exc: Exception) -> None:
    if len(result.invalid) < 20:
        result.invalid.append(f"第 {line} 行：{exc}")


def parse(name: str, data: bytes, now: Optional[datetime] = None) -> ParseResult:
    """解析上传的文件；坏行只记原因，不影响其它行。"""
    if not data:
        raise DeviceImportError("文件是空的。")
    if len(data) > MAX_BYTES:
        raise DeviceImportError("文件太大了（超过 8 MB），请先按月份拆成几个文件。")

    text = _decode(data)
    stripped = text.lstrip()
    is_json = name.lower().endswith(".json") or stripped[:1] in ("[", "{")
    headers, table = _json_table(text) if is_json else _csv_table(text, name)

    if len(table) > MAX_ROWS:
        raise DeviceImportError(
            f"一次最多导入 {MAX_ROWS} 行，这份文件有 {len(table)} 行，请分批上传。"
        )

    result = ParseResult(format_label="JSON" if is_json else "CSV")
    long_columns = _long_columns(headers)

    if long_columns:
        metric_index, value_index = long_columns
        time_index = next(
            (index for index, header in enumerate(headers) if _is_time_header(header)), None
        )
        if time_index is None:
            raise DeviceImportError("长表需要一列时间（表头写「时间」或 date/time）。")
        for line, cells in table:
            recorded_at = _parse_time(cells[time_index])
            if recorded_at is None:
                _note_invalid(result, line, "时间认不出来")
                continue
            key = _key_for_label(cells[metric_index])
            if key is None:
                _note_invalid(result, line, f"不认识的指标「{cells[metric_index]}」")
                continue
            values: Dict[str, object] = {}
            if key == "_bp":
                pair = _to_bp(cells[value_index])
                if pair is None:
                    _note_invalid(result, line, "血压要写成 120/80 这样的两个数")
                    continue
                values["sbp"], values["dbp"] = pair
            else:
                number = _to_number(cells[value_index])
                if number is None:
                    _note_invalid(result, line, "数值不是数字")
                    continue
                values[key] = number
            _collect_row(result, recorded_at, values, "随机", line, now)
    else:
        time_index = next(
            (index for index, header in enumerate(headers) if _is_time_header(header)), None
        )
        if time_index is None:
            raise DeviceImportError("表头里没有时间列。")
        mapping: Dict[int, str] = {}
        glucose_index: Optional[int] = None
        for index, header in enumerate(headers):
            if index == time_index:
                continue
            key = _key_for_header(header)
            if key:
                mapping[index] = key
            elif _norm(header) in _GLUCOSE_TYPE_NORMS:
                glucose_index = index
        if not mapping:
            mapping = _mapping_from_filename(name, headers, table, time_index)
        if not mapping:
            raise DeviceImportError(
                "表头里没有能识别的指标列。支持：心率、体重、体温、血压、血糖、"
                "身高、血脂四项；列名带单位也可以，例如「心率(bpm)」。"
                "只导出一个指标的文件，把文件名取成指标名（如 heart_rate.csv）也行。"
            )

        for line, cells in table:
            recorded_at = _parse_time(cells[time_index])
            if recorded_at is None:
                _note_invalid(result, line, "时间认不出来")
                continue
            values = {}
            for index, key in mapping.items():
                raw = cells[index]
                if key == "_bp":
                    pair = _to_bp(raw)
                    if pair is None:
                        if str(raw).strip():
                            _note_invalid(result, line, "血压要写成 120/80 这样的两个数")
                        continue
                    values["sbp"], values["dbp"] = pair
                else:
                    number = _to_number(raw)
                    if number is not None:
                        values[key] = number
            glucose_type = (
                _glucose_type_of(cells[glucose_index])
                if glucose_index is not None
                else "随机"
            )
            _collect_row(result, recorded_at, values, glucose_type, line, now)

    result.rows = _merge_rows(result.rows)
    return result


def _collect_row(
    result: ParseResult,
    recorded_at: datetime,
    values: Dict[str, object],
    glucose_type: str,
    line: int,
    now: Optional[datetime],
) -> None:
    """校验一行并收进结果；范围、成对规则都交给 checkin.normalize_values。"""
    if not values:
        result.skipped_empty += 1
        return
    try:
        recorded_at = checkin.validate_recorded_at(recorded_at, now)
        clean, extras = checkin.normalize_values(values, glucose_type)
    except checkin.ValidationError as exc:
        _note_invalid(result, line, exc)
        return

    result.rows.append(
        Row(
            recorded_at=recorded_at,
            values=clean,
            glucose_type=extras.get("glucose", "随机"),
        )
    )
    for key in clean:
        result.columns[key] = result.columns.get(key, 0) + 1


# ===================== 去重与写入 =====================


def plan(user_id: int, rows: Sequence[Row]) -> Plan:
    """查一遍库里已有的时间点，算出这次真正会写进去的记录。"""
    if not rows:
        return Plan(rows=[], duplicates=0)
    start = min(row.recorded_at for row in rows)
    end = max(row.recorded_at for row in rows)
    try:
        existing = checkin.existing_pairs(user_id, start, end)
    except checkin.CheckinError:
        raise
    except Exception as exc:  # noqa: BLE001 - 页面层统一转成友好提示
        raise DeviceImportError("读取已有打卡失败，请稍后再试。") from exc

    kept: List[Row] = []
    duplicates = 0
    seen: set = set()
    for row in rows:
        fresh: Dict[str, float] = {}
        for key, value in row.values.items():
            marker = (key, row.recorded_at)
            if marker in existing or marker in seen:
                duplicates += 1
                continue
            seen.add(marker)
            fresh[key] = value
        if fresh:
            kept.append(
                Row(
                    recorded_at=row.recorded_at,
                    values=fresh,
                    glucose_type=row.glucose_type,
                )
            )
    return Plan(rows=kept, duplicates=duplicates)


def commit(user_id: int, rows: Sequence[Row], note: str = "设备导入") -> int:
    """把 plan() 留下的记录写进打卡表，返回写入的时间点条数。"""
    if not rows:
        return 0
    items = [
        {
            "recorded_at": row.recorded_at,
            "values": row.values,
            "glucose_type": row.glucose_type,
            "note": note,
        }
        for row in rows
    ]
    try:
        return checkin.save_many(user_id, items)
    except checkin.CheckinError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DeviceImportError("写入打卡失败，请稍后再试。") from exc
