#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户自查：把表单数据判定成结构化指标，再交给模型写成一段人话。

分工（很重要）：
- 阈值判定全部走本地规则（中国成人常用参考区间），不依赖模型。0.5B 级别的小模型
  记不住指南数值，让它"判断"血压是否偏高必然出错。
- 模型只负责把已有结论写成可读的段落；模型不可用时用规则拼装兜底，功能不降级。

本模块为纯逻辑，不依赖 Streamlit，可单独测试。
"""

from dataclasses import dataclass, field
from datetime import time
from typing import Dict, List, Optional, Tuple

# 状态码 → 展示文案
STATUS_LABEL = {
    "ok": "正常",
    "watch": "关注",
    "high": "偏高",
    "low": "偏低",
    "urgent": "建议尽快就医",
}

# 整体严重度排序，取最重的一项
_SEVERITY = {"ok": 0, "watch": 1, "low": 2, "high": 2, "urgent": 3}

# ===================== 警报线 =====================
# 越过这些数值就不再是"关注/偏高"，而是"建议当天就医"：结果页会弹框提示，
# 并给出附近医院。血压、血糖、体温三条线都在这里定，改阈值只改这一处。
BP_ALARM_SBP = 180
BP_ALARM_DBP = 110
GLUCOSE_ALARM_LOW = 3.0
GLUCOSE_ALARM_HIGH = 16.7
TEMP_ALARM_HIGH = 39.5
TEMP_ALARM_LOW = 35.0

ALARM_LINES = {
    "bp": f"≥{BP_ALARM_SBP}/{BP_ALARM_DBP} mmHg",
    "glucose": f"≤{GLUCOSE_ALARM_LOW:.1f} 或 ≥{GLUCOSE_ALARM_HIGH:.1f} mmol/L",
    "temp": f"≥{TEMP_ALARM_HIGH:.1f} 或 ≤{TEMP_ALARM_LOW:.1f} ℃",
}

ALARM_LINES_TEXT = " · ".join(
    f"{label} {ALARM_LINES[key]}"
    for key, label in (("bp", "血压"), ("glucose", "血糖"), ("temp", "体温"))
)


def alarm_line(key: str) -> str:
    """该指标的警报线文案；没有警报线的指标返回空串。"""
    return ALARM_LINES.get(key, "")



@dataclass
class Metric:
    key: str
    label: str
    value: str
    reference: str
    status: str = "ok"
    note: str = ""
    status_text: str = ""  # 覆盖默认状态文案，如作息节律的"熬夜"而非"偏高"
    dept: str = ""  # 覆盖默认就诊科室（如低体温走急诊内科而不是发热门诊）

    @property
    def status_label(self) -> str:
        return self.status_text or STATUS_LABEL.get(self.status, self.status)


@dataclass
class SelfCheckResult:
    metrics: List[Metric] = field(default_factory=list)
    level: str = "ok"
    urgent_items: List[str] = field(default_factory=list)

    @property
    def abnormal(self) -> List[Metric]:
        return [m for m in self.metrics if m.status != "ok"]

    @property
    def level_label(self) -> str:
        return {"ok": "各项正常", "watch": "部分指标需关注", "alert": "存在异常指标", "urgent": "存在需要尽快就医的指标"}[
            self.level
        ]


# ===================== 单项判定 =====================

def evaluate_bmi(height_cm: float, weight_kg: float) -> Metric:
    bmi = weight_kg / ((height_cm / 100) ** 2)
    if bmi < 18.5:
        status = "low"
    elif bmi < 24:
        status = "ok"
    elif bmi < 28:
        status = "watch"
    else:
        status = "high"
    return Metric("bmi", "体质指数 BMI", f"{bmi:.1f}", "18.5–23.9", status)


def evaluate_temperature(temp_c: float) -> Metric:
    if temp_c < TEMP_ALARM_LOW:
        metric = Metric(
            "temp",
            "体温",
            f"{temp_c:.1f} °C",
            "36.0–37.2 °C",
            "urgent",
            dept="急诊内科",
        )
        metric.note = (
            f"低于 {TEMP_ALARM_LOW:.1f} ℃，属低体温，可能与失温、甲状腺功能减退、"
            "低血糖或感染性休克有关，请立即就医并注意保暖。"
        )
        return metric
    if temp_c < 36.0:
        status = "low"
    elif temp_c <= 37.2:
        status = "ok"
    elif temp_c < 37.3:
        status = "watch"
    elif temp_c < 38.1:
        status = "high"
    elif temp_c < TEMP_ALARM_HIGH:
        status = "high"
    else:
        status = "urgent"
    return Metric("temp", "体温", f"{temp_c:.1f} °C", "36.0–37.2 °C", status)


def evaluate_blood_pressure(sbp: int, dbp: int) -> Metric:
    if sbp >= BP_ALARM_SBP or dbp >= BP_ALARM_DBP:
        status = "urgent"
    elif sbp >= 140 or dbp >= 90:
        status = "high"
    elif sbp >= 120 or dbp >= 80:
        status = "watch"
    elif sbp < 90 or dbp < 60:
        status = "low"
    else:
        status = "ok"
    return Metric("bp", "血压", f"{sbp}/{dbp} mmHg", "90–119 / 60–79 mmHg", status)


def evaluate_glucose(value: float, kind: str) -> Metric:
    reference = {
        "空腹": "3.9–6.1 mmol/L",
        "餐后2小时": "＜7.8 mmol/L",
    }.get(kind, "3.9–11.1 mmol/L")

    note = ""
    if value < GLUCOSE_ALARM_LOW:
        # 严重低血糖不区分测量时点，任何情况下都要立即处理
        status = "urgent"
        note = (
            f"低于 {GLUCOSE_ALARM_LOW:.1f} mmol/L，属严重低血糖：立即进食含糖食物；"
            "若出现意识不清，请立刻呼叫 120。"
        )
    elif kind == "空腹":
        if value < 3.9:
            status = "low"
        elif value <= 6.1:
            status = "ok"
        elif value < 7.0:
            status = "watch"
        elif value < GLUCOSE_ALARM_HIGH:
            status = "high"
        else:
            status = "urgent"
    elif kind == "餐后2小时":
        if value < 3.9:
            status = "low"
        elif value < 7.8:
            status = "ok"
        elif value < 11.1:
            status = "watch"
        elif value < GLUCOSE_ALARM_HIGH:
            status = "high"
        else:
            status = "urgent"
    else:  # 随机
        if value < 3.9:
            status = "low"
        elif value <= 11.1:
            status = "ok"
        elif value < GLUCOSE_ALARM_HIGH:
            status = "high"
        else:
            status = "urgent"

    if status == "urgent" and not note:
        note = (
            f"≥{GLUCOSE_ALARM_HIGH:.1f} mmol/L，显著高血糖，存在酮症酸中毒等急性并发症风险，"
            "请立即就医。"
        )

    metric = Metric(
        "glucose", f"血糖（{kind}）", f"{value:.1f} mmol/L", reference, status
    )
    metric.note = note
    return metric


def _bedtime_minutes(start: time) -> int:
    """把入睡时间换算成"当天 12:00 之后"的分钟数，凌晨入睡算作次日。"""
    minutes = start.hour * 60 + start.minute
    if minutes < 12 * 60:
        minutes += 24 * 60
    return minutes


def _judge_bedtime(start: time) -> Tuple[str, str]:
    minutes = _bedtime_minutes(start)
    if minutes <= 23 * 60:
        return "ok", "入睡时间正常"
    if minutes <= 24 * 60:
        return "watch", "入睡偏晚"
    return "high", "熬夜"


def _judge_wake(end: time) -> Tuple[str, str]:
    minutes = end.hour * 60 + end.minute
    if 5 * 60 <= minutes <= 8 * 60:
        return "ok", "起床正常"
    if 8 * 60 < minutes <= 9 * 60 + 30:
        return "watch", "起床偏晚"
    if minutes > 9 * 60 + 30:
        return "high", "起床过晚"
    return "watch", "起床过早"


def _sleep_note(hours: float, bedtime_status: str, wake_status: str) -> str:
    issues: List[str] = []
    if bedtime_status == "high":
        issues.append("入睡时间已进入熬夜区间")
    elif bedtime_status == "watch":
        issues.append("入睡时间偏晚")
    if hours < 7:
        issues.append(f"睡眠时长只有 {hours:.1f} 小时")
    elif hours > 9:
        issues.append(f"睡眠时长达到 {hours:.1f} 小时，偏长")
    if wake_status == "high":
        issues.append("起床时间明显偏晚")
    elif wake_status == "watch" and bedtime_status == "ok":
        issues.append("起床时间偏晚")

    if not issues:
        return "时长与入睡时间都落在推荐区间，作息规律，继续保持。"

    body = "、".join(issues)
    if bedtime_status != "ok" and hours >= 7:
        return body + "；时长虽然够，但作息整体后移，长期与代谢、情绪问题相关。"
    if hours < 7 and bedtime_status != "ok":
        return body + "；既睡不够也睡得晚，建议先把入睡时间提前，再补足时长。"
    return body + "。"


def evaluate_sleep(start: time, end: time) -> List[Metric]:
    """睡眠拆成两条：时长看「够不够」，节律看「几点睡、几点起」。"""
    hours = sleep_hours(start, end)
    if hours < 6:
        duration_status = "low"
    elif hours < 7:
        duration_status = "watch"
    elif hours <= 9:
        duration_status = "ok"
    elif hours <= 10:
        duration_status = "watch"
    else:
        duration_status = "high"

    bedtime_status, _ = _judge_bedtime(start)
    wake_status, _ = _judge_wake(end)
    if "high" in (bedtime_status, wake_status):
        rhythm_status = "high"
    elif "watch" in (bedtime_status, wake_status):
        rhythm_status = "watch"
    else:
        rhythm_status = "ok"

    window = f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"
    duration = Metric(
        "sleep", "睡眠时长", f"{hours:.1f} 小时（{window}）", "7–9 小时", duration_status
    )
    duration.status_text = {"low": "偏短", "high": "偏长"}.get(duration_status, "")

    rhythm = Metric(
        "rhythm",
        "作息节律",
        f"{start.strftime('%H:%M')} 入睡 · {end.strftime('%H:%M')} 起床",
        "23:00 前入睡，07:00 前后起床",
        rhythm_status,
    )
    rhythm.status_text = {"ok": "规律", "watch": "偏晚", "high": "熬夜"}[rhythm_status]
    rhythm.note = _sleep_note(hours, bedtime_status, wake_status)

    return [duration, rhythm]


def evaluate_heart_rate(hr: int) -> Metric:
    if hr < 45:
        status = "urgent"
    elif hr < 60:
        status = "low"
    elif hr <= 100:
        status = "ok"
    elif hr <= 130:
        status = "high"
    else:
        status = "urgent"
    return Metric("hr", "心率", f"{hr} 次/分", "60–100 次/分", status)


def sleep_hours(start: time, end: time) -> float:
    """跨零点计算睡眠时长（入睡 23:30 → 起床 07:00 = 7.5 小时）。"""
    start_min = start.hour * 60 + start.minute
    end_min = end.hour * 60 + end.minute
    if end_min <= start_min:
        end_min += 24 * 60
    return round((end_min - start_min) / 60, 1)


# ===================== 汇总 =====================

def evaluate(data: Dict) -> SelfCheckResult:
    """按已填字段逐项判定；未填的项直接跳过，不会产生空行。"""
    metrics: List[Metric] = []

    height = data.get("height_cm")
    weight = data.get("weight_kg")
    if height and weight:
        metrics.append(evaluate_bmi(float(height), float(weight)))

    if data.get("temp_c") is not None:
        metrics.append(evaluate_temperature(float(data["temp_c"])))

    sbp, dbp = data.get("sbp"), data.get("dbp")
    if sbp and dbp:
        metrics.append(evaluate_blood_pressure(int(sbp), int(dbp)))

    if data.get("glucose") is not None:
        metrics.append(evaluate_glucose(float(data["glucose"]), data.get("glucose_type", "空腹")))

    if data.get("sleep_start") and data.get("sleep_end"):
        metrics.extend(evaluate_sleep(data["sleep_start"], data["sleep_end"]))

    if data.get("heart_rate"):
        metrics.append(evaluate_heart_rate(int(data["heart_rate"])))

    level = "ok"
    for metric in metrics:
        if _SEVERITY[metric.status] >= 3:
            level = "urgent"
        elif _SEVERITY[metric.status] >= 2 and level != "urgent":
            level = "alert"
        elif _SEVERITY[metric.status] >= 1 and level == "ok":
            level = "watch"

    urgent = [m.label for m in metrics if m.status == "urgent"]
    return SelfCheckResult(metrics=metrics, level=level, urgent_items=urgent)


# ===================== 文案 =====================

def build_prompt(result: SelfCheckResult, elder: bool = False) -> str:
    rows = "\n".join(
        f"- {m.label}：{m.value}（参考 {m.reference}，判定：{m.status_label}）" for m in result.metrics
    )

    if elder:
        # 老人模式：结论压到两三句，建议只留三条，术语要换成大白话
        return f"""你是全科医生助手。下面是用户本次自查的测量数据，请用老人听得懂的话写一段简短解读。

{rows}

要求：
1. 先写「结论：」，60–90 字，最多 3 句。第一句先说整体要不要紧，
   再点出最需要注意的 1–2 项，用"可能与……有关"这类措辞，不要下诊断。
2. 再写「建议：」，只写 3 条，每条单独一行，动词开头，不超过 15 字，要具体可做。
3. 不用专业名词，非用不可就在括号里换成大白话，例如「血压高（血管里的压力太大）」。
4. 不出现药名和剂量，不出现"确诊""你得了"这类判断。
5. 若存在需要尽快就医的指标，第一句就要说"要尽快去医院"。"""

    return f"""你是全科医生助手。下面是用户本次自查的测量数据，请据此写一段初步解读。

{rows}

要求：
1. 先写「结论：」，120–180 字。第一句总述整体情况，再点出最值得注意的 1–3 项，
   说明可能相关的原因（用"可能与……有关"这类措辞），不要下诊断。
2. 再写「建议：」，3–5 条，每条单独一行，动词开头，不超过 30 字，要具体可执行。
3. 不出现药名和剂量，不出现"确诊""你得了"这类判断。
4. 若存在需要尽快就医的指标，第一句就要提示就医。"""


def _conclusion_from_rules(result: SelfCheckResult) -> str:
    total = len(result.metrics)
    abnormal = result.abnormal

    if result.level == "urgent":
        head = f"本次自查中有一项或多项指标明显超出常见范围（{'、'.join(result.urgent_items)}），建议今天内前往医院就诊，不要仅在家观察。"
    elif not abnormal:
        head = f"本次自查共 {total} 项，各项均在常见参考范围内，整体情况平稳，继续保持现有作息即可。"
    elif result.level == "alert":
        parts = "、".join(f"{m.label}·{m.status_label}" for m in abnormal[:3])
        head = f"本次自查共 {total} 项，其中 {parts}，已经超出常见参考范围，建议近期安排一次门诊复查。"
    else:
        parts = "、".join(f"{m.label}·{m.status_label}" for m in abnormal[:3])
        head = f"本次自查共 {total} 项，其中 {parts}，属于需要留意的范围，暂时不必紧张，但要持续记录观察。"

    if result.level != "urgent":
        head += "单次测量会受情绪、饮食、睡眠和测量方式影响，建议在相似条件下复测一次再判断趋势。"
    return head


_SUGGESTION_BY_KEY = {
    "bmi": "控制总热量摄入，每周累计 150 分钟中等强度运动",
    "bp": "减少食盐至每日 5 克以内，每天固定时间测量并记录",
    "glucose": "减少含糖饮料与精制主食，两周后复查空腹血糖",
    "temp": "多饮水、保证休息；体温超过 38.5 ℃ 或持续三天请就医",
    "sleep": "先固定起床时间，睡前一小时离开屏幕",
    "rhythm": "把入睡时间每三天提前 15 分钟，先固定起床时间",
    "hr": "避免咖啡因与熬夜；静息心率持续异常请做心电图",
}

_FALLBACK_GENERIC = [
    "把这次数据记下来，两周后在相同条件下复测一次做对比",
    "保持规律作息与三餐时间，避免连续熬夜",
]


def fallback_analysis(result: SelfCheckResult) -> Tuple[str, List[str]]:
    """模型不可用时的兜底：结论与建议都由规则生成。"""
    conclusion = _conclusion_from_rules(result)

    suggestions: List[str] = []
    for metric in result.abnormal:
        tip = _SUGGESTION_BY_KEY.get(metric.key)
        if tip and tip not in suggestions:
            suggestions.append(tip)
    for tip in _FALLBACK_GENERIC:
        if len(suggestions) >= 3:
            break
        if tip not in suggestions:
            suggestions.append(tip)

    return conclusion, suggestions[:5]


def parse_analysis(text: str) -> Tuple[str, List[str]]:
    """把模型输出拆成 (结论, 建议列表)。缺少「建议：」时整段当结论。"""
    text = (text or "").strip()
    if not text:
        return "", []

    conclusion, suggestions = text, []
    if "结论" in text:
        after = text.split("结论", 1)[1].lstrip("：: ")
        if "建议" in after:
            conclusion, rest = after.split("建议", 1)
            rest = rest.lstrip("：: ")
            for line in rest.splitlines():
                item = line.strip().lstrip("-·•0123456789.、) ").strip()
                if len(item) >= 4:
                    suggestions.append(item)
        else:
            conclusion = after

    conclusion = conclusion.strip().strip("-—").strip()
    return conclusion, suggestions[:5]


# ===================== 逐项解读（PDF 用） =====================

_NOTES = {
    "bmi": {
        "ok": "体重与身高比例合适，处于 18.5–23.9 的常见区间。",
        "low": "低于 18.5，属偏瘦。可能与进食不足、消耗过多或代谢偏快有关。",
        "watch": "处于超重区间（24–27.9）。体重上升会同时抬高血压与血糖的风险。",
        "high": "达到肥胖区间（≥28）。建议一并关注腰围、血压与血糖。",
    },
    "temp": {
        "ok": "36.0–37.2 ℃ 为常见腋温区间，本次处于正常范围。",
        "low": "低于 36.0 ℃。常见于测量方式不当（未夹紧、刚吹过风）或代谢偏低。",
        "watch": "略微超出常见区间，建议半小时后复测一次。",
        "high": "已属发热。感染最常见，也可能与脱水、环境温度或剧烈运动有关。",
        "urgent": "高热。请立即就医，途中不要自行服用退热药掩盖体温。",
    },
    "bp": {
        "ok": "处于 90–119 / 60–79 mmHg 的理想区间。",
        "watch": "处于正常高值。单次偏高常见于紧张、刚活动完或咖啡因，建议静坐 5 分钟后复测。",
        "high": "达到高血压的常见判定线（≥140/90 mmHg）。需在不同日期复测确认，不能凭单次定论。",
        "urgent": "显著升高（≥180/110 mmHg）。若伴头痛、胸闷、视物模糊，请立即就医。",
        "low": "低于 90/60 mmHg。若伴头晕、乏力、起身发黑，需排查贫血或脱水。",
    },
    "glucose": {
        "ok": "处于该测量时点的常见区间。",
        "watch": "轻度偏高。可能处于糖尿病前期的区间，建议复查确认。",
        "high": "明显偏高，建议尽快到内分泌科复查空腹血糖与糖化血红蛋白。",
        "low": "偏低。若伴心慌、出汗、手抖，应立即补充含糖食物并复测。",
        "urgent": "严重偏离常见区间，请立即就医。",
    },
    "sleep": {
        "ok": "睡眠时长处于 7–9 小时的推荐区间。",
        "watch": "略偏离推荐区间。时长正常不代表质量好，可留意夜间醒来次数。",
        "low": "低于 6 小时。长期睡眠不足与血压、血糖、情绪问题都相关。",
        "high": "超过 10 小时。过长睡眠本身可能是疲劳、情绪或睡眠呼吸问题的信号。",
    },
    "hr": {
        "ok": "静息心率处于 60–100 次/分的常见区间。",
        "low": "偏慢。运动员常见，但若伴头晕、乏力需排查传导问题。",
        "high": "偏快。常见于紧张、发热、贫血或甲状腺功能亢进。",
        "urgent": "明显异常，建议尽快做心电图检查。",
    },
    "rhythm": {
        "ok": "入睡与起床时间都落在推荐区间。",
        "watch": "入睡或起床时间偏离推荐区间，作息有后移趋势。",
        "high": "入睡时间进入熬夜区间，属于典型的晚睡型作息。",
    },
}


def note_for(metric: Metric) -> str:
    """给单个指标一句解释，用于 PDF 的逐项解读。指标自带动态 note 时优先用它。"""
    if metric.note:
        return metric.note
    table = _NOTES.get(metric.key, {})
    return table.get(metric.status, "本次测量已记录，建议在相同条件下复测一次做对比。")


_DEPARTMENT = {
    "bp": "心内科",
    "glucose": "内分泌科",
    "temp": "发热门诊",
    "hr": "心内科",
}


def department_advice(result: SelfCheckResult) -> str:
    """存在需要尽快就医的指标时，给出就诊科室建议。"""
    if not result.urgent_items:
        return ""
    by_key = {m.key: m for m in result.metrics}
    departments = []
    for metric in result.metrics:
        if metric.status == "urgent":
            dept = metric.dept or _DEPARTMENT.get(metric.key)
            if dept and dept not in departments:
                departments.append(dept)
    if not departments:
        return "建议前往急诊内科或全科门诊。"
    return "建议就诊科室：" + "、".join(departments) + "；情况紧急请直接前往急诊。"


def followup_plan(result: SelfCheckResult) -> List[Tuple[str, str]]:
    """复测计划：按异常项给出「多久之后再测什么」。"""
    by_key = {m.key: m for m in result.metrics}
    plan: List[Tuple[str, str]] = []

    if by_key.get("bp") and by_key["bp"].status != "ok":
        plan.append(("血压", "连续 7 天，每天早晚各测一次并记录，复诊时带上记录。"))
    if by_key.get("glucose") and by_key["glucose"].status != "ok":
        plan.append(("血糖", "2 周内复查空腹血糖；条件允许时加查糖化血红蛋白。"))
    if by_key.get("bmi") and by_key["bmi"].status in ("watch", "high", "low"):
        plan.append(("体重", "每周固定时间称重一次，看趋势，不要盯单次数值。"))
    if by_key.get("sleep") and by_key["sleep"].status != "ok":
        plan.append(("睡眠", "记录两周睡眠日记：入睡时间、夜间醒来次数、起床时间。"))
    if by_key.get("rhythm") and by_key["rhythm"].status != "ok":
        plan.append(("作息", "两周内先固定起床时间，再逐步把入睡时间前移，每天记录。"))
    if by_key.get("temp") and by_key["temp"].status == "urgent":
        plan.append(("体温", "立即就诊，途中不要自行服用退热药。"))
    if by_key.get("hr") and by_key["hr"].status != "ok":
        plan.append(("心率", "静息状态下复测；持续异常建议做一次心电图。"))

    if not plan:
        plan.append(("保持", "维持现有作息，每 3–6 个月复查一次基础指标即可。"))

    return plan
