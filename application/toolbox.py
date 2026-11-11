#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具箱：一组本地公式与查表计算，不调用模型、不联网。

设计原则：
- 结果可复算。公式、阈值都写成常量，改口径只改这一处；
- 只给区间和参考，不给个体处方。用药换算只做算术，不判断该不该用；
- 每个工具统一返回 ToolResult，前端只负责画表，不参与计算。

本模块为纯逻辑，不依赖 Streamlit，可单独测试。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

# ===================== 数据结构 =====================


@dataclass
class Row:
    """结果表的一行：项目 / 数值 / 说明。"""

    label: str
    value: str
    note: str = ""
    status: str = ""  # ok / watch / high / low，仅用于给说明文字上色


@dataclass
class ToolResult:
    headline: str
    level: str = "ok"  # ok / watch / alert / info
    rows: List[Row] = field(default_factory=list)
    advice: List[str] = field(default_factory=list)
    disclaimer: str = ""


@dataclass(frozen=True)
class Tool:
    key: str
    num: str
    name: str
    summary: str
    desc: str


# 顺序即工具页的展示顺序，和导航里的 01/02/03 编号语言保持一致
TOOLS: List[Tool] = [
    Tool(
        "bmi",
        "01",
        "BMI / 标准体重",
        "身高体重 → BMI、理想体重区间",
        "按中国成人 BMI 区间判断体重位置，并换算出对应的理想体重范围。",
    ),
    Tool(
        "bmr",
        "02",
        "基础代谢与每日热量",
        "年龄性别身高体重活动量 → BMR / TDEE",
        "用 Mifflin-St Jeor 公式估算静息消耗，再乘活动系数得到每日总消耗。",
    ),
    Tool(
        "pregnancy",
        "03",
        "孕周与预产期",
        "末次月经 → 孕周、预产期、产检节点",
        "按末次月经推算孕周与预产期，并列出接下来该做的产检。",
    ),
    Tool(
        "ovulation",
        "04",
        "排卵期估算",
        "周期长度 → 易孕期区间",
        "用日历法估算排卵日与易孕期；填了末次月经就一并换算成日期。",
    ),
    Tool(
        "growth",
        "05",
        "儿童身高体重百分位",
        "年龄性别身高体重 → P3/P50/P97 位置",
        "把孩子的身高体重放回同年龄同性别的参考曲线里，看大致百分位。",
    ),
    Tool(
        "alcohol",
        "06",
        "饮酒 / 咖啡因摄入",
        "酒量、杯数 → 是否超推荐量",
        "把酒和咖啡换算成酒精克数与咖啡因毫克数，对照每日推荐上限。",
    ),
    Tool(
        "bp",
        "07",
        "血压分级",
        "收缩压 / 舒张压 → 分级与建议",
        "按中国高血压指南的诊室血压界值给出分级，并说明下一步怎么做。",
    ),
    Tool(
        "dose",
        "08",
        "用药剂量换算",
        "体重给药 → 说明书常见换算",
        "只按说明书标注的 mg/kg 做算术换算，不判断适应证，也不构成处方。",
    ),
]

TOOL_MAP: Dict[str, Tool] = {tool.key: tool for tool in TOOLS}


# ===================== 通用小工具 =====================


def _date_str(value: date) -> str:
    return f"{value.year}-{value.month:02d}-{value.day:02d}"


def _range_str(start: date, end: date) -> str:
    if start == end:
        return _date_str(start)
    return f"{_date_str(start)} – {_date_str(end)}"


# ===================== 01 BMI / 标准体重 =====================

BMI_LOW = 18.5
BMI_HIGH = 23.9
BMI_OBESE = 28.0


def bmi(height_cm: float, weight_kg: float) -> ToolResult:
    """中国成人 BMI 分级（WS/T 428-2013）与理想体重区间。"""
    height_m = height_cm / 100
    value = weight_kg / (height_m**2)

    if value < BMI_LOW:
        level, label, status = "watch", "偏瘦", "low"
    elif value <= BMI_HIGH:
        level, label, status = "ok", "正常", "ok"
    elif value < BMI_OBESE:
        level, label, status = "watch", "超重", "watch"
    else:
        level, label, status = "alert", "肥胖", "high"

    ideal_low = BMI_LOW * height_m**2
    ideal_high = BMI_HIGH * height_m**2
    std_weight = height_cm - 105

    if weight_kg < ideal_low:
        gap = f"比区间下限低 {ideal_low - weight_kg:.1f} kg"
        gap_status = "low"
    elif weight_kg > ideal_high:
        gap = f"比区间上限高 {weight_kg - ideal_high:.1f} kg"
        gap_status = "high" if value >= BMI_OBESE else "watch"
    else:
        gap = "已在理想区间内"
        gap_status = "ok"

    rows = [
        Row("体质指数 BMI", f"{value:.1f}", "中国成人 18.5–23.9 为正常", status),
        Row("体重状态", label, "按 BMI 区间判定", status),
        Row("理想体重区间", f"{ideal_low:.1f}–{ideal_high:.1f} kg", "对应 BMI 18.5–23.9"),
        Row("常用标准体重", f"{std_weight:.1f} kg", "身高(cm) − 105，仅作粗估"),
        Row("与理想区间", gap, "", gap_status),
    ]

    advice: List[str] = []
    if value < BMI_LOW:
        advice.append(
            "体重低于正常区间。若近期没有刻意减重却持续下降，建议查甲状腺功能、血糖与消化系统。"
        )
        advice.append("增重以每周 0.25–0.5 kg 为宜，优先增加优质蛋白与主食，而不是只吃零食。")
    elif value <= BMI_HIGH:
        advice.append("体重在正常区间。每周至少 150 分钟中等强度运动，久坐每小时起身活动 3–5 分钟。")
        advice.append("体重稳定比单次数字更重要，建议固定每周同一时间、同样条件称重。")
    elif value < BMI_OBESE:
        advice.append("处于超重区间。减重 5%–10% 就能明显改善血压、血糖与血脂，不必追求快速下降。")
        advice.append("先减少含糖饮料与精制主食，再逐步增加有氧运动，比极端节食更容易维持。")
    else:
        advice.append("达到肥胖区间。建议在医生或营养师指导下制定减重计划，不要自行使用减肥药。")
        advice.append("同时筛查血压、血糖、血脂、尿酸与脂肪肝，这些指标常与体重一起变化。")

    return ToolResult(
        headline=f"BMI {value:.1f} · {label}",
        level=level,
        rows=rows,
        advice=advice,
        disclaimer=(
            "BMI 不适用于孕妇、18 岁以下人群、运动员与明显水肿者；"
            "腰围（男性 ≥90 cm、女性 ≥85 cm）要一并看。"
        ),
    )


# ===================== 02 基础代谢与每日热量 =====================

# 名称 → 活动系数（Mifflin-St Jeor 常用分级）
ACTIVITY_LEVELS: List[Tuple[str, float]] = [
    ("久坐（几乎不运动）", 1.2),
    ("轻度活动（每周 1–3 次）", 1.375),
    ("中度活动（每周 3–5 次）", 1.55),
    ("高度活动（每周 6–7 次）", 1.725),
    ("极高活动（体力工作 / 一天两练）", 1.9),
]
ACTIVITY_FACTORS: Dict[str, float] = dict(ACTIVITY_LEVELS)


def bmr_tdee(
    age: int,
    sex: str,
    height_cm: float,
    weight_kg: float,
    activity: str,
) -> ToolResult:
    """Mifflin-St Jeor 估算 BMR，乘活动系数得到 TDEE。"""
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    value = base + (5 if sex == "男" else -161)
    factor = ACTIVITY_FACTORS.get(activity, 1.2)
    tdee = value * factor
    bmi_value = weight_kg / ((height_cm / 100) ** 2)

    rows = [
        Row("基础代谢 BMR", f"{value:.0f} kcal/天", "静息状态下维持生命的最低消耗"),
        Row("活动系数", f"× {factor:.3f}", activity),
        Row("每日总消耗 TDEE", f"{tdee:.0f} kcal/天", "BMR × 活动系数"),
        Row(
            "维持体重",
            f"约 {tdee - 100:.0f}–{tdee + 100:.0f} kcal/天",
            "日常波动约 ±100 kcal",
        ),
    ]

    advice: List[str] = []
    if bmi_value < BMI_LOW:
        rows.append(Row("增重参考", f"约 {tdee + 300:.0f}–{tdee + 500:.0f} kcal/天", "每周增重 0.25–0.5 kg"))
        advice.append("BMI 偏低，先以增重为目标：在维持热量上每天加 300–500 kcal。")
    else:
        rows.append(
            Row(
                "温和减重",
                f"约 {tdee - 500:.0f}–{tdee - 300:.0f} kcal/天",
                "每周减 0.25–0.5 kg，不低于 BMR",
            )
        )
        advice.append("减重缺口控制在每天 300–500 kcal；长期低于 BMR 容易掉肌肉、之后反弹。")

    advice.append("公式对孕期、哺乳期、甲状腺疾病与健美人群误差较大，只能当起点，用 2–4 周体重变化校准。")
    advice.append("蛋白质可按每公斤体重 1.0–1.6 g 安排；训练量越大，取上限。")

    return ToolResult(
        headline=f"TDEE 约 {tdee:.0f} kcal/天",
        level="info",
        rows=rows,
        advice=advice,
        disclaimer="热量只是估算值，食物标签、烹饪方式与个体代谢都会带来 10% 左右的偏差。",
    )


# ===================== 03 孕周与预产期 =====================

# 常规产检节点：(名称, 起始孕周, 结束孕周)
CHECKUP_NODES: List[Tuple[str, int, int]] = [
    ("建档 · 首次产检", 6, 13),
    ("NT 超声", 11, 13),
    ("唐筛 / 无创 DNA", 15, 20),
    ("大排畸超声", 20, 24),
    ("糖耐量试验", 24, 28),
    ("孕晚期产检（每 2 周）", 28, 36),
    ("足月后产检（每周）", 36, 40),
    ("预产期评估", 40, 41),
]


def pregnancy(
    lmp: date,
    cycle_days: int = 28,
    today: Optional[date] = None,
) -> ToolResult:
    """按末次月经推算孕周、预产期与产检节点（Naegele 法）。"""
    today = today or date.today()
    if lmp > today:
        return ToolResult(
            headline="日期有误",
            level="watch",
            rows=[Row("末次月经", _date_str(lmp), "不能晚于今天", "high")],
            advice=["请检查末次月经第一天是否填错。"],
        )

    # 周期不是 28 天时，预产期按差值校正（排卵日相应前后移动）
    edd = lmp + timedelta(days=280 + (cycle_days - 28))
    ga_days = (today - lmp).days
    weeks, days = divmod(ga_days, 7)
    remaining = (edd - today).days

    if weeks < 14:
        stage = "孕早期"
    elif weeks < 28:
        stage = "孕中期"
    else:
        stage = "孕晚期"

    # 每个节点在日历上的起止日期（按孕周算，不随周期校正移动）
    def node_dates(start_week: int, end_week: int) -> Tuple[date, date]:
        return (
            lmp + timedelta(days=start_week * 7),
            lmp + timedelta(days=end_week * 7 + 6),
        )

    def node_state(start_week: int, end_week: int) -> str:
        start, end = node_dates(start_week, end_week)
        if today > end:
            return "done"
        if today >= start:
            return "now"
        return "next"

    upcoming: List[Tuple[str, int, int, str]] = []
    for name, start_week, end_week in CHECKUP_NODES:
        state = node_state(start_week, end_week)
        if state != "done":
            upcoming.append((name, start_week, end_week, state))

    # 同时有多个节点在进行时（如 40 周），优先展示最靠后的那个
    chosen_index = 0
    ongoing = [i for i, node in enumerate(upcoming) if node[3] == "now"]
    if ongoing:
        chosen_index = ongoing[-1]

    if remaining >= 0:
        remain_text = f"还有 {remaining} 天"
    else:
        remain_text = f"已过预产期 {abs(remaining)} 天"

    rows = [
        Row("孕周", f"{weeks} 周 + {days} 天", f"从末次月经第一天算起，共 {ga_days} 天"),
        Row("预产期", _date_str(edd), f"按月经周期 {cycle_days} 天校正" if cycle_days != 28 else "按 28 天周期推算"),
        Row("距离预产期", remain_text, "预产期只是估算，37–42 周分娩都属足月"),
        Row("当前阶段", stage, "孕早期 <14 周 · 孕中期 14–27 周 · 孕晚期 ≥28 周"),
    ]

    if upcoming:
        name, start_week, end_week, state = upcoming[chosen_index]
        start, end = node_dates(start_week, end_week)
        rows.append(
            Row(
                "下一次产检",
                f"{name}（{'进行中' if state == 'now' else '待做'}）",
                f"{_range_str(start, end)} · 孕 {start_week}–{end_week} 周",
            )
        )
    else:
        rows.append(Row("下一次产检", "按医生安排", "已超过常规产检表范围", "watch"))

    advice: List[str] = []
    for name, start_week, end_week, state in upcoming[chosen_index + 1: chosen_index + 4]:
        start, end = node_dates(start_week, end_week)
        advice.append(
            f"{name}：{_range_str(start, end)}（孕 {start_week}–{end_week} 周）"
        )
    advice.append("出现阴道出血、持续腹痛、破水、胎动明显减少、剧烈头痛或视物模糊，立即就医。")
    advice.append("月经不规律或记不清末次月经时，以孕早期超声核对的孕周为准。")
    advice.append("28 周后开始规律数胎动；具体产检频率以建档医院的安排为准。")

    if ga_days > 294:
        advice.insert(0, "按末次月经已超过 42 周，请尽快联系产科核对孕周，不要继续等待。")

    return ToolResult(
        headline=f"{weeks} 周 + {days} 天",
        level="watch" if ga_days > 294 else "info",
        rows=rows,
        advice=advice,
        disclaimer="本工具只做日期推算，不替代产科检查与医生判断。",
    )


# ===================== 04 排卵期估算 =====================

def ovulation_cycle(
    cycle_days: int,
    lmp: Optional[date] = None,
    today: Optional[date] = None,
) -> ToolResult:
    """日历法：排卵日约在下次月经前 14 天，易孕期为排卵前 5 天至后 1 天。"""
    today = today or date.today()
    cycle = max(21, min(45, int(cycle_days)))
    ovulation_day = cycle - 14
    fertile_start = max(1, ovulation_day - 5)
    fertile_end = ovulation_day + 1

    rows = [
        Row("周期长度", f"{cycle} 天", "从月经第一天到下一位月经前一天"),
        Row(
            "预计排卵日",
            f"周期第 {ovulation_day} 天",
            "下次月经前约 14 天",
        ),
        Row(
            "易孕期",
            f"周期第 {fertile_start}–{fertile_end} 天",
            "排卵前 5 天到排卵后 1 天",
        ),
    ]

    if lmp is not None:
        if lmp > today:
            return ToolResult(
                headline="日期有误",
                level="watch",
                rows=[Row("末次月经", _date_str(lmp), "不能晚于今天", "high")],
                advice=["请检查末次月经第一天是否填错。"],
            )
        ovu_date = lmp + timedelta(days=ovulation_day - 1)
        fertile_start_date = lmp + timedelta(days=fertile_start - 1)
        fertile_end_date = lmp + timedelta(days=fertile_end - 1)
        next_period = lmp + timedelta(days=cycle)
        next_ovu = ovu_date + timedelta(days=cycle)

        rows[1] = Row("预计排卵日", f"{_date_str(ovu_date)}（第 {ovulation_day} 天）", "下次月经前约 14 天")
        rows[2] = Row(
            "易孕期",
            _range_str(fertile_start_date, fertile_end_date),
            f"周期第 {fertile_start}–{fertile_end} 天",
        )
        rows.append(Row("下次月经预计", _date_str(next_period), "按当前周期长度推算"))
        rows.append(
            Row(
                "下一次易孕期",
                _range_str(
                    fertile_start_date + timedelta(days=cycle),
                    fertile_end_date + timedelta(days=cycle),
                ),
                "再下一个周期",
            )
        )
        elapsed = (today - lmp).days % cycle
        rows.insert(0, Row("今天是", f"周期第 {elapsed + 1} 天", "按末次月经第一天起算"))

    advice: List[str] = []
    if cycle < 21 or cycle > 35:
        advice.append("周期短于 21 天或长于 35 天，日历法误差很大，建议妇科就诊评估。")
    advice.append("压力、熬夜、旅行和生病都可能让排卵提前或推后，日历法只适合周期规律的人。")
    advice.append("精子在体内可存活约 3–5 天，卵子排出后约 12–24 小时，所以易孕期取排卵前 5 天到后 1 天。")
    advice.append("不建议把日历法作为唯一避孕手段；备孕可以配合排卵试纸或基础体温。")

    if lmp is not None:
        headline = _range_str(
            lmp + timedelta(days=fertile_start - 1),
            lmp + timedelta(days=fertile_end - 1),
        )
    else:
        headline = f"易孕期：周期第 {fertile_start}–{fertile_end} 天"

    return ToolResult(
        headline=headline,
        level="info",
        rows=rows,
        advice=advice,
        disclaimer="排卵期估算不是避孕工具，也不能用来判断是否怀孕。",
    )


# ===================== 05 儿童身高体重百分位 =====================
# 中国 0–18 岁儿童生长参考的年度节点（P3 / P50 / P97），按年龄线性插值。
# 表内数值为公开生长标准表的简化整理，用于粗略定位，不用于诊断。

_HEIGHT_REF: Dict[str, List[Tuple[float, float, float]]] = {
    "男": [
        (46.9, 50.4, 53.9), (71.5, 76.5, 81.5), (82.1, 88.5, 94.9),
        (89.7, 96.8, 103.9), (96.1, 104.1, 112.1), (102.8, 111.3, 119.8),
        (108.6, 117.7, 126.8), (114.6, 124.0, 133.4), (119.9, 130.0, 140.1),
        (124.6, 135.4, 146.2), (128.7, 140.2, 151.7), (132.9, 145.3, 157.7),
        (138.0, 152.4, 166.8), (145.0, 160.0, 175.0), (151.5, 165.9, 180.3),
        (156.7, 169.8, 182.9), (159.9, 171.9, 183.9), (161.7, 172.8, 183.9),
        (162.9, 173.3, 183.7),
    ],
    "女": [
        (46.4, 49.7, 53.0), (70.2, 75.0, 79.8), (80.9, 87.2, 93.5),
        (88.7, 95.6, 102.5), (95.1, 103.1, 111.1), (101.8, 110.2, 118.6),
        (107.6, 116.6, 125.6), (113.3, 122.5, 131.7), (118.9, 128.5, 138.1),
        (124.0, 134.1, 144.2), (129.4, 140.1, 150.8), (135.3, 146.6, 157.9),
        (140.2, 151.2, 162.2), (143.9, 154.5, 165.1), (146.1, 156.4, 166.7),
        (147.3, 157.5, 167.7), (148.0, 158.2, 168.4), (148.5, 158.6, 168.7),
        (148.8, 158.9, 169.0),
    ],
}

_WEIGHT_REF: Dict[str, List[Tuple[float, float, float]]] = {
    "男": [
        (2.6, 3.3, 4.0), (8.3, 10.1, 12.0), (10.4, 12.5, 15.1),
        (12.1, 14.6, 17.5), (13.4, 16.3, 19.7), (14.6, 18.0, 22.0),
        (15.9, 19.8, 24.3), (17.2, 21.5, 26.7), (18.6, 23.5, 29.4),
        (20.2, 25.6, 32.5), (22.1, 28.1, 36.2), (24.2, 30.9, 40.4),
        (26.9, 34.2, 45.5), (30.0, 38.4, 52.0), (33.6, 43.4, 59.3),
        (37.6, 48.5, 66.0), (41.5, 53.0, 71.6), (44.9, 56.8, 76.0),
        (47.7, 60.0, 79.5),
    ],
    "女": [
        (2.5, 3.2, 3.9), (7.7, 9.4, 11.4), (9.8, 11.8, 14.3),
        (11.3, 13.7, 16.6), (12.6, 15.3, 18.6), (13.7, 16.9, 20.8),
        (14.9, 18.6, 23.2), (16.2, 20.3, 25.3), (17.6, 22.2, 28.0),
        (19.2, 24.4, 31.3), (21.1, 27.2, 35.4), (23.3, 30.4, 40.2),
        (25.8, 34.0, 45.5), (28.4, 37.5, 50.6), (30.7, 40.4, 54.5),
        (32.4, 42.6, 57.5), (33.7, 44.2, 60.0), (34.5, 45.2, 61.6),
        (35.0, 45.8, 62.6),
    ],
}

# P3 / P97 对应的标准正态分位数
_Z97 = 1.881


def _interpolate(table: Sequence[Tuple[float, float, float]], age: float) -> Tuple[float, float, float]:
    age = max(0.0, min(18.0, age))
    low = int(math.floor(age))
    high = min(18, low + 1)
    weight = age - low
    a, b = table[low], table[high]
    return tuple(a[i] + (b[i] - a[i]) * weight for i in range(3))  # type: ignore[return-value]


def _percentile(value: float, p3: float, p50: float, p97: float) -> float:
    """用 P3/P50/P97 三个点拟合上下半段标准差，再换算成百分位。"""
    if value <= p50:
        sd = (p50 - p3) / _Z97
    else:
        sd = (p97 - p50) / _Z97
    if sd <= 0:
        return 50.0
    z = (value - p50) / sd
    pct = 0.5 * (1 + math.erf(z / math.sqrt(2))) * 100
    return max(0.1, min(99.9, pct))


def _pct_band(pct: float) -> Tuple[str, str]:
    if pct < 3:
        return "低于 P3", "low"
    if pct < 25:
        return "P3–P25", ""
    if pct <= 75:
        return "P25–P75", "ok"
    if pct <= 97:
        return "P75–P97", ""
    return "高于 P97", "high"


def _pct_text(pct: float) -> str:
    """百分位展示：两端收敛到 P1 / P99，避免出现「P100」这种不存在的说法。"""
    return f"P{max(1, min(99, int(round(pct))))}"


def child_growth(
    age_months: int,
    sex: str,
    height_cm: float,
    weight_kg: float,
) -> ToolResult:
    """儿童身高、体重在同年龄同性别参考人群中的大致百分位。"""
    age = min(18.0, max(0.0, age_months / 12))
    table_sex = "男" if sex == "男" else "女"
    h3, h50, h97 = _interpolate(_HEIGHT_REF[table_sex], age)
    w3, w50, w97 = _interpolate(_WEIGHT_REF[table_sex], age)

    hp = _percentile(height_cm, h3, h50, h97)
    wp = _percentile(weight_kg, w3, w50, w97)
    hband, hstatus = _pct_band(hp)
    wband, wstatus = _pct_band(wp)

    outside = hp < 3 or hp > 97 or wp < 3 or wp > 97
    level = "watch" if outside else "ok"

    years = age_months // 12
    months = age_months % 12
    age_text = f"{years} 岁" + (f" {months} 个月" if months else "")
    age_note = "按周岁与月龄计" if age_months <= 216 else "超过 18 岁，按 18 岁参考值计"

    rows = [
        Row("年龄", age_text, age_note),
        Row("身高", f"{height_cm:.1f} cm", f"约 {_pct_text(hp)}（{hband}）", hstatus),
        Row("同年龄参考身高", f"{h3:.1f} / {h50:.1f} / {h97:.1f} cm", "P3 / P50 / P97"),
        Row("体重", f"{weight_kg:.1f} kg", f"约 {_pct_text(wp)}（{wband}）", wstatus),
        Row("同年龄参考体重", f"{w3:.1f} / {w50:.1f} / {w97:.1f} kg", "P3 / P50 / P97"),
    ]

    advice: List[str] = []
    if outside:
        advice.append("至少一项落在 P3–P97 之外，建议带上连续的生长记录到儿科或儿童保健科评估。")
    else:
        advice.append("身高体重都在常见范围内，继续按生长曲线随访即可，不必因为一次测量焦虑。")
    advice.append("百分位看的是在同年龄同性别孩子中的位置，P50 是中间值，不是「达标线」。")
    advice.append("一次测量的意义有限，连续曲线的走向更重要；百分位持续下滑并跨过两条主百分位线，建议就诊。")
    advice.append("睡眠、营养、运动和遗传共同决定身高；青春期前后的生长速度尤其值得记录。")

    return ToolResult(
        headline=f"身高 {_pct_text(hp)} · 体重 {_pct_text(wp)}",
        level=level,
        rows=rows,
        advice=advice,
        disclaimer=(
            "参考值按中国儿童生长标准的年度节点线性插值整理，早产儿需按矫正月龄评估；"
            "本结果不能用于诊断生长迟缓或性早熟。"
        ),
    )


# ===================== 06 饮酒 / 咖啡因 =====================

ALCOHOL_TYPES: List[Tuple[str, float]] = [
    ("啤酒", 4.5),
    ("红酒 / 葡萄酒", 12.0),
    ("黄酒", 15.0),
    ("白酒（52 度）", 52.0),
    ("自填度数", 0.0),
]

# 《中国居民膳食指南（2022）》：成年人如饮酒，一天酒精量不超过 15 g
ALCOHOL_DAILY_LIMIT = 15.0
ETHANOL_DENSITY = 0.789  # g/ml


def alcohol_intake(
    kind: str,
    cups: float,
    volume_ml: float,
    abv: Optional[float] = None,
    pregnant: bool = False,
) -> ToolResult:
    """按酒类、杯数与每杯容量换算酒精克数，对照每日推荐上限。"""
    if kind == "自填度数":
        degree = max(0.0, min(75.0, float(abv or 0.0)))
    else:
        degree = dict(ALCOHOL_TYPES).get(kind, 0.0)

    total_ml = max(0.0, cups) * max(0.0, volume_ml)
    grams = total_ml * (degree / 100) * ETHANOL_DENSITY
    limit = 0.0 if pregnant else ALCOHOL_DAILY_LIMIT
    standard_cups = grams / 10
    hours = grams / 10  # 肝脏大约每小时代谢 10 g 酒精

    if pregnant:
        level = "alert" if grams > 0 else "ok"
        headline = "孕期应完全禁酒" if grams > 0 else "孕期未饮酒"
    elif grams > limit:
        level = "alert"
        headline = f"酒精 {grams:.1f} g · 超出推荐量"
    else:
        level = "ok"
        headline = f"酒精 {grams:.1f} g"

    if pregnant:
        over_text = "应保持 0 g"
    elif grams > limit:
        over_text = f"超出 {grams - limit:.1f} g"
    else:
        over_text = f"余量 {limit - grams:.1f} g"

    rows = [
        Row("酒类 / 度数", f"{kind} · {degree:.1f}%", f"共 {cups:g} 杯 × {volume_ml:g} ml"),
        Row("总饮用量", f"{total_ml:.0f} ml", ""),
        Row("摄入酒精", f"{grams:.1f} g", f"约 {standard_cups:.1f} 个标准杯（1 杯 = 10 g）"),
        Row("每日推荐上限", f"{limit:.0f} g", "孕期 / 哺乳期按 0 g 计" if pregnant else "成人如饮酒，不超过 15 g"),
        Row("与推荐量相比", over_text, "", "high" if (grams > limit) else "ok"),
        Row("大致代谢时间", f"约 {hours:.0f} 小时", "按肝脏每小时代谢 10 g 估算"),
    ]

    advice: List[str] = []
    if pregnant:
        advice.append("孕期、备孕期和哺乳期应完全避免酒精，不存在安全的饮酒量。")
    else:
        advice.append("《中国居民膳食指南（2022）》建议：成年人如饮酒，一天酒精量不超过 15 g。")
    advice.append("空腹饮酒吸收更快；不要与头孢类、镇静催眠药、降糖药同服。")
    advice.append("所谓「适量饮酒有益健康」的证据不足，越少越好。")
    advice.append(f"15 g 酒精约相当于啤酒 450 ml、葡萄酒 150 ml 或 52 度白酒 30 ml。")

    return ToolResult(
        headline=headline,
        level=level,
        rows=rows,
        advice=advice,
        disclaimer="酒精量不等于醉酒程度，还受饮酒速度、体重、肝功能与是否空腹影响。",
    )


CAFFEINE_ITEMS: List[Tuple[str, float]] = [
    ("美式咖啡（240 ml）", 95.0),
    ("拿铁 / 卡布奇诺（240 ml）", 75.0),
    ("速溶咖啡（150 ml）", 60.0),
    ("浓缩咖啡（30 ml）", 63.0),
    ("红茶（240 ml）", 47.0),
    ("绿茶（240 ml）", 28.0),
    ("可乐（330 ml）", 34.0),
    ("能量饮料（250 ml）", 80.0),
]
CAFFEINE_LIMIT = 400.0
CAFFEINE_LIMIT_PREGNANT = 200.0


def caffeine_intake(item: str, cups: float, pregnant: bool = False) -> ToolResult:
    """按饮品类型与杯数累计咖啡因，对照每日推荐上限。"""
    per_cup = dict(CAFFEINE_ITEMS).get(item, 0.0)
    total = per_cup * max(0.0, cups)
    limit = CAFFEINE_LIMIT_PREGNANT if pregnant else CAFFEINE_LIMIT

    if total > limit:
        level = "alert"
        headline = f"咖啡因 {total:.0f} mg · 超出推荐量"
    elif total > limit * 0.7:
        level = "watch"
        headline = f"咖啡因 {total:.0f} mg · 接近上限"
    else:
        level = "ok"
        headline = f"咖啡因 {total:.0f} mg"

    rows = [
        Row("饮品", item, f"每杯约 {per_cup:.0f} mg"),
        Row("杯数", f"{cups:g} 杯", ""),
        Row("咖啡因总量", f"{total:.0f} mg", f"约相当于 {total / 95:.1f} 杯美式"),
        Row("每日推荐上限", f"{limit:.0f} mg", "孕期 / 哺乳期 200 mg" if pregnant else "健康成人 400 mg"),
        Row(
            "与推荐量相比",
            f"超出 {total - limit:.0f} mg" if total > limit else f"余量 {limit - total:.0f} mg",
            "",
            "high" if total > limit else "ok",
        ),
    ]

    advice: List[str] = []
    if pregnant:
        advice.append("孕期和哺乳期每日咖啡因建议不超过 200 mg，约 2 杯速溶咖啡。")
    else:
        advice.append("健康成人每日咖啡因不超过 400 mg；儿童青少年按每公斤体重 3 mg 估算上限。")
    advice.append("下午 2 点后尽量不再摄入，咖啡因的半衰期约 5 小时，会影响入睡。")
    advice.append("能量饮料不建议与酒同饮，也不建议儿童青少年饮用。")
    advice.append("突然停用可能出现头痛、乏力，可以每几天减半杯逐步减量。")

    return ToolResult(
        headline=headline,
        level=level,
        rows=rows,
        advice=advice,
        disclaimer="不同品牌、豆种与萃取方式的咖啡因差异很大，表中为常见均值。",
    )


# ===================== 07 血压分级 =====================

def blood_pressure(sbp: int, dbp: int) -> ToolResult:
    """中国高血压指南的诊室血压分级（取收缩压/舒张压中较高的级别）。"""
    if sbp >= 180 or dbp >= 110:
        grade, level = "3 级高血压", "alert"
    elif sbp >= 160 or dbp >= 100:
        grade, level = "2 级高血压", "alert"
    elif sbp >= 140 or dbp >= 90:
        grade, level = "1 级高血压", "watch"
    elif sbp >= 120 or dbp >= 80:
        grade, level = "正常高值", "watch"
    elif sbp < 90 or dbp < 60:
        grade, level = "血压偏低", "watch"
    else:
        grade, level = "正常血压", "ok"

    isolated = sbp >= 140 and dbp < 90
    emergency = sbp >= 180 and dbp >= 120

    rows = [
        Row("收缩压（高压）", f"{sbp} mmHg", "诊室血压 ≥140 为升高"),
        Row("舒张压（低压）", f"{dbp} mmHg", "诊室血压 ≥90 为升高"),
        Row("分级", grade, "按收缩压与舒张压中较高的级别判定", level),
        Row("家庭自测参考线", "≥135 / 85 mmHg", "家庭血压达到该值即需就医评估"),
    ]
    if isolated:
        rows.append(Row("补充", "单纯收缩期高血压", "高压 ≥140 且低压 <90", "watch"))

    advice: List[str] = []
    if emergency:
        advice.append(
            "达到高血压急症警戒线：静坐 5 分钟后复测，若仍 ≥180/120，或伴胸痛、呼吸困难、剧烈头痛、"
            "视物模糊、肢体无力，请立即拨打 120。"
        )
        advice.append("不要自行含服降压药或一次加量，以免血压骤降。")
    elif grade == "3 级高血压":
        advice.append("建议尽快到心内科或全科就诊，评估靶器官并制定用药方案。")
    elif grade == "2 级高血压":
        advice.append("建议近期就医，医生会结合其他危险因素决定是否立即用药。")
    elif grade == "1 级高血压":
        advice.append("建议不同日期复测确认；多次达到该水平需就医评估。")
        advice.append("先限盐（每日 <5 g）、减重、规律有氧运动，戒烟限酒。")
    elif grade == "正常高值":
        advice.append("尚未达到高血压标准，但属于需要干预的区间：限盐、减重、规律运动，并做家庭血压监测。")
        advice.append("建议连续 7 天，每天早晚各测一次，记录后带给医生看。")
    elif grade == "血压偏低":
        advice.append("血压偏低。如果伴头晕、乏力、眼前发黑或晕厥，建议就医排查原因。")
        advice.append("起身时动作放慢，保证饮水，避免长时间站立。")
    else:
        advice.append("血压正常。保持低盐饮食、规律运动与体重管理，至少每年测一次血压。")
        advice.append("单次正常不代表长期正常，40 岁后建议每半年测一次。")

    return ToolResult(
        headline=f"{grade} · {sbp}/{dbp} mmHg",
        level=level,
        rows=rows,
        advice=advice,
        disclaimer="单次测量不能确诊高血压，需不同日期复测；本工具不替代医生诊断。",
    )


# ===================== 08 用药剂量换算 =====================

def dose_conversion(
    weight_kg: float,
    mg_per_kg: float,
    times_per_day: int,
    tablet_mg: float = 0.0,
) -> ToolResult:
    """只做说明书 mg/kg 剂量的算术换算，不给适应证或处方建议。"""
    single = weight_kg * mg_per_kg
    daily = single * times_per_day

    rows = [
        Row("体重", f"{weight_kg:.1f} kg", ""),
        Row("说明书剂量", f"{mg_per_kg:.2f} mg/kg/次", f"每日 {times_per_day} 次"),
        Row("单次给药量", f"{single:.1f} mg", "体重 × 每公斤每次剂量"),
        Row("每日总给药量", f"{daily:.1f} mg", "单次剂量 × 每日次数"),
    ]
    if tablet_mg > 0:
        rows.append(
            Row(
                "每次折合片数",
                f"{single / tablet_mg:.2f} 片",
                f"按每片 / 每袋 {tablet_mg:g} mg 计",
            )
        )
        rows.append(
            Row("每日折合片数", f"{daily / tablet_mg:.2f} 片", "分次服用，不是一次服完")
        )

    advice: List[str] = [
        "这里只是把说明书上的 mg/kg 换算成毫克与片数，不判断该不该用、用哪种药。",
        "儿童剂量常按体表面积或年龄分段，不能只按体重线性推算；缓释片、肠溶片不可掰开。",
        "肝肾功能不全、正在服用其他药物、孕期或哺乳期，请先咨询医生或药师。",
        "实际用量以医生处方和药品说明书为准；若换算结果与说明书单次上限冲突，以说明书为准。",
    ]
    if single <= 0:
        advice.insert(0, "剂量为 0，请按说明书填写每公斤每次的毫克数。")

    return ToolResult(
        headline=f"单次 {single:.0f} mg" if single > 0 else "请填写说明书剂量",
        level="info",
        rows=rows,
        advice=advice,
        disclaimer="本页仅展示说明书常见换算方式，不构成处方或用药建议。",
    )
