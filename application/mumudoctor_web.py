#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
木木医生 · 私人健康助手（对话前端）

界面：单栏对话流（居中 760px），设置收在顶栏 popover；桌面端左侧分组导航，
移动端用独立的「☰ 导航」按钮呼出 / 收起导航抽屉（不再塞在头像弹窗里）。

账号（MySQL）：
    未登录只能看到登录 / 注册页；注册要填账号、密码、性别、年龄。
    账号、口令摘要和模式偏好都存在 MySQL 的 users 表里，见 application/auth.py。
    60 岁及以上用户登录时弹窗询问用「老年模式」还是「正常模式」，
    勾选「以后不再提醒」后把这次选择存成默认模式；之后仍可在设置里
    随时切换当前模式，或改下次登录的默认模式。

老人模式（给长辈用）：
    开启后字号更大、入口更少（藏起「工具」与精细参数）、
    每条回答和自查报告都能"朗读"（浏览器自带语音合成，不联网）。
    回答本身也换了写法：加一条 system 提示要求「150 字以内、先给做法、说人话」，
    同时压低 token 上限、少喂参考资料、只推荐 2 条追问。
    登录后按账号的默认模式进入，本次会话可在设置里随时切换。

启动:
    streamlit run application/mumudoctor_web.py

模型后端:
    默认走本地 Ollama；设置 LLM_PROVIDER=openai 后改走云端 OpenAI 兼容接口，
    详见 application/llm.py 顶部的环境变量说明。对话、自查分析、追问推荐、
    RAG 向量化四处共用同一层，切换后端不需要改这里。
"""

import hashlib
import html
import json
import random
import re
import sys
import time as time_module
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

import auth
import checkin
import device_import
import device_bind
import device_view
import document
import environment
import family
import history
import llm
import medicine
import medicine_scan
import poi
import toolbox
import triage
from followups import build_messages, fallback_followups, parse_followups
from rag import get_retriever
from report import render_pdf, render_trend_pdf
from safety import check_safety
from selfcheck import (
    ALARM_LINES_TEXT,
    alarm_line,
    build_prompt,
    department_advice,
    evaluate,
    fallback_analysis,
    parse_analysis,
)
from web_search import web_search

DEFAULT_MODEL = llm.default_model()

# 浏览器定位组件（HTTPS 下才可用）与医院列表组件（带 App 唤起检测）
GEO = components.declare_component("geo", path=str(APP_DIR / "geo"))
NAV = components.declare_component("navlist", path=str(APP_DIR / "nav"))
# 语音朗读组件：用浏览器自带的 speechSynthesis 朗读，不联网、不额外占服务器
SPEAK = components.declare_component("speak", path=str(APP_DIR / "speak"))
# 语音输入组件：圆形麦克风按钮，用浏览器自带的 SpeechRecognition 转文字，
# 识别结果直接写进输入框，不经过服务器（音色 / 识别能力取决于浏览器）
VOICE = components.declare_component("voice", path=str(APP_DIR / "voice"))

# RAG 相似度阈值：低于该值的知识视为不相关，不注入到 prompt（兜底过滤噪声）
RAG_MIN_SIMILARITY = 0.6
# RAG 检索条数上限：本地命中不足该值时用 Tavily 网页搜索补足
RAG_TOP_K = 3

# 环境信息（气温、空气质量）的刷新间隔：同一位置 30 分钟内复用，不重复打接口
ENV_TTL_SECONDS = 1800

FEEDBACK_FILE = PROJECT_ROOT / "data" / "feedback" / "feedback.jsonl"

COMPLIANCE_LINE = "回答由 AI 生成，仅供参考，不能替代执业医师的当面诊断。"

# 文件内容与医疗健康无关时的固定回复：不经过模型，保证不会顺着文件给出建议
FILE_IRRELEVANT_REPLY = (
    "上传的文件并未分析出有效信息。\n\n"
    "这份文件的内容与医疗保健、健康咨询无关，我不会对文件内容作解读，"
    "也不会据此给出任何健康建议。如果你上传的是体检报告、化验单、门诊病历、"
    "处方或健康相关资料，请确认文件完整清晰后重新上传。"
)

# 开屏示例问题池：只放能独立成问、不依赖上下文指代的句子
# （如"体检报告里这几项箭头"这类必须先有报告才能回答的，不要放进来）
QUICK_QUESTION_POOL = [
    "最近总是睡不好，怎么办？",
    "头晕是怎么回事？",
    "工作压力大，怎么放松？",
    "早上起来嗓子干痒，需要吃药吗？",
    "每天坐办公室，腰酸背痛怎么缓解？",
    "血压偏高，饮食上该怎么调整？",
    "最近总是犯困，是缺什么吗？",
    "长期熬夜对身体有哪些影响？",
    "想开始运动，从什么强度入手比较安全？",
    "经常胃胀气，日常怎么调理？",
    "换季总过敏，有什么办法能少遭罪？",
    "体检前需要注意些什么？",
]

# 每次进入主页面随机抽出的条数
QUICK_QUESTION_COUNT = 4

DISLIKE_REASONS = ["答非所问", "信息有误", "语气不当", "太啰嗦", "其他"]

# 导航里的视图（编号沿用页面的 01/02/03… 索引语言）
VIEWS = [
    ("对话", "01"),
    ("自查", "02"),
    ("打卡", "03"),
    ("数据看板", "04"),
    ("工具", "05"),
    ("历史", "06"),
    ("家庭组", "07"),
    ("设备导入", "08"),
    ("家庭药箱", "09"),
]
VIEW_NAMES = [name for name, _ in VIEWS]

# 桌面端左侧导航：分组 + Material 图标。移动端改用顶栏「☰ 导航」抽屉。
NAV_GROUPS = [
    ("聊天", [("对话", ":material/forum:"), ("历史", ":material/history:")]),
    (
        "健康",
        [
            ("自查", ":material/monitor_heart:"),
            ("打卡", ":material/event_available:"),
            ("数据看板", ":material/insights:"),
            ("工具", ":material/calculate:"),
        ],
    ),
    (
        "家庭",
        [
            ("家庭组", ":material/group:"),
            ("家庭药箱", ":material/medication:"),
            ("设备导入", ":material/watch:"),
        ],
    ),
]

# 老人模式去掉「工具」（八个计算器是给爱钻研的人用的），打卡、看板、家庭组与药箱保留
ELDER_VIEWS = [
    ("对话", "01"),
    ("自查", "02"),
    ("打卡", "03"),
    ("数据看板", "04"),
    ("历史", "05"),
    ("家庭组", "06"),
    ("家庭药箱", "07"),
]
# 老人模式的开屏示例少一条，每条更高更好点
ELDER_QUICK_QUESTION_COUNT = 3
# 朗读语速：1.0 是正常语速，取整会让长辈听不清，所以给 0.05 的步长
SPEECH_RATE_MIN = 0.7
SPEECH_RATE_MAX = 1.3
SPEECH_RATE_DEFAULT = 0.95

# 老人模式的回答要求：短、直给、说人话。
# 字数靠提示词约束，条数靠下面几个上限兜底——模型偶尔不听话，硬上限至少能让它别写长篇。
ELDER_SYSTEM_PROMPT = """你正在和一位老人说话。回答要短、要好懂，不求全。

要求：
1. 全文不超过 150 字，最多 3 句话或 3 条，一行一条，不要写成大段。
2. 第一句直接给答案或"该怎么做"；后面最多再补一句原因。不要寒暄，不要复述问题，不要总结。
3. 只说最要紧的一两件事，其余等对方追问。宁可少说，不要多说。
4. 不用专业名词。非用不可时，后面用括号换成大白话，例如「高血压（血管里的压力太大）」。
5. 不提药名、剂量、处方；需要就医就直接说「要去医院看看」。
6. 用短句，一句一个意思。不用"综上所述""因此""建议您"这类书面腔。"""

# 老人模式单次生成的 token 上限（约 200–300 字，拦在提示词的字数要求后面当保险）
ELDER_MAX_TOKENS = 320
# 老人模式只注入 2 条参考资料：资料越多，模型越想罗列
ELDER_RAG_TOP_K = 2
# 老人模式只推荐 2 条追问
ELDER_FOLLOWUP_LIMIT = 2
# 自查解读在老人模式下更短，token 也给得更紧
ELDER_SELFCHECK_MAX_TOKENS = 260
ELDER_SELFCHECK_SUGGESTIONS = 3

# ===================== 打卡 / 数据看板 =====================

# 看板的时间范围按「天」筛选；None 表示全部记录
DASH_RANGES = (
    ("近 7 天", 7),
    ("近 30 天", 30),
    ("近 90 天", 90),
    ("近 180 天", 180),
    ("近 365 天", 365),
    ("全部", None),
)
DASH_RANGE_LABELS = [label for label, _ in DASH_RANGES]
DASH_RANGE_DAYS = dict(DASH_RANGES)
DEFAULT_DASH_RANGE = "近 30 天"

# 身高 / 体重 / BMI 不再单独出图：体重是手表之外的秤数据，BMI 是派生值，
# 放在「设备」和打卡表里就够用了。数据照旧入库，只是不上看板。
DASH_HIDDEN_GROUPS = frozenset({"height", "weight", "bmi"})

# 对话里问「分析一下打卡数据」时，末尾固定加这句，把人引到看板看完整趋势
CHECKIN_ANALYSIS_HINT = "需要了解更多，请前往「数据看板」查看完整趋势和详细报告。"

# 打卡数据意图：只要用户明确在说「打卡 / 我的指标数据」，就走本地数据，
# 不交给模型自由发挥，避免编造没有记录过的数值。
_CHECKIN_INTENT_RE = re.compile(
    r"(打卡.{0,8}(数据|记录|情况|趋势|分析|总结|怎么样|如何|咋样|看看|看一下|看下|回顾|状况)"
    r"|(分析|总结|回顾|评估|解读|看看|看一下|看下|查一下|查看).{0,16}"
    r"(打卡|健康数据|我的数据|我的全部记录|全部记录|所有记录|健康记录|"
    r"(?:我的|我|最近|近|过去|这|本|今天|今日).{0,10}"
    r"(?:血压|血糖|血脂|体重|BMI|bmi|身高|心率|心跳|体温))"
    r"|(?:分析|总结|回顾|评估|解读|看看|看一下|看下|查一下|查看).{0,16}"
    r"的\s*(?:血压|血糖|血脂|体重|BMI|bmi|身高|心率|心跳|体温)"
    r"|(我的|我最近|最近|帮我|给我|近|过去|这).{0,12}"
    r"(血压|血糖|血脂|体重|BMI|bmi|身高|心率|心跳|体温).{0,12}"
    r"(数据|记录|趋势|怎么样|如何|咋样|状况|情况|控制|水平|正常|高不高|好不好)"
    r"|(血压|血糖|血脂|体重|BMI|bmi|身高|心率|心跳|体温).{0,10}"
    r"(数据|记录|趋势|怎么样|如何|咋样|状况|情况|控制|水平)"
    r"|(?:近|最近|过去|这|本)\s*(?:[0-9]+|[零一二两三四五六七八九十]+|半)\s*个?\s*"
    r"(?:天|日|周|星期|礼拜|月|年).{0,6}"
    r"(血压|血糖|血脂|体重|BMI|bmi|身高|心率|心跳|体温)"
    r"|(打开|去|进入|看看|查看|分析).{0,6}数据看板"
    r"|数据看板.{0,8}(看|分析|打开|去|进))"
)
# 用户只是在对话里报了一个测量值时（「今天打卡记录血压 120/80」），
# 不要抢着做趋势分析；除非同一句里还有明确的「分析/看看/趋势」等请求。
_CHECKIN_MEASUREMENT_RE = re.compile(
    r"(血压|血糖|血脂|体重|体温|心率|BMI|bmi).{0,6}\d"
)
_CHECKIN_ANALYSIS_VERB_RE = re.compile(
    r"分析|看看|看一下|看下|总结|回顾|趋势|怎么样|如何|情况|评估|解读|控制|状况|水平"
)

# 用户点名某个指标时，只分析这个指标，不再把所有打卡项混在一起。
# 值对应 checkin.GROUPS 的 key；顺序由 checkin.GROUPS 统一决定。
_METRIC_GROUP_PATTERNS = (
    ("bp", re.compile(r"血压|收缩压|舒张压")),
    ("glucose", re.compile(r"血糖")),
    ("lipid", re.compile(r"血脂|胆固醇|甘油三酯|低密度|高密度")),
    ("weight", re.compile(r"体重")),
    ("bmi", re.compile(r"BMI|bmi|体质指数")),
    ("height", re.compile(r"身高")),
    ("hr", re.compile(r"心率|心跳")),
    ("temp", re.compile(r"体温")),
)

# 「近一周 / 最近 7 天 / 这一个月」这类时间范围。支持阿拉伯数字和常见中文数字。
_CN_NUMBERS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_WINDOW_RE = re.compile(
    r"(?P<prefix>近|最近|过去|这|本)?\s*"
    r"(?P<num>[0-9]+|[零一二两三四五六七八九十]+|半)\s*"
    r"(?P<ge>个)?\s*"
    r"(?P<unit>天|日|周|星期|礼拜|月|年)(?![前后])"
)
_ALL_TIME_RE = re.compile(r"全部记录|所有记录|全部历史|所有历史|从开始到现在|一直以来|全部时间|所有时间")
_DEFAULT_WINDOW_DAYS = 30


@dataclass(frozen=True)
class CheckinQuery:
    """从一句问话里解析出来的分析范围。"""

    group_keys: Tuple[str, ...]
    window_days: Optional[int]
    window_label: str
    explicit_window: bool
    focus_label: str


def _cn_to_int(text: str) -> Optional[int]:
    """把「七」「十二」「二十」转成整数；不认识时返回 None。"""
    text = (text or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in _CN_NUMBERS:
        return _CN_NUMBERS[text]
    if "十" in text:
        left, right = text.split("十", 1)
        if left and left not in _CN_NUMBERS:
            return None
        if right and right not in _CN_NUMBERS:
            return None
        tens = _CN_NUMBERS[left] if left else 1
        ones = _CN_NUMBERS[right] if right else 0
        return tens * 10 + ones
    return None


def _window_from_text(text: str) -> Tuple[Optional[int], str, bool]:
    """解析时间范围，返回 (天数, 展示文案, 是否用户明确指定)。

    天数 None 表示全部记录；没有明确说明时默认近 30 天。
    """
    if _ALL_TIME_RE.search(text):
        return None, "全部记录", True

    for match in _WINDOW_RE.finditer(text):
        raw = match.group("num")
        unit = match.group("unit")
        prefix = match.group("prefix")
        ge = match.group("ge")

        # 「3 月」「三月」更可能是日期而不是三个月，必须带「个」或「近」才算范围
        if unit == "月" and not prefix and not ge:
            continue

        if raw == "半":
            if unit == "年":
                days = 180
            elif unit == "月":
                days = 15
            else:
                continue
        else:
            number = _cn_to_int(raw)
            if not number:
                continue
            if unit in ("天", "日"):
                days = number
            elif unit in ("周", "星期", "礼拜"):
                days = number * 7
            elif unit == "月":
                days = number * 30
            else:
                days = number * 365

        # 2026 年这类数字不是时间范围，直接忽略
        if not (1 <= days <= 3650):
            continue
        return days, f"近 {days} 天", True

    return _DEFAULT_WINDOW_DAYS, f"近 {_DEFAULT_WINDOW_DAYS} 天", False


def parse_checkin_query(prompt: str) -> CheckinQuery:
    """从问话里取出要看的指标和时间范围。"""
    text = prompt or ""
    found = {
        key
        for key, pattern in _METRIC_GROUP_PATTERNS
        if pattern.search(text)
    }
    group_keys = tuple(
        group.key for group in checkin.GROUPS if group.key in found
    )
    window_days, window_label, explicit_window = _window_from_text(text)
    focus_label = "、".join(
        checkin.GROUP_MAP[key].label for key in group_keys if key in checkin.GROUP_MAP
    )
    return CheckinQuery(
        group_keys=group_keys,
        window_days=window_days,
        window_label=window_label,
        explicit_window=explicit_window,
        focus_label=focus_label,
    )


# 对话里怎么认出「某位已绑定的家人」。优先匹配账号，其次匹配关系称谓。
_FAMILY_REFERENCE_RE = re.compile(
    r"爸|妈|父亲|母亲|儿子|女儿|丈夫|妻子|老公|老婆|配偶|爱人|"
    r"子女|父母|爸妈|家人|亲属|亲戚|孩子|家属|亲人|"
    r"爷爷|奶奶|外公|外婆|姥爷|姥姥|岳父|岳母|公公|婆婆|"
    r"哥哥|姐姐|弟弟|妹妹|叔叔|阿姨|舅舅|舅妈|姑姑|姑父|"
    r"伯父|伯母|孙子|孙女|外孙|外孙女|老伴"
)
_RELATION_ALIASES = {
    "父亲": ("父亲", "爸爸", "爸", "老爸", "爹"),
    "母亲": ("母亲", "妈妈", "妈", "老妈", "娘"),
    "儿子": ("儿子",),
    "女儿": ("女儿", "闺女"),
    "丈夫": ("丈夫", "老公"),
    "妻子": ("妻子", "老婆", "媳妇"),
    "配偶": ("配偶", "爱人", "老伴"),
    "父母": ("父母", "爸妈", "家长"),
    "子女": ("子女",),
    "家人": (),
}
_COLLECTIVE_ALIASES = {
    "父母": ("父亲", "母亲"),
    "爸妈": ("父亲", "母亲"),
    "子女": ("儿子", "女儿"),
    "孩子": ("儿子", "女儿"),
}


def family_member_label(member: Dict) -> str:
    """对话里指代家人时用的名字：你的父亲（father）。"""
    title = str(member.get("relation_title") or "家人")
    account = str(member.get("account") or "")
    return f"你的{title}（{account}）"


def family_member_options(members: List[Dict]) -> str:
    options = [
        f"{str(member.get('relation_title') or '家人')}（{member.get('account')}）"
        for member in members[:6]
    ]
    text = "、".join(options)
    if len(members) > 6:
        text += " 等"
    return text


def family_member_ambiguous(members: List[Dict]) -> Tuple[Optional[Dict], str, str]:
    first = members[0].get("account") if members else ""
    return (
        None,
        "",
        f"家庭组里有多位可能的成员：{family_member_options(members)}。"
        f"请直接说账号，例如「分析一下 {first} 近一周的血压」。",
    )


def resolve_chat_family_member(
    prompt: str, viewer_id: int
) -> Tuple[Optional[Dict], str, str]:
    """从对话里找出要分析的家庭组成员。

    返回 (成员, 称谓, 错误信息)。成员为 None 且错误信息为空时表示分析自己。
    只查 active 关系，pending / rejected 成员不会出现在这里。
    """
    text = prompt or ""
    try:
        members = family.active_members(int(viewer_id))
    except family.FamilyError as exc:
        return None, "", str(exc)

    if not members:
        if not _FAMILY_REFERENCE_RE.search(text):
            return None, "", ""
        try:
            owners = family.active_owners(int(viewer_id))
        except family.FamilyError:
            owners = []
        if owners:
            return (
                None,
                "",
                "你目前是家庭组成员，只有家庭组管理员可以分析其他成员的数据。"
                "要分析自己的数据，可以说「分析一下我的打卡数据」。",
            )
        return (
            None,
            "",
            "你的家庭组里还没有已绑定的家人，暂时不能分析对方的数据。"
            "请先在「家庭组」发起邀请，对方确认后再试；"
            "要分析自己的数据，可以说「分析一下我的打卡数据」。",
        )

    # 1) 账号名优先：直接说账号最不容易歧义
    folded = text.casefold()
    account_matches = []
    for member in members:
        account = str(member.get("account") or "")
        if len(account) >= 2 and account.casefold() in folded:
            account_matches.append(member)
    if account_matches:
        longest = max(
            len(str(member.get("account") or "")) for member in account_matches
        )
        top = [
            member
            for member in account_matches
            if len(str(member.get("account") or "")) == longest
        ]
        if len(top) == 1:
            return top[0], family_member_label(top[0]), ""
        return family_member_ambiguous(top)

    # 2) 关系称谓：爸 / 妈妈 / 女儿 / 老公 等
    relation_matches: List[Dict] = []
    for member in members:
        title = str(member.get("relation_title") or "家人")
        if any(alias in text for alias in _RELATION_ALIASES.get(title, ())):
            relation_matches.append(member)

    # 3) 父母 / 子女 / 孩子这类集合称谓
    if not relation_matches:
        for phrase, titles in _COLLECTIVE_ALIASES.items():
            if phrase in text:
                relation_matches.extend(
                    member
                    for member in members
                    if str(member.get("relation_title") or "") in titles
                )

    unique = {int(member["id"]): member for member in relation_matches}
    if len(unique) == 1:
        member = next(iter(unique.values()))
        return member, family_member_label(member), ""
    if len(unique) > 1:
        return family_member_ambiguous(list(unique.values()))

    # 4) 只说「家人 / 亲属」：只有一位成员时可以直接指向他/她
    if re.search(r"家人|亲属|亲戚|家属|亲人", text):
        if len(members) == 1:
            member = members[0]
            return member, family_member_label(member), ""
        return family_member_ambiguous(members)

    if _FAMILY_REFERENCE_RE.search(text):
        return (
            None,
            "",
            f"没有在已绑定成员里找到对应的人。当前家庭组有："
            f"{family_member_options(members)}。"
            f"请直接说账号，例如「分析一下 {members[0]['account']} 近一周的血压」。",
        )
    return None, "", ""

# 每张图可以在柱状图和折线图之间切换
CHART_KINDS = ("柱状图", "折线图")

# 打卡表单的控件 key：保存成功后统一清空，下一次录入从空白开始
CHECKIN_INPUT_KEYS = (
    "ci-height",
    "ci-weight",
    "ci-sbp",
    "ci-dbp",
    "ci-glucose",
    "ci-glucose-type",
    "ci-tc",
    "ci-tg",
    "ci-ldl",
    "ci-hdl",
    "ci-heart-rate",
    "ci-temp",
    "ci-note",
)
CHECKIN_RESET_KEY = "_checkin_reset"
CHECKIN_FLASH_KEY = "_checkin_flash"
CHECKIN_REPORT_KEY = "checkin_report"
CHECKIN_DIALOG_KEY = "_checkin_dialog_open"
# 设备导入成功后的一次性提示：和打卡分开，免得两个页面互相清掉
DEVICE_FLASH_KEY = "_device_flash"
DEVICE_CLEAR_UPLOAD_KEY = "_device_clear_upload"
# 站内绑定设备账号：弹窗开关、待确认的配置、成功提示
DEVICE_BIND_KEY = "_device_bind_open"
DEVICE_BIND_PAYLOAD_KEY = "_device_bind_payload"
DEVICE_BIND_FLASH_KEY = "_device_bind_flash"

# 家庭药箱：新增 / 修改弹窗、一次性提示、以及表单控件 key
# （保存成功后统一清空，下一次新增从空白开始；退出登录也按 "med-" 前缀清掉）
MED_ADD_KEY = "_med_add_open"
MED_EDIT_KEY = "_med_edit_id"
MED_FLASH_KEY = "_med_flash"
MED_RESET_KEY = "_med_reset"
MED_SYNC_KEY = "_med_alert_sync"
# 药盒识图：按图片内容指纹缓存一次结果，避免每次 rerun 都重新 OCR + 调模型
MED_SCAN_KEY = "_med_scan_digest"
MED_SCAN_RESULT = "_med_scan_result"
MED_SCAN_MAX_MB = medicine_scan.MAX_IMAGE_BYTES // (1024 * 1024)
# 药品照片：看大图的弹窗目标，以及列表缩略图的会话缓存（按 id + 更新时间失效）
MED_PHOTO_KEY = "_med_photo_id"
MED_IMAGE_CACHE_KEY = "_med_image_cache"
MED_IMAGE_CACHE_MAX = 40
MED_IMAGE_MAX_MB = medicine.IMAGE_MAX_INPUT_BYTES // (1024 * 1024)
# 提醒同步的节流间隔：临期按天变化，不需要每次 rerun 都重算一遍
MED_SYNC_TTL = 120
MED_FORM_KEYS = (
    "med-name",
    "med-category",
    "med-manufacturer",
    "med-efficacy",
    "med-spec",
    "med-unit",
    "med-package-unit",
    "med-packages",
    "med-production",
    "med-shelf-life",
    "med-has-expiry",
    "med-note",
)

# 对话里待发送的文件：文件本身只放在会话里，不写库，退出登录即清
CHAT_DOC_KEY = "_chat_doc"
CHAT_DOC_PANEL_KEY = "_chat_doc_panel"
CHAT_DOC_EPOCH_KEY = "_chat_doc_epoch"
# 附件来源：从文件里选，还是直接调摄像头拍一张
CHAT_DOC_SOURCE_KEY = "_chat_doc_source"
CHAT_DOC_SOURCES = ("file", "camera")
# 上传并分析过的文件正文：只留一小段给后续追问当背景
CHAT_DOC_CONTEXT_KEY = "_chat_doc_context"
CHAT_DOC_FOLLOWUP_CHARS = 4000
CHAT_DOC_MAX_MB = document.MAX_FILE_BYTES // (1024 * 1024)

st.set_page_config(
    page_title="木木医生 · 私人健康助手",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ===================== 基础设施 =====================

def load_css(elder: bool = False) -> str:
    """读取同目录下的 theme.css；缺失时静默降级为 Streamlit 默认样式。

    老人模式再叠一层 theme_elder.css：只覆盖字号与点按尺寸，
    放在基础样式之后，同优先级下自然覆盖，不用改基础文件。
    刻意不加 cache：这两个文件很小，每次 rerun 重读一遍，改完 CSS 存盘刷新即可看到。
    """
    # 顺序有讲究：外壳最后加载，基础样式先立住；老人模式再压在最上面，
    # 这样它的字号与点按尺寸能盖过外壳。
    if elder:
        names = ("theme.css", "theme_shell.css", "theme_elder.css")
    else:
        names = ("theme.css", "theme_shell.css")
    chunks: List[str] = []
    for name in names:
        try:
            chunks.append((APP_DIR / name).read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n".join(chunks)


# ===================== 账号 =====================

AUTH_USER_KEY = "auth_user"
MODE_CHOICE_PENDING_KEY = "_mode_prompt_pending"
LOGOUT_PENDING_KEY = "_logout_pending"
DEFAULT_MODE_REVERT_KEY = "_default_mode_revert"

# 家庭组：当前正在查看 / 代操作的家人 id（None = 自己）
FAMILY_SUBJECT_KEY = "family_subject_id"
FAMILY_SUBJECT_PICKER_KEY = "family-subject-picker"
FAMILY_FLASH_KEY = "_family_flash"

GENDER_LABELS = {"male": "男", "female": "女", "other": "其他"}
GENDER_BY_LABEL = {label: key for key, label in GENDER_LABELS.items()}
GENDER_OPTIONS = list(GENDER_LABELS.values())


def member_label(member: Dict) -> str:
    """家庭组对象在界面上的一行名字：称谓 · 账号（性别 · 年龄）。"""
    gender = GENDER_LABELS.get(member.get("gender"), "未填写")
    title = member.get("relation_title") or family.FALLBACK_TITLE
    return f"{title} · {member.get('account', '')}（{gender} · {member.get('age', '?')} 岁）"


def subject_display(subject: Dict, viewer: Optional[Dict]) -> str:
    """看板 / 报告里的分析对象：我自己，或你的父亲（father）。"""
    if not subject:
        return ""
    if viewer and int(subject.get("id") or 0) == int(viewer.get("id") or 0):
        return f"我自己（{viewer.get('account', '')}）"
    title = subject.get("relation_title") or family.FALLBACK_TITLE
    return f"你的{title}（{subject.get('account', '')}）"

# 返回栈：记着"从哪一页来"，返回按钮据此回上一级
NAV_STACK_KEY = "_nav_stack"
# 移动端导航抽屉是否展开；切换视图时由 goto_view 收起
MOBILE_NAV_KEY = "_mobile_nav_open"

# 退出登录 / 换账号时清掉的会话状态：别把上一个人的对话和模式带过来
SESSION_RESET_KEYS = (
    "messages",
    "chat_history",
    "conv_id",
    "feedback_status",
    "pending_down",
    "view",
    "tool",
    "quick_batch",
    "current_question",
    "hospital_index",
    "_view_prev",
    "_menu_open",
    "urgent_open",
    "urgent_result",
    "urgent_geo",
    "urgent_geo_error",
    "urgent_geo_ts",
    "urgent_city",
    "elder_mode",
    "default_elder_mode",
    "prompt_on_login",
    "speech_rate",
    MODE_CHOICE_PENDING_KEY,
    # 家庭组：当前查看对象和提示不跨账号保留
    FAMILY_SUBJECT_KEY,
    FAMILY_SUBJECT_PICKER_KEY,
    FAMILY_FLASH_KEY,
    # 打卡 / 看板：报告和筛选条件不跨账号保留
    CHECKIN_REPORT_KEY,
    CHECKIN_DIALOG_KEY,
    CHECKIN_FLASH_KEY,
    CHECKIN_RESET_KEY,
    "dash-range",
    # 环境建议：定位与缓存不跨账号保留
    "env_geo",
    "env_geo_error",
    "env_geo_pending",
    "env_geo_asked",
    "env_geo_ts",
    "env_city",
    "env_data",
    "env_fetched_at",
    # 对话文件：附件、上传方式（文件 / 拍照）和读出的正文不跨账号保留
    CHAT_DOC_KEY,
    CHAT_DOC_PANEL_KEY,
    CHAT_DOC_EPOCH_KEY,
    CHAT_DOC_SOURCE_KEY,
    CHAT_DOC_CONTEXT_KEY,
    # 家庭药箱：弹窗状态、一次性提示与同步节流时间不跨账号保留
    MED_ADD_KEY,
    MED_EDIT_KEY,
    MED_FLASH_KEY,
    MED_RESET_KEY,
    MED_SYNC_KEY,
    MED_SCAN_KEY,
    MED_SCAN_RESULT,
    MED_PHOTO_KEY,
    MED_IMAGE_CACHE_KEY,
    NAV_STACK_KEY,
    MOBILE_NAV_KEY,
)


# ===================== 上一级导航 =====================
# 每个子界面左上角都有一个「← 返回…」按钮，回到进这一页之前的那一页。
# 记的是"来路"而不是固定父子关系：从看板点「去打卡」，打卡页的返回就回看板；
# 从历史打开一条对话，对话页的返回就回历史。

# 只记最近几层：够来回就够了，别让一个会话无限长
NAV_STACK_MAX = 8


def view_number(name: str) -> str:
    """界面编号从当前模式的菜单里取，别在页面里写死。

    菜单顺序会调整（比如后来插入「设备」），写死的 09 会和菜单对不上；
    优先按当前模式取，老人模式的编号和正常模式不一样。
    """
    views = ELDER_VIEWS if st.session_state.get("elder_mode") else VIEWS
    for view_name, num in views:
        if view_name == name:
            return num
    for view_name, num in VIEWS:
        if view_name == name:
            return num
    return ""


def view_available(view: str) -> bool:
    """这个界面在当前模式下能不能去：老人模式把「工具」收起来了。"""
    if view not in VIEW_NAMES:
        return False
    if not st.session_state.get("elder_mode"):
        return True
    return view in [name for name, _ in ELDER_VIEWS]


def goto_view(target: str) -> None:
    """切到另一个界面，并把当前界面压进返回栈。"""
    st.session_state.pop(MOBILE_NAV_KEY, None)
    current = st.session_state.get("view") or "对话"
    if target == current:
        return
    stack = list(st.session_state.get(NAV_STACK_KEY) or [])
    stack.append(current)
    st.session_state[NAV_STACK_KEY] = stack[-NAV_STACK_MAX:]
    st.session_state.view = target


def nav_back_target(default: str = "对话") -> str:
    """返回按钮会回到哪一页：栈顶那一页，没有来路就是主页。

    栈里如果有当前模式下已经藏起来的界面（老人模式的「工具」），往下找一层。
    """
    for target in reversed(st.session_state.get(NAV_STACK_KEY) or []):
        if view_available(target):
            return target
    return default


def go_back(default: str = "对话") -> None:
    """返回上一级：弹掉栈顶，切过去。"""
    stack = list(st.session_state.get(NAV_STACK_KEY) or [])
    target = default
    while stack:
        candidate = stack.pop()
        if view_available(candidate):
            target = candidate
            break
    st.session_state[NAV_STACK_KEY] = stack
    st.session_state.view = target


def current_user() -> Optional[Dict]:
    user = st.session_state.get(AUTH_USER_KEY)
    return user if isinstance(user, dict) else None


def reset_user_session() -> None:
    for key in SESSION_RESET_KEYS:
        st.session_state.pop(key, None)
    # 每张图各自的时间范围 / 图形选择：key 按指标拼出来，统一按前缀清掉；
    # 家庭药箱的表单控件同理（med-name / med-edit-name-3 …）
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(
            ("chart-range-", "chart-kind-", "report-range", "med-")
        ):
            st.session_state.pop(key, None)


def sign_in(user: Dict) -> None:
    """登录成功：清掉上一份会话，再按账号的默认模式初始化本次会话。"""
    reset_user_session()
    st.session_state[AUTH_USER_KEY] = user
    st.session_state["elder_mode"] = user["default_mode"] == auth.MODE_ELDER
    st.session_state[MODE_CHOICE_PENDING_KEY] = bool(
        user["age"] >= 60 and user["prompt_on_login"]
    )


def sign_out() -> None:
    """先摘掉登录态；按钮所在的这一轮不动 elder_mode 等控件状态，下一轮开头再清。"""
    st.session_state.pop(AUTH_USER_KEY, None)
    st.session_state[MODE_CHOICE_PENDING_KEY] = False
    st.session_state[LOGOUT_PENDING_KEY] = True


def flush_logout() -> None:
    if st.session_state.pop(LOGOUT_PENDING_KEY, False):
        reset_user_session()


def dismiss_mode_prompt() -> None:
    """弹窗被关掉（Esc / 点遮罩）时保留当前默认模式，本次登录不再反复弹。"""
    st.session_state.pop(MODE_CHOICE_PENDING_KEY, None)


def render_login_form() -> None:
    with st.form("login-form"):
        account = st.text_input(
            "账号", placeholder="请输入账号", max_chars=auth.ACCOUNT_MAX
        )
        password = st.text_input(
            "密码",
            type="password",
            placeholder="请输入密码",
            max_chars=auth.PASSWORD_MAX,
        )
        submitted = st.form_submit_button("登录", type="primary", width="stretch")
    if not submitted:
        return

    try:
        user = auth.authenticate(account, password)
    except auth.AuthError as exc:
        st.error(str(exc))
        return
    if not user:
        st.error("账号或密码不对，请重试。")
        return

    sign_in(user)
    st.rerun()


def render_register_form() -> None:
    with st.form("register-form"):
        account = st.text_input(
            "账号",
            placeholder="2–32 个字符",
            max_chars=auth.ACCOUNT_MAX,
            help="中文、字母、数字和 . @ + - _ 都可以，不能有空格。",
        )
        password = st.text_input(
            "密码",
            type="password",
            placeholder="至少 6 位",
            max_chars=auth.PASSWORD_MAX,
        )
        confirm = st.text_input(
            "确认密码",
            type="password",
            placeholder="再输入一次",
            max_chars=auth.PASSWORD_MAX,
        )
        col_gender, col_age = st.columns(2, gap="small")
        gender_label = col_gender.selectbox("性别", GENDER_OPTIONS)
        age = col_age.number_input("年龄（岁）", 1, 120, 30, step=1)
        submitted = st.form_submit_button("注册并登录", type="primary", width="stretch")
    if not submitted:
        return
    if password != confirm:
        st.error("两次输入的密码不一致。")
        return

    try:
        user = auth.register(
            account, password, GENDER_BY_LABEL[gender_label], int(age)
        )
    except auth.AuthError as exc:
        st.error(str(exc))
        return

    sign_in(user)
    st.rerun()


def render_auth_page() -> None:
    inject_styles(elder=False)
    st.markdown(
        """
        <section class="hero auth-hero">
          <div class="eyebrow">
            <span class="eyebrow-num">00</span>
            <span class="eyebrow-sep"></span>
            <span>账号</span>
          </div>
          <h1 class="hero-title">木木医生</h1>
          <p class="hero-sub">
            登录后继续你的健康对话。账号用于区分使用者，并记住你的界面模式偏好。
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="auth-card"):
        login_tab, register_tab = st.tabs(["登录", "注册"])
        with login_tab:
            render_login_form()
        with register_tab:
            render_register_form()

    st.markdown(
        '<div class="compliance">密码只保存加密摘要，不保存明文；请勿使用其他网站的同款密码。</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=5, show_spinner=False)
def probe_models():
    """探活并返回 (是否可用, 可对话模型列表)。缓存 5 秒，避免每次 rerun 都等超时。"""
    return llm.probe()


@st.cache_data(ttl=600, show_spinner=False)
def cached_nearby(lat: float, lng: float, keyword: str):
    return poi.nearby(lat, lng, keyword=keyword)


@st.cache_data(ttl=600, show_spinner=False)
def cached_by_city(city: str, keyword: str):
    return poi.by_city(city, keyword=keyword)


@st.cache_data(ttl=600, show_spinner=False)
def cached_urgent_hospitals(lat: float, lng: float):
    """弹框只要两家医院：先搜 5 公里，搜不到再放宽到 20 公里。"""
    items = poi.nearby(lat, lng, keyword="医院", radius=5000, limit=2)
    if not items:
        items = poi.nearby(lat, lng, keyword="医院", radius=20000, limit=2)
    return items[:2]


def dismiss_urgent_dialog() -> None:
    """严重指标弹框被关掉（Esc / 点遮罩 / 关闭按钮）时调用，避免下次 rerun 又弹出来。"""
    st.session_state.urgent_open = False
    st.session_state.pop("urgent_result", None)


def close_checkin_dialog() -> None:
    """打卡分析弹窗被关掉时清掉开关，下一次 rerun 不再重开。"""
    st.session_state[CHECKIN_DIALOG_KEY] = False


def close_device_bind_dialog() -> None:
    """设备绑定弹窗关闭：清掉开关和半截的配置，下次重新来。"""
    st.session_state[DEVICE_BIND_KEY] = False
    st.session_state.pop(DEVICE_BIND_PAYLOAD_KEY, None)


def inject_styles(elder: bool = False) -> None:
    css = load_css(elder)
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


class MumuDoctorApp:
    """对话页面的状态与渲染。业务逻辑（RAG/安全/反馈）保持原样。"""

    def __init__(self):
        self.init_state()
        online, models = probe_models()
        st.session_state.online = online
        st.session_state.available_models = models

    def init_state(self) -> None:
        defaults = {
            "messages": [],
            "chat_history": [],
            "online": False,
            "available_models": [],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 512,
            "feedback_status": {},
            "pending_down": None,
            "suggest_followups": True,
            "speech_rate": SPEECH_RATE_DEFAULT,
            # 环境建议：默认开启，并且进入对话页就自动请求定位、取当地天气和空气，
            # 不需要再点「允许定位」。浏览器会弹一次权限询问，拒绝后本次会话不再打扰。
            # 坐标只放在本次会话里，不写库、不进对话历史。
            "env_enabled": True,
            "env_auto_locate": True,
            "env_geo": None,
            "env_geo_error": None,
            "env_geo_pending": False,
            "env_geo_asked": False,
            "env_geo_ts": None,
            "env_city": "",
            "env_data": None,
            "env_fetched_at": 0.0,
            "view": "对话",
            "tool": None,
            FAMILY_SUBJECT_KEY: None,
        }
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)

    # ---------- 环境（气温 / 空气质量） ----------

    def env_stale(self) -> bool:
        fetched = float(st.session_state.get("env_fetched_at") or 0)
        return time_module.time() - fetched >= ENV_TTL_SECONDS

    def auto_start_location(self) -> None:
        """启动时自动定位：进入应用就挂载定位组件，不用等用户点「允许定位」。

        每个会话只自动试一次（env_geo_asked）：成功就顺手取天气；被拒或失败
        不再反复弹权限框，用户仍可点「重新定位」或直接输入城市。
        定位组件本身只能挂在页面上，所以由对话页的环境条负责触发；
        对话页就是登录后的落点，等价于「启动即获取」。
        """
        if not st.session_state.get("env_enabled", True):
            return
        if not st.session_state.get("env_auto_locate", True):
            return
        if st.session_state.get("env_geo_pending"):
            return
        if st.session_state.get("env_geo_asked"):
            return
        if st.session_state.get("env_geo"):
            return
        if str(st.session_state.get("env_city") or "").strip():
            return
        st.session_state.env_geo_asked = True
        st.session_state.env_geo_pending = True

    def ensure_environment(self, force: bool = False) -> Optional[Dict]:
        """按本次会话里的定位或城市取环境信息；拿不到返回 None。

        结果放在 session_state：成功缓存 30 分钟，失败只缓存 1 分钟，
        免得服务挂掉时每次 rerun 都去等超时。坐标不出现在返回值里，
        页面只展示城市名和天气数值。
        """
        if not st.session_state.get("env_enabled", True):
            return None

        cached = st.session_state.get("env_data")
        if cached and not force:
            ttl = ENV_TTL_SECONDS if cached.get("ok") else 60
            if time_module.time() - float(st.session_state.get("env_fetched_at") or 0) < ttl:
                return cached

        coords = st.session_state.get("env_geo")
        city = str(st.session_state.get("env_city") or "").strip()
        if not coords and not city:
            return None

        try:
            if coords:
                env = environment.fetch(float(coords[0]), float(coords[1]))
            else:
                hit = environment.lookup_city(city)
                if not hit:
                    env = {"ok": False, "error": "没找到这个城市"}
                else:
                    env = environment.fetch(
                        float(hit["lat"]),
                        float(hit["lng"]),
                        city=str(hit.get("label") or city),
                    )
        except Exception:  # noqa: BLE001
            env = {"ok": False, "error": "环境服务暂时不可用"}

        env["fetched_at"] = time_module.time()
        st.session_state.env_data = env
        st.session_state.env_fetched_at = env["fetched_at"]
        return env

    def clear_environment(self) -> None:
        """清掉定位与缓存，但保留「是否结合环境」这个开关。"""
        st.session_state.env_geo = None
        st.session_state.env_geo_error = None
        st.session_state.env_geo_pending = False
        st.session_state.env_geo_asked = False
        st.session_state.env_geo_ts = None
        st.session_state.env_city = ""
        st.session_state.env_data = None
        st.session_state.env_fetched_at = 0.0

    @staticmethod
    def env_chip_html(env: Dict) -> str:
        """环境条上那一行：城市 · 12°C 阴 · 空气良。"""
        tone = environment.aqi_tone(env.get("aqi"))
        bits: List[str] = []
        if env.get("temp") is not None:
            bits.append(f"{env['temp']:.0f}°C")
        if env.get("text") and env.get("text") != "天气未知":
            bits.append(html.escape(str(env["text"])))
        if env.get("aqi") is not None:
            bits.append(f"空气{env.get('aqi_level') or ''}")
        city = html.escape(str(env.get("city") or "当前位置"))
        value = html.escape(" · ".join(bits))
        return (
            f'<div class="envbar"><span class="envbar-dot {tone}"></span>'
            f'<span class="envbar-city">{city}</span>'
            f'<span class="envbar-value">{value}</span></div>'
        )

    def render_env_status_caption(self) -> None:
        """设置里的状态行：现在参考的是哪儿；没设位置时告诉用户会自动定位。"""
        if not st.session_state.get("env_enabled", True):
            st.caption("已关闭：回答不会参考天气和空气质量。")
            return
        env = st.session_state.get("env_data")
        if env and env.get("ok"):
            st.caption(f"当前参考：{environment.describe(env)}")
        elif st.session_state.get("env_geo_pending"):
            st.caption("正在获取定位…（浏览器可能会问一次定位权限）")
        elif st.session_state.get("env_geo") or st.session_state.get("env_city"):
            st.caption("位置已设置，环境数据暂时取不到。")
        elif st.session_state.get("env_geo_error"):
            st.caption(
                f"没能获取定位（{st.session_state.get('env_geo_error')}）："
                "可以到对话页点「允许定位」，或直接输入城市。"
            )
        elif st.session_state.get("env_auto_locate", True):
            st.caption("已开启自动定位：进入对话页就会获取位置和当地天气。")
        else:
            st.caption("还没设置位置：到对话页输入框上方点「允许定位」或「输入城市」。")

    def render_env_auto_toggle(self) -> None:
        """设置里的「启动时自动定位」：关掉后不再自动弹浏览器定位权限。"""
        was = bool(st.session_state.get("env_auto_locate", True))
        now = st.toggle(
            "启动时自动定位",
            value=was,
            help="开启后，进入应用就自动请求浏览器定位并取当地天气和空气，"
            "浏览器可能会问一次权限。关掉后只能手动点「允许定位」或输入城市。"
            "位置只在本次登录内使用，不写进对话历史。",
        )
        st.session_state.env_auto_locate = now
        if now != was:
            # 重新打开时允许立刻再试一次；关掉时收起正在等待结果的定位组件
            st.session_state.env_geo_asked = False
            if not now:
                st.session_state.env_geo_pending = False

    @property
    def elder(self) -> bool:
        """当前是否使用老年模式。

        登录时 sign_in() 已经按账号的 default_mode 把 elder_mode 放进 session_state；
        之后设置里的开关随时改它。没有登录态不会渲染主页面，所以不需要 URL 兜底。
        """
        return bool(st.session_state.get("elder_mode", False))

    # ---------- 模型 ----------

    def resolve_model(self) -> str:
        """使用服务端配置的模型。

        页面上不再提供模型选择，所以只看服务端配置：优先精确命中默认模型，
        其次按「冒号前的名字」匹配（本地 Ollama 的模型名通常带 tag），
        云端没命中时仍用默认模型，避免误选到列表里的其他型号。
        """
        models: List[str] = st.session_state.available_models or []
        if DEFAULT_MODEL in models:
            return DEFAULT_MODEL

        target = DEFAULT_MODEL.split(":")[0]
        for name in models:
            if name.split(":")[0] == target:
                return name

        if llm.is_cloud():
            return DEFAULT_MODEL
        return models[0] if models else DEFAULT_MODEL

    def stream_chat(self, messages: List[Dict], model: str):
        """流式对话生成；本地 Ollama 还是云端由 llm 模块决定。"""
        max_tokens = st.session_state.max_tokens
        if self.elder:
            # 老人模式不要长回答：模型偶尔不理会提示词里的字数要求，这里再压一道
            max_tokens = min(max_tokens, ELDER_MAX_TOKENS)
        yield from llm.chat_stream(
            messages,
            model=model,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            top_k=st.session_state.top_k,
            max_tokens=max_tokens,
        )

    def request_messages(self) -> List[Dict]:
        """把会话整理成这次要发出去的消息。

        老人模式在最前面加一条 system：短、先给做法、说人话。
        上传过文件时再补一条 system，把文件正文作为背景，方便接着追问数值。
        历史消息原样带上，只换「怎么说」，不换「说什么」。
        """
        messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in st.session_state.messages
        ]
        context = st.session_state.get(CHAT_DOC_CONTEXT_KEY) or {}
        context_text = str(context.get("text") or "").strip()
        if context_text:
            messages.insert(
                0,
                document.followup_system_message(
                    str(context.get("name") or "上传的文件"), context_text
                ),
            )
        if self.elder:
            return [{"role": "system", "content": ELDER_SYSTEM_PROMPT}, *messages]
        return messages

    def inject_knowledge(self, messages: List[Dict]) -> List[Dict]:
        """检索知识并注入到最后一条 user 消息（Tavily 优先 + 本地 RAG 兜底）。

        检索用的查询要带上最近的上下文：追问常常只有「平时该怎么做」几个字，
        单拿它去检索必然跑偏，注入无关知识后模型反而会拒绝回答。
        """
        try:
            if not messages or messages[-1]["role"] != "user":
                return messages
            prompt = messages[-1]["content"]
            # 老人模式少喂两条资料：资料越多，越容易把回答写成文献综述
            top_k = ELDER_RAG_TOP_K if self.elder else RAG_TOP_K

            # 最近几轮的提问拼成检索查询，让向量知道"在聊什么"
            recent = " ".join(
                m["content"] for m in messages[:-1] if m.get("role") == "user"
            )
            query = f"{recent} {prompt}".strip()[:200]

            docs = web_search(query, max_results=top_k)
            if len(docs) < top_k:
                local_docs = get_retriever().retrieve(
                    query, top_k=top_k, min_score=RAG_MIN_SIMILARITY
                )
                for doc in local_docs:
                    if doc not in docs:
                        docs.append(doc)
                    if len(docs) >= top_k:
                        break

            if not docs:
                return messages

            doc_context = st.session_state.get(CHAT_DOC_CONTEXT_KEY) or {}
            block = (
                "以下是可能相关的医学资料，仅供参考。\n"
                "如果这些资料与患者当前的问题无关，直接忽略它们，按你自己的医学知识正常回答，"
                "不要提及资料内容，也不要说「资料与问题无关」。\n"
                "可以引用相关资料，但不要编造资料里没有的内容。\n"
            )
            if doc_context.get("text"):
                block += (
                    "如果问题和用户此前上传过的药盒 / 说明书有关，可以引用文件里印的"
                    "用法用量；文件里没写的剂量不要编，也不要推荐购买新药或自行加量。\n"
                )
            else:
                block += (
                    "另外：不要给出任何具体药名、剂量或处方建议，也不要推荐购买药品；"
                    "只提供生活方式调整与是否需要就医的判断。\n"
                )
            if self.elder:
                block += "这些资料只是背景：最多取一句最相关的意思，不要罗列资料内容。\n"
            block += "\n".join(f"- {doc}" for doc in docs)
            messages[-1]["content"] = f"{block}\n\n患者问题：{prompt}"
        except Exception:  # noqa: BLE001
            pass
        return messages

    def inject_environment(
        self, messages: List[Dict]
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """把当地环境作为一条 system 消息插到最前面（老人模式的写法要求之后）。

        只带城市名和几个数值，不带坐标。取不到环境就原样返回，
        对这一轮回答没有任何影响。
        """
        env = st.session_state.get("env_data")
        if not env or not env.get("ok"):
            return messages, None
        block = environment.prompt_block(env, elder=self.elder)
        if not block:
            return messages, None

        at = 0
        while at < len(messages) and messages[at].get("role") == "system":
            at += 1
        messages.insert(at, {"role": "system", "content": block})
        return messages, env

    # ---------- 渲染 ----------

    def render_shell_nav(self) -> None:
        """桌面端左侧固定导航。

        移动端整块隐藏（CSS），改用顶栏「☰ 导航」呼出的独立导航抽屉；
        手机上不再把功能列表塞进头像弹窗。
        """
        view = st.session_state.get("view", "对话")
        allowed = {name for name, _ in (ELDER_VIEWS if self.elder else VIEWS)}

        with st.container(key="shell-nav"):
            st.markdown(
                """
                <div class="nav-brand">
                  <div class="nav-mark">木</div>
                  <div class="nav-brand-text">
                    <div class="nav-brand-name">木木医生</div>
                    <div class="nav-brand-role">全科医生 · 私人健康助手</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for label, items in NAV_GROUPS:
                items = [(name, icon) for name, icon in items if name in allowed]
                if not items:
                    continue
                st.markdown(
                    f'<div class="nav-group">{label}</div>', unsafe_allow_html=True
                )
                for name, icon in items:
                    if st.button(
                        name,
                        key=f"nav-{name}",
                        icon=icon,
                        width="stretch",
                        type="primary" if name == view else "tertiary",
                    ):
                        st.session_state.view = name
                        st.rerun()

            st.markdown(
                '<div class="nav-foot">回答由 AI 生成，仅供参考</div>',
                unsafe_allow_html=True,
            )

    def render_mobile_nav(self) -> None:
        """移动端导航抽屉：和桌面侧栏同一套 NAV_GROUPS，由顶栏按钮呼出。"""
        view = st.session_state.get("view", "对话")
        allowed = {name for name, _ in (ELDER_VIEWS if self.elder else VIEWS)}
        user = current_user()

        with st.container(key="mobile-nav"):
            col_brand, col_close = st.columns(
                [4, 1], gap="small", vertical_alignment="center"
            )
            with col_brand:
                st.markdown(
                    """
                    <div class="nav-brand">
                      <div class="nav-mark">木</div>
                      <div class="nav-brand-text">
                        <div class="nav-brand-name">木木医生</div>
                        <div class="nav-brand-role">全科医生 · 私人健康助手</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_close:
                if st.button("收起", key="mobile-nav-close", type="tertiary"):
                    st.session_state[MOBILE_NAV_KEY] = False
                    st.rerun()

            for label, items in NAV_GROUPS:
                items = [(name, icon) for name, icon in items if name in allowed]
                if not items:
                    continue
                st.markdown(
                    f'<div class="nav-group">{label}</div>', unsafe_allow_html=True
                )
                for name, icon in items:
                    suffix = ""
                    if name == "家庭药箱" and user:
                        try:
                            _, alerts = self.load_medicine_alerts(user)
                            suffix = f"　· {len(alerts)} 条提醒" if alerts else ""
                        except medicine.MedicineError:
                            suffix = ""
                    if st.button(
                        f"{name}{suffix}",
                        key=f"mnav-{name}",
                        icon=icon,
                        width="stretch",
                        type="primary" if name == view else "tertiary",
                    ):
                        if name == "工具" and view != "工具":
                            st.session_state.tool = None
                        st.session_state[MOBILE_NAV_KEY] = False
                        goto_view(name)
                        st.rerun()

            st.markdown(
                '<div class="nav-foot">回答由 AI 生成，仅供参考</div>',
                unsafe_allow_html=True,
            )

    def render_header(self) -> None:
        online = st.session_state.online
        dot = "on" if online else "off"
        status = "在线" if online else "未连接"
        view = st.session_state.get("view", "对话")

        # 顶栏：移动端「头像 + 品牌」，桌面端「页面标题 + 操作」。
        # 两套都在 DOM 里，靠 CSS 按宽度取舍，避免为断点写两份页面。
        with st.container(
            key="topbar", horizontal=True, vertical_alignment="center", gap="small"
        ):
            with st.container(key="topbar-avatar"):
                if st.button("木", key="avatar-btn", type="secondary"):
                    # 头像只负责开门，弹窗在下一次 rerun 里画出来
                    st.session_state._menu_open = True
                    st.rerun()

            with st.container(key="mobile-nav-toggle"):
                nav_open = bool(st.session_state.get(MOBILE_NAV_KEY))
                if st.button(
                    "✕ 收起" if nav_open else "☰ 导航",
                    key="mobile-nav-toggle-btn",
                    type="tertiary",
                ):
                    st.session_state[MOBILE_NAV_KEY] = not nav_open
                    st.rerun()

            st.markdown(
                f"""
                <div class="brand only-mobile">
                  <div class="brand-line">
                    <span class="brand-name">木木医生</span>
                    <span class="status"><span class="dot {dot}"></span>{status}</span>
                  </div>
                  <div class="brand-role">全科医生 · 私人健康助手</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="page-title only-desktop">{html.escape(view)}'
                f'<span class="page-status"><span class="dot {dot}"></span>{status}</span></div>',
                unsafe_allow_html=True,
            )

            with st.container(key="topbar-new"):
                if st.button("新对话", key="new_chat_top", type="tertiary"):
                    self.clear_chat()
                    st.rerun()

            with st.popover(
                "设置", key="settings-menu", type="tertiary", width="content"
            ):
                self.render_settings()

        st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)

    def render_back(self, target_label: str, key: str, action) -> None:
        """子界面左上角的「← 返回xx」：点是回到上一级界面。

        目标写在字面上（「← 返回对话」），点之前就知道会去哪。
        """
        if st.button(f"← 返回{target_label}", key=f"back-{key}", type="tertiary"):
            action()
            st.rerun()

    def back_to_tool_list(self) -> None:
        """工具详情 → 工具列表。"""
        st.session_state.tool = None

    def back_to_chat(self) -> None:
        """收起就近医院 / 就诊流程，回对话。"""
        st.session_state.hospital_index = None
        st.session_state.pop("guide_open", None)

    @st.dialog("功能")
    def render_view_menu(self) -> None:
        """头像弹窗只留快捷动作；功能导航已经拆到顶栏「☰ 导航」。

        手机端顶栏的「新对话」被 CSS 隐藏，这里保留一个入口。
        """
        with st.container(key="view-menu"):
            st.caption("功能导航请点顶栏的「☰ 导航」。")
            if st.button("新对话", key="menu-new-chat", width="stretch", type="tertiary"):
                self.clear_chat()
                goto_view("对话")
                st.rerun()

    def on_mode_change(self) -> None:
        """模式变了，开屏示例条数也变了，旧的一批作废重抽。"""
        st.session_state.pop("quick_batch", None)

    def render_mode_switch(self) -> None:
        """当前这次登录用哪种模式：改完立即重绘，不写回数据库。"""
        st.toggle(
            "当前使用老年模式",
            key="elder_mode",
            on_change=self.on_mode_change,
            help="字号更大、选项更少，每条回答下面都能点「朗读」。只影响这次登录。",
        )

    def save_default_mode(self) -> None:
        """把「下次登录默认用哪种模式」写回数据库。"""
        user = current_user()
        if not user:
            return
        target = (
            auth.MODE_ELDER
            if st.session_state.get("default_elder_mode")
            else auth.MODE_NORMAL
        )
        try:
            # 60 岁以上用户手动定了默认模式，就不再每次登录都问；想恢复询问用下面的按钮
            updated = auth.update_mode_preferences(
                user["id"],
                default_mode=target,
                prompt_on_login=False if user["age"] >= 60 else None,
            )
        except auth.AuthError as exc:
            st.toast(f"默认模式没保存成功：{exc}")
            # 控件已经翻转，下一轮开头把它恢复成数据库里的值（本轮不能直接改控件状态）
            st.session_state[DEFAULT_MODE_REVERT_KEY] = True
            return
        st.session_state[AUTH_USER_KEY] = updated
        st.toast("已保存为默认模式。")

    def render_mode_defaults(self) -> None:
        user = current_user()
        if not user:
            return

        toggle_kwargs: Dict = {}
        if "default_elder_mode" not in st.session_state:
            # 换账号 / 首次登录时按数据库里的默认值定初值；已有控件状态时不再传 value
            toggle_kwargs["value"] = user["default_mode"] == auth.MODE_ELDER
        st.toggle(
            "下次登录默认使用老年模式",
            key="default_elder_mode",
            on_change=self.save_default_mode,
            help="只改下次登录时的初始模式；本次随时可以用上面的开关切换。",
            **toggle_kwargs,
        )

        if user["age"] < 60:
            st.caption("登录时不会询问，直接按默认模式进入。")
            return
        if user["prompt_on_login"]:
            st.caption("您已满 60 岁：下次登录时会再问一次用哪种模式。")
            return

        st.caption("您已满 60 岁：下次登录直接按默认模式进入。")
        if st.button("登录时重新询问模式", key="reenable-mode-prompt", width="stretch"):
            try:
                updated = auth.update_mode_preferences(user["id"], prompt_on_login=True)
            except auth.AuthError as exc:
                st.error(str(exc))
            else:
                st.session_state[AUTH_USER_KEY] = updated
                st.rerun()

    def render_account(self) -> None:
        user = current_user()
        if not user:
            return
        gender = GENDER_LABELS.get(user["gender"], "未填写")
        st.caption(
            f"当前账号：`{user['account']}`　·　{gender}　·　{user['age']} 岁"
        )
        if st.button("退出登录", key="logout", width="stretch"):
            sign_out()
            st.rerun()

    # ---------- 家庭组 ----------

    def clear_subject_caches(self) -> None:
        """切换查看对象时，把上一个账号的打卡输入、报告和图表筛选清掉。"""
        for key in CHECKIN_INPUT_KEYS:
            st.session_state.pop(key, None)
        for key in (
            CHECKIN_FLASH_KEY,
            CHECKIN_REPORT_KEY,
            CHECKIN_DIALOG_KEY,
            "dash-range",
            "report-range",
            "sleep-night",
            "device-account",
            "device-unbind-confirm",
        ):
            st.session_state.pop(key, None)
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and key.startswith(("chart-range-", "chart-kind-")):
                st.session_state.pop(key, None)

    def resolve_subject(self, warn: bool = False) -> Optional[Dict]:
        """当前要查看 / 代操作的账号；没有授权就退回自己。

        所有打卡、看板、历史都从这里拿 user_id，页面层不直接信 session_state
        里的数字：每次渲染都重新查一遍 active 关联。
        """
        viewer = current_user()
        if not viewer:
            return None
        selected = st.session_state.get(FAMILY_SUBJECT_KEY)
        if selected is None:
            return viewer
        try:
            member = family.authorized_member(viewer["id"], int(selected))
        except (family.FamilyError, TypeError, ValueError):
            member = None
        if member:
            return member
        st.session_state[FAMILY_SUBJECT_KEY] = None
        st.session_state.pop(FAMILY_SUBJECT_PICKER_KEY, None)
        self.clear_subject_caches()
        if warn:
            st.warning("这条家庭组关联已被解除，已切回你自己的数据。")
        return viewer

    def on_subject_change(self) -> None:
        """下拉框换人：重新校验授权，并清掉上一个账号的页面缓存。"""
        viewer = current_user()
        raw = st.session_state.get(FAMILY_SUBJECT_PICKER_KEY)
        subject_id = None
        if raw is not None and viewer:
            try:
                member = family.authorized_member(viewer["id"], int(raw))
            except (family.FamilyError, TypeError, ValueError):
                member = None
            if member:
                subject_id = int(raw)
            else:
                st.session_state[FAMILY_SUBJECT_PICKER_KEY] = None
        st.session_state[FAMILY_SUBJECT_KEY] = subject_id
        self.clear_subject_caches()

    def render_subject_selector(self, view: str) -> None:
        """打卡 / 看板 / 历史顶部的「数据对象」。没有已授权家人时不显示。"""
        viewer = current_user()
        if not viewer:
            return
        try:
            members = family.active_members(viewer["id"])
        except family.FamilyError as exc:
            st.error(str(exc))
            return

        if not members:
            st.session_state[FAMILY_SUBJECT_KEY] = None
            st.session_state.pop(FAMILY_SUBJECT_PICKER_KEY, None)
            return

        options = [None] + [int(m["id"]) for m in members]
        labels = {int(m["id"]): member_label(m) for m in members}
        try:
            current = st.session_state.get(FAMILY_SUBJECT_KEY)
            current = int(current) if current is not None else None
        except (TypeError, ValueError):
            current = None

        if FAMILY_SUBJECT_PICKER_KEY in st.session_state:
            picker = st.session_state.get(FAMILY_SUBJECT_PICKER_KEY)
            if picker in options:
                current = picker
            else:
                st.session_state[FAMILY_SUBJECT_PICKER_KEY] = current
        else:
            st.session_state[FAMILY_SUBJECT_PICKER_KEY] = current

        if current not in options:
            current = None
            st.session_state[FAMILY_SUBJECT_KEY] = None
            st.session_state[FAMILY_SUBJECT_PICKER_KEY] = None
        else:
            st.session_state[FAMILY_SUBJECT_KEY] = current

        st.selectbox(
            "数据对象",
            options,
            key=FAMILY_SUBJECT_PICKER_KEY,
            format_func=lambda uid: (
                f"我自己（{viewer['account']}）"
                if uid is None
                else labels.get(int(uid), "已解除关联")
            ),
            on_change=self.on_subject_change,
            help="选择自己或已授权的家人；打卡、看板和历史都会切换到该账号。",
        )

        if current is None:
            caption = "当前查看的是你自己的数据。"
            if view == "数据看板" and members:
                caption += (
                    " 在上方「数据对象」选择家人后，图表、分析报告和 PDF "
                    "都会按对方的数据生成。"
                )
            st.caption(caption)
            return
        member = next((m for m in members if int(m["id"]) == current), None)
        if member:
            self.render_proxy_banner(member, view)

    @staticmethod
    def render_proxy_banner(member: Dict, view: str) -> None:
        name = html.escape(str(member.get("account") or ""))
        title = html.escape(str(member.get("relation_title") or family.FALLBACK_TITLE))
        text = {
            "打卡": f"正在为你的 <b>{title}</b> <b>{name}</b> 代为打卡；保存后记在对方账号下。",
            "数据看板": (
                f"正在查看你的 <b>{title}</b> <b>{name}</b> 的打卡数据与趋势；"
                "分析报告和 PDF 也会按对方的数据生成。"
            ),
            "历史": f"正在查看你的 <b>{title}</b> <b>{name}</b> 的问询历史（只读）。",
            "设备导入": (
                f"正在为你的 <b>{title}</b> <b>{name}</b> 导入设备数据；"
                "导入的记录会记在对方账号下。"
            ),
        }.get(view, f"当前对象：你的 <b>{title}</b> <b>{name}</b>。")
        st.markdown(f'<div class="proxy-banner">{text}</div>', unsafe_allow_html=True)

    def open_subject(self, member_id: int, view: str) -> None:
        """从家庭组页直接打开某个家人的打卡 / 看板 / 历史。"""
        st.session_state[FAMILY_SUBJECT_KEY] = int(member_id)
        st.session_state[FAMILY_SUBJECT_PICKER_KEY] = int(member_id)
        goto_view(view)
        self.clear_subject_caches()
        st.rerun()

    def family_action(self, action, success_text: str) -> None:
        try:
            action()
        except family.FamilyError as exc:
            st.error(str(exc))
            return
        st.session_state[FAMILY_FLASH_KEY] = success_text
        st.rerun()

    @staticmethod
    def family_time(value) -> str:
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M")
        return "时间未知"

    def render_family_request(self, request: Dict, can_accept: bool = True) -> None:
        user = current_user()
        if not user:
            return
        message = (
            f'<div class="family-message">留言：{html.escape(request["message"])}</div>'
            if request.get("message")
            else ""
        )
        st.markdown(
            f'<div class="family-card">'
            f'<div class="family-name">{html.escape(request["account"])}'
            f'<span class="family-badge pending">请求</span></div>'
            f'<div class="family-meta">关系：{html.escape(request["relation_title"])}'
            f' · {html.escape(GENDER_LABELS.get(request["gender"], "未填写"))}'
            f' · {request["age"]} 岁 · {self.family_time(request.get("created_at"))}</div>'
            f"{message}</div>",
            unsafe_allow_html=True,
        )
        if not can_accept:
            st.caption("你已属于一个家庭组，需先解除现有关系才能接受新的请求。")
            if st.button(
                "拒绝",
                key=f"family-reject-{request['id']}",
                width="stretch",
            ):
                self.family_action(
                    lambda: family.respond(user["id"], request["link_id"], False),
                    f"已拒绝 {request['account']} 的关联请求。",
                )
            return

        col_yes, col_no = st.columns(2, gap="small")
        if col_yes.button(
            "接受",
            key=f"family-accept-{request['id']}",
            type="primary",
            width="stretch",
        ):
            self.family_action(
                lambda: family.respond(user["id"], request["link_id"], True),
                f"已接受 {request['account']} 的关联请求。",
            )
        if col_no.button(
            "拒绝",
            key=f"family-reject-{request['id']}",
            width="stretch",
        ):
            self.family_action(
                lambda: family.respond(user["id"], request["link_id"], False),
                f"已拒绝 {request['account']} 的关联请求。",
            )

    def render_family_member(self, member: Dict) -> None:
        user = current_user()
        if not user:
            return
        st.markdown(
            f'<div class="family-card">'
            f'<div class="family-name">{html.escape(member["account"])}'
            f'<span class="family-badge active">已授权</span></div>'
            f'<div class="family-meta">关系：{html.escape(member["relation_title"])}'
            f' · {html.escape(GENDER_LABELS.get(member["gender"], "未填写"))}'
            f' · {member["age"]} 岁 · 关联于 {self.family_time(member.get("linked_at"))}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns(2, gap="small")
        if col_a.button(
            "查看数据看板",
            key=f"family-dash-{member['id']}",
            width="stretch",
        ):
            self.open_subject(member["id"], "数据看板")
        if col_b.button(
            "代为打卡",
            key=f"family-checkin-{member['id']}",
            width="stretch",
        ):
            self.open_subject(member["id"], "打卡")

        col_a, col_b = st.columns(2, gap="small")
        if col_a.button(
            "查看问询历史",
            key=f"family-history-{member['id']}",
            width="stretch",
        ):
            self.open_subject(member["id"], "历史")
        if col_b.button(
            "解除关联",
            key=f"family-unlink-{member['id']}",
            width="stretch",
        ):
            try:
                family.unlink(user["id"], member["link_id"])
            except family.FamilyError as exc:
                st.error(str(exc))
            else:
                if st.session_state.get(FAMILY_SUBJECT_KEY) == int(member["id"]):
                    st.session_state[FAMILY_SUBJECT_KEY] = None
                    st.session_state.pop(FAMILY_SUBJECT_PICKER_KEY, None)
                    self.clear_subject_caches()
                st.session_state[FAMILY_FLASH_KEY] = (
                    f"已解除与 {member['account']} 的关联。"
                )
                st.rerun()

    def render_family_outgoing(self, link: Dict, can_invite: bool = True) -> None:
        user = current_user()
        if not user:
            return
        status = link["status"]
        label, css = (
            ("等待确认", "pending")
            if status == family.STATUS_PENDING
            else ("已拒绝", "rejected")
        )
        st.markdown(
            f'<div class="family-card">'
            f'<div class="family-name">{html.escape(link["account"])}'
            f'<span class="family-badge {css}">{label}</span></div>'
            f'<div class="family-meta">关系：{html.escape(link["relation_title"])}'
            f' · {html.escape(GENDER_LABELS.get(link["gender"], "未填写"))}'
            f' · {link["age"]} 岁 · {self.family_time(link.get("created_at"))}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if status == family.STATUS_PENDING:
            if st.button(
                "取消请求",
                key=f"family-cancel-{link['id']}",
                width="stretch",
            ):
                self.family_action(
                    lambda: family.unlink(user["id"], link["link_id"]),
                    f"已取消发给 {link['account']} 的请求。",
                )
        else:
            if not can_invite:
                st.caption("你已加入一个家庭组，不能再发起新的邀请。")
                return
            key = f"family-reinvite-relation-{link['id']}"
            if key not in st.session_state:
                st.session_state[key] = link.get("relation") or None
            st.selectbox(
                "你与对方的关系",
                options=list(family.RELATIONS),
                key=key,
                format_func=family.RELATION_OPTION_LABELS.get,
                placeholder="请选择你与对方的关系",
                help="只支持直系亲属：父母、子女、配偶。称谓按双方注册的性别自动显示。",
            )
            if st.button(
                "重新邀请",
                key=f"family-reinvite-{link['id']}",
                width="stretch",
            ):
                relation = st.session_state.get(key)
                self.family_action(
                    lambda: family.invite(
                        user["id"],
                        link["account"],
                        relation,
                        link.get("message", ""),
                    ),
                    f"已重新向 {link['account']} 发送关联请求。",
                )

    def render_family_owner(self, link: Dict) -> None:
        user = current_user()
        if not user:
            return
        st.markdown(
            f'<div class="family-card">'
            f'<div class="family-name">{html.escape(link["account"])}'
            f'<span class="family-badge active">可查看我的数据</span></div>'
            f'<div class="family-meta">关系：{html.escape(link["relation_title"])}'
            f' · {html.escape(GENDER_LABELS.get(link["gender"], "未填写"))}'
            f' · {link["age"]} 岁 · 授权于 {self.family_time(link.get("linked_at"))}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "解除授权",
            key=f"family-revoke-{link['id']}",
            width="stretch",
        ):
            self.family_action(
                lambda: family.unlink(user["id"], link["link_id"]),
                f"已撤回对 {link['account']} 的授权。",
            )

    def render_family(self) -> None:
        """家庭组主页：收到的请求、已关联家人、邀请入口、我发出的请求。"""
        user = current_user()
        if not user:
            return

        flash = st.session_state.pop(FAMILY_FLASH_KEY, None)
        if flash:
            st.success(flash)

        st.markdown(
            """
            <section class="hero">
              <div class="eyebrow">
                <span class="eyebrow-num">07</span>
                <span class="eyebrow-sep"></span>
                <span>家庭组</span>
              </div>
              <h1 class="hero-title">把家人的健康，<br>放在一个放心的位置</h1>
              <p class="hero-sub">
                一个家庭组由发起邀请的账号担任管理者，可以邀请多位家人；每个账号只能加入一个家庭组。
                发起时请选择你与对方的关系，对方确认后称谓会按双方注册的性别自动显示。
              </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        try:
            incoming = family.incoming(user["id"])
            members = family.active_members(user["id"])
            outgoing = family.outgoing(user["id"])
            owners = family.active_owners(user["id"])
        except family.FamilyError as exc:
            st.error(str(exc))
            return

        # 一个账号只能属于一个家庭组：已经有任何 active 关系时不能再接受新请求
        can_accept = not owners and not members

        if incoming:
            st.markdown(
                '<div class="rule-label"><span>收到的关联请求</span></div>',
                unsafe_allow_html=True,
            )
            for request in incoming:
                self.render_family_request(request, can_accept=can_accept)

        st.markdown(
            '<div class="rule-label"><span>我的家人</span></div>',
            unsafe_allow_html=True,
        )
        if not members:
            if owners:
                st.caption(
                    "你目前是家庭组成员，没有自己管理的家人。"
                    "一个账号只能加入一个家庭组，详见下方「能看到我数据的家人」。"
                )
            else:
                st.caption("还没有已授权的家人。在下面输入对方的账号发起邀请。")
        for member in members:
            self.render_family_member(member)

        if owners:
            owner_names = "、".join(link["account"] for link in owners)
            st.info(
                f"你已加入 {owner_names} 的家庭组。一个账号只能加入一个家庭组，"
                "如需自己作为管理者邀请家人，请先在页面底部解除现有授权。"
            )
        else:
            st.markdown(
                '<div class="rule-label"><span>邀请家人</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "你是家庭组管理者，可以邀请多位家人。对方会收到一条关联请求，"
                "只有他本人确认后，你才能看到他的数据。"
            )
            with st.form("family-invite-form"):
                account = st.text_input(
                    "家人账号",
                    placeholder="对方注册时用的账号",
                    max_chars=auth.ACCOUNT_MAX,
                )
                relation = st.selectbox(
                    "你与对方的关系",
                    options=list(family.RELATIONS),
                    index=None,
                    format_func=family.RELATION_OPTION_LABELS.get,
                    placeholder="请选择你与对方的关系",
                    help="只支持直系亲属：父母、子女、配偶。称谓按双方注册的性别自动显示。",
                )
                st.caption(
                    "只支持直系亲属：父母、子女、配偶。"
                    "例如选「子女」，对方接受后你看到对方是父亲/母亲，"
                    "对方看到你是儿子/女儿。"
                )
                message = st.text_input(
                    "留言（可选）",
                    max_chars=family.MESSAGE_MAX,
                    placeholder="例如：我是你女儿，想帮你记血压",
                )
                submitted = st.form_submit_button(
                    "发送关联请求", type="primary", width="stretch"
                )
            if submitted:
                try:
                    link = family.invite(user["id"], account, relation, message)
                except family.FamilyError as exc:
                    st.error(str(exc))
                else:
                    st.session_state[FAMILY_FLASH_KEY] = (
                        f"已向 {link['account']} 发送关联请求"
                        f"（关联后对方显示为你的{link['relation_title']}），"
                        "等待对方确认。"
                    )
                    st.rerun()

        pending_out = [
            link
            for link in outgoing
            if link["status"] in (family.STATUS_PENDING, family.STATUS_REJECTED)
        ]
        if pending_out:
            st.markdown(
                '<div class="rule-label"><span>我发出的请求</span></div>',
                unsafe_allow_html=True,
            )
            for link in pending_out:
                self.render_family_outgoing(link, can_invite=not owners)

        if owners:
            st.markdown(
                '<div class="rule-label"><span>能看到我数据的家人</span></div>',
                unsafe_allow_html=True,
            )
            st.caption("这些账号已经获得你的授权；不想继续时，随时可以解除。")
            for link in owners:
                self.render_family_owner(link)

    def render_family_notice(self) -> None:
        """对话页顶部的提醒：有未处理的关联请求时指个路。"""
        user = current_user()
        if not user:
            return
        try:
            count = family.incoming_count(user["id"])
        except family.FamilyError:
            return
        if not count:
            return
        st.markdown(
            f'<div class="family-notice">'
            f'<b>{count} 个家庭组关联请求</b>　有人想关联你的账号，接受后对方才能查看你的数据。'
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("去「家庭组」查看", key="family-notice-go", width="stretch"):
            goto_view("家庭组")
            st.rerun()

    @st.dialog("选择使用模式", width="small", on_dismiss=dismiss_mode_prompt)
    def render_mode_prompt(self) -> None:
        """60 岁以上用户登录时问一次；勾选「以后不再提醒」就把这次选择存成默认。"""
        user = current_user()
        if not user:
            st.session_state.pop(MODE_CHOICE_PENDING_KEY, None)
            return

        st.markdown(
            "您已满 **60 岁**。老年模式字更大、按钮更少，每条回答下面还能点「朗读」；"
            "正常模式信息更全。"
        )
        remember = st.checkbox("以后不再提醒（默认使用这次选择的模式）")
        st.caption("之后仍可在「设置」里随时切换当前模式和默认模式。")
        if st.button(
            "使用老年模式",
            key="mode-prompt-elder",
            type="primary",
            width="stretch",
        ):
            self.apply_mode_choice(True, remember)
        if st.button(
            "使用正常模式",
            key="mode-prompt-normal",
            width="stretch",
        ):
            self.apply_mode_choice(False, remember)

    def apply_mode_choice(self, use_elder: bool, remember: bool) -> None:
        """应用弹窗里的选择；勾了「不再提醒」才把这次选择写成默认模式。"""
        user = current_user()
        # 弹窗在 render_header 之前调用，这里改控件状态不会被「已实例化」拦住
        st.session_state["elder_mode"] = use_elder
        st.session_state.pop(MODE_CHOICE_PENDING_KEY, None)
        st.session_state.pop("quick_batch", None)
        if remember and user:
            try:
                updated = auth.update_mode_preferences(
                    user["id"],
                    default_mode=auth.MODE_ELDER if use_elder else auth.MODE_NORMAL,
                    prompt_on_login=False,
                )
            except auth.AuthError as exc:
                st.toast(f"偏好没保存成功：{exc}")
            else:
                st.session_state[AUTH_USER_KEY] = updated
                # 设置里的「默认模式」开关可能已经按旧值画过一轮，这里同步成新值
                st.session_state["default_elder_mode"] = (
                    updated["default_mode"] == auth.MODE_ELDER
                )
        st.rerun()

    def render_settings(self) -> None:
        # 面板内容多，用一个带 key 的容器包起来，样式上把上下间距收紧
        with st.container(key="settings-panel"):
            self.render_mode_switch()
            self.render_mode_defaults()
            st.divider()

            if self.elder:
                self.render_simple_settings()
            else:
                self.render_advanced_settings()

            st.divider()
            self.render_account()

    def render_simple_settings(self) -> None:
        """老人模式的设置：只留朗读语速和清空对话，模型参数全部收起来。"""
        st.caption("已开启：字更大、按钮更少。每条回答下面都有「朗读」。")
        st.session_state.speech_rate = st.slider(
            "朗读语速",
            SPEECH_RATE_MIN,
            SPEECH_RATE_MAX,
            float(st.session_state.speech_rate),
            0.05,
            help="1.0 是正常语速，往左更慢、往右更快。",
        )

        st.session_state.env_enabled = st.toggle(
            "结合当地天气与空气",
            value=st.session_state.get("env_enabled", True),
            help="开启后，回答会参考当地气温、湿度和空气质量。"
            "位置在对话页底部设置，不会写进对话历史。",
        )
        self.render_env_auto_toggle()
        self.render_env_status_caption()

        st.divider()
        st.caption("对话会自动保存，可在「历史」里翻看。")
        if st.button("清空当前对话", key="clear_chat", width="stretch"):
            self.clear_chat()
            st.rerun()

    def render_advanced_settings(self) -> None:
        # 用哪个模型完全由服务端 .env 的 LLM_MODEL 决定：页面上不显示模型名、
        # 也不提供切换入口，避免把后端配置暴露给终端用户。
        st.caption("回答风格")

        if llm.is_cloud():
            # 云端不支持 top_k，隐藏以免误解
            st.session_state.temperature = st.slider(
                "temperature · 随机性", 0.1, 1.2, st.session_state.temperature, 0.1
            )
            st.session_state.top_p = st.slider(
                "top_p · 用词范围", 0.1, 1.0, st.session_state.top_p, 0.1
            )
            st.session_state.max_tokens = st.slider(
                "回答长度上限", 128, 4096, max(1024, st.session_state.max_tokens), 64
            )
        else:
            st.session_state.temperature = st.slider(
                "temperature · 随机性", 0.1, 1.2, st.session_state.temperature, 0.1
            )
            st.session_state.top_p = st.slider(
                "top_p · 用词范围", 0.1, 1.0, st.session_state.top_p, 0.1
            )
            st.session_state.top_k = st.slider(
                "top_k · 候选词数", 1, 100, st.session_state.top_k, 1
            )
            st.session_state.max_tokens = st.slider(
                "回答长度上限", 128, 1024, st.session_state.max_tokens, 64
            )

        st.session_state.suggest_followups = st.toggle(
            "回答后推荐 3 条追问",
            value=st.session_state.suggest_followups,
            help="回答结束后再调一次模型，生成你可能想问的问题。关闭可省一次请求。",
        )

        st.session_state.env_enabled = st.toggle(
            "结合当地天气与空气",
            value=st.session_state.get("env_enabled", True),
            help="开启后，回答会参考当地气温、湿度、降水与空气质量；"
            "位置在对话页底部设置，不写进对话历史。",
        )
        self.render_env_auto_toggle()
        self.render_env_status_caption()

        st.divider()
        st.caption("对话会自动保存，可在「历史」里翻看和恢复。")
        if st.button("清空当前对话", key="clear_chat", width="stretch"):
            self.clear_chat()
            st.rerun()


    def quick_questions(self) -> List[str]:
        """当前这批开屏示例：进入主页面时抽一次，同一次停留内保持不变。"""
        count = ELDER_QUICK_QUESTION_COUNT if self.elder else QUICK_QUESTION_COUNT
        count = min(count, len(QUICK_QUESTION_POOL))
        batch = st.session_state.get("quick_batch")
        # 条数跟着模式走：切到老人模式后旧的那批要作废重抽
        if not batch or len(batch) != count:
            batch = random.sample(QUICK_QUESTION_POOL, count)
            st.session_state.quick_batch = batch
        return batch

    def refresh_quick_questions(self) -> None:
        """作废当前这批，下一次渲染重新抽样。"""
        st.session_state.pop("quick_batch", None)

    def render_empty_state(self) -> None:
        st.markdown(
            """
            <section class="hero">
              <div class="eyebrow">
                <span class="eyebrow-num">01</span>
                <span class="eyebrow-sep"></span>
                <span>开始</span>
              </div>
              <h1 class="hero-title">我是木木，<br>你的私人健康助手</h1>
              <p class="hero-sub">
                可以陪你梳理症状、看懂检查结果、整理日常用药。
                急症或持续加重的症状，请直接拨打 120 或前往急诊。
              </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="rule-label"><span>可以这样开始</span></div>',
            unsafe_allow_html=True,
        )

        with st.container(key="suggestions"):
            for i, question in enumerate(self.quick_questions()):
                if st.button(f"0{i + 1}　{question}", key=f"quick_{i}", width="stretch"):
                    st.session_state.current_question = question
                    # 这条已经点过，下次回到主页面换一批
                    self.refresh_quick_questions()

    def render_env_bar(self) -> None:
        """对话页的环境条。

        默认开启并自动定位：进入对话页就挂载浏览器定位组件（浏览器会问一次
        定位权限），拿到坐标后立刻取当地天气和空气。被拒或失败只在本次会话
        提示一次，之后可以点「重新定位」或直接输入城市。
        """
        if not st.session_state.get("env_enabled", True):
            return

        # 启动即获取：登录后落到对话页，第一轮就自动请求定位
        self.auto_start_location()

        if st.session_state.get("env_geo_pending"):
            payload = GEO(key="env-geo", default=None)
            if payload:
                stamp = payload.get("ts")
                if stamp and stamp != st.session_state.get("env_geo_ts"):
                    st.session_state.env_geo_ts = stamp
                    st.session_state.env_geo_pending = False
                    if payload.get("lat") and payload.get("lng"):
                        st.session_state.env_geo = (
                            round(float(payload["lat"]), 3),
                            round(float(payload["lng"]), 3),
                        )
                        st.session_state.env_geo_error = None
                        st.session_state.env_city = ""
                        st.session_state.env_data = None
                        st.session_state.env_fetched_at = 0.0
                    else:
                        code = str(payload.get("error") or "")
                        st.session_state.env_geo_error = {
                            "1": "权限被拒绝",
                            "2": "定位不可用",
                            "3": "定位超时",
                        }.get(code, "定位失败")
                    st.rerun()
            # 组件自己显示「正在获取定位…」和重试按钮，这一轮不再画环境条
            return

        env = st.session_state.get("env_data")
        coords = st.session_state.get("env_geo")
        city = str(st.session_state.get("env_city") or "").strip()

        if (coords or city) and env is None:
            with st.spinner("正在获取当地环境…"):
                env = self.ensure_environment()

        with st.container(
            key="envbar",
            horizontal=True,
            vertical_alignment="center",
            wrap=True,
            gap="small",
        ):
            if env and env.get("ok"):
                st.markdown(self.env_chip_html(env), unsafe_allow_html=True)
                with st.popover(
                    "环境", key="env-menu", type="tertiary", width="content"
                ):
                    self.render_env_actions()
                return

            if coords or city:
                message = (env or {}).get("error") or "环境数据暂时取不到，不影响对话"
                st.markdown(
                    '<div class="envbar envbar-empty"><span class="envbar-dot"></span>'
                    f"{html.escape(str(message))}</div>",
                    unsafe_allow_html=True,
                )
                if st.button("重试", key="env-retry", type="tertiary"):
                    st.session_state.env_fetched_at = 0.0
                    st.session_state.env_data = None
                    st.rerun()
            else:
                error = st.session_state.get("env_geo_error")
                hint = "结合当地天气和空气，回答会更贴合当下（会自动获取定位）"
                if error == "权限被拒绝":
                    hint = (
                        "没能获取定位（权限被拒绝）：在浏览器地址栏的站点设置里"
                        "允许定位后点「重新定位」，或直接输入城市"
                    )
                elif error:
                    hint = f"没能获取定位（{error}），可以点「重新定位」或输入城市"
                st.markdown(
                    f'<div class="envbar envbar-empty"><span class="envbar-dot"></span>'
                    f"{html.escape(hint)}</div>",
                    unsafe_allow_html=True,
                )
                if st.button("允许定位", key="env-locate", type="tertiary"):
                    st.session_state.env_geo_pending = True
                    st.session_state.env_geo_error = None
                    st.session_state.env_geo_asked = True
                    st.rerun()

            with st.popover(
                "输入城市", key="env-city-menu", type="tertiary", width="content"
            ):
                self.render_env_actions()

    def render_env_actions(self) -> None:
        """环境设置：城市输入、重新定位、关闭。只放在环境条的 popover 里。"""
        st.caption(
            "回答会参考当地气温、湿度和空气质量，只用来调整生活建议；"
            "位置不会写进对话历史。"
        )
        if st.session_state.get("env_auto_locate", True):
            st.caption("已开启自动定位：进入对话页就会获取位置和当地天气，可在「设置」里关闭。")
        env = st.session_state.get("env_data")
        if env and env.get("ok"):
            when = env.get("observed_at") or ""
            caption = environment.describe(env)
            if when:
                caption += f" · {when} 更新"
            st.caption(caption)

        with st.form("env-city-form"):
            typed = st.text_input(
                "城市",
                value=str(st.session_state.get("env_city") or ""),
                placeholder="输入城市名，例如：杭州",
                label_visibility="collapsed",
            )
            use = st.form_submit_button("用这个城市", width="stretch")
        if use:
            text = typed.strip()
            if not text:
                st.warning("先输入城市名。")
            else:
                hit = environment.lookup_city(text)
                if not hit:
                    st.warning("没找到这个城市，换个写法试试（例如「杭州」）。")
                else:
                    st.session_state.env_city = text
                    st.session_state.env_geo = (float(hit["lat"]), float(hit["lng"]))
                    st.session_state.env_geo_error = None
                    st.session_state.env_geo_pending = False
                    st.session_state.env_geo_ts = None
                    st.session_state.env_data = None
                    st.session_state.env_fetched_at = 0.0
                    st.rerun()

        if st.button("更新数据", key="env-refresh", width="stretch"):
            st.session_state.env_data = None
            st.session_state.env_fetched_at = 0.0
            st.rerun()

        if st.button("重新定位", key="env-relocate", width="stretch"):
            self.clear_environment()
            st.session_state.env_geo_pending = True
            st.rerun()

        if st.button("不再结合环境", key="env-disable", width="stretch"):
            st.session_state.env_enabled = False
            self.clear_environment()
            st.rerun()

    def render_hospital_panel(self, index: int) -> None:
        """就近医院：先按症状定科室与紧迫度，再按定位检索。"""
        self.render_back("对话", f"hospital-{index}", self.back_to_chat)
        messages = st.session_state.messages
        question = messages[index - 1]["content"] if index > 0 else ""
        info = triage.assess(question, blocked=bool(messages[index].get("redline")))

        st.markdown(
            f'<div class="triage-note {info["level"]}">'
            f'<div class="triage-head"><span class="triage-level">{info["label"]}</span>'
            f'<span class="triage-dept">建议科室 · {html.escape(info["department"])}</span></div>'
            f'<div class="triage-advice">{html.escape(info["advice"])}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        keyword = "医院|社区卫生服务中心"
        coords = st.session_state.get("geo")
        geo_error = st.session_state.get("geo_error")

        # 只在"还没定位且没失败"时挂载组件：组件每次挂载都会重新请求定位并上报，
        # 若无条件渲染会形成 rerun 死循环。
        if coords is None and geo_error is None:
            payload = GEO(key=f"geo-{index}", default=None)
            if isinstance(payload, dict):
                stamp = payload.get("ts")
                if stamp and stamp != st.session_state.get("geo_ts"):
                    st.session_state.geo_ts = stamp
                    if payload.get("lat"):
                        st.session_state.geo = (float(payload["lat"]), float(payload["lng"]))
                    else:
                        st.session_state.geo_error = str(payload.get("error"))
                    st.rerun()
        elif geo_error:
            st.caption("没能获取定位（可能是权限被拒或非 HTTPS）。可以手动输入城市：")
            if st.button("重新定位", key=f"relocate-{index}"):
                st.session_state.geo_error = None
                st.session_state.geo = None
                st.rerun()
        else:
            col_state, col_relocate = st.columns([4, 1])
            col_state.caption("按距离排序")
            if col_relocate.button("重新定位", key=f"relocate2-{index}"):
                st.session_state.geo = None
                st.session_state.geo_ts = None
                st.rerun()

        results = []
        if coords:
            results = cached_nearby(round(coords[0], 3), round(coords[1], 3), keyword)

        with st.form(key=f"cityform-{index}"):
            col_city, col_go = st.columns([3, 1])
            city = col_city.text_input(
                "城市",
                placeholder="或手动输入城市，例如：北京",
                label_visibility="collapsed",
            )
            submitted = col_go.form_submit_button("搜索")
        if submitted and city.strip():
            results = cached_by_city(city.strip(), keyword)

        if results:
            # 医院列表交给组件渲染：导航按钮需要 JS 判断 App 是否已安装
            NAV(items=results, key=f"nav-{index}", default=None)
        elif coords or (submitted and city.strip()):
            st.caption("没有检索到医院，可以换个城市，或直接打开地图搜索。")

        # 就诊流程与定位无关，始终提供
        st.markdown(
            '<div class="guide-ask">需要的话，我还可以说明完整的就诊流程。</div>',
            unsafe_allow_html=True,
        )
        col_ask, _ = st.columns([2, 5])
        if col_ask.button("看看就诊流程", key=f"guide-btn-{index}", width="stretch"):
            st.session_state.guide_open = index
            st.rerun()

        if st.session_state.get("guide_open") == index:
            steps = triage.visit_guide(info["level"], info["department"])
            rows = "".join(
                f'<li><span class="guide-title">{html.escape(title)}</span>'
                f'<span class="guide-detail">{html.escape(detail)}</span></li>'
                for title, detail in steps
            )
            st.markdown(f'<ol class="guide-list">{rows}</ol>', unsafe_allow_html=True)

    def render_message(self, index: int, message: Dict) -> None:
        role = message.get("role", "assistant")
        content = message.get("content", "") or ""
        with st.chat_message(role):
            if message.get("redline"):
                st.markdown(
                    '<div class="redline-flag">需要立即处理</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(content)
            note = message.get("env_note")
            if note:
                st.markdown(
                    f'<div class="env-note">已结合当时环境 · {html.escape(str(note))}</div>',
                    unsafe_allow_html=True,
                )
        if role == "assistant":
            # 只在最后一轮引导追问，避免整屏都是推荐
            if index == len(st.session_state.messages) - 1:
                self.render_followups(index, message.get("followups") or [])
            if self.elder:
                self.render_elder_actions(index, content)
            else:
                self.render_feedback(index, content)
            cta = message.get("checkin_cta")
            if message.get("checkin_brief") and cta:
                label = message.get("checkin_cta_label") or (
                    "前往数据看板" if cta == "数据看板" else "去打卡"
                )
                if st.button(
                    f"{label} →",
                    key=f"checkin-brief-cta-{index}",
                    type="secondary",
                    width="content",
                ):
                    goto_view(cta)
                    st.rerun()
            if st.session_state.get("hospital_index") == index:
                self.render_hospital_panel(index)

    def render_speak(self, text: str, key: str, big: bool = False) -> None:
        """语音朗读入口：把这段文字交给浏览器朗读，不联网、不经过服务器。

        读不出来（旧浏览器）时组件自己会给一句提示，这里不做兜底。
        """
        SPEAK(
            text=text,
            label="朗读这段话" if big else "朗读",
            big=big,
            rate=float(st.session_state.get("speech_rate") or SPEECH_RATE_DEFAULT),
            key=key,
            default=None,
        )

    def render_elder_actions(self, index: int, content: str) -> None:
        """老人模式下每条回答只留两个动作：朗读、就近医院。

        👍/👎 反馈在这里收起来——选项一多，长辈就不点了；
        少掉的反馈量由「附近医院」带来的实际帮助补回来。
        """
        emergency = bool(st.session_state.messages[index].get("redline"))
        with st.container(key=f"elder-actions-{index}"):
            col_speak, col_geo = st.columns(2, gap="small")
            with col_speak:
                self.render_speak(content, key=f"speak-{index}", big=True)
            with col_geo:
                if st.button(
                    "立即就医" if emergency else "附近医院",
                    key=f"elder_geo_{index}",
                    width="stretch",
                ):
                    st.session_state.hospital_index = index
                    st.rerun()

    def render_followups(self, index: int, suggestions: List[str]) -> None:
        if not suggestions:
            return
        st.markdown(
            '<div class="followup-label"><span>你可能想问</span></div>',
            unsafe_allow_html=True,
        )
        with st.container(key=f"followup-{index}", horizontal=True, wrap=True, gap="small"):
            for pos, question in enumerate(suggestions):
                if st.button(question, key=f"fu_{index}_{pos}"):
                    st.session_state.current_question = question

    def generate_followups(self, question: str, answer: str) -> List[str]:
        """回答结束后再调一次模型；失败或解析为空时退回通用追问。"""
        if not st.session_state.suggest_followups:
            return []
        if "连接错误" in answer or "未知错误" in answer:
            return []  # 模型没连上，别在报错下面再推荐问题

        text = llm.chat_once(
            build_messages(question, answer),
            model=self.resolve_model(),
            temperature=0.6,
            top_p=0.9,
            max_tokens=160,
            timeout=30,
        )

        suggestions = parse_followups(text, question) if text else []
        suggestions = suggestions or fallback_followups(question)
        # 老人模式少给一条：选项越少，越不用挑
        if self.elder:
            return suggestions[:ELDER_FOLLOWUP_LIMIT]
        return suggestions

    def render_feedback(self, index: int, content: str = "") -> None:
        """回答下面那排小字：朗读 / 有帮助 / 没帮助 / 附近医院。"""
        status = st.session_state.feedback_status.get(index)

        with st.container(key=f"feedback-{index}"):
            if st.session_state.pending_down == index:
                reason = st.radio(
                    "哪里不好",
                    DISLIKE_REASONS,
                    key=f"reason_{index}",
                    horizontal=True,
                    label_visibility="collapsed",
                )
                col_ok, col_cancel, _ = st.columns([1, 1, 5])
                with col_ok:
                    if st.button("提交", key=f"fb_ok_{index}", width="stretch"):
                        self.save_feedback(index, "down", reason)
                        st.session_state.pending_down = None
                        st.rerun()
                with col_cancel:
                    if st.button("取消", key=f"fb_no_{index}", width="stretch"):
                        st.session_state.pending_down = None
                        st.rerun()
                return

            col_speak, col_up, col_down, col_geo, _ = st.columns([1.2, 1, 1, 1.6, 2.4])
            with col_speak:
                self.render_speak(content, key=f"speak-{index}")
            with col_up:
                label = "已认可" if status == "up" else "有帮助"
                if st.button(label, key=f"fb_up_{index}", width="stretch"):
                    self.save_feedback(index, "up", "")
                    st.rerun()
            with col_down:
                label = "已反对" if status == "down" else "没帮助"
                if st.button(label, key=f"fb_down_{index}", width="stretch"):
                    st.session_state.pending_down = index
                    st.rerun()
            with col_geo:
                emergency = bool(st.session_state.messages[index].get("redline"))
                if st.button(
                    "立即就医" if emergency else "附近医院",
                    key=f"geo_btn_{index}",
                    width="stretch",
                ):
                    st.session_state.hospital_index = index
                    st.rerun()

    # ---------- 对话 ----------

    @staticmethod
    def checkin_intent(prompt: str) -> bool:
        """这句话是不是在让我分析打卡数据。"""
        text = prompt or ""
        if not _CHECKIN_INTENT_RE.search(text):
            return False
        if _CHECKIN_MEASUREMENT_RE.search(text) and not _CHECKIN_ANALYSIS_VERB_RE.search(
            text
        ):
            return False
        return True

    def latest_checkin_line(
        self, points: List[Dict], group_keys: Tuple[str, ...]
    ) -> str:
        """最近一次记录里「本次要看的指标」的数值，不再混入其他指标。"""
        latest = checkin.latest_values(points)
        keys: List[str] = []
        for group in checkin.GROUPS:
            if group.key in group_keys:
                keys.extend(group.series)
        if not keys:
            return ""

        bits: List[str] = []
        if "sbp" in keys or "dbp" in keys:
            sbp = latest.get("sbp")
            dbp = latest.get("dbp")
            if sbp and dbp:
                bits.append(f"血压 {sbp['value']:.0f}/{dbp['value']:.0f} mmHg")
            elif sbp or dbp:
                item = sbp or dbp
                series = checkin.SERIES[item["metric_key"]]
                bits.append(
                    f"{series.label} {self.format_metric(series.key, item['value'])}"
                )

        for key in keys:
            if key in ("sbp", "dbp"):
                continue
            item = latest.get(key)
            if not item or len(bits) >= 4:
                continue
            series = checkin.SERIES[key]
            value_text = self.format_metric(key, item["value"])
            if key == "bmi":
                text = f"BMI {value_text}"
            else:
                text = f"{series.label} {value_text}"
            if key == "glucose" and item.get("extra"):
                text += f"（{item['extra']}）"
            bits.append(text)

        if not bits:
            return ""
        return "最近一次记录：" + "、".join(bits) + "。"

    def checkin_missing_message(
        self,
        query: CheckinQuery,
        points: List[Dict],
        window_label: str,
        subject_label: str = "",
    ) -> Tuple[str, str]:
        """没有可分析记录时，说清楚是谁、哪个指标、哪段时间没有。"""
        owner = subject_label or "你"
        owner_poss = f"{owner}的"
        focus = f"「{query.focus_label}」" if query.focus_label else "打卡"
        if not points:
            return (
                f"{owner}还没有打卡记录，暂时没有可以分析的数据。\n\n"
                "先去「打卡」记一笔（血压、血糖、体重等都可以），"
                "之后再来问我，我就能帮你看趋势。\n\n"
                + CHECKIN_ANALYSIS_HINT,
                "打卡",
            )

        if query.group_keys:
            wanted = set()
            for key in query.group_keys:
                group = checkin.GROUP_MAP.get(key)
                if group:
                    wanted.update(group.series)
            related = [point for point in points if point["metric_key"] in wanted]
        else:
            related = points

        if not related:
            return (
                f"{owner_poss}打卡记录里还没有{focus}的数据。\n\n"
                "先去「打卡」补记一项，之后再来看趋势。\n\n"
                + CHECKIN_ANALYSIS_HINT,
                "打卡",
            )

        latest = max(point["recorded_at"] for point in related)
        return (
            f"{window_label}没有{owner_poss}{focus}记录。"
            f"最近一次相关记录在 {latest.strftime('%Y-%m-%d')}。\n\n"
            "可以去「数据看板」把时间范围放宽看看，或者先补记一次。\n\n"
            + CHECKIN_ANALYSIS_HINT,
            "数据看板",
        )

    def build_checkin_brief(
        self, user: Dict, prompt: str = "", subject_label: str = ""
    ) -> Tuple[str, str]:
        """用本地打卡数据拼一段简要分析，返回 (正文, 引导按钮目标视图)。

        不走模型：打卡记录是事实，让模型改写反而可能编造没有测过的数值。
        用户点名「血压」就只分析血压，说了「近一周」就只看这一周。
        subject_label 非空时表示分析的是家人，例如「你的父亲（father）」。
        """
        query = parse_checkin_query(prompt)
        try:
            points = checkin.list_points(user["id"])
        except checkin.CheckinError as exc:
            return (f"暂时读不到{subject_label or '你'}的打卡数据：{exc}", "")

        if not points:
            return self.checkin_missing_message(
                query, points, query.window_label, subject_label
            )

        now = datetime.now()
        if query.explicit_window:
            if query.window_days is None:
                scoped = points
                report_days: Optional[int] = None
                window_label = "全部记录"
            else:
                scoped = [
                    point
                    for point in points
                    if point["recorded_at"] >= now - timedelta(days=query.window_days)
                ]
                report_days = query.window_days
                window_label = query.window_label
            prefix = ""
            if not scoped:
                return self.checkin_missing_message(
                    query, points, window_label, subject_label
                )
        else:
            recent = [
                point
                for point in points
                if point["recorded_at"]
                >= now - timedelta(days=_DEFAULT_WINDOW_DAYS)
            ]
            if recent:
                scoped = recent
                report_days = _DEFAULT_WINDOW_DAYS
                window_label = f"近 {_DEFAULT_WINDOW_DAYS} 天"
                prefix = ""
            else:
                scoped = points
                report_days = None
                window_label = "全部记录"
                newest = max(point["recorded_at"] for point in points)
                prefix = (
                    f"近 {_DEFAULT_WINDOW_DAYS} 天没有新的打卡，我按全部记录"
                    f"（最近一次 {newest.strftime('%Y-%m-%d')}）来看。"
                )

        if query.group_keys:
            groups = [
                key
                for key in query.group_keys
                if any(
                    point["metric_key"] in checkin.GROUP_MAP[key].series
                    for point in scoped
                )
            ]
            missing_groups = [
                checkin.GROUP_MAP[key].label
                for key in query.group_keys
                if key not in groups
            ]
        else:
            groups = [
                group.key
                for group in checkin.GROUPS
                if any(point["metric_key"] in group.series for point in scoped)
            ]
            missing_groups = []

        if not groups:
            return self.checkin_missing_message(
                query, points, window_label, subject_label
            )

        report = checkin.analyze(scoped, groups, window_days=report_days)
        if not report.series:
            return self.checkin_missing_message(
                query, points, window_label, subject_label
            )

        # 一次要看多个指标时，每个指标单独成点，结论、数值和建议都只针对它；
        # 不再把多个指标揉成一段「整体结论」。
        multi = (
            len(query.group_keys) > 1
            if query.group_keys
            else len(groups) > 1
        )

        parts: List[str] = []
        if subject_label:
            parts.append(f"下面是{subject_label}的打卡数据分析。")
        if prefix:
            parts.append(prefix)

        if multi:
            if query.focus_label:
                parts.append(f"本次只看：{query.focus_label}。下面按指标分别看：")
            else:
                parts.append(f"{window_label}按指标分别看：")

            section_limit = 1 if self.elder else 2
            position = 0
            for key in groups:
                group_report = checkin.analyze(
                    scoped, [key], window_days=report_days
                )
                if not group_report.series:
                    continue
                group_label = checkin.GROUP_MAP[key].label
                group_conclusion, group_suggestions = checkin.fallback_analysis(
                    group_report
                )
                group_conclusion = group_conclusion.replace(
                    f"分析了 {len(group_report.series)} 项指标的记录",
                    f"分析了{group_label}的记录",
                )
                # 这句总括只保留在末尾一次，避免每个指标都重复一遍
                group_conclusion = group_conclusion.replace(
                    "单次测量会受情绪、饮食、睡眠和测量方式影响，趋势比单个数值更可靠。",
                    "",
                ).strip()

                position += 1
                parts.append(f"**{position}. {group_label}**")
                parts.append(group_conclusion)
                latest_line = self.latest_checkin_line(scoped, (key,))
                if latest_line:
                    parts.append(latest_line)
                if group_suggestions:
                    parts.append("可以先做这几件事：")
                    parts.append(
                        "\n".join(
                            f"- {tip}"
                            for tip in group_suggestions[:section_limit]
                        )
                    )

            if missing_groups:
                parts.append(f"{window_label}没有{'、'.join(missing_groups)}的记录。")
            if position:
                parts.append(
                    "单次测量会受情绪、饮食、睡眠和测量方式影响，趋势比单个数值更可靠。"
                )
            parts.append(CHECKIN_ANALYSIS_HINT)
            return ("\n\n".join(parts), "数据看板")

        conclusion, suggestions = checkin.fallback_analysis(report)
        if query.focus_label and groups:
            available_focus = "、".join(
                checkin.GROUP_MAP[key].label for key in groups
            )
            conclusion = conclusion.replace(
                f"分析了 {len(report.series)} 项指标的记录",
                f"分析了{available_focus}的记录",
            )
        limit = 2 if self.elder else 3
        suggestions = suggestions[:limit]

        if query.focus_label:
            parts.append(f"本次只看：{query.focus_label}。")
        parts.append(conclusion)
        if missing_groups:
            parts.append(f"{window_label}没有{'、'.join(missing_groups)}的记录。")
        latest_line = self.latest_checkin_line(scoped, tuple(groups))
        if latest_line:
            parts.append(latest_line)
        if suggestions:
            parts.append("**可以先做这几件事**")
            parts.append(
                "\n".join(
                    f"{index}. {tip}"
                    for index, tip in enumerate(suggestions, start=1)
                )
            )
        parts.append(CHECKIN_ANALYSIS_HINT)
        return ("\n\n".join(parts), "数据看板")

    def remember_turn(
        self,
        prompt: str,
        response: str,
        blocked: bool = False,
        followups: Optional[List[str]] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        """把一轮问答写进内存和数据库；分析类回答可以带额外标记。"""
        message = {
            "role": "assistant",
            "content": response,
            "redline": blocked,
            "followups": followups or [],
        }
        if extra:
            message.update(extra)
        st.session_state.messages.append(message)

        st.session_state.chat_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "user": prompt,
                "assistant": response,
                "params": {
                    "temperature": st.session_state.temperature,
                    "top_p": st.session_state.top_p,
                    "top_k": st.session_state.top_k,
                    "max_tokens": st.session_state.max_tokens,
                },
            }
        )
        # 每轮结束自动落盘，用户不用手动保存（对话始终属于当前登录账号）
        if not st.session_state.get("conv_id"):
            st.session_state.conv_id = history.new_id()
        viewer = current_user()
        history.save(
            st.session_state.conv_id,
            st.session_state.messages,
            owner_id=viewer["id"] if viewer else None,
        )

    def sync_chat_subject(self, viewer: Dict, member: Optional[Dict]) -> None:
        """对话里分析了谁的打卡，看板就切到谁，保证「前往数据看板」看到同一个人。"""
        target = (
            int(member["id"])
            if member and int(member["id"]) != int(viewer["id"])
            else None
        )
        current = st.session_state.get(FAMILY_SUBJECT_KEY)
        try:
            current = int(current) if current is not None else None
        except (TypeError, ValueError):
            current = None
        if current == target:
            return
        st.session_state[FAMILY_SUBJECT_KEY] = target
        if target is None:
            st.session_state.pop(FAMILY_SUBJECT_PICKER_KEY, None)
        else:
            st.session_state[FAMILY_SUBJECT_PICKER_KEY] = target
        self.clear_subject_caches()

    def answer_notice(self, prompt: str, content: str) -> None:
        """不走模型的一次性提示：家人未绑定、指代不清等。"""
        with st.chat_message("assistant"):
            st.markdown(content)
        self.remember_turn(prompt, content, followups=[])
        st.rerun()

    def answer_checkin_brief(
        self,
        prompt: str,
        subject: Optional[Dict] = None,
        subject_label: str = "",
    ) -> None:
        """命中打卡数据意图：直接读本地记录，给一段简要分析并引导去数据看板。

        subject 非空时分析的是家庭组成员；subject_label 是给用户看的称谓。
        """
        viewer = current_user()
        if not viewer:
            return
        target = subject or viewer
        content, cta = self.build_checkin_brief(
            target, prompt, subject_label=subject_label
        )
        with st.chat_message("assistant"):
            st.markdown(content)
        extra: Dict = {"checkin_brief": True}
        if cta:
            extra["checkin_cta"] = cta
        if subject:
            extra["checkin_cta_label"] = (
                "前往 TA 的数据看板" if cta == "数据看板" else "去 TA 的打卡页"
            )
        self.sync_chat_subject(viewer, subject)
        self.remember_turn(prompt, content, followups=[], extra=extra)
        st.rerun()

    def answer(self, prompt: str) -> None:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 追问药盒/说明书时，「一次吃几粒」是正常问法，不要按急症弹 120；
        # 只有此前确实上传过文件（有 doc context）才放开这一分类。
        followup_doc = st.session_state.get(CHAT_DOC_CONTEXT_KEY) or {}
        blocked, safety_reply = check_safety(
            prompt,
            skip=("用药剂量",) if followup_doc.get("text") else (),
        )

        # 打卡数据分析走本地规则，直接读该账号的真实记录；
        # 急症/红线问题仍然优先走安全兜底，不被数据意图截走。
        if not blocked and self.checkin_intent(prompt):
            viewer = current_user()
            member, subject_label, error = (
                resolve_chat_family_member(prompt, viewer["id"])
                if viewer
                else (None, "", "")
            )
            if error:
                self.answer_notice(prompt, error)
                return
            self.answer_checkin_brief(
                prompt, subject=member, subject_label=subject_label
            )
            return

        env: Optional[Dict] = None
        with st.chat_message("assistant"):
            if blocked:
                st.markdown(
                    '<div class="redline-flag">需要立即处理</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(safety_reply)
                full_response = safety_reply
            else:
                placeholder = st.empty()
                placeholder.markdown(
                    '<span class="typing">正在整理…</span>', unsafe_allow_html=True
                )

                # 环境信息可能已经放了半小时，回答前顺手刷新一次；
                # 取不到就不注入，回答照常生成。
                self.ensure_environment(force=self.env_stale())
                messages, env = self.inject_environment(
                    self.inject_knowledge(self.request_messages())
                )

                full_response = ""
                for chunk in self.stream_chat(messages, self.resolve_model()):
                    full_response += chunk
                    placeholder.markdown(full_response + " ▍")

                if not full_response.strip():
                    full_response = "这次没能生成回答，请稍后再试。"
                    placeholder.markdown(
                        f'<div class="error-inline">{full_response}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # 高温、重污染、雷雨这类情况补一条确定性的提醒，
                    # 模型没提到时也不会漏。普通天气不打扰。
                    tip = environment.notice(env)
                    if tip and not any(
                        key.lower() in full_response.lower() for key in tip["keys"]
                    ):
                        full_response += f"\n\n> **环境提醒**：{tip['text']}"
                    placeholder.markdown(full_response)

        # 追问引导：回答结束后再问一次模型，失败退回通用追问
        followups: List[str] = []
        if not blocked:
            slot = st.empty()
            slot.markdown(
                '<div class="followup-label"><span>正在整理追问…</span></div>',
                unsafe_allow_html=True,
            )
            followups = self.generate_followups(prompt, full_response)
            slot.empty()

        note = {"env_note": environment.describe(env)} if env else None
        self.remember_turn(prompt, full_response, blocked, followups, note)
        st.rerun()

    # ---------- 输入区：文字 + 文件 ----------

    def pending_doc(self) -> Optional[Dict]:
        pending = st.session_state.get(CHAT_DOC_KEY)
        return pending if isinstance(pending, dict) else None

    def reset_pending_doc(self) -> None:
        """丢掉待发送的附件，并换一个上传器 key，免得旧文件又冒出来。"""
        st.session_state.pop(CHAT_DOC_KEY, None)
        st.session_state[CHAT_DOC_PANEL_KEY] = False
        st.session_state[CHAT_DOC_EPOCH_KEY] = (
            int(st.session_state.get(CHAT_DOC_EPOCH_KEY) or 0) + 1
        )

    @staticmethod
    def doc_chip_html(pending: Dict) -> str:
        name = html.escape(str(pending.get("name") or "文件"))
        meta = " · ".join(
            str(part)
            for part in (pending.get("kind_label"), pending.get("size_label"))
            if part
        )
        bad = "" if pending.get("supported") else " bad"
        return (
            f'<div class="attach-chip{bad}">'
            '<span class="attach-dot"></span>'
            f'<span class="attach-name">{name}</span>'
            f'<span class="attach-meta">{html.escape(meta)}</span>'
            "</div>"
        )

    def store_pending_doc(self, name: str, data: bytes) -> None:
        """把一份附件收进会话：上传的文件和刚拍的照片都走这里。

        不支持的类型也留着，好让用户看到原因；面板立刻收起、换一个上传器 key，
        下一次打开是干净的，不会自动再挂上这份文件。
        """
        data = data or b""
        info = document.inspect(name, len(data))
        st.session_state[CHAT_DOC_KEY] = {
            "name": info.name,
            "size": info.size,
            "kind": info.kind,
            "kind_label": info.kind_label,
            "size_label": info.size_label,
            "supported": info.supported,
            "reason": info.reason,
            "data": data if info.supported else b"",
        }
        st.session_state[CHAT_DOC_PANEL_KEY] = False
        st.session_state[CHAT_DOC_EPOCH_KEY] = (
            int(st.session_state.get(CHAT_DOC_EPOCH_KEY) or 0) + 1
        )
        st.rerun()

    def accept_upload(self, uploaded) -> None:
        """把上传器里选中的文件收进会话。"""
        data = uploaded.getvalue() if uploaded is not None else b""
        self.store_pending_doc(str(getattr(uploaded, "name", "") or ""), data)

    def accept_capture(self, shot) -> None:
        """把摄像头拍下的画面当成一张图片收进会话。

        拍照组件给的文件名是英文的 camera-input.jpg，换成带时间的「拍照-…」，
        附件条和对话里都能看出这是刚拍的，不是从相册里选的。
        """
        data = shot.getvalue() if shot is not None else b""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.store_pending_doc(f"拍照-{stamp}.jpg", data)

    def render_attach_strip(self) -> Optional[Dict]:
        """输入框上方那一条：上传 / 拍照入口、已选文件、类型不支持的提示。"""
        pending = self.pending_doc()
        panel_open = bool(st.session_state.get(CHAT_DOC_PANEL_KEY))

        with st.container(
            key="attachbar",
            horizontal=True,
            vertical_alignment="center",
            gap="small",
            wrap=True,
        ):
            if st.button(
                "收起" if panel_open else "📎 上传 / 拍照",
                key="attach-toggle",
                type="tertiary",
            ):
                st.session_state[CHAT_DOC_PANEL_KEY] = not panel_open
                st.rerun()
            if pending:
                st.markdown(
                    self.doc_chip_html(pending), unsafe_allow_html=True
                )
                if st.button("移除", key="attach-remove", type="tertiary"):
                    self.reset_pending_doc()
                    st.rerun()
            else:
                st.markdown(
                    '<div class="attach-hint">支持 PDF / Word / Excel / 图片，'
                    f"单个不超过 {CHAT_DOC_MAX_MB} MB</div>",
                    unsafe_allow_html=True,
                )

        if pending and not pending.get("supported"):
            st.markdown(
                f'<div class="attach-error">{html.escape(str(pending.get("reason") or "文件类型暂不支持"))}</div>',
                unsafe_allow_html=True,
            )

        if panel_open:
            epoch = int(st.session_state.get(CHAT_DOC_EPOCH_KEY) or 0)
            source = st.segmented_control(
                "附件来源",
                options=CHAT_DOC_SOURCES,
                format_func=lambda value: (
                    "📁 从文件上传" if value == "file" else "📷 直接拍照"
                ),
                default=st.session_state.get(CHAT_DOC_SOURCE_KEY) or "file",
                label_visibility="collapsed",
                width="content",
                required=True,
            )
            # 记下来：拍完一张面板会收起，下次打开还停在拍照这一档
            st.session_state[CHAT_DOC_SOURCE_KEY] = source or "file"

            if source == "camera":
                st.markdown(
                    '<div class="attach-hint">点「拍照」后浏览器会请求使用摄像头，'
                    "允许后就能拍，拍完自动收进附件；拍药盒、化验单时把文字拍清楚。"
                    "电脑没有摄像头或不允许时，请改用「从文件上传」。</div>",
                    unsafe_allow_html=True,
                )
                with st.container(key="attach-camera"):
                    shot = st.camera_input(
                        "拍照",
                        key=f"chat-cam-{epoch}",
                        label_visibility="collapsed",
                        resolution="1080p",
                    )
                if shot is not None:
                    self.accept_capture(shot)
            else:
                uploaded = st.file_uploader(
                    "上传文件",
                    key=f"chat-doc-upload-{epoch}",
                    label_visibility="collapsed",
                    max_upload_size=CHAT_DOC_MAX_MB,
                    help=(
                        f"支持 {document.SUPPORTED_HINT}；"
                        "PDF 和图片会尽量识别里面的文字，扫描件也可以试。"
                    ),
                )
                if uploaded is not None:
                    self.accept_upload(uploaded)
        return pending

    def render_composer(self) -> Tuple[str, Optional[Dict], bool]:
        """底部输入区：附件条 + 一行文字 + 语音键 + 发送键。

        发送键只在挂着不支持的文件时变灰。空输入时仍然可以点，提交后什么都不做——
        表单里的控件不会触发 rerun，若按「有没有文字」去禁用提交键，它会一直停在
        第一次渲染的置灰状态，连回车都提交不了（Streamlit 的表单限制）。
        """
        with st.bottom:
            with st.container(key="composer"):
                pending = self.render_attach_strip()
                supported = bool(pending and pending.get("supported"))
                if pending and not supported:
                    placeholder = "请先移除不支持的文件"
                elif supported:
                    placeholder = "可以补充一句想问的问题，直接发送就开始分析"
                else:
                    placeholder = "说说你的情况，比如凌晨总是醒来"

                with st.form("chat-composer-form", clear_on_submit=True, border=False):
                    col_text, col_actions = st.columns(
                        [7.2, 1.9], gap="small", vertical_alignment="bottom"
                    )
                    with col_text:
                        typed = st.text_input(
                            "消息",
                            placeholder=placeholder,
                            label_visibility="collapsed",
                            key="chat-composer-text",
                        )
                    with col_actions:
                        # 语音键和发送键放在一个零间距的行里：圆形麦克风紧贴发送键
                        with st.container(
                            key="send-group",
                            horizontal=True,
                            gap=None,
                            vertical_alignment="center",
                        ):
                            with st.container(key="voice-slot"):
                                VOICE(key="chat-voice", default=None)
                            submitted = st.form_submit_button(
                                "发送",
                                type="primary",
                                width="stretch",
                                disabled=bool(pending and not supported),
                            )

                st.markdown(
                    f'<div class="compliance">{COMPLIANCE_LINE}</div>',
                    unsafe_allow_html=True,
                )
        return (typed or "").strip(), pending, bool(submitted)

    def collect_prompt(self) -> Tuple[str, Optional[Dict]]:
        typed, pending, submitted = self.render_composer()
        if submitted:
            doc = pending if (pending and pending.get("supported")) else None
            self.reset_pending_doc()
            return typed, doc

        queued = st.session_state.get("current_question")
        if queued:
            del st.session_state["current_question"]
            return str(queued), None
        return "", None

    # ---------- 上传文件的解读 ----------

    def remember_doc_context(self, name: str, text: str) -> None:
        text = str(text or "").strip()
        if not text:
            return
        st.session_state[CHAT_DOC_CONTEXT_KEY] = {
            "name": name,
            "text": text[:CHAT_DOC_FOLLOWUP_CHARS],
        }

    def classify_document(self, messages: List[Dict]) -> str:
        """拿不准文件是不是健康内容时问一次模型；只让它回 YES / NO。"""
        return llm.chat_once(
            messages,
            model=self.resolve_model(),
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            max_tokens=16,
            timeout=30,
        )

    def stream_document(self, messages: List[Dict]):
        """解读文件时用更低的温度：报告类内容不需要发挥。"""
        max_tokens = min(int(st.session_state.max_tokens or 600), 900)
        if self.elder:
            max_tokens = min(max_tokens, ELDER_MAX_TOKENS)
        yield from llm.chat_stream(
            messages,
            model=self.resolve_model(),
            temperature=0.3,
            top_p=0.9,
            top_k=st.session_state.top_k,
            max_tokens=max_tokens,
        )

    def stream_vision(self, messages: List[Dict]):
        """图片解读：视觉模型需要更多输出空间，老人模式仍然收短。"""
        max_tokens = ELDER_MAX_TOKENS if self.elder else 900
        yield from llm.chat_stream(
            messages,
            model=self.resolve_model(),
            temperature=0.3,
            top_p=0.9,
            top_k=st.session_state.top_k,
            max_tokens=max_tokens,
        )

    def answer_image_vision(
        self, name: str, data: bytes, question: str, slot
    ) -> Tuple[str, bool]:
        """优先让视觉模型直接看图片；不可用或失败时返回 ("", False) 交给 OCR 兜底。"""
        if not data or not llm.is_cloud():
            return "", False
        uri = document.vision_data_uri(name, data)
        if not uri:
            return "", False

        try:
            reply = llm.chat_once(
                document.vision_classifier_messages(uri, question),
                model=self.resolve_model(),
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                max_tokens=32,
                timeout=45,
            )
        except Exception:  # noqa: BLE001 - 视觉失败就回退 OCR
            return "", False
        verdict = document.parse_classifier_reply(reply)
        if verdict is False:
            return FILE_IRRELEVANT_REPLY, True
        if verdict is not True:
            return "", False

        slot.markdown(
            '<span class="typing">正在看这张图片…</span>',
            unsafe_allow_html=True,
        )
        messages = document.vision_analysis_messages(
            name, uri, question, elder=self.elder
        )
        full_response = ""
        for chunk in self.stream_vision(messages):
            full_response += chunk
            slot.markdown(full_response + " ▍")

        if not full_response.strip():
            return "", False
        slot.markdown(full_response)
        if "连接错误" not in full_response and "未知错误" not in full_response:
            self.remember_doc_context(name, "图片解读：\n" + full_response)
        return full_response, True

    def answer_file_ocr(self, name: str, data: bytes, prompt: str, slot) -> str:
        """文字提取兜底：PDF / Word / Excel，以及视觉不可用时的图片。"""
        extraction = document.extract(name, data)
        if not extraction.ok:
            response = (
                "**这份文件没能读出可用内容**\n\n"
                f"{extraction.error}"
            )
            slot.markdown(response)
            return response

        slot.markdown(
            '<span class="typing">正在判断文件内容…</span>',
            unsafe_allow_html=True,
        )
        relevance = document.judge(
            extraction.text,
            ask=self.classify_document,
            question=prompt,
        )
        if not relevance.relevant:
            slot.markdown(FILE_IRRELEVANT_REPLY)
            return FILE_IRRELEVANT_REPLY

        slot.markdown(
            '<span class="typing">正在整理这份文件的解读…</span>',
            unsafe_allow_html=True,
        )
        messages = document.analysis_messages(
            name, extraction, prompt, elder=self.elder
        )
        response = ""
        for chunk in self.stream_document(messages):
            response += chunk
            slot.markdown(response + " ▍")

        if not response.strip():
            response = "这次没能生成解读，请稍后再试。"
            slot.markdown(
                f'<div class="error-inline">{response}</div>',
                unsafe_allow_html=True,
            )
        else:
            slot.markdown(response)
            # 只给健康相关的文件留追问背景；无关文件不重复喂给模型
            if "连接错误" not in response and "未知错误" not in response:
                self.remember_doc_context(name, extraction.text)
        return response

    def answer_file(self, prompt: str, doc: Dict) -> None:
        """解读一份上传的文件；不相关的文件只给固定提示，不交给模型发挥。"""
        # 新文件进来，上一份文件留下的追问背景就作废
        st.session_state.pop(CHAT_DOC_CONTEXT_KEY, None)
        name = str(doc.get("name") or "文件")
        meta = " · ".join(
            str(part)
            for part in (doc.get("kind_label"), doc.get("size_label"))
            if part
        )
        # 文件名放进代码段，避免文件名里的 _ / * 被 markdown 当成格式
        safe_name = name.replace("`", "'").strip() or "文件"
        label = f"**已上传文件**：`{safe_name}`"
        if meta:
            label += f"（{html.escape(meta)}）"
        user_content = f"{label}\n\n{prompt}" if prompt else label

        st.session_state.messages.append(
            {"role": "user", "content": user_content}
        )
        with st.chat_message("user"):
            st.markdown(user_content)

        # 药盒解读里「一次吃几粒」是正常问法，不要按急症弹 120；
        # 自伤、急症、药物过量仍然照常拦截。
        blocked, safety_reply = (
            check_safety(prompt, skip=("用药剂量",)) if prompt else (False, "")
        )

        with st.chat_message("assistant"):
            if blocked:
                st.markdown(
                    '<div class="redline-flag">需要立即处理</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(safety_reply)
                full_response = safety_reply
            else:
                slot = st.empty()
                slot.markdown(
                    '<span class="typing">正在读取文件…</span>',
                    unsafe_allow_html=True,
                )
                full_response = ""
                handled = False
                # 图片优先交给视觉模型直接看：药盒、伤口、化验单照片比 OCR 稳得多；
                # 视觉不可用或失败时，answer_file_ocr 还会走原来的文字提取兜底。
                if doc.get("kind") == "image":
                    full_response, handled = self.answer_image_vision(
                        name, doc.get("data") or b"", prompt, slot
                    )
                if not handled:
                    full_response = self.answer_file_ocr(
                        name, doc.get("data") or b"", prompt, slot
                    )

        self.remember_turn(
            user_content,
            full_response,
            blocked,
            followups=[],
            extra={"file_name": name},
        )
        st.rerun()

    # ---------- 反馈与历史 ----------

    def clear_chat(self) -> None:
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.feedback_status = {}
        st.session_state.pending_down = None
        st.session_state.conv_id = None
        st.session_state.hospital_index = None
        st.session_state.pop("current_question", None)
        # 附件和文件正文属于这一轮对话，清空时一并丢掉
        st.session_state.pop(CHAT_DOC_CONTEXT_KEY, None)
        self.reset_pending_doc()
        # 清空后回到主页面，示例问题换一批
        self.refresh_quick_questions()

    def save_feedback(self, index: int, feedback: str, reason: str = "") -> bool:
        """保存用户对某条回答的反馈（👍/👎 互斥，改判追加新记录）。"""
        messages = st.session_state.messages
        if index <= 0 or index >= len(messages):
            return False
        user_msg = messages[index - 1]
        assistant_msg = messages[index]
        if user_msg["role"] != "user" or assistant_msg["role"] != "assistant":
            return False

        record = {
            "id": str(uuid.uuid4()),
            "prompt": user_msg["content"],
            "response": assistant_msg["content"],
            "feedback": feedback,
            "reason": reason,
            "model": self.resolve_model(),
            "timestamp": datetime.now().isoformat(),
            "params": {
                "temperature": st.session_state.temperature,
                "top_p": st.session_state.top_p,
                "top_k": st.session_state.top_k,
            },
        }

        try:
            FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(FEEDBACK_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            st.session_state.feedback_status[index] = feedback
            return True
        except Exception as exc:  # noqa: BLE001
            st.toast(f"反馈保存失败：{exc}")
            return False

    # ---------- 历史对话 ----------

    def render_history(self) -> None:
        viewer = current_user()
        if not viewer:
            return
        subject = self.resolve_subject(warn=True)
        if not subject:
            return
        proxy = int(subject["id"]) != int(viewer["id"])

        st.markdown(
            """
            <section class="hero">
              <div class="eyebrow">
                <span class="eyebrow-num">06</span>
                <span class="eyebrow-sep"></span>
                <span>历史</span>
              </div>
              <h1 class="hero-title">过去的对话</h1>
              <p class="hero-sub">每轮问答都会自动保存。打开任意一条可以继续追问，上下文会一并恢复。</p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        if proxy:
            self.render_proxy_banner(subject, "历史")

        items = history.list_all(subject["id"])
        if not items:
            st.caption("还没有对话记录。回到「对话」问点什么，会自动出现在这里。")
            return

        if proxy:
            st.caption("只读查看家人的问询记录，不会恢复成你的对话，也不会写入你的历史。")
            for item in items:
                stamp = datetime.fromtimestamp(item["updated_at"]).strftime(
                    "%m-%d %H:%M"
                )
                with st.expander(
                    f"{item['title']}　·　{stamp}　·　{item['count']} 条消息"
                ):
                    data = history.load(item["id"], subject["id"])
                    if not data:
                        st.caption("这条记录读不出来了。")
                        continue
                    for message in data.get("messages") or []:
                        role = (
                            "user"
                            if message.get("role") == "user"
                            else "assistant"
                        )
                        with st.chat_message(role):
                            st.markdown(message.get("content") or "")
            return

        for item in items:
            stamp = datetime.fromtimestamp(item["updated_at"]).strftime("%m-%d %H:%M")
            col_main, col_open, col_del = st.columns(
                [6, 1, 1], vertical_alignment="center"
            )
            with col_main:
                st.markdown(
                    f'<div class="hist-item">'
                    f'<div class="hist-title">{html.escape(item["title"])}</div>'
                    f'<div class="hist-meta">{stamp}　·　{item["count"]} 条消息</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_open:
                if st.button("打开", key=f"open-{item['id']}", width="stretch"):
                    self.open_conversation(item["id"])
            with col_del:
                if st.button("删除", key=f"del-{item['id']}", width="stretch"):
                    history.delete(item["id"], viewer["id"])
                    st.rerun()

    def open_conversation(self, conv_id: str) -> None:
        viewer = current_user()
        if not viewer:
            return
        data = history.load(conv_id, viewer["id"])
        if not data:
            st.toast("这条记录读不出来了")
            return
        st.session_state.messages = data.get("messages") or []
        st.session_state.conv_id = conv_id
        st.session_state.feedback_status = {}
        st.session_state.hospital_index = None
        # 历史里没有文件正文，换回旧对话时不再引用上一份文件
        st.session_state.pop(CHAT_DOC_CONTEXT_KEY, None)
        self.reset_pending_doc()
        goto_view("对话")
        st.rerun()

    # ---------- 用户自查 ----------

    def render_selfcheck(self) -> None:
        st.markdown(
            """
            <section class="hero">
              <div class="eyebrow">
                <span class="eyebrow-num">02</span>
                <span class="eyebrow-sep"></span>
                <span>自查</span>
              </div>
              <h1 class="hero-title">填一份数据，<br>带走一份可存档的报告</h1>
              <p class="hero-sub">
                身高体重、体温、睡眠、血压、血糖。参考区间由本地规则判定，AI 只负责把结果写成人话；
                模型不可用时也能出报告。留空的项不会被计入。
              </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        with st.form("selfcheck_form"):
            st.markdown('<div class="form-section"><span>基本</span></div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2, gap="medium")
            age = col_a.number_input("年龄（岁）", 1, 120, 30)
            sex = col_b.selectbox("性别", ["男", "女"])

            st.markdown(
                '<div class="form-section"><span>体型与体温</span></div>',
                unsafe_allow_html=True,
            )
            col_a, col_b = st.columns(2, gap="medium")
            height = col_a.number_input("身高（cm）", 100.0, 250.0, 170.0, 0.5, format="%.1f")
            weight = col_b.number_input("体重（kg）", 20.0, 200.0, 65.0, 0.5, format="%.1f")

            col_a, col_b = st.columns(2, gap="medium")
            temp = col_a.number_input("体温（°C）", 34.0, 43.0, 36.5, 0.1, format="%.1f")
            heart_rate = col_b.number_input("心率（次/分 · 可留空）", 30, 220, None)

            st.markdown(
                '<div class="form-section"><span>血压与血糖</span></div>',
                unsafe_allow_html=True,
            )
            col_a, col_b = st.columns(2, gap="medium")
            sbp = col_a.number_input("收缩压（mmHg）", 70, 260, 120)
            dbp = col_b.number_input("舒张压（mmHg）", 40, 160, 78)

            col_a, col_b = st.columns(2, gap="medium")
            glucose = col_a.number_input(
                "血糖（mmol/L · 可留空）", 2.0, 35.0, None, 0.1, format="%.1f"
            )
            glucose_type = col_b.selectbox("血糖测量类型", ["空腹", "餐后2小时", "随机"])

            st.markdown('<div class="form-section"><span>作息</span></div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2, gap="medium")
            sleep_start = col_a.time_input("入睡时间", value=time(23, 0))
            sleep_end = col_b.time_input("起床时间", value=time(7, 0))

            submitted = st.form_submit_button("生成分析报告")
            st.caption(f"警报线：{ALARM_LINES_TEXT}。越过会立刻弹出就医提醒，并给出附近医院。")

        if submitted:
            data = {
                "age": age,
                "sex": sex,
                "height_cm": height,
                "weight_kg": weight,
                "temp_c": temp,
                "heart_rate": heart_rate,
                "sleep_start": sleep_start,
                "sleep_end": sleep_end,
                "sbp": sbp,
                "dbp": dbp,
                "glucose": glucose,
                "glucose_type": glucose_type,
            }
            result = evaluate(data)
            # 规则判定出"建议尽快就医"的指标：报告照常生成，同时挂一个弹框（run() 里统一弹出）
            if result.urgent_items:
                st.session_state.urgent_result = result
                st.session_state.urgent_open = True
            with st.spinner("正在生成分析并排版报告…"):
                conclusion, suggestions = self.analyze_selfcheck(result)
                pdf = render_pdf(
                    result,
                    conclusion,
                    suggestions,
                    profile=f"{age} 岁 · {sex}",
                )
            st.session_state.selfcheck_result = {
                "result": result,
                "conclusion": conclusion,
                "suggestions": suggestions,
                "pdf": pdf,
                "at": datetime.now(),
            }

        cached = st.session_state.get("selfcheck_result")
        if not cached:
            st.caption("填好后点「生成分析报告」，报告可下载为 PDF。")
            return
        self.render_selfcheck_result(cached)

    def analyze_selfcheck(self, result) -> Tuple[str, List[str]]:
        """让模型把判定结果写成人话；失败或解析为空时用规则兜底。"""
        text = llm.chat_once(
            [{"role": "user", "content": build_prompt(result, elder=self.elder)}],
            model=self.resolve_model(),
            temperature=0.4,
            top_p=0.9,
            max_tokens=ELDER_SELFCHECK_MAX_TOKENS if self.elder else 420,
            timeout=60,
        )

        conclusion, suggestions = parse_analysis(text)
        if not conclusion:
            return fallback_analysis(result)

        # 模型给的条数不足时，用规则建议补齐，保证报告里总有可执行的下一步
        if len(suggestions) < 3:
            _, fallback_suggestions = fallback_analysis(result)
            for tip in fallback_suggestions:
                if len(suggestions) >= 3:
                    break
                if tip not in suggestions:
                    suggestions.append(tip)
        # 老人模式只留 3 条：建议一多，反而一条都记不住
        if self.elder:
            suggestions = suggestions[:ELDER_SELFCHECK_SUGGESTIONS]
        return conclusion, suggestions

    def render_selfcheck_result(self, cached: Dict) -> None:
        result = cached["result"]
        at = cached["at"]

        rows = "".join(
            f"<tr>"
            f"<td>{m.label}</td>"
            f'<td class="num">{m.value}</td>'
            f"<td>{m.reference}</td>"
            f'<td class="st {m.status}">{m.status_label}</td>'
            f"</tr>"
            for m in result.metrics
        )
        st.markdown(
            f"""
            <div class="check-banner {result.level}">
              <span class="check-banner-label">整体评估</span>
              <span class="check-banner-value">{result.level_label}</span>
            </div>
            <table class="check-table">
              <thead><tr><th>项目</th><th>数值</th><th>参考范围</th><th>状态</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="rule-label"><span>分析结论</span></div>', unsafe_allow_html=True
        )
        with st.container(key="check-conclusion"):
            st.markdown(cached["conclusion"])

        st.markdown('<div class="rule-label"><span>建议</span></div>', unsafe_allow_html=True)
        with st.container(key="check-suggestions"):
            st.markdown(
                "\n".join(
                    f"{i}. {item}" for i, item in enumerate(cached["suggestions"], start=1)
                )
            )

        # 整份报告可以听一遍：老人模式下把它做成大按钮
        with st.container(key="check-speak"):
            self.render_speak(
                self.selfcheck_speech(cached),
                key="speak-selfcheck",
                big=self.elder,
            )

        st.caption("PDF 含逐项解读与复测计划，比本页更详细。")
        st.download_button(
            "下载 PDF 报告",
            data=cached["pdf"],
            file_name=f"健康自查报告_{at.strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )

        # 弹框关掉后还能再打开，避免一失手就得重新生成报告
        if result.urgent_items:
            col_open, _ = st.columns([2, 3])
            if col_open.button("查看附近医院", key="urgent-reopen", width="stretch"):
                st.session_state.urgent_result = result
                st.session_state.urgent_open = True
                st.rerun()

    @staticmethod
    def selfcheck_speech(cached: Dict) -> str:
        """把自查报告摊平成一段能顺口读出来的话。"""
        result = cached["result"]
        parts = [f"整体评估：{result.level_label}。", cached["conclusion"]]
        for i, tip in enumerate(cached["suggestions"], start=1):
            parts.append(f"第{i}条建议：{tip}")
        return "\n".join(parts)

    # ---------- 打卡 ----------

    @staticmethod
    def format_metric(key: str, value: float, with_unit: bool = True) -> str:
        """按指标的精度格式化数值；BMI 的"单位"就是 BMI 本身，不重复写。"""
        series = checkin.SERIES.get(key)
        if series is None:
            return f"{value:g}"
        text = f"{value:.{series.decimals}f}"
        if with_unit and series.unit and series.unit != "BMI":
            text += f" {series.unit}"
        return text

    @classmethod
    def format_checkin_items(cls, items: List[Dict]) -> str:
        parts = []
        for item in items:
            text = f"{item['label']} {cls.format_metric(item['key'], item['value'])}"
            if item["key"] == "glucose" and item.get("extra"):
                text += f"（{item['extra']}）"
            parts.append(text)
        return "　·　".join(parts) if parts else "（没有数值）"

    def render_checkin(self) -> None:
        """打卡页：只填这次测到的项目，保存后写进 MySQL（可为已授权的家人代记）。"""
        user = current_user()
        if not user:
            return
        subject = self.resolve_subject(warn=True)
        if not subject:
            return

        flash = st.session_state.pop(CHECKIN_FLASH_KEY, None)
        if flash:
            st.success(flash)

        # 上一轮保存成功时清空输入框；必须在控件实例化之前改 session_state
        if st.session_state.pop(CHECKIN_RESET_KEY, False):
            for key in CHECKIN_INPUT_KEYS:
                st.session_state.pop(key, None)

        st.markdown(
            """
            <section class="hero">
              <div class="eyebrow">
                <span class="eyebrow-num">03</span>
                <span class="eyebrow-sep"></span>
                <span>打卡</span>
              </div>
              <h1 class="hero-title">记一笔，<br>把身体的变化留下来</h1>
              <p class="hero-sub">
                只填这次测到的项目，留空的不会被记录；身高和体重同时填时，BMI 由系统自动算好。
                数据会保存到当前选择的账号下。
              </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        with st.container(key="checkin-form", border=True):
            st.markdown(
                '<div class="form-section"><span>身高与体重</span></div>',
                unsafe_allow_html=True,
            )
            col_height, col_weight = st.columns(2, gap="medium")
            height = col_height.number_input(
                "身高（cm）",
                min_value=80.0,
                max_value=250.0,
                value=None,
                step=0.5,
                format="%.1f",
                placeholder="留空",
                key="ci-height",
            )
            weight = col_weight.number_input(
                "体重（kg）",
                min_value=15.0,
                max_value=300.0,
                value=None,
                step=0.1,
                format="%.1f",
                placeholder="留空",
                key="ci-weight",
            )
            if height and weight:
                bmi = checkin.bmi_of(height, weight)
                _, bmi_label = checkin.bmi_status(bmi)
                st.markdown(
                    f'<div class="check-banner info">'
                    f'<span class="check-banner-label">自动计算</span>'
                    f'<span class="check-banner-value">BMI {bmi:.1f} · {bmi_label}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="form-section"><span>血压</span></div>',
                unsafe_allow_html=True,
            )
            col_sbp, col_dbp = st.columns(2, gap="medium")
            sbp = col_sbp.number_input(
                "收缩压（mmHg）",
                min_value=70,
                max_value=260,
                value=None,
                step=1,
                placeholder="留空",
                key="ci-sbp",
            )
            dbp = col_dbp.number_input(
                "舒张压（mmHg）",
                min_value=40,
                max_value=160,
                value=None,
                step=1,
                placeholder="留空",
                key="ci-dbp",
            )
            st.caption("血压要同时填两个数，这样才能画出完整的一条曲线。")

            st.markdown(
                '<div class="form-section"><span>血糖</span></div>',
                unsafe_allow_html=True,
            )
            col_glucose, col_gtype = st.columns([1.2, 1], gap="medium")
            glucose = col_glucose.number_input(
                "血糖（mmol/L）",
                min_value=2.0,
                max_value=35.0,
                value=None,
                step=0.1,
                format="%.1f",
                placeholder="留空",
                key="ci-glucose",
            )
            glucose_type = col_gtype.selectbox(
                "测量时点", list(checkin.GLUCOSE_TYPES), key="ci-glucose-type"
            )

            st.markdown(
                '<div class="form-section"><span>血脂</span></div>',
                unsafe_allow_html=True,
            )
            col_tc, col_tg = st.columns(2, gap="medium")
            tc = col_tc.number_input(
                "总胆固醇（mmol/L）",
                min_value=1.0,
                max_value=20.0,
                value=None,
                step=0.01,
                format="%.2f",
                placeholder="留空",
                key="ci-tc",
            )
            tg = col_tg.number_input(
                "甘油三酯（mmol/L）",
                min_value=0.3,
                max_value=30.0,
                value=None,
                step=0.01,
                format="%.2f",
                placeholder="留空",
                key="ci-tg",
            )
            col_ldl, col_hdl = st.columns(2, gap="medium")
            ldl = col_ldl.number_input(
                "低密度脂蛋白（mmol/L）",
                min_value=0.3,
                max_value=15.0,
                value=None,
                step=0.01,
                format="%.2f",
                placeholder="留空",
                key="ci-ldl",
            )
            hdl = col_hdl.number_input(
                "高密度脂蛋白（mmol/L）",
                min_value=0.2,
                max_value=5.0,
                value=None,
                step=0.01,
                format="%.2f",
                placeholder="留空",
                key="ci-hdl",
            )

            st.markdown(
                '<div class="form-section"><span>其他</span></div>',
                unsafe_allow_html=True,
            )
            col_hr, col_temp = st.columns(2, gap="medium")
            heart_rate = col_hr.number_input(
                "心率（次/分）",
                min_value=30,
                max_value=220,
                value=None,
                step=1,
                placeholder="留空",
                key="ci-heart-rate",
            )
            temp = col_temp.number_input(
                "体温（°C）",
                min_value=34.0,
                max_value=43.0,
                value=None,
                step=0.1,
                format="%.1f",
                placeholder="留空",
                key="ci-temp",
            )

            st.markdown(
                '<div class="form-section"><span>时间与备注</span></div>',
                unsafe_allow_html=True,
            )
            col_date, col_time = st.columns(2, gap="medium")
            recorded_date = col_date.date_input(
                "打卡日期", value=date.today(), max_value=date.today(), key="ci-date"
            )
            recorded_time = col_time.time_input(
                "打卡时间",
                value=datetime.now().time().replace(second=0, microsecond=0),
                key="ci-time",
            )
            note = st.text_input(
                "备注（可留空）",
                max_chars=200,
                placeholder="例如：早餐后、运动前、服药前",
                key="ci-note",
            )

            saved = st.button(
                "保存这次打卡", type="primary", width="stretch", key="ci-save"
            )

        if saved:
            values = {
                "height": height,
                "weight": weight,
                "sbp": sbp,
                "dbp": dbp,
                "glucose": glucose,
                "tc": tc,
                "tg": tg,
                "ldl": ldl,
                "hdl": hdl,
                "heart_rate": heart_rate,
                "temp": temp,
            }
            recorded_at = datetime.combine(recorded_date, recorded_time)
            try:
                checkin.save(
                    subject["id"], recorded_at, values, note, glucose_type
                )
            except checkin.CheckinError as exc:
                st.error(str(exc))
            else:
                target = (
                    ""
                    if int(subject["id"]) == int(user["id"])
                    else f"（{subject['account']}）"
                )
                st.session_state[CHECKIN_FLASH_KEY] = (
                    f"已保存{target} {recorded_at.strftime('%Y-%m-%d %H:%M')} 的打卡。"
                    "去「数据看板」就能看到趋势。"
                )
                st.session_state[CHECKIN_RESET_KEY] = True
                # 新数据进来，旧报告就过期了
                st.session_state.pop(CHECKIN_REPORT_KEY, None)
                st.rerun()

        self.render_recent_checkins(subject)

    def render_recent_checkins(self, user: Dict) -> None:
        """最近几条打卡：给一个改正手误的入口。"""
        try:
            items = checkin.recent(user["id"], limit=8)
        except checkin.CheckinError as exc:
            st.error(str(exc))
            return

        if not items:
            st.caption("还没有打卡记录。填好上面的项目，点「保存这次打卡」就是第一条。")
            return

        st.markdown(
            '<div class="rule-label"><span>最近打卡</span></div>', unsafe_allow_html=True
        )
        with st.container(key="recent-checkins"):
            for item in items:
                stamp = item["recorded_at"].strftime("%Y-%m-%d %H:%M")
                summary = self.format_checkin_items(item["items"])
                col_main, col_del = st.columns([8, 1], vertical_alignment="center")
                with col_main:
                    st.markdown(
                        f'<div class="hist-item">'
                        f'<div class="hist-title">{stamp}</div>'
                        f'<div class="hist-meta">{html.escape(summary)}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col_del:
                    if st.button(
                        "删除", key=f"ci-del-{item['id']}", width="stretch"
                    ):
                        try:
                            checkin.delete(user["id"], item["id"])
                        except checkin.CheckinError as exc:
                            st.error(str(exc))
                        else:
                            st.session_state.pop(CHECKIN_REPORT_KEY, None)
                            st.rerun()

    # ---------- 数据看板 ----------

    # 看板卡片用的小图标：内联 SVG，颜色跟随 CSS，不额外引资源
    CHART_ICONS = {
        "bp": '<path d="M3 12h3l2-5 3.5 10 2.5-6h7"/>',
        "glucose": '<path d="M12 3.5c3 4.2 5 6.7 5 9.1a5 5 0 0 1-10 0c0-2.4 2-4.9 5-9.1z"/>',
        "lipid": (
            '<path d="M12 3.5 21 8l-9 4.5L3 8z"/>'
            '<path d="m3 12.5 9 4.5 9-4.5"/>'
            '<path d="m3 17 9 4.5 9-4.5"/>'
        ),
        "weight": (
            '<circle cx="12" cy="12" r="8.5"/>'
            '<path d="M8.5 9.5a4.5 4.5 0 0 1 7 0"/>'
            '<path d="M12 13v2"/>'
        ),
        "bmi": (
            '<circle cx="12" cy="12" r="8.5"/>'
            '<circle cx="12" cy="12" r="4"/>'
            '<circle cx="12" cy="12" r="1"/>'
        ),
        "height": '<path d="M7 4v16M17 4v16M7 8h10M7 12h10M7 16h10"/>',
        "hr": '<path d="M12 20s-7-4.4-7-9a4 4 0 0 1 7-2.6A4 4 0 0 1 19 11c0 4.6-7 9-7 9z"/>',
        "steps": (
            '<path d="M7.5 4c1.4 0 2.4 1.1 2.4 2.8S8.9 12 7.5 12 5.1 9.5 5.1 6.8 6.1 4 7.5 4z"/>'
            '<path d="M16.5 12c1.4 0 2.4 1.1 2.4 2.8S17.9 20 16.5 20s-2.4-2.5-2.4-5.2 1-2.8 2.4-2.8z"/>'
        ),
        "sleep": '<path d="M20 14.2A8.2 8.2 0 0 1 9.8 4 8.4 8.4 0 1 0 20 14.2z"/>',
        "spo2": (
            '<path d="M12 3.5c3 4.2 5 6.7 5 9.1a5 5 0 0 1-10 0c0-2.4 2-4.9 5-9.1z"/>'
            '<path d="M9.6 13.6h4.8"/>'
        ),
        "temp": '<path d="M10 4a2 2 0 0 1 4 0v9a4 4 0 1 1-4 0z"/><path d="M12 13.5V7"/>',
        "calendar": (
            '<rect x="3.5" y="5" width="17" height="15.5" rx="2"/>'
            '<path d="M8 3v4M16 3v4M3.5 10h17"/>'
        ),
        "records": '<path d="M5 19.5V11M12 19.5V5M19 19.5v-6"/>',
        "clock": '<circle cx="12" cy="12" r="8"/><path d="M12 7.5V12l3 1.8"/>',
        "coverage": (
            '<circle cx="12" cy="12" r="8.5"/>'
            '<path d="M12 3.5v4M12 16.5v4M3.5 12h4M16.5 12h4"/>'
        ),
        "report": (
            '<path d="M6 3.5h8l4 4v13H6z"/>'
            '<path d="M14 3.5v4h4M9 12h6M9 15.5h6M9 8.5h2"/>'
        ),
    }

    @classmethod
    def chart_icon(cls, name: str, tone: str = "") -> str:
        paths = cls.CHART_ICONS.get(name, "")
        return (
            f'<span class="kpi-icon {tone}">'
            f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
            f'stroke-linecap="round" stroke-linejoin="round">{paths}</svg>'
            f"</span>"
        )

    def render_dashboard(self) -> None:
        st.markdown(
            """
            <section class="hero dashboard-hero">
              <div class="eyebrow">
                <span class="eyebrow-num">04</span>
                <span class="eyebrow-sep"></span>
                <span>数据看板</span>
              </div>
              <h1 class="hero-title">数据看板</h1>
              <p class="hero-sub">
                每个指标一张图，柱状图与折线图一键切换；时间范围按天筛选，
                横轴为时间、纵轴为数值，悬停可以看到具体时刻。
              </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        user = current_user()
        if not user:
            return
        subject = self.resolve_subject(warn=True)
        if not subject:
            return

        try:
            totals = checkin.stats(subject["id"])
            all_points = checkin.list_points(subject["id"])
        except checkin.CheckinError as exc:
            st.error(str(exc))
            return

        groups = [
            group
            for group in checkin.GROUPS
            if group.key not in DASH_HIDDEN_GROUPS
            and any(point["metric_key"] in group.series for point in all_points)
        ]

        # 统计范围与「生成分析报告」放在页面最上面：下面所有图表都跟着它走
        if "dash-range" not in st.session_state:
            st.session_state["dash-range"] = DEFAULT_DASH_RANGE
        col_range, col_report = st.columns(
            [3.1, 1.35], gap="small", vertical_alignment="bottom"
        )
        with col_range:
            st.selectbox(
                "统计范围",
                DASH_RANGE_LABELS,
                key="dash-range",
                help="这个范围对页面里所有图表生效，包括下面的设备区。",
            )
        with col_report:
            if st.button(
                "生成分析报告",
                key="dash-report-open",
                type="primary",
                width="stretch",
            ):
                st.session_state[CHECKIN_DIALOG_KEY] = True
                st.rerun()

        range_label = st.session_state.get("dash-range", DEFAULT_DASH_RANGE)
        range_days = DASH_RANGE_DAYS.get(range_label)

        self.render_dashboard_stats(totals, all_points, groups)

        # 手表数据并进同一页：先看今天，再看打卡指标的趋势
        self.render_device_section(subject, user, range_days, range_label)

        if not all_points:
            with st.container(border=True, key="dash-empty"):
                st.markdown(
                    f'<div class="chart-head">{self.chart_icon("records")}'
                    f'<span class="chart-title">还没有打卡记录</span></div>',
                    unsafe_allow_html=True,
                )
                st.caption("先去「打卡」记一笔，这里就会出现趋势图、打卡天数和分析报告。")
                if st.button(
                    "去打卡", key="dash-go-checkin", type="primary", width="stretch"
                ):
                    goto_view("打卡")
                    st.rerun()
        else:
            st.markdown(
                '<div class="section-head"><span class="section-title">指标趋势</span>'
                '<span class="section-sub">顶部统计范围对下面所有图表生效</span></div>',
                unsafe_allow_html=True,
            )

            for group in groups:
                self.render_checkin_chart(group, all_points, range_days)

            missing = [
                group.label
                for group in checkin.GROUPS
                if group.key not in DASH_HIDDEN_GROUPS
                and not any(point["metric_key"] in group.series for point in all_points)
            ]
            if missing:
                st.caption("这些指标还没有记录：" + "、".join(missing))

        # 弹窗放在最后画。开关留在 session_state 里：弹窗内的每次交互只重跑弹窗本身，
        # 生成报告或手动关闭时才把它置回 False。
        # 注意不能在上面「还没有打卡记录」时提前 return：绑定设备、生成报告
        # 与有没有打卡数据无关，否则点按钮会看起来没反应（弹窗根本没被调用）。
        if st.session_state.get(DEVICE_BIND_KEY):
            self.render_device_bind_dialog()

        if st.session_state.get(CHECKIN_DIALOG_KEY):
            self.render_checkin_report_dialog()

    def render_dashboard_stats(
        self, totals: Dict, points: List[Dict], groups: List[checkin.Group]
    ) -> None:
        """看板顶部的四张指标卡：先看「已打卡 xx 天」，再看记录与覆盖情况。"""
        days = int(totals.get("days") or 0)
        records = int(totals.get("records") or 0)
        latest = totals.get("latest")
        first = min((point["recorded_at"] for point in points), default=None)

        cutoff = datetime.now() - timedelta(days=30)
        recent = len(
            {
                point["checkin_id"]
                for point in points
                if point["recorded_at"] >= cutoff
            }
        )

        if latest:
            latest_day = latest.strftime("%m-%d")
            gap_days = (datetime.now().date() - latest.date()).days
            gap_text = (
                "今天" if gap_days <= 0 else ("昨天" if gap_days == 1 else f"{gap_days} 天前")
            )
            latest_foot = f"{latest.strftime('%H:%M')} · {gap_text}"
        else:
            latest_day, latest_foot = "—", "还没有打卡"

        if first:
            first_foot = f"从 {first.strftime('%Y-%m-%d')} 开始记录"
        else:
            first_foot = "还没有开始记录"

        missing = [
            group.label
            for group in checkin.GROUPS
            if group.key not in DASH_HIDDEN_GROUPS
            and group.key not in {g.key for g in groups}
        ]
        if missing:
            coverage_foot = "还缺 " + "、".join(missing[:3]) + ("…" if len(missing) > 3 else "")
        else:
            coverage_foot = "全部指标都有记录"

        def card(
            icon: str,
            tone: str,
            label: str,
            value: object,
            unit: str,
            foot: str,
            date_value: bool = False,
        ) -> str:
            unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
            value_class = "kpi-value kpi-value-date" if date_value else "kpi-value"
            return (
                '<div class="kpi">'
                f'<div class="kpi-head">{self.chart_icon(icon, tone)}'
                f'<span class="kpi-label">{label}</span></div>'
                f'<div class="{value_class}">{value}{unit_html}</div>'
                f'<div class="kpi-foot">{foot}</div>'
                "</div>"
            )

        st.markdown(
            '<div class="kpi-grid">'
            + card("calendar", "", "已打卡", days, "天", first_foot)
            + card("records", "amber", "累计记录", records, "次", f"近 30 天 {recent} 次")
            + card("clock", "slate", "最近打卡", latest_day, "", latest_foot, date_value=True)
            + card("coverage", "rose", "覆盖指标", len(groups), "项", coverage_foot)
            + "</div>",
            unsafe_allow_html=True,
        )

    # ---------- 设备（手表数据，并进数据看板） ----------

    def render_device_section(
        self,
        subject: Dict,
        user: Dict,
        days: Optional[int],
        range_label: str,
    ) -> None:
        """设备区入口：只看当前「数据对象」自己的手表数据。

        设备库按站点账号隔离（device_bridge/accounts/<账号>/mihealth.db），
        所以这里和上面的打卡图表用同一个对象：看自己就是自己的设备，把上方
        「数据对象」切成家人才会显示家人的设备。不再单独提供设备账号下拉，
        免得数据对象是别人时还画着别人的心率、血氧和睡眠分期。
        """
        viewer_id = int(user.get("id") or 0)
        is_self = int(subject.get("id") or 0) == viewer_id
        account = str(subject.get("account") or "")

        st.markdown(
            '<div class="section-head"><span class="section-title">设备</span>'
            f'<span class="section-sub">手表每天自动同步 · {html.escape(range_label)}；'
            "心率是抽查值，看趋势就好</span></div>",
            unsafe_allow_html=True,
        )

        flash = st.session_state.pop(DEVICE_BIND_FLASH_KEY, None)
        if flash:
            st.success(flash)

        if not account or device_view.db_path_for(account) is None:
            if is_self:
                st.info(
                    "还没有绑定设备账号。绑定后，这里会显示手表同步过来的步数、"
                    "心率、睡眠分期和血氧。"
                )
                if st.button(
                    "绑定设备账号",
                    key="device-bind-empty",
                    type="primary",
                    width="stretch",
                ):
                    st.session_state[DEVICE_BIND_KEY] = True
                    st.rerun()
            else:
                label = html.escape(account) if account else "这位家人"
                st.info(
                    f"「{label}」还没有绑定设备账号，所以这里没有设备图表。"
                    " 设备数据跟随上方「数据对象」；设备要用该账号本人登录后绑定，"
                    "要看自己的手表数据把数据对象切回自己即可。"
                )
            return

        st.caption(f"设备账号：{account} · 跟随上方「数据对象」")
        self.render_device_block(days, range_label, account)

        # 只有看自己时才允许绑定 / 解绑：操作的是当前登录账号的设备目录
        if not is_self:
            return

        with st.expander("设备账号管理", expanded=False):
            current = account
            st.caption(
                "绑定需要小米运动健康的登录态：桥接工具生成的 config.json，"
                "或者其中的 passToken + cUserId。"
            )
            if st.button(
                "绑定新的设备账号", key="device-bind-more", width="stretch"
            ):
                st.session_state[DEVICE_BIND_KEY] = True
                st.rerun()
            confirmed = st.checkbox(
                f"我确认解绑「{current}」", key="device-unbind-confirm"
            )
            if st.button(
                "解绑并删除本地数据",
                key="device-unbind",
                width="stretch",
                disabled=not confirmed,
            ):
                device_bind.unbind(current)
                st.session_state[DEVICE_BIND_FLASH_KEY] = f"已解绑 {current}。"
                st.rerun()

    @st.dialog("绑定设备账号", width="large", on_dismiss=close_device_bind_dialog)
    def render_device_bind_dialog(self) -> None:
        """站内绑定：上传 config.json 或手填 passToken，校验后立刻起一次同步。"""
        user = current_user()
        if not user:
            return
        account = str(user["account"])
        st.caption(f"绑定后，手表数据会记在「{account}」名下，每天自动同步。")

        with st.expander("怎么拿到登录态？", expanded=False):
            st.markdown(
                """
                1. 在电脑上打开桥接工具的本地看板（默认 `http://127.0.0.1:8567`），
                   用要绑定的**小米账号**登录一次（会过短信 / 邮箱验证）；
                2. 打开 `mihealth-api` 目录下的 `config.json`，整份传到下面；
                3. 也可以只复制里面的 `pass_token` 和 `cuser_id` 手填。
                """
            )

        payload = st.session_state.get(DEVICE_BIND_PAYLOAD_KEY) or {}

        tab_upload, tab_manual = st.tabs(["上传 config.json", "手填 passToken"])
        with tab_upload:
            file = st.file_uploader("选择 config.json", type=("json",), key="bind-file")
            if file is not None:
                try:
                    parsed = device_bind.parse_config_text(
                        file.getvalue().decode("utf-8", "ignore")
                    )
                except device_bind.BindError as exc:
                    st.error(str(exc))
                else:
                    if parsed != payload:
                        st.session_state[DEVICE_BIND_PAYLOAD_KEY] = parsed
                        payload = parsed
                    st.success("已读取配置：" + "、".join(sorted(parsed)))
        with tab_manual:
            token = st.text_input("passToken", type="password", key="bind-token")
            cuser = st.text_input("cUserId", key="bind-cuser")
            uid = st.text_input("userId（可留空）", key="bind-uid")
            if st.button("用 passToken 换票", key="bind-manual", width="stretch"):
                try:
                    parsed = device_bind.parse_manual(token, cuser, uid)
                except device_bind.BindError as exc:
                    st.error(str(exc))
                else:
                    st.session_state[DEVICE_BIND_PAYLOAD_KEY] = parsed
                    st.success("凭据有效，可以绑定了。")
                    st.rerun()

        if payload:
            st.caption("待绑定字段：" + "、".join(sorted(payload)))
            if not payload.get("pass_token"):
                st.warning(
                    "这份配置里没有 passToken：只能用到当前 serviceToken 过期为止，"
                    "之后需要重新绑定。建议传桥接工具完整生成的 config.json。"
                )
            if st.button(
                "确认绑定并开始同步",
                type="primary",
                width="stretch",
                key="bind-save",
            ):
                try:
                    device_bind.verify(dict(payload), account)
                    device_bind.save(account, payload)
                    device_bind.start_sync(account)
                except device_bind.BindError as exc:
                    st.error(str(exc))
                else:
                    st.session_state[DEVICE_BIND_FLASH_KEY] = (
                        "已绑定，正在后台同步（约 3 分钟），稍后回看板就能看到手表数据。"
                    )
                    close_device_bind_dialog()
                    st.rerun()

    def render_device_block(
        self, days: Optional[int], range_label: str, account: Optional[str] = None
    ) -> None:
        """数据看板里的「设备」区：步数、心率、睡眠分期、血氧。

        数据来自 device_view（accounts/<站点账号>/mihealth.db），和打卡指标放在
        同一页里对照着看；曲线与分期跟着页面顶部的「统计范围」走。
        """
        summary = device_view.today_summary(account)

        def kpi(icon: str, tone: str, label: str, value: str, unit: str, foot: str) -> str:
            unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
            return (
                '<div class="kpi">'
                f'<div class="kpi-head">{self.chart_icon(icon, tone)}'
                f'<span class="kpi-label">{label}</span></div>'
                f'<div class="kpi-value">{value}{unit_html}</div>'
                f'<div class="kpi-foot">{foot}</div>'
                "</div>"
            )

        steps_value = f"{summary['steps']:,}" if summary["steps"] else "—"
        if summary["hr_avg"] is None:
            hr_value, hr_foot = "—", "今天还没有抽查"
        else:
            hr_value = f"{summary['hr_avg']}"
            hr_foot = (
                f"{summary['hr_min']}–{summary['hr_max']} · {summary['hr_count']} 次抽查"
            )
        if summary["spo2_min"] is None:
            spo2_value, spo2_foot = "—", "今天还没有记录"
        else:
            spo2_value = f"{summary['spo2_min']}"
            spo2_foot = f"{summary['spo2_count']} 次测量里的最低值"
        if summary["sleep_hours"] is None:
            sleep_value, sleep_foot = "—", "没有找到昨晚的记录"
        else:
            sleep_value = f"{summary['sleep_hours']:g}"
            sleep_foot = f"{summary['sleep_day'][5:]} 醒来"

        st.markdown(
            '<div class="kpi-grid">'
            + kpi(
                "steps",
                "amber",
                "今日步数",
                steps_value,
                "步",
                f"{summary['date'].strftime('%m-%d')} 累计",
            )
            + kpi("hr", "", "今日心率均值", hr_value, "次/分", hr_foot)
            + kpi("spo2", "slate", "血氧最低", spo2_value, "%", spo2_foot)
            + kpi("sleep", "rose", "昨夜睡眠", sleep_value, "小时", sleep_foot)
            + "</div>",
            unsafe_allow_html=True,
        )

        # ----- 睡眠分期时间轴：一次只看一晚，按钟点画三条泳道 -----
        st.markdown(
            '<div class="rule-label"><span>睡眠分期</span></div>',
            unsafe_allow_html=True,
        )
        nights = [
            night
            for night in device_view.sleep_nights(min(days or 14, 14), account=account)
            if night["stages"]
        ]
        if not nights:
            st.caption("最近几晚还没有分期数据。")
        else:
            labels = [f"{night['day'][5:]} 夜 · {night['hours']:g} 小时" for night in nights]
            if st.session_state.get("sleep-night") not in (None, *labels):
                st.session_state.pop("sleep-night", None)
            picked = st.selectbox(
                "看哪一晚",
                labels,
                key="sleep-night",
                label_visibility="collapsed",
            )
            night = nights[labels.index(picked)]

            minutes = {"深睡": 0.0, "浅睡": 0.0, "REM": 0.0}
            for stage in night["stages"]:
                minutes[stage["label"]] = minutes.get(stage["label"], 0.0) + (
                    stage["end"] - stage["start"]
                ).total_seconds() / 60
            total_minutes = sum(minutes.values()) or 1.0
            summary = " · ".join(
                f"{label} {minutes[label] / 60:.1f} 小时"
                f"（{minutes[label] / total_minutes * 100:.0f}%）"
                for label in ("深睡", "浅睡", "REM")
            )

            rows = [
                {
                    "阶段": stage["label"],
                    "开始": stage["start"],
                    "结束": stage["end"],
                    "持续": f"{(stage['end'] - stage['start']).total_seconds() / 60:.0f} 分钟",
                }
                for stage in night["stages"]
            ]
            frame = pd.DataFrame(rows)
            chart = (
                alt.Chart(frame)
                .mark_bar(size=24, cornerRadius=2)
                .encode(
                    x=alt.X(
                        "开始:T",
                        title="时间（钟点）",
                        axis=alt.Axis(
                            format="%H:%M",
                            labelAngle=0,
                            tickCount=8,
                            grid=False,
                            titlePadding=8,
                        ),
                    ),
                    x2=alt.X2("结束:T"),
                    y=alt.Y(
                        "阶段:N",
                        sort=["深睡", "浅睡", "REM"],
                        title=None,
                        axis=alt.Axis(labelAngle=0),
                    ),
                    color=alt.Color(
                        "阶段:N",
                        scale=alt.Scale(
                            domain=["深睡", "浅睡", "REM"],
                            range=["#1C5B4A", "#8FB3A6", "#8A5A00"],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("阶段:N", title="分期"),
                        alt.Tooltip("开始:T", title="开始", format="%H:%M"),
                        alt.Tooltip("结束:T", title="结束", format="%H:%M"),
                        alt.Tooltip("持续:N", title="持续"),
                    ],
                )
                .properties(height=150)
            )
            with st.container(border=True, key="device-sleep"):
                st.markdown(
                    f'<div class="chart-head">{self.chart_icon("sleep")}'
                    f'<span class="chart-title">{night["day"]} 夜</span>'
                    f'<span class="chart-unit">共 {night["hours"]:g} 小时</span></div>',
                    unsafe_allow_html=True,
                )
                st.altair_chart(chart, width="stretch")
                st.caption(f"{summary}；越靠上的色块，说明那段时间睡得越沉。")

        # ----- 心率与血氧曲线 -----
        st.markdown(
            '<div class="rule-label"><span>心率与血氧</span></div>',
            unsafe_allow_html=True,
        )
        # 跟页面顶部的统计范围走：按天换算成小时，选「全部」时给足一整年
        hours = (days or 365) * 24

        hr_points = device_view.heart_rate_points(hours, account=account)
        with st.container(border=True, key="device-hr"):
            st.markdown(
                f'<div class="chart-head">{self.chart_icon("hr")}'
                f'<span class="chart-title">心率</span>'
                f'<span class="chart-unit">次/分 · {range_label}</span></div>',
                unsafe_allow_html=True,
            )
            if hr_points:
                st.altair_chart(
                    self.build_device_line(
                        hr_points, "次/分", "#1C5B4A", rules=(60, 100)
                    ),
                    width="stretch",
                )
                st.caption(
                    f"共 {len(hr_points)} 次抽查；虚线是 60–100 次/分的常见静息区间。"
                )
            else:
                st.caption("这个时间范围内没有心率抽查。")

        spo2_points = device_view.spo2_points(hours, account=account)
        with st.container(border=True, key="device-spo2"):
            st.markdown(
                f'<div class="chart-head">{self.chart_icon("spo2")}'
                f'<span class="chart-title">血氧</span>'
                f'<span class="chart-unit">% · {range_label}</span></div>',
                unsafe_allow_html=True,
            )
            if spo2_points:
                st.altair_chart(
                    self.build_device_line(spo2_points, "%", "#5B6B7A", rules=(95,)),
                    width="stretch",
                )
                st.caption(f"共 {len(spo2_points)} 次测量；虚线是 95% 的参考线。")
            else:
                st.caption("这个时间范围内没有血氧记录。")

    @staticmethod
    def build_device_line(
        points: List[Dict], unit: str, color: str, rules: Sequence[float] = ()
    ):
        """设备曲线：细线 + 半透明点，另加几条参考虚线。"""
        frame = pd.DataFrame(
            {
                "时间": [point["when"] for point in points],
                "数值": [point["value"] for point in points],
            }
        )
        base = alt.Chart(frame)
        line = base.mark_line(
            strokeWidth=1.6, color=color, interpolate="monotone"
        ).encode(
            x=alt.X(
                "时间:T",
                title=None,
                axis=alt.Axis(format="%m-%d", labelAngle=0, tickCount=6, grid=False),
            ),
            y=alt.Y(
                "数值:Q",
                title=f"数值（{unit}）",
                axis=alt.Axis(format="~g", tickCount=5, titlePadding=8),
            ),
            tooltip=[
                alt.Tooltip("时间:T", title="时间", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("数值:Q", title="数值"),
            ],
        )
        dots = base.mark_circle(size=7, color=color, opacity=0.35).encode(
            x="时间:T", y="数值:Q"
        )
        chart = line + dots
        for rule in rules:
            rule_frame = pd.DataFrame({"参考": [rule]})
            chart = chart + alt.Chart(rule_frame).mark_rule(
                strokeDash=[4, 4], color="#B9B2A6"
            ).encode(y="参考:Q")
        return chart

    def render_checkin_chart(
        self, group: checkin.Group, points: List[Dict], days: Optional[int] = None
    ) -> None:
        keys = set(group.series)
        rows: List[Dict] = []
        for point in points:
            key = point["metric_key"]
            if key not in keys:
                continue
            series = checkin.SERIES[key]
            shown = self.format_metric(key, point["value"])
            if key == "glucose" and point.get("extra"):
                shown += f"（{point['extra']}）"
            rows.append(
                {
                    "时间": point["recorded_at"],
                    "数值": point["value"],
                    "系列": series.label,
                    "显示": shown,
                }
            )
        if not rows:
            return

        data = pd.DataFrame(rows).sort_values("时间")
        kind_key = f"chart-kind-{group.key}"
        st.session_state.setdefault(kind_key, "折线图")

        with st.container(border=True, key=f"chart-card-{group.key}"):
            head_title, head_kind = st.columns(
                [3.4, 1.5], gap="small", vertical_alignment="center"
            )
            with head_title:
                st.markdown(
                    f'<div class="chart-head">'
                    f"{self.chart_icon(group.key)}"
                    f'<span class="chart-title">{html.escape(group.label)}</span>'
                    f'<span class="chart-unit">{html.escape(group.unit)}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with head_kind:
                kind = (
                    st.segmented_control(
                        "图形",
                        options=list(CHART_KINDS),
                        key=kind_key,
                        label_visibility="collapsed",
                        width="stretch",
                    )
                    or "折线图"
                )

            shown = data
            if days:
                shown = data[
                    data["时间"] >= datetime.now() - timedelta(days=days)
                ]
            if shown.empty:
                st.caption("这个统计范围内没有记录。")
                return

            latest = shown.iloc[-1]
            st.markdown(
                f'<div class="chart-meta">'
                f"<span>共 {len(shown)} 个数据点</span>"
                f'<span class="chart-latest">最近 {html.escape(str(latest["显示"]))}'
                f' · {latest["时间"].strftime("%m-%d %H:%M")}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.altair_chart(
                self.build_trend_chart(shown, kind, group.unit), width="stretch"
            )

    @staticmethod
    def build_trend_chart(data: "pd.DataFrame", kind: str, unit: str):
        """横轴时间（日期 + 时刻）、纵轴数值；柱状图与折线图共用同一份数据。"""
        axis_x = alt.X(
            "时间:T",
            title="时间（日期 + 时刻）",
            axis=alt.Axis(
                format="%m-%d", labelAngle=0, tickCount=6, grid=False, titlePadding=8
            ),
        )
        axis_y = alt.Y(
            "数值:Q",
            title=f"数值（{unit}）" if unit else "数值",
            scale=alt.Scale(zero=kind == "柱状图"),
            axis=alt.Axis(format="~g", tickCount=5, titlePadding=8),
        )
        tooltip = [
            alt.Tooltip("时间:T", title="时间", format="%Y-%m-%d %H:%M"),
            alt.Tooltip("系列:N", title="指标"),
            alt.Tooltip("显示:N", title="数值"),
        ]
        multiple = data["系列"].nunique() > 1
        palette = ["#1C5B4A", "#8A5A00", "#5B6B7A", "#9E3524"]
        color = (
            alt.Color(
                "系列:N",
                title=None,
                scale=alt.Scale(range=palette),
                legend=alt.Legend(
                    orient="bottom",
                    columns=4,
                    symbolSize=110,
                    labelFontSize=11,
                    title=None,
                ),
            )
            if multiple
            else alt.value(palette[0])
        )

        if kind == "柱状图":
            encoding = {
                "x": axis_x,
                "y": axis_y,
                "color": color,
                "tooltip": tooltip,
            }
            if multiple:
                encoding["xOffset"] = alt.XOffset("系列:N", title=None)
            chart = alt.Chart(data).mark_bar(
                size=14,
                cornerRadiusTopLeft=2,
                cornerRadiusTopRight=2,
                opacity=0.92,
            ).encode(**encoding)
        else:
            chart = (
                alt.Chart(data)
                .mark_line(
                    strokeWidth=2,
                    point=alt.OverlayMarkDef(size=50, filled=True, strokeWidth=0),
                )
                .encode(x=axis_x, y=axis_y, color=color, tooltip=tooltip)
            )

        return (
            chart.properties(height=300)
            .configure_view(strokeWidth=0)
            .configure_axis(
                grid=True,
                gridColor="#E5E1D5",
                gridDash=[2, 3],
                domainColor="#CFC9B8",
                tickColor="#CFC9B8",
                labelColor="#3E3E38",
                titleColor="#85857A",
                labelFontSize=11,
                titleFontSize=11,
            )
            .configure_legend(
                labelColor="#3E3E38", symbolSize=110, padding=8, labelFontSize=11
            )
        )

    @st.dialog("生成分析报告", width="large", on_dismiss=close_checkin_dialog)
    def render_checkin_report_dialog(self) -> None:
        """点「生成分析报告」后先选指标和时间范围，再决定要简要版还是详细版。"""
        subject = self.resolve_subject(warn=True)
        if not subject:
            st.caption("登录状态已失效，请重新登录。")
            return
        viewer = current_user()
        st.caption(f"当前分析对象：{subject_display(subject, viewer)}")

        try:
            points = checkin.list_points(subject["id"])
        except checkin.CheckinError as exc:
            st.error(str(exc))
            return

        available = [
            group
            for group in checkin.GROUPS
            if any(point["metric_key"] in group.series for point in points)
        ]
        if not available:
            st.caption("还没有任何打卡数据，先去「打卡」记一笔再回来。")
            return

        labels = [group.label for group in available]
        picked = st.multiselect(
            "选择要分析的指标",
            options=labels,
            default=labels,
            help="可以只挑关心的几项；血压、血脂会按同组的多条曲线一起分析。",
        )

        if "report-range" not in st.session_state:
            st.session_state["report-range"] = st.session_state.get(
                "dash-range", DEFAULT_DASH_RANGE
            )
        range_label = st.select_slider(
            "分析时间范围（天）", options=DASH_RANGE_LABELS, key="report-range"
        )
        st.caption(
            "简要报告直接生成在这个弹窗里；详细版会额外生成一份可下载的 PDF，含趋势图与逐项解读。"
        )

        col_brief, col_detail = st.columns(2, gap="small")
        brief = col_brief.button(
            "生成简要报告", type="primary", width="stretch", key="report-brief"
        )
        detailed = col_detail.button(
            "生成详细版 PDF", width="stretch", key="report-detailed"
        )
        if brief or detailed:
            if not picked:
                st.warning("至少选择一个指标。")
            else:
                keys = [group.key for group in available if group.label in picked]
                self.generate_checkin_report(
                    keys, DASH_RANGE_DAYS.get(range_label), detailed
                )

        # 报告就在本弹窗里显示，不再插到页面中间
        pdf_error = st.session_state.pop("_checkin_pdf_error", None)
        if pdf_error:
            st.error(f"详细版 PDF 生成失败：{pdf_error}")

        cached = st.session_state.get(CHECKIN_REPORT_KEY)
        if cached:
            pdf = cached.get("pdf")
            if pdf:
                st.success(
                    f"详细版 PDF 已生成（{len(pdf) / 1024:.0f} KB），点下面的按钮保存到本地。"
                )
                st.download_button(
                    "下载详细版 PDF",
                    data=pdf,
                    file_name=(
                        f"健康数据趋势报告_{cached['at'].strftime('%Y%m%d_%H%M')}.pdf"
                    ),
                    mime="application/pdf",
                    key="dialog-pdf-download",
                    type="primary",
                    width="stretch",
                )
            st.markdown(
                '<div class="rule-label"><span>最近一次生成的报告</span></div>',
                unsafe_allow_html=True,
            )
            self.render_checkin_report_card(cached, compact=bool(pdf))

    def generate_checkin_report(
        self, group_keys: List[str], days: Optional[int], detailed: bool
    ) -> None:
        subject = self.resolve_subject(warn=True)
        if not subject:
            return

        since = datetime.now() - timedelta(days=days) if days else None
        with st.spinner("正在汇总趋势并生成分析…"):
            try:
                points = checkin.list_points(subject["id"], since=since)
            except checkin.CheckinError as exc:
                st.error(str(exc))
                return

            report = checkin.analyze(points, group_keys, window_days=days)
            if not report.series:
                st.warning("所选指标在统计范围内没有记录，换个时间范围再试。")
                return

            conclusion, suggestions = self.analyze_trend(report)
            report.conclusion = conclusion
            report.suggestions = suggestions

            pdf = None
            if detailed:
                profile = (
                    f"{subject['age']} 岁 · "
                    f"{GENDER_LABELS.get(subject['gender'], '未填写')}"
                )
                try:
                    pdf = render_trend_pdf(report, profile=profile)
                    st.session_state.pop("_checkin_pdf_error", None)
                except Exception as exc:  # noqa: BLE001
                    # 弹窗会 rerun，toast 会被刷掉；存进 session，下一轮在弹窗里红字提示
                    st.session_state["_checkin_pdf_error"] = str(exc)

        st.session_state[CHECKIN_REPORT_KEY] = {
            "report": report,
            "conclusion": conclusion,
            "suggestions": suggestions,
            "pdf": pdf,
            "at": datetime.now(),
            "subject_label": subject_display(subject, current_user()),
        }
        # 不关弹窗：rerun 之后报告直接出现在这个弹窗里
        st.session_state[CHECKIN_DIALOG_KEY] = True
        st.rerun()

    def analyze_trend(self, report: checkin.CheckinReport) -> Tuple[str, List[str]]:
        """让模型把趋势统计写成人话；模型不可用时退回规则拼装。"""
        text = llm.chat_once(
            [
                {
                    "role": "user",
                    "content": checkin.build_prompt(report, elder=self.elder),
                }
            ],
            model=self.resolve_model(),
            temperature=0.4,
            top_p=0.9,
            max_tokens=ELDER_SELFCHECK_MAX_TOKENS if self.elder else 480,
            timeout=60,
        )

        conclusion, suggestions = parse_analysis(text)
        if not conclusion:
            return checkin.fallback_analysis(report)

        if len(suggestions) < 3:
            _, fallback_suggestions = checkin.fallback_analysis(report)
            for tip in fallback_suggestions:
                if len(suggestions) >= 3:
                    break
                if tip not in suggestions:
                    suggestions.append(tip)
        if self.elder:
            suggestions = suggestions[:ELDER_SELFCHECK_SUGGESTIONS]
        return conclusion, suggestions

    def render_checkin_report_card(self, cached: Dict, compact: bool = False) -> None:
        """报告卡片：在弹窗里显示。

        compact=True 表示这次是详细版（PDF 已经生成、下载按钮在卡片上方），
        卡片只留标题与折叠的简要解读，避免看起来像又出了一份简要报告。
        """
        report: checkin.CheckinReport = cached["report"]
        with st.container(key="dash-report-card"):
            if compact:
                st.markdown(
                    f'<div class="chart-head">'
                    f"{self.chart_icon('report')}"
                    f'<span class="chart-title">分析报告</span>'
                    f'<span class="chart-unit">{html.escape(report.window_label)}'
                    f" · {len(report.series)} 项指标</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                head_left, head_right = st.columns(
                    [3.0, 1.35], gap="small", vertical_alignment="center"
                )
                with head_left:
                    st.markdown(
                        f'<div class="chart-head">'
                        f"{self.chart_icon('report')}"
                        f'<span class="chart-title">分析报告</span>'
                        f'<span class="chart-unit">{html.escape(report.window_label)}'
                        f" · {len(report.series)} 项指标</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with head_right:
                    if st.button(
                        "生成详细版 PDF", key="dash-detail", width="stretch"
                    ):
                        self.build_checkin_pdf(cached)

            subject_label = cached.get("subject_label")
            if subject_label:
                st.caption(f"分析对象：{subject_label}")

            level = "alert" if report.urgent else "info"
            headline = (
                f"最近一次越过警报线：{'、'.join(report.urgent)}"
                if report.urgent
                else f"打卡 {report.records} 次 · 覆盖 {report.days} 天"
            )
            st.markdown(
                f'<div class="check-banner {level}">'
                f'<span class="check-banner-label">趋势分析</span>'
                f'<span class="check-banner-value">{html.escape(headline)}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

            if compact:
                with st.expander("查看简要解读", expanded=False):
                    self.render_report_insight(cached)
                return

            rows = "".join(
                f"<tr>"
                f"<td>{html.escape(item.label)}</td>"
                f'<td class="num">{item.count} 次</td>'
                f'<td class="num">{item.fmt(item.last)} {html.escape(item.unit)}</td>'
                f'<td class="num">{item.fmt(item.mean)}</td>'
                f'<td class="num">{item.change_pct:+.1f}%</td>'
                f'<td class="st {"high" if item.abnormal else "ok"}">{item.abnormal} 次</td>'
                f"</tr>"
                for item in report.series
            )
            st.markdown(
                f"""
                <table class="check-table">
                  <thead><tr><th>指标</th><th>记录</th><th>最新</th><th>平均</th><th>变化</th><th>超出参考</th></tr></thead>
                  <tbody>{rows}</tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )

            self.render_report_insight(cached)

    def render_report_insight(self, cached: Dict) -> None:
        """报告正文：分析结论 + 建议 + 朗读。"""
        st.markdown(
            '<div class="rule-label"><span>分析结论</span></div>',
            unsafe_allow_html=True,
        )
        with st.container(key="check-conclusion"):
            st.markdown(cached["conclusion"])

        st.markdown(
            '<div class="rule-label"><span>建议</span></div>', unsafe_allow_html=True
        )
        with st.container(key="check-suggestions"):
            st.markdown(
                "\n".join(
                    f"{i}. {item}"
                    for i, item in enumerate(cached["suggestions"], start=1)
                )
            )

        with st.container(key="check-speak"):
            self.render_speak(
                self.checkin_speech(cached), key="speak-report", big=self.elder
            )

    def build_checkin_pdf(self, cached: Dict) -> None:
        """已经生成过简要报告时，详细版只补 PDF，不再调一遍模型。"""
        subject = self.resolve_subject(warn=True)
        report: checkin.CheckinReport = cached["report"]
        profile = ""
        if subject:
            profile = (
                f"{subject['age']} 岁 · "
                f"{GENDER_LABELS.get(subject['gender'], '未填写')}"
            )
        with st.spinner("正在排版详细版报告…"):
            try:
                pdf = render_trend_pdf(report, profile=profile)
                st.session_state.pop("_checkin_pdf_error", None)
            except Exception as exc:  # noqa: BLE001
                st.session_state["_checkin_pdf_error"] = str(exc)
                return
        cached["pdf"] = pdf
        st.session_state[CHECKIN_REPORT_KEY] = cached
        st.rerun()

    @staticmethod
    def checkin_speech(cached: Dict) -> str:
        report: checkin.CheckinReport = cached["report"]
        parts = [f"{report.window_label}的打卡趋势分析。", cached["conclusion"]]
        for i, tip in enumerate(cached["suggestions"], start=1):
            parts.append(f"第{i}条建议：{tip}")
        return "\n".join(parts)

    # ---------- 设备数据导入 ----------

    def render_device_import(self) -> None:
        """设备导入：把手表导出的 CSV / JSON 写进打卡（服务端连不到手表，只接文件）。"""
        user = current_user()
        if not user:
            return
        subject = self.resolve_subject(warn=True)
        if not subject:
            return

        # 上一轮导入成功：先清掉上传控件，再画新的一页
        if st.session_state.pop(DEVICE_CLEAR_UPLOAD_KEY, False):
            st.session_state.pop("device-file", None)

        flash = st.session_state.pop(DEVICE_FLASH_KEY, None)
        if flash:
            st.success(flash)

        st.markdown(
            """
            <section class="hero">
              <div class="eyebrow">
                <span class="eyebrow-num">08</span>
                <span class="eyebrow-sep"></span>
                <span>设备导入</span>
              </div>
              <h1 class="hero-title">表上的记录，<br>接进你的趋势里</h1>
              <p class="hero-sub">
                手表的数据不会自己飞进来：服务端在云上，没有蓝牙，厂商也没有开放接口。
                先在手机上把数据导出成文件，再传到这里，它就和手填的打卡并到同一张看板上。
              </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Redmi Watch 6 怎么导出数据（Android）"):
            st.markdown(
                """
                按这个顺序试，前一条走不通再看下一条：

                1. **先在 App 里翻一遍**：`我的 → 设置`，看有没有「导出」「数据导出」「隐私」
                   这类入口（不同版本位置会变，也可能在「我的 → 关于」里）。
                2. **小米账号的隐私中心**：在 [privacy.mi.com/support](https://privacy.mi.com/support)
                   提交「获取个人信息副本」的请求，小米会把 Mi Fitness 的数据打包发给你，
                   通常几个工作日。这条路不挑 App 版本，最稳。
                3. **Health Connect**（Android 14 以上或装了「健康连接」）：去系统设置里看
                   「小米运动健康」有没有 Health Connect 的写入权限；有的话，数据已经在那儿，
                   用任意 Health Connect 导出工具存成 CSV 即可。
                4. **都没有就先手填**：下载下面的模板，每周记一次静息心率、体重、血压，
                   数字照手环屏幕上抄，也能看出趋势。
                """
            )

        st.markdown(
            '<div class="form-section"><span>第一步 · 准备文件</span></div>',
            unsafe_allow_html=True,
        )
        col_tpl, col_hint = st.columns([1, 1.6], gap="medium")
        col_tpl.download_button(
            "下载 CSV 模板",
            data=device_import.TEMPLATE_CSV.encode("utf-8-sig"),
            file_name=device_import.TEMPLATE_NAME,
            mime="text/csv",
            width="stretch",
            key="device-template",
        )
        col_hint.caption(
            "模板里是宽表：一行一个时间点，一列一个指标。"
            "长表（时间 / 指标 / 数值）和 JSON 也支持。"
        )

        uploaded = st.file_uploader(
            "选择导出的 CSV 或 JSON",
            type=("csv", "json", "txt"),
            key="device-file",
            help=f"一次最多 {device_import.MAX_ROWS} 行、8 MB；超了就按月拆成几个文件。",
        )
        if uploaded is None:
            return

        st.markdown(
            '<div class="form-section"><span>第二步 · 预览</span></div>',
            unsafe_allow_html=True,
        )
        try:
            parsed = device_import.parse(uploaded.name, uploaded.getvalue())
        except device_import.DeviceImportError as exc:
            st.error(str(exc))
            return
        except checkin.CheckinError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - 文件千奇百怪，别让整页崩掉
            st.error(f"文件读不进来：{exc}")
            return

        if parsed.invalid:
            with st.expander(f"有 {len(parsed.invalid)} 行没读进来，看看原因"):
                st.markdown("\n".join(f"- {line}" for line in parsed.invalid))
        if parsed.skipped_empty:
            st.caption(f"另有 {parsed.skipped_empty} 行整行都是空的，已跳过。")

        if not parsed.rows:
            st.warning(
                "这份文件里没有能导入的记录。检查一下时间列是不是「2026-11-05 08:30」"
                "这种格式，以及指标列名是不是心率 / 体重 / 体温 / 血压 / 血糖。"
            )
            return

        summary = " · ".join(
            f"{checkin.SERIES[key].label} {count} 条"
            for key, count in sorted(
                parsed.columns.items(), key=lambda item: -item[1]
            )
            if key in checkin.SERIES
        )
        st.caption(
            f"{parsed.format_label} · 共 {len(parsed.rows)} 个时间点、"
            f"{parsed.value_count} 个数值"
            + (f"：{summary}" if summary else "")
        )

        preview: List[Dict] = []
        for row in parsed.rows[:60]:
            for key, value in row.values.items():
                series = checkin.SERIES[key]
                preview.append(
                    {
                        "时间": row.recorded_at.strftime("%Y-%m-%d %H:%M"),
                        "指标": series.label,
                        "数值": f"{value:g}",
                        "单位": series.unit,
                    }
                )
        st.dataframe(pd.DataFrame(preview), width="stretch", hide_index=True)
        if len(parsed.rows) > 60:
            st.caption(f"表格只列前 60 个时间点，导入时会写全部 {len(parsed.rows)} 个。")

        st.markdown(
            '<div class="form-section"><span>第三步 · 写入</span></div>',
            unsafe_allow_html=True,
        )
        try:
            plan = device_import.plan(subject["id"], parsed.rows)
        except checkin.CheckinError as exc:
            st.error(str(exc))
            return
        except device_import.DeviceImportError as exc:
            st.error(str(exc))
            return

        if plan.duplicates:
            st.caption(
                f"其中 {plan.duplicates} 个数值和已有记录在同一时间点，会跳过、不覆盖。"
            )
        if not plan.rows:
            st.info("这个文件里的记录都已经在库里了，不需要再导入。")
            return

        target = (
            ""
            if int(subject["id"]) == int(user["id"])
            else f"（{subject['account']}）"
        )
        label = "我的打卡" if not target else f"{subject['account']} 的打卡"
        if st.button(
            f"导入 {len(plan.rows)} 个时间点到{label}",
            type="primary",
            width="stretch",
            key="device-save",
        ):
            try:
                written = device_import.commit(
                    subject["id"],
                    plan.rows,
                    note=f"设备导入 · {uploaded.name}"[:200],
                )
            except checkin.CheckinError as exc:
                st.error(str(exc))
                return
            except device_import.DeviceImportError as exc:
                st.error(str(exc))
                return
            st.session_state[DEVICE_FLASH_KEY] = (
                f"已导入 {written} 条打卡{target}，"
                "去「数据看板」就能和手填的记录一起看趋势。"
            )
            # 新数据进来，旧报告就过期了
            st.session_state.pop(CHECKIN_REPORT_KEY, None)
            # 清掉上传控件（控件已实例化，只能下一轮开头再删）
            st.session_state[DEVICE_CLEAR_UPLOAD_KEY] = True
            st.rerun()

    # ---------- 家庭药箱 ----------

    def load_medicine_alerts(
        self, user: Dict, force: bool = False
    ) -> Tuple[int, List[Dict]]:
        """家庭组药箱的提醒，以及「我」还没读的那几条。

        重算提醒有节流：临期按天变化，不必每次 rerun 都跑一遍；新增 / 修改 /
        补货 / 删除会在保存后立即重算，所以页面里的状态总是最新的。
        """
        group_id = medicine.group_of(int(user["id"]))
        stamps = st.session_state.get(MED_SYNC_KEY)
        if not isinstance(stamps, dict):
            stamps = {}
        now = time_module.time()
        try:
            last = float(stamps.get(group_id) or 0)
        except (TypeError, ValueError):
            last = 0.0
        if force or now - last > MED_SYNC_TTL:
            medicine.sync_alerts(group_id)
            stamps[group_id] = now
            st.session_state[MED_SYNC_KEY] = stamps
        return group_id, medicine.pending_alerts(group_id, int(user["id"]))

    def render_medicine_notice(self) -> None:
        """全局提醒条：家里有临期或缺量时，每个家庭组成员在每一页都能看到。"""
        user = current_user()
        if not user:
            return
        try:
            _, alerts = self.load_medicine_alerts(user)
        except medicine.MedicineError:
            return
        if not alerts:
            return
        names = "、".join(
            f"{alert['item']['name']}（{alert['label']}）" for alert in alerts[:3]
        )
        more = f" 等 {len(alerts)} 条" if len(alerts) > 3 else ""
        st.markdown(
            f'<div class="family-notice"><b>家庭药箱有 {len(alerts)} 条提醒</b>　'
            f"{html.escape(names)}{more}，该补药或清理了。</div>",
            unsafe_allow_html=True,
        )
        if st.button("去「家庭药箱」看看", key="med-notice-go", width="stretch"):
            goto_view("家庭药箱")
            st.rerun()

    def reset_medicine_form(self) -> None:
        """清空新增表单（含药盒照片）；只能在控件还没实例化的这一轮开头调用。"""
        for key in MED_FORM_KEYS:
            st.session_state.pop(key, None)
        for key in (MED_SCAN_KEY, MED_SCAN_RESULT, "med-scan-file"):
            st.session_state.pop(key, None)

    def reset_medicine_edit_form(self) -> None:
        """点「修改」时清掉上一条药留下的编辑框内容。"""
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and (
                key.startswith("med-edit-") or key.startswith("med-del-")
            ):
                st.session_state.pop(key, None)

    def init_medicine_form(self) -> None:
        """新增表单的初始值：数量先按 1 盒、保质期先按 24 个月。"""
        defaults = {
            "med-category": medicine.CATEGORY_DRUG,
            "med-manufacturer": "",
            "med-efficacy": "",
            "med-spec": 1.0,
            "med-unit": "颗",
            "med-package-unit": "盒",
            "med-packages": 1.0,
            "med-production": date.today(),
            "med-shelf-life": 24,
            "med-has-expiry": True,
            "med-note": "",
        }
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)
        st.session_state.setdefault("med-name", None)

    def fill_medicine_from_catalog(self) -> None:
        """名称选到以前录入过的药：厂商 / 功效 / 规格自动带出来。"""
        user = current_user()
        name = st.session_state.get("med-name")
        if not user or not isinstance(name, str) or not name.strip():
            return
        try:
            entry = medicine.catalog_lookup(
                medicine.group_of(int(user["id"])), name
            )
        except medicine.MedicineError:
            return
        if not entry:
            return
        if entry["manufacturer"]:
            st.session_state["med-manufacturer"] = entry["manufacturer"]
        if entry["efficacy"]:
            st.session_state["med-efficacy"] = entry["efficacy"]
        if entry["spec_amount"] > 0:
            st.session_state["med-spec"] = float(entry["spec_amount"])
        if entry["unit"]:
            st.session_state["med-unit"] = entry["unit"]
        if entry["package_unit"]:
            st.session_state["med-package-unit"] = entry["package_unit"]

    def apply_medicine_scan(self, result: Dict) -> None:
        """把识别结果写进「新增药品」表单；必须在控件创建之前调用。"""
        for key, value in medicine_scan.form_values(result).items():
            st.session_state[key] = value

    def render_medicine_scan_result(self, result: Dict) -> None:
        """识别结果说明：读到了什么、还缺什么、原文是什么（失败时也摊开原文）。"""
        error = str(result.get("error") or "")
        filled = result.get("filled") or []
        missing = result.get("missing") or []
        warnings = result.get("warnings") or []
        if error:
            st.warning(error)
        rows = []
        if filled:
            rows.append("识别到：" + "、".join(str(item) for item in filled))
        if missing:
            rows.append("还需要你补充 / 确认：" + "、".join(str(item) for item in missing))
        if rows:
            st.markdown(
                '<div class="med-scan-box">'
                + "".join(f"<div>{html.escape(row)}</div>" for row in rows)
                + "</div>",
                unsafe_allow_html=True,
            )
        for warning in warnings:
            st.caption(str(warning))
        raw = str(result.get("raw_text") or "")
        if raw:
            with st.expander("查看照片里读到的文字"):
                st.text(raw)

    def render_medicine_scan_panel(self, uploaded) -> None:
        """药盒照片 → 识别一次 → 预填表单。

        同一张照片只识别一次（按内容指纹缓存）；换一张才会重新跑 OCR 和模型，
        用户改过的字段不会被重复覆盖。
        """
        data = uploaded.getvalue()
        digest = hashlib.sha1(data).hexdigest()
        if st.session_state.get(MED_SCAN_KEY) != digest:
            with st.spinner("正在识别药盒，请稍等几秒…"):
                try:
                    result = medicine_scan.scan(data, uploaded.name)
                    payload = result.as_dict()
                except medicine_scan.ScanError as exc:
                    payload = medicine_scan.ScanResult.failed(
                        str(exc), getattr(exc, "raw_text", "")
                    ).as_dict()
                except Exception as exc:  # noqa: BLE001 - 照片千奇百怪，别让弹窗崩掉
                    payload = medicine_scan.ScanResult.failed(
                        f"这张照片没能识别：{exc}"
                    ).as_dict()
            st.session_state[MED_SCAN_KEY] = digest
            st.session_state[MED_SCAN_RESULT] = payload
            if not payload.get("error"):
                self.apply_medicine_scan(payload)

        result = st.session_state.get(MED_SCAN_RESULT)
        if not isinstance(result, dict):
            result = {}
        col_photo, col_info = st.columns([1, 2.2], gap="medium")
        col_photo.image(uploaded, width=180)
        with col_info:
            self.render_medicine_scan_result(result)
            if result.get("error") and col_info.button(
                "重新识别这张照片", key="med-scan-retry", width="stretch"
            ):
                # 清掉这次的缓存，下一次 rerun 会重新跑 OCR + 模型
                st.session_state.pop(MED_SCAN_KEY, None)
                st.session_state.pop(MED_SCAN_RESULT, None)
                st.rerun()

    def render_medicine_math(
        self,
        spec,
        packages,
        unit: str,
        package_unit: str,
        production: Optional[date],
        shelf_months,
    ) -> None:
        """把「规格 × 包装数」和到期日当场算给用户看，不用等保存。"""
        spec = float(spec or 0)
        packages = float(packages or 0)
        unit = (unit or "个").strip() or "个"
        package_unit = (package_unit or "个").strip() or "个"
        quantity = medicine.quantity_of(spec, packages)
        pieces = [
            f"共 {medicine.fmt_number(quantity)} {unit}"
            f"（{medicine.fmt_number(packages)} {package_unit} × "
            f"{medicine.fmt_number(spec)} {unit}/{package_unit}）"
        ]
        expiry = medicine.expiry_of(production, shelf_months)
        if expiry:
            days = (expiry - date.today()).days
            if days < 0:
                tail = f"已过期 {abs(days)} 天"
            elif days == 0:
                tail = "今天到期"
            else:
                tail = f"剩 {days} 天"
            pieces.append(f"有效期至 {expiry} · {tail}")
        else:
            pieces.append("未设保质期，不会提醒临期")
        st.markdown(
            f'<div class="check-banner info">'
            f'<span class="check-banner-label">自动计算</span>'
            f'<span class="check-banner-value">{html.escape("；".join(pieces))}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    @staticmethod
    def medicine_card_html(item: Dict) -> str:
        """药箱卡片：名称 + 类别 / 状态标签 + 规格数量 + 有效期。"""
        tags = (
            f'<span class="med-badge plain">'
            f'{html.escape(item["category_label"])}</span>'
        )
        tags += "".join(
            f'<span class="med-badge {cls}">{label}</span>'
            for label, cls in item["badges"]
        )
        meta = " · ".join(
            part for part in (item["manufacturer"], item["efficacy"]) if part
        )
        note = (
            f'<div class="med-note">{html.escape(item["note"])}</div>'
            if item["note"]
            else ""
        )
        return f"""
        <div class="med-card">
          <div class="med-card-head">
            <span class="med-name">{html.escape(item["name"])}</span>
            {tags}
          </div>
          <div class="med-meta">
            {html.escape(meta) if meta else "厂商 / 功效未填写"}
          </div>
          <div class="med-figures">
            <span><b>{html.escape(item["quantity_label"])}</b></span>
            <span>{html.escape(item["stock_label"])}</span>
            <span>{html.escape(item["expiry_label"])}</span>
          </div>
          {note}
        </div>
        """

    def medicine_image(self, group_id: int, item: Dict) -> Optional[bytes]:
        """取列表缩略图；会话里按 (id, 更新时间) 缓存，避免每次 rerun 都读 BLOB。"""
        if not item.get("has_image"):
            return None
        cache = st.session_state.get(MED_IMAGE_CACHE_KEY)
        if not isinstance(cache, dict):
            cache = {}
        key = (int(item["id"]), str(item.get("image_updated_at") or ""))
        if key not in cache:
            try:
                fetched = medicine.get_image(group_id, int(item["id"]))
            except medicine.MedicineError:
                return None
            if not fetched:
                return None
            cache[key] = fetched[0]
            if len(cache) > MED_IMAGE_CACHE_MAX:
                for stale in list(cache)[: len(cache) - MED_IMAGE_CACHE_MAX]:
                    cache.pop(stale, None)
            st.session_state[MED_IMAGE_CACHE_KEY] = cache
        return cache.get(key)

    @st.dialog("药品照片", width="medium")
    def render_medicine_photo_dialog(self, group_id: int, item: Dict) -> None:
        """点缩略图旁边的「看照片」：大图 + 名称。"""
        try:
            fetched = medicine.get_image(group_id, int(item["id"]))
        except medicine.MedicineError as exc:
            st.error(str(exc))
            return
        if not fetched:
            st.warning("这条药品还没有照片，或者照片已经被删除了。")
            return
        st.image(fetched[0], width=380)
        st.caption(f"{item['name']} · 照片仅家庭组成员可见")
        if st.button("关闭", key="med-photo-close", width="stretch"):
            st.session_state.pop(MED_PHOTO_KEY, None)
            st.rerun()

    def adjust_medicine_packages(
        self, group_id: int, item_id: int, delta: float, package_unit: str
    ) -> None:
        """补货 / 用掉一盒；数量按规格自动重算，提醒也跟着刷新。"""
        try:
            updated = medicine.adjust_packages(group_id, item_id, delta)
        except medicine.MedicineError as exc:
            st.error(str(exc))
            return
        if delta >= 0:
            action = f"补了 {medicine.fmt_number(delta)} {package_unit}"
        else:
            action = f"用掉 {medicine.fmt_number(abs(delta))} {package_unit}"
        st.session_state[MED_FLASH_KEY] = (
            f"「{updated['name']}」{action}，现在共 {updated['quantity_label']}。"
        )
        st.rerun()

    def render_medicine_cabinet(self) -> None:
        """家庭药箱：一个家庭组共用；数量自动算，临期 / 缺量在这里提醒。"""
        user = current_user()
        if not user:
            return

        # 上一轮保存成功：清空新增表单（控件还没实例化，可以在这一轮开头删）
        if st.session_state.pop(MED_RESET_KEY, False):
            self.reset_medicine_form()

        flash = st.session_state.pop(MED_FLASH_KEY, None)
        if flash:
            st.success(flash)

        try:
            group_id = medicine.group_of(int(user["id"]))
            medicine.sync_alerts(group_id)
            items = medicine.list_items(group_id)
            alerts = medicine.pending_alerts(group_id, int(user["id"]))
            members = family.group_members(group_id)
        except (medicine.MedicineError, family.FamilyError) as exc:
            st.error(str(exc))
            return

        expiring = sum(1 for item in items if item["expired"] or item["expiring"])
        low = sum(1 for item in items if item["low"])

        st.markdown(
            f"""
            <section class="hero">
              <div class="eyebrow">
                <span class="eyebrow-num">{view_number("家庭药箱")}</span>
                <span class="eyebrow-sep"></span>
                <span>家庭药箱</span>
              </div>
              <h1 class="hero-title">家里有什么药，<br>一起记清楚</h1>
              <p class="hero-sub">
                录入规格和盒数，系统自动算出总数量；保质期只剩两周、或者只剩不到半盒时，
                会提醒家里每个人补药。同一个家庭组共用这一个药箱。
              </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        summary = [f"共 {len(items)} 种"]
        if expiring:
            summary.append(f"临期 / 过期 {expiring} 种")
        if low:
            summary.append(f"需补药 {low} 种")
        if len(members) > 1:
            share = (
                f"共享给家庭组里的 {len(members)} 个账号"
                "（谁都能增删改，提醒会发给每个人）"
            )
        else:
            share = "只有你能看到；建立家庭组后，药箱会自动共享给家人"
        st.caption(f"{' · '.join(summary)}　｜　{share}")

        if alerts:
            st.markdown(
                '<div class="form-section"><span>待处理提醒</span></div>',
                unsafe_allow_html=True,
            )
            for alert in alerts:
                item = alert["item"]
                css = (
                    "danger"
                    if alert["kind"] == medicine.ALERT_EXPIRED
                    else "warn"
                )
                st.markdown(
                    f'<div class="med-alert {css}">'
                    f'<b>{alert["label"]}</b>'
                    f'<span class="med-alert-name">'
                    f'{html.escape(item["name"])}</span>'
                    f'{html.escape(alert["message"])}</div>',
                    unsafe_allow_html=True,
                )
                col_restock, col_read, _ = st.columns([1.1, 1.1, 2.8])
                if col_restock.button(
                    f"+1 {item['package_unit']}",
                    key=f"med-alert-plus-{alert['id']}",
                    width="stretch",
                ):
                    self.adjust_medicine_packages(
                        group_id, item["id"], 1, item["package_unit"]
                    )
                if col_read.button(
                    "知道了",
                    key=f"med-alert-read-{alert['id']}",
                    width="stretch",
                ):
                    try:
                        medicine.mark_read(
                            group_id, int(user["id"]), alert["id"]
                        )
                    except medicine.MedicineError as exc:
                        st.error(str(exc))
                    else:
                        st.rerun()
            if len(alerts) > 1:
                if st.button(
                    "全部标记为已读", key="med-alert-read-all", width="stretch"
                ):
                    try:
                        medicine.mark_all_read(group_id, int(user["id"]))
                    except medicine.MedicineError as exc:
                        st.error(str(exc))
                    else:
                        st.rerun()

        st.markdown(
            '<div class="form-section"><span>药箱清单</span></div>',
            unsafe_allow_html=True,
        )
        col_search, col_filter, col_add = st.columns(
            [2.2, 1.2, 1.1], vertical_alignment="bottom"
        )
        keyword = col_search.text_input(
            "搜索",
            key="med-search",
            placeholder="按名称 / 厂商 / 功效搜索",
            label_visibility="collapsed",
        )
        filter_label = col_filter.selectbox(
            "筛选",
            ("全部", "需处理", "需补药", "临期 / 过期"),
            key="med-filter",
            label_visibility="collapsed",
        )
        if col_add.button(
            "＋ 新增药品", key="med-add", type="primary", width="stretch"
        ):
            self.reset_medicine_form()
            st.session_state[MED_ADD_KEY] = True
            st.rerun()

        text = (keyword or "").strip().casefold()
        visible: List[Dict] = []
        for item in items:
            if text and not any(
                text in str(item.get(key) or "").casefold()
                for key in ("name", "manufacturer", "efficacy")
            ):
                continue
            if filter_label == "需处理" and not (
                item["low"] or item["expiring"] or item["expired"]
            ):
                continue
            if filter_label == "需补药" and not item["low"]:
                continue
            if filter_label == "临期 / 过期" and not (
                item["expiring"] or item["expired"]
            ):
                continue
            visible.append(item)

        if not items:
            st.info(
                "药箱还是空的。点右上角「＋ 新增药品」，先把家里的常用药记进来。"
            )
        elif not visible:
            st.info("没有符合条件的药品。换个关键词或筛选条件试试。")
        for item in visible:
            with st.container(border=True):
                photo = self.medicine_image(group_id, item)
                if photo:
                    col_photo, col_body = st.columns([0.9, 4.1], gap="medium")
                    col_photo.image(photo, width=84)
                    with col_body:
                        st.markdown(
                            self.medicine_card_html(item),
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        self.medicine_card_html(item), unsafe_allow_html=True
                    )
                if item.get("has_image"):
                    col_edit, col_plus, col_photo_btn, _ = st.columns(
                        [1, 1.2, 1, 2]
                    )
                else:
                    col_edit, col_plus, _ = st.columns([1, 1.2, 3.2])
                    col_photo_btn = None
                if col_edit.button(
                    "修改 / 删除",
                    key=f"med-edit-{item['id']}",
                    width="stretch",
                ):
                    self.reset_medicine_edit_form()
                    st.session_state[MED_EDIT_KEY] = int(item["id"])
                    st.rerun()
                if col_plus.button(
                    f"+1 {item['package_unit']}",
                    key=f"med-plus-{item['id']}",
                    width="stretch",
                ):
                    self.adjust_medicine_packages(
                        group_id, item["id"], 1, item["package_unit"]
                    )
                if col_photo_btn is not None and col_photo_btn.button(
                    "看照片",
                    key=f"med-photo-{item['id']}",
                    width="stretch",
                ):
                    # 一次只能开一个弹窗：先把新增 / 修改的开关收起来
                    st.session_state.pop(MED_ADD_KEY, None)
                    st.session_state.pop(MED_EDIT_KEY, None)
                    st.session_state[MED_PHOTO_KEY] = int(item["id"])
                    st.rerun()

        # 一次只开一个弹窗：优先级是 看照片 > 修改 > 新增。
        # Streamlit 同一个脚本轮次里只允许开一个 dialog，残留的开关要先收起来。
        photo_id = st.session_state.get(MED_PHOTO_KEY)
        edit_id = st.session_state.get(MED_EDIT_KEY)
        if photo_id:
            st.session_state.pop(MED_ADD_KEY, None)
            st.session_state.pop(MED_EDIT_KEY, None)
            target = next(
                (it for it in items if int(it["id"]) == int(photo_id)), None
            )
            if target:
                self.render_medicine_photo_dialog(group_id, target)
            else:
                st.session_state.pop(MED_PHOTO_KEY, None)
        elif edit_id:
            st.session_state.pop(MED_ADD_KEY, None)
            self.render_medicine_edit_dialog(group_id, int(edit_id))
        elif st.session_state.get(MED_ADD_KEY):
            self.render_medicine_add_dialog(group_id)

    @st.dialog("新增药品 / 卫材", width="large")
    def render_medicine_add_dialog(self, group_id: int) -> None:
        """新增弹窗：名称做联想输入，规格 × 包装数当场算出总数。"""
        user = current_user()
        if not user:
            return
        self.init_medicine_form()
        try:
            options = medicine.catalog_names(group_id)
        except medicine.MedicineError as exc:
            st.error(str(exc))
            options = []

        st.markdown(
            '<div class="form-section"><span>拍药盒 · 自动填</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "上传药盒正面照片，系统会读出名称、厂商、规格、生产日期等；"
            "读不到的字段再请你补充。照片会随这条药品一起保存，方便下次辨认"
            "（仅家庭组成员可见）。尽量只拍有字的一面、让药名占满画面；"
            "iPhone 的 HEIC 也可以直接传。"
        )
        uploaded = st.file_uploader(
            "药盒照片（可选）",
            type=("png", "jpg", "jpeg", "webp", "bmp", "heic", "heif"),
            key="med-scan-file",
            help=f"一张清楚的正面照就够了，最大 {MED_SCAN_MAX_MB} MB。",
        )
        if uploaded is not None:
            self.render_medicine_scan_panel(uploaded)

        # 识别到的名称 / 单位可能不在联想词库里。只写 session_state 的话，
        # 下次 rerun 下拉框可能把它当成无效值清空；先并进 options 就稳了。
        scan_fields = {}
        scan_payload = st.session_state.get(MED_SCAN_RESULT)
        if isinstance(scan_payload, dict):
            scan_fields = scan_payload.get("fields") or {}
        scan_name = str(scan_fields.get("name") or "").strip()
        if scan_name and scan_name not in options:
            options = [scan_name] + options
        unit_options = list(medicine.UNIT_SUGGESTIONS)
        scan_unit = str(scan_fields.get("unit") or "").strip()
        if scan_unit and scan_unit not in unit_options:
            unit_options.append(scan_unit)
        package_options = list(medicine.PACKAGE_UNIT_SUGGESTIONS)
        scan_package = str(scan_fields.get("package_unit") or "").strip()
        if scan_package and scan_package not in package_options:
            package_options.append(scan_package)

        st.markdown(
            '<div class="form-section"><span>药品信息</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "名称、规格、计数单位是必填。以前录过的药会出现在名称的联想里，"
            "选中后自动带出厂商、功效和规格。"
        )
        name = st.selectbox(
            "名称",
            options,
            index=None,
            accept_new_options=True,
            placeholder="输入药名 / 卫材名，或从录入过的里面选",
            key="med-name",
            on_change=self.fill_medicine_from_catalog,
        )
        col_category, col_manufacturer = st.columns([1, 2], gap="medium")
        category = col_category.selectbox(
            "类别",
            list(medicine.CATEGORIES),
            format_func=lambda key: medicine.CATEGORIES.get(key, key),
            key="med-category",
        )
        manufacturer = col_manufacturer.text_input(
            "厂商",
            key="med-manufacturer",
            max_chars=medicine.MANUFACTURER_MAX,
            placeholder="例如：同仁堂",
        )
        efficacy = st.text_area(
            "功效 / 用途",
            key="med-efficacy",
            max_chars=medicine.EFFICACY_MAX,
            height=70,
            placeholder="例如：缓解发热、头痛",
        )

        st.markdown(
            '<div class="form-section"><span>规格与数量</span></div>',
            unsafe_allow_html=True,
        )
        col_spec, col_unit, col_pkg_unit, col_pkg = st.columns(
            4, gap="small"
        )
        spec = col_spec.number_input(
            "每盒数量",
            min_value=0.01,
            max_value=medicine.MAX_SPEC,
            step=1.0,
            format="%.2f",
            key="med-spec",
            help="一个包装里有多少：一盒 5 颗就填 5。",
        )
        unit = col_unit.selectbox(
            "计数单位",
            unit_options,
            accept_new_options=True,
            key="med-unit",
        )
        package_unit = col_pkg_unit.selectbox(
            "包装单位",
            package_options,
            accept_new_options=True,
            key="med-package-unit",
        )
        packages = col_pkg.number_input(
            f"已有几{package_unit}",
            min_value=0.0,
            max_value=medicine.MAX_PACKAGES,
            step=1.0,
            format="%.2f",
            key="med-packages",
            help="有几盒 / 几瓶就填几，数量会按规格自动算。",
        )

        st.markdown(
            '<div class="form-section"><span>生产日期与保质期</span></div>',
            unsafe_allow_html=True,
        )
        has_expiry = st.checkbox(
            "填写生产日期和保质期（系统自动算到期日，只剩两周时提醒）",
            key="med-has-expiry",
        )
        production: Optional[date] = None
        shelf = 0
        if has_expiry:
            col_prod, col_shelf = st.columns(2, gap="medium")
            production = col_prod.date_input(
                "生产日期",
                key="med-production",
                max_value=date.today(),
                help="药品包装上印的生产日期。",
            )
            shelf = col_shelf.number_input(
                "保质期（月）",
                min_value=1,
                max_value=medicine.MAX_SHELF_LIFE_MONTHS,
                step=1,
                key="med-shelf-life",
                help="保质期 2 年就填 24。",
            )
        note = st.text_input(
            "备注",
            key="med-note",
            max_chars=medicine.NOTE_MAX,
            placeholder="例如：放在玄关柜",
        )

        self.render_medicine_math(
            spec, packages, unit, package_unit, production, shelf
        )

        if st.button(
            "保存到家庭药箱",
            type="primary",
            width="stretch",
            key="med-save",
        ):
            payload = {
                "category": category,
                "name": name or "",
                "manufacturer": manufacturer,
                "efficacy": efficacy,
                "spec_amount": spec,
                "unit": unit,
                "package_unit": package_unit,
                "packages": packages,
                "production_date": production,
                "shelf_life_months": shelf,
                "note": note,
            }
            try:
                saved = medicine.save(
                    group_id,
                    int(user["id"]),
                    payload,
                    image=uploaded.getvalue() if uploaded is not None else None,
                )
            except medicine.MedicineError as exc:
                st.error(str(exc))
                return
            st.session_state[MED_FLASH_KEY] = (
                f"已把「{saved['name']}」放进家庭药箱："
                f"{saved['quantity_label']}。"
            )
            # 新照片进来，旧缩略图缓存作废
            st.session_state.pop(MED_IMAGE_CACHE_KEY, None)
            st.session_state.pop(MED_ADD_KEY, None)
            st.session_state[MED_RESET_KEY] = True
            st.rerun()
        if st.button("取消", key="med-cancel", width="stretch"):
            st.session_state.pop(MED_ADD_KEY, None)
            st.rerun()

    @st.dialog("修改 / 删除药品", width="large")
    def render_medicine_edit_dialog(self, group_id: int, item_id: int) -> None:
        """修改数量 / 日期，或者把这条药删掉。"""
        user = current_user()
        if not user:
            return
        try:
            item = medicine.get_item(group_id, item_id)
        except medicine.MedicineError as exc:
            st.error(str(exc))
            st.session_state.pop(MED_EDIT_KEY, None)
            return
        if not item:
            st.warning("这条药品不存在，可能已经被家人删除了。")
            st.session_state.pop(MED_EDIT_KEY, None)
            return

        st.caption(f"当前：{item['stock_label']}；{item['expiry_label']}")

        st.markdown(
            '<div class="form-section"><span>照片</span></div>',
            unsafe_allow_html=True,
        )
        current_photo = None
        if item.get("has_image"):
            try:
                current_photo = medicine.get_image(group_id, item_id)
            except medicine.MedicineError:
                current_photo = None
        if current_photo:
            col_now, col_tip = st.columns([1, 2.2], gap="medium")
            col_now.image(current_photo[0], width=140)
            col_tip.caption(
                "现在的照片。可以换一张，或勾选下面的「删除现有照片」。"
                "照片仅家庭组成员可见。"
            )
        new_photo = st.file_uploader(
            "换一张照片（可选）",
            type=("png", "jpg", "jpeg", "webp", "bmp", "heic", "heif"),
            key=f"med-edit-image-{item_id}",
            help=f"最大 {MED_IMAGE_MAX_MB} MB，保存时会自动压缩到 1200px。",
        )
        remove_photo = st.checkbox(
            "删除现有照片",
            value=False,
            key=f"med-edit-remove-image-{item_id}",
            disabled=not bool(current_photo),
        )

        name = st.text_input(
            "名称",
            value=item["name"],
            max_chars=medicine.NAME_MAX,
            key=f"med-edit-name-{item_id}",
        )
        col_category, col_manufacturer = st.columns([1, 2], gap="medium")
        category = col_category.selectbox(
            "类别",
            list(medicine.CATEGORIES),
            index=(
                list(medicine.CATEGORIES).index(item["category"])
                if item["category"] in medicine.CATEGORIES
                else 0
            ),
            format_func=lambda key: medicine.CATEGORIES.get(key, key),
            key=f"med-edit-category-{item_id}",
        )
        manufacturer = col_manufacturer.text_input(
            "厂商",
            value=item["manufacturer"],
            max_chars=medicine.MANUFACTURER_MAX,
            key=f"med-edit-manufacturer-{item_id}",
        )
        efficacy = st.text_area(
            "功效 / 用途",
            value=item["efficacy"],
            max_chars=medicine.EFFICACY_MAX,
            height=68,
            key=f"med-edit-efficacy-{item_id}",
        )

        col_spec, col_unit, col_pkg_unit, col_pkg = st.columns(
            4, gap="small"
        )
        spec = col_spec.number_input(
            "每盒数量",
            min_value=0.01,
            max_value=medicine.MAX_SPEC,
            value=float(item["spec_amount"]),
            step=1.0,
            format="%.2f",
            key=f"med-edit-spec-{item_id}",
        )
        unit = col_unit.text_input(
            "计数单位",
            value=item["unit"],
            max_chars=medicine.UNIT_MAX,
            key=f"med-edit-unit-{item_id}",
        )
        package_unit = col_pkg_unit.text_input(
            "包装单位",
            value=item["package_unit"],
            max_chars=medicine.UNIT_MAX,
            key=f"med-edit-package-unit-{item_id}",
        )
        packages = col_pkg.number_input(
            f"已有几{package_unit}",
            min_value=0.0,
            max_value=medicine.MAX_PACKAGES,
            value=float(item["packages"]),
            step=1.0,
            format="%.2f",
            key=f"med-edit-packages-{item_id}",
        )

        has_expiry = st.checkbox(
            "填写生产日期和保质期",
            value=bool(item["shelf_life_months"]),
            key=f"med-edit-has-expiry-{item_id}",
        )
        production: Optional[date] = None
        shelf = 0
        if has_expiry:
            col_prod, col_shelf = st.columns(2, gap="medium")
            production = col_prod.date_input(
                "生产日期",
                value=item["production_date"] or date.today(),
                max_value=date.today(),
                key=f"med-edit-production-{item_id}",
            )
            shelf = col_shelf.number_input(
                "保质期（月）",
                min_value=1,
                max_value=medicine.MAX_SHELF_LIFE_MONTHS,
                value=int(item["shelf_life_months"] or 24),
                step=1,
                key=f"med-edit-shelf-{item_id}",
            )
        note = st.text_input(
            "备注",
            value=item["note"],
            max_chars=medicine.NOTE_MAX,
            key=f"med-edit-note-{item_id}",
        )

        if st.button(
            "保存修改",
            type="primary",
            width="stretch",
            key=f"med-edit-save-{item_id}",
        ):
            payload = {
                "category": category,
                "name": name,
                "manufacturer": manufacturer,
                "efficacy": efficacy,
                "spec_amount": spec,
                "unit": unit,
                "package_unit": package_unit,
                "packages": packages,
                "production_date": production,
                "shelf_life_months": shelf,
                "note": note,
            }
            try:
                saved = medicine.save(
                    group_id,
                    int(user["id"]),
                    payload,
                    item_id=item_id,
                    image=(
                        new_photo.getvalue() if new_photo is not None else None
                    ),
                    remove_image=bool(remove_photo and new_photo is None),
                )
            except medicine.MedicineError as exc:
                st.error(str(exc))
            else:
                st.session_state[MED_FLASH_KEY] = (
                    f"「{saved['name']}」已更新：{saved['stock_label']}。"
                )
                # 换了 / 删了照片，缩略图缓存作废
                st.session_state.pop(MED_IMAGE_CACHE_KEY, None)
                st.session_state.pop(MED_EDIT_KEY, None)
                st.rerun()

        st.divider()
        confirm = st.checkbox(
            "确认删除这条记录（删除后不可恢复）",
            key=f"med-del-confirm-{item_id}",
        )
        col_delete, col_close = st.columns(2)
        if col_delete.button(
            "删除",
            key=f"med-del-{item_id}",
            width="stretch",
            disabled=not confirm,
        ):
            try:
                medicine.delete_item(group_id, item_id)
            except medicine.MedicineError as exc:
                st.error(str(exc))
            else:
                st.session_state[MED_FLASH_KEY] = (
                    f"已从药箱删除「{item['name']}」。"
                )
                st.session_state.pop(MED_IMAGE_CACHE_KEY, None)
                st.session_state.pop(MED_EDIT_KEY, None)
                st.session_state.pop(MED_SYNC_KEY, None)
                st.rerun()
        if col_close.button(
            "关闭", key=f"med-edit-close-{item_id}", width="stretch"
        ):
            st.session_state.pop(MED_EDIT_KEY, None)
            st.rerun()

    # ---------- 工具箱 ----------

    def render_toolbox(self) -> None:
        """工具箱入口：没选工具时是列表，选了就进对应的计算页。"""
        tool = toolbox.TOOL_MAP.get(st.session_state.get("tool") or "")
        if tool:
            self.render_tool_detail(tool)
        else:
            self.render_tool_home()

    def render_tool_home(self) -> None:
        self.render_back(nav_back_target(), "工具", go_back)
        st.markdown(
            """
            <section class="hero">
              <div class="eyebrow">
                <span class="eyebrow-num">05</span>
                <span class="eyebrow-sep"></span>
                <span>工具</span>
              </div>
              <h1 class="hero-title">先算清楚，<br>再决定要不要紧张</h1>
              <p class="hero-sub">
                八个常用计算，全部在本地按固定公式完成，不联网、不调用模型。
                给出的是参考区间和下一步建议，不是诊断，也不替代医生。
              </p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="rule-label"><span>选择工具</span></div>',
            unsafe_allow_html=True,
        )

        with st.container(key="tool-grid"):
            col_left, col_right = st.columns(2, gap="medium")
            for index, tool in enumerate(toolbox.TOOLS):
                target = col_left if index % 2 == 0 else col_right
                with target:
                    with st.container(border=True, key=f"toolcard-{tool.key}"):
                        if st.button(
                            f"{tool.num}　{tool.name}",
                            key=f"tool-{tool.key}",
                            width="stretch",
                            type="tertiary",
                        ):
                            st.session_state.tool = tool.key
                            st.rerun()
                        st.caption(tool.summary)

    def render_tool_detail(self, tool: toolbox.Tool) -> None:
        self.render_back("工具", "tool-detail", self.back_to_tool_list)

        st.markdown(
            f"""
            <section class="hero tool-hero">
              <div class="eyebrow">
                <span class="eyebrow-num">{tool.num}</span>
                <span class="eyebrow-sep"></span>
                <span>工具</span>
              </div>
              <h1 class="hero-title">{html.escape(tool.name)}</h1>
              <p class="hero-sub">{html.escape(tool.desc)}</p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        handlers = {
            "bmi": self.tool_bmi,
            "bmr": self.tool_bmr,
            "pregnancy": self.tool_pregnancy,
            "ovulation": self.tool_ovulation,
            "growth": self.tool_growth,
            "alcohol": self.tool_lifestyle,
            "bp": self.tool_bp,
            "dose": self.tool_dose,
        }
        handlers[tool.key]()

    def render_tool_result(self, result: toolbox.ToolResult) -> None:
        rows = "".join(
            f"<tr>"
            f"<td>{html.escape(row.label)}</td>"
            f'<td class="num">{html.escape(row.value)}</td>'
            f'<td class="tip {row.status}">{html.escape(row.note)}</td>'
            f"</tr>"
            for row in result.rows
        )
        st.markdown(
            f"""
            <div class="check-banner {result.level}">
              <span class="check-banner-label">结果</span>
              <span class="check-banner-value">{html.escape(result.headline)}</span>
            </div>
            <table class="check-table tool-table">
              <thead><tr><th>项目</th><th>数值</th><th>说明</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )

        if result.advice:
            st.markdown(
                '<div class="rule-label"><span>提示</span></div>',
                unsafe_allow_html=True,
            )
            items = "".join(f"<li>{html.escape(item)}</li>" for item in result.advice)
            st.markdown(f'<ol class="tool-advice">{items}</ol>', unsafe_allow_html=True)

        if result.disclaimer:
            st.caption(result.disclaimer)

    # ---------- 工具箱 · 各计算页 ----------

    def tool_bmi(self) -> None:
        with st.form("tool-bmi-form"):
            col_height, col_weight = st.columns(2, gap="medium")
            height = col_height.number_input(
                "身高（cm）", 100.0, 250.0, 170.0, 0.5, format="%.1f", key="bmi-height"
            )
            weight = col_weight.number_input(
                "体重（kg）", 20.0, 250.0, 65.0, 0.5, format="%.1f", key="bmi-weight"
            )
            submitted = st.form_submit_button("计算")

        if submitted:
            self.render_tool_result(toolbox.bmi(height, weight))

    def tool_bmr(self) -> None:
        with st.form("tool-bmr-form"):
            col_sex, col_age = st.columns(2, gap="medium")
            sex = col_sex.selectbox("性别", ["男", "女"], key="bmr-sex")
            age = col_age.number_input("年龄（岁）", 18, 90, 30, key="bmr-age")

            col_height, col_weight = st.columns(2, gap="medium")
            height = col_height.number_input(
                "身高（cm）", 100.0, 250.0, 170.0, 0.5, format="%.1f", key="bmr-height"
            )
            weight = col_weight.number_input(
                "体重（kg）", 20.0, 250.0, 65.0, 0.5, format="%.1f", key="bmr-weight"
            )

            activity = st.selectbox(
                "日常活动量",
                [name for name, _ in toolbox.ACTIVITY_LEVELS],
                key="bmr-activity",
            )
            submitted = st.form_submit_button("计算")

        if submitted:
            self.render_tool_result(
                toolbox.bmr_tdee(age, sex, height, weight, activity)
            )

    def tool_pregnancy(self) -> None:
        today = date.today()
        with st.form("tool-pregnancy-form"):
            col_lmp, col_cycle = st.columns(2, gap="medium")
            lmp = col_lmp.date_input(
                "末次月经第一天",
                value=today - timedelta(days=70),
                min_value=today - timedelta(days=300),
                max_value=today,
                key="preg-lmp",
            )
            cycle = col_cycle.number_input(
                "平时月经周期（天）", 21, 45, 28, key="preg-cycle"
            )
            submitted = st.form_submit_button("计算")

        if submitted:
            self.render_tool_result(toolbox.pregnancy(lmp, cycle, today))

    def tool_ovulation(self) -> None:
        today = date.today()
        with st.form("tool-ovulation-form"):
            cycle = st.number_input(
                "月经周期长度（天）", 21, 45, 28, key="ovu-cycle"
            )
            lmp = st.date_input(
                "末次月经第一天（可留空，留空只给周期第几天）",
                value=None,
                min_value=today - timedelta(days=120),
                max_value=today,
                key="ovu-lmp",
            )
            submitted = st.form_submit_button("计算")

        if submitted:
            self.render_tool_result(toolbox.ovulation_cycle(cycle, lmp, today))

    def tool_growth(self) -> None:
        with st.form("tool-growth-form"):
            col_sex, col_year, col_month = st.columns([1.2, 1, 1], gap="medium")
            sex = col_sex.selectbox("性别", ["男", "女"], key="growth-sex")
            years = col_year.number_input("年龄（岁）", 0, 18, 8, key="growth-years")
            months = col_month.number_input("月", 0, 11, 0, key="growth-months")

            col_height, col_weight = st.columns(2, gap="medium")
            height = col_height.number_input(
                "身高 / 身长（cm）", 45.0, 200.0, 128.0, 0.5, format="%.1f", key="growth-height"
            )
            weight = col_weight.number_input(
                "体重（kg）", 2.0, 120.0, 26.0, 0.5, format="%.1f", key="growth-weight"
            )
            submitted = st.form_submit_button("计算")

        if submitted:
            self.render_tool_result(
                toolbox.child_growth(years * 12 + months, sex, height, weight)
            )

    def tool_lifestyle(self) -> None:
        """饮酒与咖啡因共用一个入口：两者都是「算总量、对上限」。"""
        tab_alcohol, tab_caffeine = st.tabs(["酒精", "咖啡因"])

        with tab_alcohol:
            col_kind, col_abv = st.columns([1.6, 1], gap="medium")
            kind = col_kind.selectbox(
                "酒类", [name for name, _ in toolbox.ALCOHOL_TYPES], key="alc-kind"
            )
            abv = None
            if kind == "自填度数":
                abv = col_abv.number_input(
                    "酒精度数（%vol）", 0.0, 75.0, 10.0, 0.5, key="alc-abv"
                )
            else:
                col_abv.caption(
                    f"常见度数约 {dict(toolbox.ALCOHOL_TYPES)[kind]:.1f}%vol"
                )

            col_cups, col_volume = st.columns(2, gap="medium")
            cups = col_cups.number_input(
                "杯数", 0.0, 20.0, 1.0, 0.5, format="%.1f", key="alc-cups"
            )
            volume = col_volume.number_input(
                "每杯容量（ml）", 10.0, 1000.0, 330.0, 10.0, key="alc-volume"
            )
            st.caption("参考：一罐啤酒约 330 ml，一杯红酒约 150 ml，一两白酒约 50 ml。")

            if st.button("换算", key="alc-calc", width="stretch"):
                self.render_tool_result(
                    toolbox.alcohol_intake(kind, cups, volume, abv)
                )

        with tab_caffeine:
            col_item, col_cups = st.columns([1.6, 1], gap="medium")
            item = col_item.selectbox(
                "饮品", [name for name, _ in toolbox.CAFFEINE_ITEMS], key="caf-item"
            )
            cups = col_cups.number_input(
                "杯数", 0.0, 12.0, 1.0, 0.5, format="%.1f", key="caf-cups"
            )
            pregnant = st.checkbox("孕期 / 哺乳期", key="caf-pregnant")
            st.caption("咖啡因含量按常见中杯（240 ml）估算，不同品牌差异较大。")

            if st.button("换算", key="caf-calc", width="stretch"):
                self.render_tool_result(
                    toolbox.caffeine_intake(item, cups, pregnant)
                )

    def tool_bp(self) -> None:
        with st.form("tool-bp-form"):
            col_sbp, col_dbp = st.columns(2, gap="medium")
            sbp = col_sbp.number_input("收缩压（mmHg）", 70, 260, 128, key="bp-sbp")
            dbp = col_dbp.number_input("舒张压（mmHg）", 40, 160, 82, key="bp-dbp")
            st.caption("按静坐 5 分钟后、上臂式电子血压计的诊室读数填写。")
            submitted = st.form_submit_button("分级")

        if submitted:
            self.render_tool_result(toolbox.blood_pressure(sbp, dbp))

    def tool_dose(self) -> None:
        st.caption(
            "只把说明书上的 mg/kg 换算成毫克与片数，不判断适应证，也不构成处方。"
        )
        with st.form("tool-dose-form"):
            col_weight, col_dose = st.columns(2, gap="medium")
            weight = col_weight.number_input(
                "体重（kg）", 1.0, 200.0, 20.0, 0.5, format="%.1f", key="dose-weight"
            )
            mg_per_kg = col_dose.number_input(
                "说明书每次剂量（mg/kg/次）",
                0.0,
                100.0,
                0.0,
                0.5,
                format="%.2f",
                key="dose-mgkg",
            )

            col_times, col_tablet = st.columns(2, gap="medium")
            times = col_times.number_input(
                "每日给药次数", 1, 6, 2, key="dose-times"
            )
            tablet = col_tablet.number_input(
                "每片 / 每袋规格（mg，0 表示不换算）",
                0.0,
                2000.0,
                0.0,
                0.5,
                format="%.1f",
                key="dose-tablet",
            )
            submitted = st.form_submit_button("换算")

        if submitted:
            self.render_tool_result(
                toolbox.dose_conversion(weight, mg_per_kg, times, tablet)
            )

    # ---------- 严重指标弹框 ----------

    @st.dialog("指标异常 · 请尽快就医", width="large", on_dismiss=dismiss_urgent_dialog)
    def render_urgent_dialog(self) -> None:
        result = st.session_state.get("urgent_result")
        if result is None:
            st.caption("这条提示已失效，重新生成报告即可再次查看。")
            return

        urgent = [m for m in result.metrics if m.status == "urgent"]

        def row(m) -> str:
            line = alarm_line(m.key)
            tail = f"警报线 {line}" if line else f"参考 {m.reference}"
            return (
                f'<li><span class="ua-name">{html.escape(m.label)}</span>'
                f'<span class="ua-value">{html.escape(m.value)}</span>'
                f'<span class="ua-ref">{html.escape(tail)}</span></li>'
            )

        rows = "".join(row(m) for m in urgent)
        st.markdown(
            f"""
            <div class="urgent-alert">
              <div class="ua-eyebrow">以下指标已超出常见范围，建议今天内就医</div>
              <ul class="ua-list">{rows}</ul>
              <p class="ua-advice">{html.escape(department_advice(result))}</p>
              <p class="ua-advice">若同时出现胸痛、呼吸困难、意识模糊、单侧肢体无力或说话含糊，请立即拨打 120。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="rule-label"><span>附近医院</span></div>', unsafe_allow_html=True
        )
        self.render_urgent_hospitals()
        st.caption("点击医院会用高德地图导航；手机上装了高德地图会直接唤起 App，没装则打开网页版。")

        if st.button("我知道了，稍后就医", key="urgent-ack", width="stretch"):
            dismiss_urgent_dialog()
            st.rerun()

    def render_urgent_hospitals(self) -> None:
        """弹框里的两家附近医院：优先按定位查，定位失败时退回城市检索。"""
        coords = st.session_state.get("urgent_geo")
        geo_error = st.session_state.get("urgent_geo_error")

        # 只在"还没定位且没失败"时挂载组件，避免每次 rerun 重复请求定位
        if coords is None and geo_error is None:
            payload = GEO(key="urgent-geo", default=None)
            if isinstance(payload, dict):
                stamp = payload.get("ts")
                if stamp != st.session_state.get("urgent_geo_ts"):
                    st.session_state.urgent_geo_ts = stamp
                    if payload.get("lat"):
                        st.session_state.urgent_geo = (float(payload["lat"]), float(payload["lng"]))
                        st.session_state.urgent_geo_error = None
                        coords = st.session_state.urgent_geo
                    else:
                        st.session_state.urgent_geo_error = str(payload.get("error") or "定位失败")
                        geo_error = st.session_state.urgent_geo_error

        city = st.session_state.get("urgent_city")

        results: List[Dict] = []
        if city:
            results = cached_by_city(city, "医院")[:2]
        elif coords:
            results = cached_urgent_hospitals(round(coords[0], 3), round(coords[1], 3))

        if results:
            # 整张卡片可点：医院列表组件内部判断 App 是否已安装并决定唤起还是回退网页
            NAV(items=results[:2], key="urgent-nav", default=None, card_click=True)
        elif city or (coords is not None) or geo_error:
            st.caption("暂时没有检索到医院，可以直接打开高德地图搜索。")
            st.markdown(
                f'<a class="ua-link" href="{html.escape(poi.search_url("医院"))}" '
                'target="_blank" rel="noopener">在高德地图中搜索附近医院</a>',
                unsafe_allow_html=True,
            )

        if geo_error:
            st.caption("没能获取定位，可以重试，或手动输入城市。")
            if st.button("重新定位", key="urgent-retry-geo", width="stretch"):
                st.session_state.urgent_geo = None
                st.session_state.urgent_geo_error = None
                st.session_state.pop("urgent_geo_ts", None)
                st.rerun()
        with st.form("urgent-city-form"):
            col_city, col_go = st.columns([3, 1])
            typed = col_city.text_input(
                "城市",
                value=city or "",
                placeholder="输入城市即可按城市检索，例如：北京",
                label_visibility="collapsed",
            )
            go = col_go.form_submit_button("搜索")
        if go and typed.strip():
            st.session_state.urgent_city = typed.strip()
            st.rerun()
        if city and st.button("按当前位置重新检索", key="urgent-relocate", width="stretch"):
            st.session_state.urgent_city = None
            st.session_state.urgent_geo = None
            st.session_state.urgent_geo_error = None
            st.session_state.pop("urgent_geo_ts", None)
            st.rerun()

    # ---------- 入口 ----------

    def run(self) -> None:
        # 60 岁以上用户登录后的模式询问：先画弹窗，再画页面。
        # 放在 render_header 之前，apply_mode_choice 改 elder_mode 时才不会被
        # 「控件已实例化，不能再改 session_state」拦住。
        if st.session_state.get(MODE_CHOICE_PENDING_KEY) and current_user():
            self.render_mode_prompt()

        elder = self.elder
        inject_styles(elder)

        # 桌面端左侧导航（移动端由 CSS 隐藏），必须在顶栏之前画
        self.render_shell_nav()

        self.render_header()

        # 移动端导航抽屉：顶栏按钮只翻转开关，这里按状态画固定面板
        if st.session_state.get(MOBILE_NAV_KEY):
            self.render_mobile_nav()

        # 头像触发一次：弹窗只在这一轮画出来，下一轮（选完功能）自动收起
        if st.session_state.pop("_menu_open", False):
            self.render_view_menu()

        # 视图由导航按钮 / 头像快捷入口写入 session_state
        view = st.session_state.get("view") or "对话"
        if view not in VIEW_NAMES:
            view = "对话"
        # 老人模式藏起了「工具」和「设备导入」，停在这些页面的话直接回对话页
        if elder and view in ("工具", "设备导入"):
            view = "对话"
        st.session_state.view = view

        # 从自查/历史切回对话页也算"重新进入主页面"，示例问题换一批
        if view == "对话" and st.session_state.get("_view_prev") not in (None, "对话"):
            self.refresh_quick_questions()
        st.session_state._view_prev = view

        # 每个子界面左上角一个「← 返回xx」，点了回上一级。
        # 对话页只在"从别处进来"时才给（比如从历史打开一条对话），工具页由工具自己画。
        if view == "对话":
            if st.session_state.get(NAV_STACK_KEY):
                self.render_back(nav_back_target(), "对话", go_back)
        elif view != "工具":
            self.render_back(nav_back_target(), view, go_back)

        # 打卡 / 看板 / 历史 / 设备导入 / 设备支持切换到自己或已授权的家人；没有家人时不会画出选择器
        if view in ("打卡", "数据看板", "历史", "设备导入"):
            self.render_subject_selector(view)

        # 家庭药箱的补药提醒是发给整个家庭组的：无论在哪个页面，顶上都能看到；
        # 药箱页自己会列出逐条提醒，不重复画。
        if view != "家庭药箱":
            self.render_medicine_notice()

        if view == "自查":
            self.render_selfcheck()
            # 弹框状态放在 session_state 里：用户关掉（on_dismiss）后不再重开
            if st.session_state.get("urgent_open"):
                self.render_urgent_dialog()
            return

        if view == "打卡":
            self.render_checkin()
            return

        if view == "数据看板":
            self.render_dashboard()
            return

        if view == "工具":
            self.render_toolbox()
            return

        if view == "历史":
            self.render_history()
            return

        if view == "家庭组":
            self.render_family()
            return

        if view == "设备导入":
            self.render_device_import()
            return

        if view == "家庭药箱":
            self.render_medicine_cabinet()
            return

        if view == "对话":
            self.render_family_notice()

        if not st.session_state.messages:
            self.render_empty_state()
        else:
            for index, message in enumerate(st.session_state.messages):
                self.render_message(index, message)

        # 环境条放在输入框上方：既能看见当前参考的天气，也不打断对话流
        self.render_env_bar()

        prompt, doc = self.collect_prompt()
        if doc:
            self.answer_file(prompt, doc)
        elif prompt:
            self.answer(prompt)


def migrate_legacy_history_once() -> None:
    """把账号体系上线前的旧对话归到最早注册的账号，避免历史突然消失。

    只在一个浏览器会话里试一次；数据库暂时不可用时下轮再试。
    """
    if st.session_state.get("_history_owner_migrated"):
        return
    try:
        oldest = auth.oldest_user_id()
    except auth.AuthError:
        return
    if oldest:
        try:
            history.migrate_legacy(oldest)
        except OSError:
            return
    st.session_state["_history_owner_migrated"] = True


def main() -> None:
    flush_logout()

    # 写库失败时把翻转过的「默认模式」开关恢复成账号里的值
    if st.session_state.pop(DEFAULT_MODE_REVERT_KEY, False):
        st.session_state.pop("default_elder_mode", None)

    user = current_user()
    if user is None:
        render_auth_page()
        return

    # 正常路径下 sign_in 已经写好；这里兜住热更新等情况下缺少开关值的情况
    if "elder_mode" not in st.session_state:
        st.session_state["elder_mode"] = user["default_mode"] == auth.MODE_ELDER

    migrate_legacy_history_once()
    MumuDoctorApp().run()


if __name__ == "__main__":
    main()
