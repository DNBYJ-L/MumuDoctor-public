#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
追问推荐：给定一轮问答，产出 3 条患者可能想接着问的问题。

纯逻辑模块，不依赖 Streamlit，便于单独测试。模型输出经常不合规
（带序号、带解释、返回对象而不是数组），因此解析层必须宽容：
先找 JSON 数组，失败则按行兜底，最后再统一做长度/重复/安全过滤。
"""

import json
import re
from typing import List

from safety import check_safety

FOLLOWUP_SYSTEM = """你是一名全科问诊助手，负责在医生回答之后，替患者想出他接下来最可能追问的问题。

要求：
1. 只输出 3 条，患者第一人称口语，每条不超过 18 个汉字，不加序号；
2. 必须能从刚才的回答自然延伸，优先这几类：具体怎么做、要不要去医院、多久能好、平时注意什么、需要做什么检查；
3. 禁止出现药名、剂量、具体诊断结论，也不要复述已经问过的问题；
4. 只输出 JSON 数组，不要任何解释，例如：["要不要去医院看看","平时要注意什么","大概多久能好"]"""

MIN_LEN = 4
MAX_LEN = 30
MAX_ITEMS = 3

# 模型不可用或输出不可解析时的兜底，按问题长度错开，避免每次都一模一样
FALLBACKS = [
    "平时要注意些什么？",
    "需要做哪些检查？",
    "大概多久能改善？",
    "这种情况要不要去医院？",
    "饮食上需要忌口吗？",
]

# 模型给的"问题"经常是半句话，把常见前缀去掉
_PREFIX_RE = re.compile(r"^\s*(?:[-*•·]|\d+\s*[.、)）]|问题\s*\d+\s*[：:])\s*")

# 一句"像问题"的话至少要有疑问或求指引的痕迹，否则多半是模型的解释性废话
_QUESTION_HINTS = (
    "？", "?", "要不要", "需不需要", "需要", "多久", "怎么", "如何", "能不能",
    "可不可以", "是否", "平时", "注意", "忌口", "检查", "该", "有没有", "适合",
)


def _looks_like_question(text: str) -> bool:
    return any(hint in text for hint in _QUESTION_HINTS)


def _clean(item: str) -> str:
    text = _PREFIX_RE.sub("", str(item or "")).strip()
    text = text.strip("「」『』\"“”'‘’ ")
    return text.replace("\n", " ").strip()


def _from_json(text: str) -> List[str]:
    match = re.search(r"\[[\s\S]*?\]", text)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [_clean(x) for x in data]
    return []


def _from_lines(text: str) -> List[str]:
    return [_clean(line) for line in text.splitlines() if line.strip()]


def parse_followups(text: str, question: str = "") -> List[str]:
    """把模型输出解析成最多 3 条安全的追问；无法解析时返回空列表。"""
    candidates = _from_json(text) or _from_lines(text)

    result: List[str] = []
    for item in candidates:
        if not (MIN_LEN <= len(item) <= MAX_LEN):
            continue
        if not _looks_like_question(item):
            continue
        if item == question or item in result:
            continue
        # 推荐出去的问题将来会被点击，必须自己先能过安全闸
        blocked, _ = check_safety(item)
        if blocked:
            continue
        result.append(item)
        if len(result) >= MAX_ITEMS:
            break
    return result


def fallback_followups(question: str = "", count: int = MAX_ITEMS) -> List[str]:
    """模型不可用时的兜底追问：按问题内容错开起点，避免每次完全一样。"""
    start = sum(ord(ch) for ch in question) % len(FALLBACKS) if question else 0
    return [FALLBACKS[(start + i) % len(FALLBACKS)] for i in range(count)]


def build_messages(question: str, answer: str) -> List[dict]:
    """组装第二次调用的消息体。"""
    return [
        {"role": "system", "content": FOLLOWUP_SYSTEM},
        {"role": "user", "content": f"患者问：{question}\n\n医生答：{answer}"},
    ]
