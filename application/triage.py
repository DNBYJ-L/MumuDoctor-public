#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
症状 → 就诊科室 + 紧迫度。

纯规则，不调模型：每轮回答后都可能要展示医院入口，多一次模型调用不划算，
而且科室映射是确定性的，规则比小模型更稳。
"""

from typing import Dict, List, Tuple

# 命中顺序即优先级，越靠前越先匹配
_DEPT_RULES: List[Tuple[Tuple[str, ...], str]] = [
    (
        (
            "胸痛", "胸闷", "胸口", "心口", "心悸", "心慌", "心慌意乱",
            "血压", "高血压", "低血压", "心跳", "憋气",
        ),
        "心血管内科",
    ),
    (("头晕", "眩晕", "头痛", "偏头痛", "手麻", "脚麻", "晕厥", "抽搐"), "神经内科"),
    (("咳嗽", "咳痰", "气喘", "哮喘", "呼吸困难", "肺", "气短"), "呼吸内科"),
    (("腹痛", "胃痛", "腹泻", "拉肚子", "便秘", "恶心", "呕吐", "反酸", "胀气"), "消化内科"),
    (("血糖", "糖尿病", "甲亢", "甲减", "甲状腺", "尿酸", "痛风"), "内分泌科"),
    (("皮疹", "疹", "瘙痒", "湿疹", "荨麻疹", "痘", "脱发", "皮肤"), "皮肤科"),
    (("关节", "腰痛", "腰疼", "颈椎", "骨折", "扭伤", "腿疼", "膝盖", "肩"), "骨科"),
    (("失眠", "睡不着", "焦虑", "抑郁", "情绪", "压力大"), "心理科"),
    (("牙", "牙龈", "口腔", "嘴破"), "口腔科"),
    (("眼睛", "视力", "眼干", "眼花", "红眼"), "眼科"),
    (("耳朵", "耳鸣", "鼻", "咽", "喉咙", "扁桃体"), "耳鼻喉科"),
    (("月经", "痛经", "白带", "怀孕", "孕", "妇科"), "妇科"),
    (("宝宝", "小孩", "儿童", "孩子", "婴儿"), "儿科"),
]

# 出现这些词说明不是"要不要去看看"，而是"尽快去"
_URGENT_HINTS = (
    "剧烈", "难忍", "得厉害", "受不了", "持续加重", "越来越重",
    "一直不好", "反复发作", "高烧", "发烧不退", "大出血", "昏迷",
    "意识", "抽搐", "肿块", "消瘦",
)

DEFAULT_DEPARTMENT = "全科 / 内科"


def suggest_department(question: str) -> str:
    """按症状关键词猜一个就诊科室。"""
    text = question or ""
    for keywords, dept in _DEPT_RULES:
        if any(k in text for k in keywords):
            return dept
    return DEFAULT_DEPARTMENT


def assess(question: str, blocked: bool = False) -> Dict[str, str]:
    """返回 {level, department, label, advice}。

    level: emergency（红线拦截）/ soon（建议尽快）/ routine（常规）
    """
    department = suggest_department(question)

    if blocked:
        return {
            "level": "emergency",
            "department": "急诊",
            "label": "需要立即处理",
            "advice": "请直接拨打 120 或前往最近的急诊科，不要等待。",
        }

    text = question or ""
    if any(hint in text for hint in _URGENT_HINTS):
        return {
            "level": "soon",
            "department": department,
            "label": "建议尽快就诊",
            "advice": f"建议今明两天内到{department}看一下，若明显加重请直接去急诊。",
        }

    return {
        "level": "routine",
        "department": department,
        "label": "需要时可就近就诊",
        "advice": f"如果方便，可以到附近的{department}或社区卫生服务中心看看。",
    }


def visit_guide(level: str, department: str) -> List[Tuple[str, str]]:
    """就诊流程。急诊和门诊走两套完全不同的步骤。"""
    if level == "emergency":
        return [
            ("直接去急诊分诊台", "不要排队挂号，先到分诊台说明情况，护士会按危重程度安排。"),
            ("带上证件", "身份证、医保卡；有既往病历和正在吃的药一并带上。"),
            ("说清三件事", "什么时候开始的、最难受的是什么、有没有在用药。"),
            ("先救治后补手续", "急诊遵循先抢救后付费，不要因为没号或没带钱而等待。"),
            ("陪同与留观", "尽量有人陪同；医生建议留观时不要坚持回家。"),
        ]

    return [
        ("预约挂号", f"用医院公众号、官方 APP 或 114 预约{department}门诊，优先选上午号。"),
        ("带齐材料", "身份证、医保卡、既往检查报告，以及正在服用的药盒。"),
        ("提前 30 分钟到院", "自助机取号，到分诊台报到后候诊。"),
        ("就诊时说清症状", "起始时间、诱因、变化过程、用过什么药、有无家族史。"),
        ("按医嘱检查取药", "化验和影像多在当天出结果，取药时核对用法用量。"),
        ("记录并复诊", "记下诊断和用药，按医生约定的时间复诊。"),
    ]
