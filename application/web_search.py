#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网页搜索增强模块（Tavily）

在本地 RAG 检索命中不足时，调用 Tavily 实时搜索权威网页作为补充知识，
与 application/rag.py 的本地知识库形成「混合增强」。

用法:
    from web_search import web_search
    docs = web_search("高血压平时要注意什么", max_results=3)

配置:
    通过环境变量 TAVILY_API_KEY 提供 Tavily API key（https://tavily.com 申请，
    免费额度 1000 credits/月）。未配置时 web_search 直接返回空列表，静默降级，
    不影响本地 RAG 逻辑。
"""

import os
import requests
from typing import List

TAVILY_URL = "https://api.tavily.com/search"


def web_search(query: str, max_results: int = 3, timeout: float = 15.0) -> List[str]:
    """调用 Tavily 搜索，返回可用于注入的知识文本列表（失败时返回空列表）。

    返回的每条文本为纯字符串：优先 Tavily 的 AI 摘要（answer），随后为网页
    标题 + 正文。总条数不超过 max_results。
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        resp = requests.post(
            TAVILY_URL,
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": True,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    docs: List[str] = []

    # Tavily 的 AI 摘要质量最高、最易被小模型遵循，放在最前
    answer = (data.get("answer") or "").strip()
    if answer:
        docs.append(answer)

    for r in data.get("results", []):
        content = (r.get("content") or "").strip()
        if not content:
            continue
        title = (r.get("title") or "").strip()
        if title:
            content = f"{title}\n{content}"
        docs.append(content)
        if len(docs) >= max_results:
            break

    return docs
