#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一模型调用层：本地 Ollama / OpenAI 兼容云端。

默认走本地 Ollama；设置 LLM_PROVIDER=openai 后改为调用云端 OpenAI 兼容接口
（DeepSeek、硅基流动、通义、智谱、OpenAI 都支持）。

环境变量
--------
LLM_PROVIDER     ollama | openai        默认 ollama
OLLAMA_HOST      本地 Ollama 地址        默认 http://localhost:11434
LLM_BASE_URL     云端 base url          https://api.openai.com/v1
LLM_API_KEY      云端 api key
LLM_MODEL        云端默认模型
LLM_MODELS       云端可选模型，逗号分隔；不填则只提供 LLM_MODEL
LLM_REASONING_EFFORT  可选：reasoning_effort（off/low/medium/high…）；
                      设为 off 可避免 reasoning token 吃掉 max_tokens
EMBED_PROVIDER   默认跟随 LLM_PROVIDER
EMBED_BASE_URL   默认跟随 LLM_BASE_URL
EMBED_API_KEY    默认跟随 LLM_API_KEY
EMBED_MODEL      云端 embedding 模型，默认 bge-m3
RAG_EMBED_MODEL  本地 embedding 模型，默认 bge-m3

为什么 embedding 必须指定模型：知识库里的向量是 bge-m3（1024 维）算出来的，
换一个模型（哪怕维度相同）查询向量和文档向量就不在同一语义空间，相似度会失效。
"""

import json
import os
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import requests

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", PROVIDER).strip().lower()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip()
LLM_MODELS = [m.strip() for m in os.environ.get("LLM_MODELS", "").split(",") if m.strip()]
# 可选：部分 OpenAI 兼容网关支持 reasoning_effort（off/low/medium/high/...）。
# 对 reasoning 模型设为 off，可以把 max_tokens 全部留给正文，避免小预算时正文为空。
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "").strip().lower()

EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", LLM_BASE_URL).rstrip("/")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", LLM_API_KEY).strip()
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3").strip()

# 本地模式的默认模型
_OLLAMA_DEFAULT_MODEL = os.environ.get("MUMUDOCTOR_MODEL", "DeepSeek-chat:latest")
_OLLAMA_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "bge-m3")

# 模型选择器里不出现的模型：embedding 模型与内部模型别名
HIDDEN_MODEL_PREFIXES = ("bge", "mumudoctor", "nomic")


def is_cloud() -> bool:
    return PROVIDER == "openai"


def is_embed_cloud() -> bool:
    return EMBED_PROVIDER == "openai"


def default_model() -> str:
    if is_cloud():
        return LLM_MODEL or "gpt-4o-mini"
    return _OLLAMA_DEFAULT_MODEL


def list_chat_models(names: Sequence[str]) -> List[str]:
    """过滤出可对话的模型名，保持原有顺序。"""
    return [
        name for name in names if name and not name.lower().startswith(HIDDEN_MODEL_PREFIXES)
    ]


# ===================== 探活 / 模型列表 =====================

def probe() -> Tuple[bool, List[str]]:
    """返回 (是否可用, 可选模型列表)。云端不发起网络请求，只检查 key 是否配置。"""
    if is_cloud():
        models = LLM_MODELS or ([LLM_MODEL] if LLM_MODEL else [])
        return bool(LLM_API_KEY and models), models

    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2.5)
        if resp.status_code == 200:
            names = [m.get("name", "") for m in resp.json().get("models", [])]
            return True, list_chat_models(names)
    except requests.exceptions.RequestException:
        pass
    return False, []


# ===================== 对话 =====================

def _ollama_payload(
    messages: List[Dict],
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    stream: bool,
) -> Dict:
    return {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_predict": int(max_tokens),
        },
    }


def _openai_payload(
    messages: List[Dict],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stream: bool,
) -> Dict:
    # OpenAI 协议没有 top_k，这里不下发
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": int(max_tokens),
    }
    if LLM_REASONING_EFFORT:
        # 只对明确支持该参数的服务配置；不配置时保持标准 OpenAI 请求
        payload["reasoning_effort"] = LLM_REASONING_EFFORT
    return payload


def _openai_headers(key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def iter_openai_chunks(lines) -> Iterator[str]:
    """解析 OpenAI 兼容接口的 SSE 流，逐段吐出正文。"""
    for raw in lines:
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip() if isinstance(raw, bytes) else str(raw).strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content")
        if piece:
            yield piece


def chat_stream(
    messages: List[Dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    max_tokens: int = 512,
    timeout: float = 120.0,
) -> Iterator[str]:
    """流式对话。失败时以 [连接错误] 开头的文本返回，调用方按普通文本处理即可。"""
    model = model or default_model()

    try:
        if is_cloud():
            resp = requests.post(
                f"{LLM_BASE_URL}/chat/completions",
                json=_openai_payload(messages, model, temperature, top_p, max_tokens, True),
                headers=_openai_headers(LLM_API_KEY),
                stream=True,
                timeout=timeout,
            )
            resp.raise_for_status()
            yield from iter_openai_chunks(resp.iter_lines())
            return

        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=_ollama_payload(messages, model, temperature, top_p, top_k, max_tokens, True),
            stream=True,
            timeout=timeout,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if "message" in data and "content" in data["message"]:
                yield data["message"]["content"]
            if data.get("done", False):
                break

    except requests.exceptions.RequestException as exc:
        yield f"\n\n[连接错误] {exc}"
    except Exception as exc:  # noqa: BLE001
        yield f"\n\n[未知错误] {exc}"


def chat_once(
    messages: List[Dict],
    model: Optional[str] = None,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 40,
    max_tokens: int = 400,
    timeout: float = 60.0,
) -> str:
    """非流式对话，返回整段文本；失败返回空字符串。"""
    model = model or default_model()

    try:
        if is_cloud():
            resp = requests.post(
                f"{LLM_BASE_URL}/chat/completions",
                json=_openai_payload(messages, model, temperature, top_p, max_tokens, False),
                headers=_openai_headers(LLM_API_KEY),
                timeout=timeout,
            )
            resp.raise_for_status()
            choice = (resp.json().get("choices") or [{}])[0]
            return ((choice.get("message") or {}).get("content") or "").strip()

        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=_ollama_payload(messages, model, temperature, top_p, top_k, max_tokens, False),
            timeout=timeout,
        )
        resp.raise_for_status()
        return ((resp.json().get("message") or {}).get("content") or "").strip()

    except Exception:  # noqa: BLE001
        return ""


# ===================== 向量化 =====================

def embed(texts: List[str], timeout: float = 60.0) -> List[List[float]]:
    """把文本编码成向量。失败时抛异常，由调用方决定是否降级。"""
    if is_embed_cloud():
        resp = requests.post(
            f"{EMBED_BASE_URL}/embeddings",
            json={"model": EMBED_MODEL, "input": texts},
            headers=_openai_headers(EMBED_API_KEY),
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json().get("data") or []
        # 部分服务不保证顺序，按 index 排序
        data = sorted(data, key=lambda item: item.get("index", 0))
        return [item.get("embedding") or [] for item in data]

    resp = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": _OLLAMA_EMBED_MODEL, "input": texts},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]
