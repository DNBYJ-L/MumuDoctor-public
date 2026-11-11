#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
循证 RAG 检索模块

向量化走 llm.embed：本地模式下调用 Ollama 的 bge-m3，云端模式下调用
OpenAI 兼容的 embeddings 接口（模型同样必须是 bge-m3，否则与现有
embeddings.npy 不在同一语义空间）。相似度计算仍用 numpy，无额外依赖。

用法:
    from application.rag import get_retriever
    retriever = get_retriever()
    docs = retriever.retrieve("最近总是头晕怎么办", top_k=3)
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from llm import embed as embed_texts

# 项目根目录（application/ 的上一级）
PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"

_retriever: Optional["MedicalRetriever"] = None


class MedicalRetriever:
    """
    医学知识检索器（懒加载 + 单例）

    知识库文件放在 data/knowledge/：
    - documents.json : [{"id", "text", "content"}, ...]
    - embeddings.npy : (N, dim) float32，已做 L2 归一化（bge-m3）
    """

    def __init__(self, knowledge_dir: Optional[Path] = None):
        self.knowledge_dir = Path(knowledge_dir) if knowledge_dir else KNOWLEDGE_DIR
        self.documents: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self._loaded = False

    def load(self) -> bool:
        """加载知识库文件，成功返回 True"""
        docs_path = self.knowledge_dir / "documents.json"
        emb_path = self.knowledge_dir / "embeddings.npy"
        if not docs_path.exists() or not emb_path.exists():
            return False

        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            self.embeddings = np.load(emb_path)
        except Exception:
            return False

        self._loaded = True
        return True

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0) -> List[str]:
        """检索与 query 最相关的 top_k 条知识内容（失败时返回空列表）。

        min_score: 余弦相似度阈值，低于该值的知识视为不相关而丢弃，0.0 表示不过滤。
        """
        if not self._loaded and not self.load():
            return []
        if not self.documents or self.embeddings is None:
            return []

        try:
            qvec = np.asarray(embed_texts([query])[0], dtype=np.float32)
            # 归一化 query 向量，使点积等价于余弦相似度
            norm = np.linalg.norm(qvec)
            if norm > 0:
                qvec = qvec / norm
        except Exception:
            return []  # 向量化失败时静默降级

        # 维度对不上说明 embedding 模型换了，直接放弃检索而不是算出噪声
        if qvec.shape[0] != self.embeddings.shape[1]:
            return []

        sims = self.embeddings @ qvec
        # 按相似度降序取 top_k，并丢弃低于阈值的条目（已排序，遇低分即可停止）
        picked = []
        for i in np.argsort(-sims):
            if sims[i] < min_score:
                break
            picked.append(i)
            if len(picked) >= top_k:
                break

        return [self.documents[i]["content"] for i in picked]


def get_retriever() -> MedicalRetriever:
    """获取全局单例检索器"""
    global _retriever
    if _retriever is None:
        _retriever = MedicalRetriever()
    return _retriever
