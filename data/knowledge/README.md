# 知识库（可选）

本目录默认不包含知识库数据。

如果要启用 RAG，请准备以下两个文件，并确保数据来源和再分发授权合法：

- `documents.json`：数组，每条包含 `id`、`text`、`content`；`text` 是问题/标题，`content` 是检索时注入的知识正文。
- `embeddings.npy`：`(N, dim)` 的 float32 矩阵，行顺序与 `documents.json` 一一对应，向量需与查询模型（默认 bge-m3）在同一语义空间。

示例：

```json
[
  {"id": 0, "text": "示例问题", "content": "示例知识正文"}
]
```

加载逻辑见 `application/rag.py`；文件缺失时检索返回空列表，应用不会崩溃。

> 不要在这里提交来源不明的抓取数据、广告、联系方式或包含个人健康信息的内容。
