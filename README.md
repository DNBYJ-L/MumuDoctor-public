# 木木医生 · MumuDoctor（可公开脱敏版）

面向个人与家庭的 AI 健康助手：日常健康对话、健康自查、打卡看板、家庭药箱、设备数据导入与文件解读，都在同一个 Streamlit 应用里完成。

> 这是一个**脱敏导出仓库**：不包含真实服务器地址、部署脚本、SSH 配置、数据库、设备桥接凭据，也不包含原项目的医疗问答知识库。
> 正式公开前请先确定许可证，并自行审查知识库/第三方数据的授权。

## 功能

| 模块 | 说明 |
| --- | --- |
| 对话 | 多轮健康咨询、流式回答、追问推荐、反馈与历史记录 |
| 老人模式 | 更大字号、更少入口、短回答与语音朗读 |
| 账号 | 注册 / 登录、性别年龄、默认模式（正常 / 老年） |
| 家庭组 | 邀请与授权、代家人打卡、查看家人的打卡与问询历史 |
| 自查 | 体温、血压、血糖、心率、睡眠、BMI 等指标判定与解读 |
| 打卡 / 看板 | 指标记录、趋势图、分析报告与详细版 PDF |
| 设备数据 | 导入手机导出的 CSV / JSON；可选桥接工具每日同步 |
| 家庭药箱 | 药品 / 卫材管理、数量与效期提醒、药盒拍照识别、照片存档 |
| 文件解读 | PDF、Word、Excel、图片；图片优先交给视觉模型，OCR 兜底 |
| 环境建议 | 自动定位并参考当地气温、空气质量，给出提醒 |
| 工具 | BMI、基础代谢、孕期、排卵、儿童生长、饮酒、咖啡因、血压、剂量换算 |

## 目录

```text
application/          Streamlit 应用与业务模块
  mumudoctor_web.py   入口：页面、视图与交互
  auth.py             账号与 MySQL 连接
  checkin.py          打卡指标与统计
  family.py           家庭组
  medicine.py         家庭药箱
  medicine_scan.py    药盒照片识别
  document.py         上传文件解析
  environment.py      天气与空气质量
  llm.py              模型调用（Ollama / OpenAI 兼容）
  report.py           PDF 报告
  theme*.css          页面样式
  geo/ nav/ speak/    浏览器定位、医院列表、语音朗读组件
data/
  knowledge/          可选知识库（本仓库不附带数据，见该目录 README）
.streamlit/           Streamlit 主题配置
```

## 本地运行

需要 Python 3.13+、MySQL 8（账号体系与打卡数据都在 MySQL 里）。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

# 数据库连接（不要把真实密码提交进仓库）
set MYSQL_HOST=127.0.0.1
set MYSQL_PORT=3306
set MYSQL_USER=mumudoctor
set MYSQL_PASSWORD=<数据库密码>
set MYSQL_DATABASE=mumudoctor

streamlit run application/mumudoctor_web.py
```

模型后端由环境变量决定，默认走本地 Ollama：

| 变量 | 说明 |
| --- | --- |
| `LLM_PROVIDER` | `ollama`（默认）或 `openai` |
| `OLLAMA_HOST` | 本地 Ollama 地址，默认 `http://localhost:11434` |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | 云端接口地址、密钥、默认模型 |
| `LLM_REASONING_EFFORT` | 仅对支持该参数的网关设置；不需要时留空 |
| `EMBED_*` / `RAG_EMBED_MODEL` | 向量模型配置，知识库使用 bge-m3 时需保持一致 |
| `AMAP_KEY` | 高德地图 key，用于定位反查与附近医院 |
| `TAVILY_API_KEY` | 可选，本地知识库命中不足时补充网页检索 |

### 可选依赖

- PDF：`pypdf`
- Word：标准库即可
- Excel：`openpyxl`
- 图片 OCR：系统 `tesseract` + `chi_sim` 语言包、`pdftoppm`
- 药盒识别：`pillow-heif`、`rapidocr-onnxruntime`，系统 `libgl1`

## 知识库

本仓库**不附带** `data/knowledge/documents.json` 和 `embeddings.npy`。没有知识库时 RAG 会自动降级为空结果，对话、打卡、药箱等功能仍可运行。

如需启用，请把有合法授权的数据放到 `data/knowledge/`，格式见 [data/knowledge/README.md](data/knowledge/README.md)。

## 设备数据

仓库不包含 `device_bridge/`、小米账号登录态或任何 `passToken`。设备导入只接受手机导出的 CSV / JSON；每日自动同步是可选的本地部署能力，需要自行放置第三方桥接工具和凭据。

## 安全与隐私

- 不在仓库中提交 `.env`、数据库密码、模型 API key、设备登录态或用户数据。
- 上传的文件只存在于当前会话内存，不写库；解读结果按正常对话历史保存。
- 定位坐标只放在会话里，不写库、不进对话历史。
- 部署脚本、服务器信息、真实数据库与设备配置不属于本脱敏仓库。

## 免责声明

本项目是健康信息辅助工具，**不是医疗器械，不构成诊断、处方或治疗建议**。AI 生成内容可能不准确；出现急症、持续或加重的不适，请及时前往正规医疗机构就诊。用药、剂量、孕期、儿童等问题请咨询医生或药师。

## 许可证

本仓库暂未附许可证。公开前请项目所有者选择并添加合适的许可证；第三方数据、字体、图标和依赖各自的许可证需要单独确认。
