# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangChain learning/demo project containing example code and documentation for building LLM-powered applications with the LangChain framework. The project is organized by date-based folders, each containing demos for different LangChain concepts.

## Environment Setup

This project uses `uv` for package management. Install dependencies:

```bash
uv sync
```

## Running Demo Files

Demo files are located in date-prefixed folders (e.g., `20260208/`, `20260212/`). Run any demo:

```bash
uv run python 20260208/chatModelDemo.py
uv run python 20260212/combPrompt.py
```

## Environment Variables

Each date folder contains its own `.env` file with API credentials. The project uses OpenAI-compatible APIs with support for multiple providers. Required environment variables:

```
OPENAI_API_KEY=<your-api-key>
OPENAI_BASE_URL=<api-base-url>
DASHSCOPE_API_KEY=<alibaba-dashscope-key>  # For Alibaba/Qwen models
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## LLM Provider Configuration

The project supports multiple LLM providers through OpenAI-compatible interfaces:

- **OpenAI/DashScope**: `ChatOpenAI` with appropriate `base_url`
- **DeepSeek**: Set `OPENAI_BASE_URL=https://api.deepseek.com`
- **ZhipuAI, Baidu, Alibaba Bailian**: See examples in `20260209/`

Common model initialization pattern:
```python
from langchain_openai import ChatOpenAI
import dotenv, os

dotenv.load_dotenv()
llm = ChatOpenAI(
    model='qwen3-max-2026-01-23',  # or 'deepseek-chat', 'gpt-4', etc.
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_BASE_URL')
)
```

## Architecture & Key Concepts

### LangChain Six Core Modules

1. **Model I/O** (`20260208/`, `20260209/`, `20260210/`): Prompts, models, output parsers
2. **Chains**: Connecting components using LCEL (`|` operator)
3. **Memory** (`20260212/store_history.py`): Conversation history management
4. **Agents** (`20260207/agent_demo.py`): Autonomous tool-using AI
5. **Retrieval** (`20260207/vector_store.py`, `20260207/rag_demo.py`): RAG with vector stores
6. **Callbacks**: Monitoring and logging

### Common Patterns

**Chain construction with LCEL:**
```python
chain = prompt | llm | output_parser
result = chain.invoke({'input': 'your question'})
```

**Streaming responses:**
```python
for chunk in llm.stream('question'):
    print(chunk.content, end='', flush=True)
```

**Vector store with FAISS:**
```python
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(documents, embedding_model)
results = vectorstore.similarity_search('query')
```

## Documentation

Detailed learning notes are in markdown files:
- `LangChain.md` - Framework overview and concepts
- `LangChain-ModelIO.md` - Model I/O module details
- `LangChain-Prompt Template.md` - Prompt template usage

---

## 量子网络设备知识库问答系统 (quantum-rag)

### 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    React SPA (Ant Design)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ 左侧历史列表 │  │  对话主区域  │  │  引用来源展示区     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │ HTTP API
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ /chat    │  │ /history │  │ /datasource│  │ /sources   │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌────────────────┐
│ FAISS 向量库   │   │ SQLite 历史存储  │   │ 阿里百炼 API   │
│ (数据源文档)   │   │ (对话记录)       │   │ qwen3.5-flash  │
│               │   │                 │   │ text-embedding │
└───────────────┘   └─────────────────┘   │ qwen3-vl-rerank│
                                          └────────────────┘
```

### 技术选型

| 组件 | 技术选型 |
|------|---------|
| 前端 | React SPA + Ant Design |
| 后端 | FastAPI |
| 向量数据库 | FAISS 本地文件 |
| 对话存储 | SQLite |
| Embedding | 百炼 text-embedding-v3 (在线) |
| Rerank | 百炼 qwen3-vl-rerank (在线) |
| Chat | 百炼 qwen3.5-flash (在线) |
| 数据源 | 支持 Excel、PDF、TXT、Markdown |

### 项目目录结构

```
quantum-rag/
├── backend/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置加载 (.env)
│   ├── routers/
│   │   ├── chat.py          # 问答接口
│   │   ├── history.py       # 历史记录接口
│   │   └── datasource.py    # 数据源管理接口
│   ├── services/
│   │   ├── embedding.py     # 向量化服务 (text-embedding-v3)
│   │   ├── rerank.py        # 重排序服务 (qwen3-vl-rerank)
│   │   ├── vectorstore.py   # FAISS 向量库管理
│   │   └── llm.py           # LLM 调用 (qwen3.5-flash)
│   ├── models/
│   │   └── database.py      # SQLite 模型
│   ├── data/
│   │   └── vector_index/    # FAISS 索引文件
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatBox.jsx      # 对话输入框
│   │   │   ├── MessageList.jsx  # 消息列表
│   │   │   ├── HistoryPanel.jsx # 左侧历史
│   │   │   ├── SourceRef.jsx    # 引用来源组件
│   │   │   └── DataSourceCard.jsx # 数据源卡片
│   │   ├── pages/
│   │   │   ├── Chat.jsx         # 问答页面
│   │   │   └── DataSource.jsx   # 数据源管理页面
│   │   ├── App.jsx
│   │   └── api/
│   └── package.json
├── .env                       # API 配置
└── init_vectordb.py          # 初始化向量数据库脚本
```

### 问答流程

```
用户问题
    │
    ▼
┌─────────────────┐
│ 问题向量化       │ (text-embedding-v3 在线)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ FAISS 检索      │ → 召回 Top-20 候选文档
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Rerank 重排序   │ → 精排 Top-5 最相关文档
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 构建提示词       │ (上下文 + 问题)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ LLM 生成回答    │ (qwen3.5-flash)
└─────────────────┘
    │
    ▼
返回回答 + 引用来源
```

### 环境变量配置 (.env)

```env
DASHSCOPE_API_KEY=sk-xxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
RERANK_MODEL=qwen3-vl-rerank
CHAT_MODEL=qwen3.5-flash
```

### 数据源管理功能

- **增**：上传文档（Excel/PDF/TXT/Markdown），自动向量化
- **删**：删除数据源及其向量索引
- **改**：编辑数据源名称、描述
- **查**：查看数据源列表、统计信息（文档数、模块数等）
- **重新向量化**：重新处理文档内容

### 前端页面设计

**左侧边栏：**
- 系统名称
- 菜单：数据源管理 / 知识问答
- 对话历史列表

**数据源管理页面：**
- 数据源卡片网格展示
- 卡片信息：名称、状态（就绪/处理中）、描述、统计（文档数、模块数、向量维度）
- 操作：编辑、重新向量化、删除
- 新增数据源入口

**知识问答页面：**
- 消息列表（用户/AI对话）
- 引用来源展示（模块、子模块、原文片段、相关度）
- 数据源选择下拉框
- 输入框 + 发送按钮
- 新建对话按钮

### API 接口设计

```
POST /api/chat              # 问答接口
GET  /api/history           # 获取对话历史列表
GET  /api/history/{id}      # 获取单个对话详情
DELETE /api/history/{id}    # 删除对话

GET  /api/datasource        # 获取数据源列表
POST /api/datasource        # 创建数据源（上传文件）
PUT  /api/datasource/{id}   # 更新数据源信息
DELETE /api/datasource/{id} # 删除数据源
POST /api/datasource/{id}/reindex  # 重新向量化
```

### 响应数据结构

**问答响应：**
```json
{
  "answer": "根据需求文档，QKM-S600等27个型号...",
  "sources": [
    {
      "module": "引擎管理",
      "sub_module": "SSH采集引擎",
      "content": "采集引擎适配QKM-S600...",
      "score": 0.89
    }
  ]
}
```
