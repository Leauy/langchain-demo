# 量子网络设备知识库问答系统 (quantum-rag)

基于 RAG (Retrieval-Augmented Generation) 的知识库问答系统。

## 技术栈

- **后端**: FastAPI + LangChain
- **前端**: React + Ant Design
- **向量数据库**: FAISS
- **对话存储**: SQLite
- **LLM**: 阿里百炼 qwen3.5-flash
- **Embedding**: text-embedding-v3
- **Rerank**: qwen3-vl-rerank

## 快速开始

### 1. 安装后端依赖

```bash
cd quantum-rag
pip install -r backend/requirements.txt
```

### 2. 配置环境变量

确保 `.env` 文件存在并配置了正确的 API Key：

```env
DASHSCOPE_API_KEY=your-api-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 3. 初始化数据库并创建示例数据

```bash
cd quantum-rag
python init_vectordb.py --sample
```

### 4. 启动后端服务

```bash
cd quantum-rag
uvicorn backend.main:app --reload --port 8000
```

后端 API 文档: http://localhost:8000/docs

### 5. 安装前端依赖

```bash
cd quantum-rag/frontend
npm install
```

### 6. 启动前端开发服务器

```bash
npm run dev
```

前端访问地址: http://localhost:3000

## 项目结构

```
quantum-rag/
├── backend/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置管理
│   ├── routers/
│   │   ├── chat.py          # 问答接口
│   │   ├── history.py       # 历史记录接口
│   │   └── datasource.py    # 数据源管理接口
│   ├── services/
│   │   ├── embedding.py     # 向量化服务
│   │   ├── rerank.py        # 重排序服务
│   │   ├── vectorstore.py   # FAISS 向量库管理
│   │   ├── llm.py           # LLM 调用
│   │   └── document_loader.py # 文档加载
│   ├── models/
│   │   └── database.py      # SQLite 模型
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/      # React 组件
│   │   ├── pages/           # 页面
│   │   └── api/             # API 客户端
│   └── package.json
├── data/                    # 数据目录
│   ├── uploads/             # 上传文件
│   └── vector_index/        # FAISS 索引
├── .env                     # 环境变量
├── init_vectordb.py         # 初始化脚本
└── README.md
```

## API 接口

### 问答接口

- `POST /api/chat` - 发送问题并获取回答

### 历史记录接口

- `GET /api/history` - 获取对话列表
- `GET /api/history/{id}` - 获取对话详情
- `DELETE /api/history/{id}` - 删除对话

### 数据源接口

- `GET /api/datasource` - 获取数据源列表
- `POST /api/datasource` - 创建数据源（上传文件）
- `PUT /api/datasource/{id}` - 更新数据源
- `DELETE /api/datasource/{id}` - 删除数据源
- `POST /api/datasource/{id}/reindex` - 重新向量化

## 支持的文件格式

- Excel (.xlsx, .xls)
- PDF
- TXT
- Markdown (.md)

## 功能特点

1. **RAG 问答**: 基于检索增强生成的智能问答
2. **Rerank 重排**: 使用 qwen3-vl-rerank 对检索结果重排序
3. **多数据源管理**: 支持创建、编辑、删除数据源
4. **对话历史**: 自动保存对话历史，支持历史回顾
5. **引用来源**: 回答附带相关文档来源
6. **流式响应**: 支持 SSE 流式输出（可选）
