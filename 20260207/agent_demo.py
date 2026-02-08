import traceback
from http.client import responses

import dotenv, os
from langchain_classic.agents import AgentExecutor
from langchain_community.agent_toolkits import create_openapi_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from torch import hub

dotenv.load_dotenv()

from langchain_core.embeddings import Embeddings
from dashscope import MultiModalEmbedding
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class QwenEmbedding(Embeddings):
    def embed_documents(self, texts):
        vectors = []
        BATCH_SIZE = 16  # <=20

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]

            resp = MultiModalEmbedding.call(
                model="multimodal-embedding-v1",
                input=batch,
                api_key=os.getenv("DASHSCOPE_API_KEY")
            )

            embs = resp["output"]["embeddings"]
            vectors.extend([e["embedding"] for e in embs])

        return vectors

    def embed_query(self, text):
        resp = MultiModalEmbedding.call(
            model="multimodal-embedding-v1",
            input=[text],
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        return resp["output"]["embeddings"][0]["embedding"]


def process_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [Document(page_content=f.read())]


docs = process_text_file("E:/learn/AI-demo/langchain-demo/LangChain.md")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)
try:
    embedding_model = QwenEmbedding()
    vector = FAISS.from_documents(documents, embedding_model)
    mm = vector.similarity_search("langchain smith")
    print(mm)
    question = 'LangGraph有什么功能？'
    retriver = vector.as_retriever(search_kwargs={'k': 3})
    docs = retriver.invoke(question)

    retriver_tool = create_retriever_tool(
        retriver,
        "CivilCodeRetriver",
        '搜索LangGraph有什么功能，关于LangGraph相关的任何问题，必须使用此工具'

    )
    llm = ChatOpenAI(model='kimi-k2.5', api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'))
    tools = [retriver_tool]
    prompt = hub.pull('hwchase17/openai-functions-agent')
    agent = create_openapi_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.invoke(question)
except Exception as e:
    import pdb;pdb.set_trace()
    traceback.print_exc()
