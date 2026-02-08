import traceback
from http.client import responses

import dotenv, os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

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
    prompt_template = '''
    你是一个问答机器人。
    你的任务是根据下述给定的已知信息回答用户问题
    确保你的回复完全根据下述已知信息，不要编造答案。
    如果下述已知信息不足以回答用户的问题，请直接回复：我不知道

    已知信息:
    {info}

    用户问题：
    {question}
    '''
    template = PromptTemplate.from_template(prompt_template)

    prompt = template.format(
        info=docs,
        question=question,
    )
    llm = ChatOpenAI(model='kimi-k2.5', api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'))
    response = llm.invoke(prompt)
    print(response.content)
except Exception as e:
    traceback.print_exc()
