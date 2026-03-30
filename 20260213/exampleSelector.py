import os

import dotenv
from dashscope import MultiModalEmbedding
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate

dotenv.load_dotenv()


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

    def embed_question(self, text):
        resp = MultiModalEmbedding.call(
            model="multimodal-embedding-v1",
            input=[text],
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        return resp["output"]["embeddings"][0]["embedding"]


embedding_model = QwenEmbedding()

prompt = PromptTemplate.from_template(
    template='输入如下：{input}, 输出如下：{output}',
)

examples = [
    {
        'question': '那个活得更久，甘地还是马丁路德金？',
        'answer': '接下来还有问什么问题呢？追问甘地多少岁去世的？'
    },
    {
        'question': '高中理科，考试有哪几门？',
        'answer': '语数外，生物化学物理'
    },
    {
        'question': '西安电子科技大学大学那个专业比较牛逼？',
        'answer': '通信工程、计算机科学与技术'
    },
]
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embedding_model,
    Chroma,
    k=1,
)

question = '高中文科有哪几门？'
select_examples = example_selector.select_examples({'question': question})
print('select_examples:', select_examples)