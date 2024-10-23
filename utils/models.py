from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer


# embedding model

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=MODEL_ID)

# rag chat model
RAG_LLM = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

REASONING_LLM_ID = "gpt-4o"
REASONING_LLM = ChatOpenAI(model=REASONING_LLM_ID)

# semantic model
SEMANTIC_MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')