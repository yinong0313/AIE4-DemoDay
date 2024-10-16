from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# embedding model

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=MODEL_ID)

# rag chat model
RAG_LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

REASONING_LLM = ChatOpenAI(model="gpt-3.5-turbo")