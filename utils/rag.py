from operator import itemgetter
from pydantic import BaseModel, InstanceOf

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


class RAGRunnables(BaseModel):
    rag_prompt_template: InstanceOf[ChatPromptTemplate]
    vector_store: InstanceOf[QdrantVectorStore]
    llm: InstanceOf[ChatOpenAI]
        

def create_rag_chain(rag_prompt_template, vector_store, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    rag_chain = ({"context": itemgetter("question") | retriever, "question": itemgetter("question")}
                    | RunnablePassthrough.assign(context=itemgetter("context"))
                    | {"response": rag_prompt_template | llm | StrOutputParser(), "context": itemgetter("context")})
    return rag_chain

