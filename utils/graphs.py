from utils.agent_helper import *
from utils.models import REASONING_LLM, RAG_LLM, EMBEDDING_MODEL
from utils.rag import *
from utils.vector_store import *
from utils.prompts import QUERY_AGENT_PROMPT, TEXTBOOK_RAG_PROMPT

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from langgraph.graph import END, StateGraph

import functools
import operator
from typing import Annotated, List, TypedDict
import os

def get_scholar_query_node(llm=REASONING_LLM, prompt=QUERY_AGENT_PROMPT, name="ScholarQuery", top_k_results = 2, load_max_docs = 2):
    
    api_wrapper = SemanticScholarAPIWrapper(top_k_results, load_max_docs)
    semantic_query_tool = SemanticScholarQueryRun(api_wrapper=api_wrapper)

    query_agent = create_agent(llm, [semantic_query_tool], prompt)
    query_node = functools.partial(agent_node, agent=query_agent, name=name)
    return query_node


def create_textbook_rag(path='data/text_books/Textbook-of-Diabetes-2024-shortened.pdf'):

    textbook_documents = get_markdown_documents(path, chunk_size=500, chunk_overlap=50)

    rag_runnables = RAGRunnables(
                                rag_prompt_template = ChatPromptTemplate.from_template(TEXTBOOK_RAG_PROMPT),
                                vector_store = get_vector_store(textbook_documents, EMBEDDING_MODEL, emb_dim=384, collection_name='textbook_collection'),
                                llm = RAG_LLM
                            )
    textbook_chain = create_rag_chain(rag_runnables.rag_prompt_template, 
                                        rag_runnables.vector_store, 
                                        rag_runnables.llm)
    return textbook_chain


def create_paper_rag(folder='data/literature'):
   
    paths = [os.path.join(folder, file) for file in  os.listdir(folder)]

    paper_documents = []
    for path in paths:
        document = get_markdown_documents(path, chunk_size=500, chunk_overlap=50)
        paper_documents.extend(document)

    rag_runnables = RAGRunnables(
                                rag_prompt_template = ChatPromptTemplate.from_template(PAPER_RAG_PROMPT),
                                vector_store = get_vector_store(paper_documents, EMBEDDING_MODEL, emb_dim=384, collection_name='paper_collection'),
                                llm = RAG_LLM)
        
    paper_chain = create_rag_chain(rag_runnables.rag_prompt_template, 
                                        rag_runnables.vector_store, 
                                        rag_runnables.llm)
    return paper_chain

class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str
    
def compile_graph(research_node, query_node, supervisor_agent):

    graph = StateGraph(ResearchTeamState)
    graph.add_node("Research", research_node)
    graph.add_node("Query", query_node)
    graph.add_node("Supervisor", supervisor_agent)

    graph.add_edge("Query", "Supervisor")
    graph.add_edge("Research", "Supervisor")

    def next_step(state):
        return state['next']

    graph.add_conditional_edges(
        "Supervisor",
        next_step,
        {"ScholarQuery": "Query", "LocalInformationRetriever": "Research", "FINISH": END},
    )

    graph.set_entry_point("Supervisor")
    chain = graph.compile()

    def enter_chain(message: str):
        results = {
            "messages": [HumanMessage(content=message)],
        }
        return results

    research_chain = enter_chain | chain
    
    return research_chain