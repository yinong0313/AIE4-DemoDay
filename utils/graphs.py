from utils.agent_helper import *
from utils.models import REASONING_LLM, RAG_LLM, EMBEDDING_MODEL
from utils.rag import *
from utils.vector_store import *
from utils.prompts import QUERY_AGENT_PROMPT, TEXTBOOK_RAG_PROMPT, PAPER_RAG_PROMPT, SUPERVISOR_PROMPT, RESEARCH_AGENT_PROMPT
from utils.data_analysis import data_visualization_node

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from langgraph.graph import END, StateGraph
# from streamlit.runtime.uploaded_file_manager import UploadedFile

import functools
import operator
from typing import Annotated, List, TypedDict
import os
import pickle

##############################################
######## Semantic scholar query node #########

def get_scholar_query_node(llm=REASONING_LLM, prompt=QUERY_AGENT_PROMPT, name="ScholarQuery", top_k_results = 1, load_max_docs = 1):
    
    api_wrapper = SemanticScholarAPIWrapper(top_k_results=top_k_results, load_max_docs=load_max_docs)
    semantic_query_tool = SemanticScholarQueryRun(api_wrapper=api_wrapper, description=prompt)
    
    query_agent = create_agent(
                    llm,
                    [semantic_query_tool],
                    prompt
)
    query_node = functools.partial(agent_node, agent=query_agent, name=name)
    return query_node

###########################################
############ RAG research node ############

def create_textbook_rag(path='data/text_books/Textbook-of-Diabetes-2024.pdf', 
                        pkl_file="data/text_books/textbook_docs.pkl"):
    try:
        with open(pkl_file, "rb") as f:
            textbook_documents = pickle.load(f)
            print(f'{len(textbook_documents)} documents loaded!')
    except:
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


def create_paper_rag(folder='data/literature', pkl_file="paper_docs.pkl"):
   
    try:
        with open(os.path.join(folder, pkl_file), "rb") as f:
            paper_documents = pickle.load(f)
            print(f'{len(paper_documents)} documents loaded!')
    except:    
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

def get_rag_tools():
    paper_chain = create_paper_rag()
    @tool
    def retrieve_paper_information(
        query: Annotated[str, "query to ask the retrieve information tool"]
        ):
      """Use Retrieval Augmented Generation to retrieve information about the papers provided."""
      return paper_chain.invoke({"question" : query})['response']

    textbook_chain = create_textbook_rag()
    @tool
    def retrieve_textbook_information(
        query: Annotated[str, "query to ask the retrieve information tool"]
        ):
      """Use Retrieval Augmented Generation to retrieve information about the book 'Textbook of Diabetes'."""
      return textbook_chain.invoke({"question" : query})['response']

    return [retrieve_paper_information, retrieve_textbook_information]

def get_research_node(llm=REASONING_LLM):
    tools = get_rag_tools()
    research_agent = create_agent(
                        llm,
                        tools,
                        RESEARCH_AGENT_PROMPT
    )

    research_node = functools.partial(agent_node, agent=research_agent, name="LocalInformationRetriever")
    return research_node    

########### construct graph ############

class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    file_path: str
    questions: List[BaseMessage]
    team_members: List[str]
    next: str
    
def compile_analysis_graph():

    supervisor_agent = create_team_supervisor(
                            REASONING_LLM,
                            SUPERVISOR_PROMPT,
                            ["ScholarQuery", "LocalInformationRetriever"],
                        )
    analysis_node = data_visualization_node()
    query_node = get_scholar_query_node()
    research_node = get_research_node()
    
    graph = StateGraph(ResearchTeamState)
    
    graph.add_node("Visualisation", analysis_node)
    graph.add_node("Research", research_node)
    graph.add_node("Query", query_node)
    graph.add_node("Supervisor", supervisor_agent)

    graph.add_edge("Query", "Supervisor")
    graph.add_edge("Research", "Supervisor")
    graph.add_edge("Visualisation", "Supervisor")

    def next_step(state):
        # Truncate messages to keep the token count manageable
        state["messages"] = truncate_messages(state["messages"])
        
        # Supervisor decides the next step based on state['next']
        if "next" in state:
            if state["next"] in {"ScholarQuery", "LocalInformationRetriever", "FINISH"}:
                return state["next"]
        # If 'next' isn't set, return a default behavior (could be research or scholar query)
        return "FINISH"
    
    graph.add_conditional_edges(
        "Supervisor",
        next_step,
        {
            "ScholarQuery": "Query", 
            "LocalInformationRetriever": "Research", 
            "FINISH": END
         },
    )

    graph.set_entry_point("Visualisation")
    chain = graph.compile()

    def enter_chain(question: str, file_path: str):
        results = {
            "messages": [HumanMessage(content=question)],
            "questions": [HumanMessage(content=question)],
            "file_path": file_path,
            "team_members": ["ScholarQuery", "LocalInformationRetriever"], 
            "next": ""
        }
        return results

    research_chain = enter_chain | chain
    
    return research_chain