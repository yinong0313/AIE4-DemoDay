import chainlit as cl
from Bio import Entrez
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from IPython.display import display, Markdown
from sentence_transformers import SentenceTransformer, util
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from typing import List, TypedDict, Annotated
import xml.etree.ElementTree as ET
import uuid
import re
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from transformers import GPT2Tokenizer

# Load the pre-trained model for embeddings (you can choose a different model if preferred)
semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def pretty_print(message: str) -> None:
    display(Markdown(f"```markdown\n{message}\n```"))

# Set your Entrez email for PubMed queries
Entrez.email = "your-email@example.com"

# 1. Define PubMed Search Tool
class PubMedSearchInput(BaseModel):
    query: str
    #max_results: int = 5

# PubMed search tool using Entrez (now with structured inputs)
def pubmed_search(query: str, max_results: int = 5):
    """Search PubMed using Entrez API and return abstracts."""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    pmids = record["IdList"]
    
    # Fetch abstracts
    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    abstracts = []
    for record in records['PubmedArticle']:
        try:
            title = record['MedlineCitation']['Article']['ArticleTitle']
            abstract = record['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
            pmid = record['MedlineCitation']['PMID']
            abstracts.append({"PMID": pmid, "Title": title, "Abstract": abstract})
        except KeyError:
            pass
    return abstracts

# Define the AbstractScreeningInput using Pydantic BaseModel
class AbstractScreeningInput(BaseModel):
    abstracts: List[dict]
    criteria: str

def screen_abstracts_semantic(abstracts: List[dict], criteria: str, similarity_threshold: float = 0.4):
    """Screen abstracts based on semantic similarity to the criteria."""
    
    # Compute the embedding of the criteria
    criteria_embedding = semantic_model.encode(criteria, convert_to_tensor=True)
    
    screened = []
    for paper in abstracts:
        abstract_text = paper['Abstract']
        
        # Compute the embedding of the abstract
        abstract_embedding = semantic_model.encode(abstract_text, convert_to_tensor=True)
        
        # Compute cosine similarity between the abstract and the criteria
        similarity_score = util.cos_sim(abstract_embedding, criteria_embedding).item()
        
        if similarity_score >= similarity_threshold:
            screened.append({
                "PMID": paper['PMID'], 
                "Decision": "Include", 
                "Reason": f"Similarity score {similarity_score:.2f} >= threshold {similarity_threshold}"
            })
        else:
            screened.append({
                "PMID": paper['PMID'], 
                "Decision": "Exclude", 
                "Reason": f"Similarity score {similarity_score:.2f} < threshold {similarity_threshold}"
            })
    
    return screened

# Define the PubMed Search Tool as a StructuredTool with proper input schema
pubmed_tool = StructuredTool(
    name="PubMed_Search_Tool",
    func=pubmed_search,
    description="Search PubMed for research papers and retrieve abstracts. Pass the abstracts (returned results) to another tool.",
    args_schema=PubMedSearchInput  # Use Pydantic BaseModel for schema
)

# Define the Abstract Screening Tool with semantic screening
semantic_screening_tool = StructuredTool(
    name="Semantic_Abstract_Screening_Tool",
    func=screen_abstracts_semantic,
    description="""Screen PubMed abstracts based on semantic similarity to inclusion/exclusion criteria. Uses cosine similarity between abstracts and criteria. Requires 'abstracts' and 'screening criteria' as input.
    The 'abstracts' is a list of dictionary with keys as PMID, Title, Abstract.
    Output a similarity scores for each abstract and send the list of pmids that passed the screening to Fetch_Extract_Tool.""",
    args_schema=AbstractScreeningInput  # Pydantic schema remains the same
)

# 3. Define Full-Text Retrieval Tool
class FetchExtractInput(BaseModel):
    pmids: List[str]  # List of PubMed IDs to fetch full text for
    query: str

def extract_text_from_pmc_xml(xml_content: str) -> str:
    """a function to format and clean text from PMC full-text XML."""
    try:
        root = ET.fromstring(xml_content)
        
        # Find all relevant text sections (e.g., <body>, <sec>, <p>)
        body_text = []
        for elem in root.iter():
            if elem.tag in ['p', 'sec', 'title', 'abstract', 'body']:  # Add more tags as needed
                if elem.text:
                    body_text.append(elem.text.strip())
        
        # Join all the text elements to form the complete full text
        full_text = "\n\n".join(body_text)
        
        return full_text
    except ET.ParseError:
        print("Error parsing XML content.")
        return ""

def fetch_and_extract(pmids: List[str], query: str):
    """Fetch full text from PubMed Central for given PMIDs, split into chunks, 
    store in a Qdrant vector database, and perform RAG for each paper.
    Retrieves exactly 3 chunks per paper (if available) and generates a consolidated answer for each paper.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    corpus = {}
    consolidated_results={}

    # Fetch records from PubMed Central (PMC)
    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    full_articles = []
    for record in records['PubmedArticle']:
        try:
            title = record['MedlineCitation']['Article']['ArticleTitle']
            pmid = record['MedlineCitation']['PMID']
            pmc_id = 'nan'
            pmc_id_temp = record['PubmedData']['ArticleIdList']
            
            # Extract PMC ID if available
            for ele in pmc_id_temp:
                if ele.attributes['IdType'] == 'pmc':
                    pmc_id = ele.replace('PMC', '')
                    break

            # Fetch full article from PMC
            if pmc_id != 'nan':
                handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="full", retmode="xml")
                full_article = handle.read()
                handle.close()

                # Split the full article into chunks
                cleaned_full_article = extract_text_from_pmc_xml(full_article)
                full_articles.append({
                    "PMID": pmid,
                    "Title": title,
                    "FullText": cleaned_full_article   # Add chunked text
                })
            else:
                full_articles.append({"PMID": pmid, "Title": title, "FullText": "cannot fetch"})
        except KeyError:
            pass

    # Create corpus for each chunk
    for article in full_articles:
        article_id = str(uuid.uuid4())
        corpus[article_id] = {
            "page_content": article["FullText"],
            "metadata": {
                "PMID": article["PMID"],
                "Title": article["Title"]
            }
        }

    documents = [
        Document(page_content=content["page_content"], metadata=content["metadata"]) 
        for content in corpus.values()
    ]
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    split_chunks = text_splitter.split_documents(documents)
    
    id_set = set()
    for document in split_chunks:
        id = str(uuid.uuid4())
        while id in id_set:
            id = uuid.uuid4()
        id_set.add(id)
        document.metadata["uuid"] = id

    LOCATION = ":memory:"
    COLLECTION_NAME = "pmd_data"
    VECTOR_SIZE = 384

    # Initialize Qdrant client
    qdrant_client = QdrantClient(location=LOCATION)

    # Create a collection in Qdrant
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Initialize the Qdrant vector store without the embedding argument
    vdb = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )

    # Add embedded documents to Qdrant
    vdb.add_documents(split_chunks)

    # Query for each paper and consolidate answers
    for pmid in pmids:
        # Correctly structure the filter using Qdrant Filter model
        qdrant_filter = Filter(
            must=[
                FieldCondition(key="metadata.PMID", match=MatchValue(value=pmid))
            ]
        )

        # Custom filtering for the retriever to only fetch chunks related to the current PMID
        retriever_with_filter = vdb.as_retriever(
            search_kwargs={
                "filter": qdrant_filter,  # Correctly passing the Qdrant filter
                "k": 3  # Retrieve 3 chunks per PMID
            }
        )

        # Reset message history and memory for each query to avoid interference
        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", chat_memory=message_history, return_messages=True)

        # Create the ConversationalRetrievalChain with the filtered retriever
        qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            retriever=retriever_with_filter,
            memory=memory,
            return_source_documents=True
        )

        # Query the vector store for relevant documents and extract information
        result = qa_chain({"question": query})

        # Generate the final answer based on the retrieved chunks
        generated_answer = result["answer"]  # This contains the LLM's generated answer based on the retrieved chunks
        generated_source = result["source_documents"]

        # Consolidate the results for each paper
        paper_info = {
            "PMID": pmid,
            "Title": result["source_documents"][0].metadata["Title"] if result["source_documents"] else "Unknown Title",
            "Generated Answer": generated_answer,  # Store the generated answer,
            "Sources": generated_source 
        }

        consolidated_results[pmid] = paper_info

    # Return consolidated results for all papers
    return consolidated_results

rag_tool = StructuredTool(
    name="Fetch_Extract_Tool",
    func=fetch_and_extract,
    description="""Fetch full-text articles based on PMIDs and store them in a Qdrant vector database.
    Then extract information based on user's query via Qdrant retriever using a RAG pipeline.
    Requires list of PMIDs and user query as input.""",
    args_schema=FetchExtractInput
)


tool_belt = [
    pubmed_tool,
    semantic_screening_tool,
    rag_tool
]


# Model setup with tools bound
model = ChatOpenAI(model="gpt-4o", temperature=0)
model = model.bind_tools(tool_belt)

# Agent state to handle the messages
class AgentState(dict):
    messages: Annotated[list, add_messages]
    cycle_count: int  # Add a counter to track the number of cycles

# Function to call the model and handle the flow automatically
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response], "cycle_count": state["cycle_count"] + 1}  # Increment cycle count

tool_node = ToolNode(tool_belt)

# Create the state graph for managing the flow between the agent and tools
uncompiled_graph = StateGraph(AgentState)
uncompiled_graph.add_node("agent", call_model)
uncompiled_graph.add_node("action", tool_node)

# Set the entry point for the graph
uncompiled_graph.set_entry_point("agent")

# Define a function to check if the process should continue
def should_continue(state):
    # Check if the cycle count exceeds the limit (e.g., 10)
    if state["cycle_count"] > 20:
        print(f"Reached the cycle limit of {state['cycle_count']} cycles. Ending the process.")
        return END
    
    # If there are tool calls, continue to the action node
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "action"
    
    return END

# Add conditional edges for the agent to action
uncompiled_graph.add_conditional_edges("agent", should_continue)
uncompiled_graph.add_edge("action", "agent")

# Compile the state graph
compiled_graph = uncompiled_graph.compile()

# Function to run the compiled graph asynchronously
async def run_graph(inputs):
    final_message_content = None  # Variable to store the final message content

    async for chunk in compiled_graph.astream(inputs, stream_mode="updates"):
        for node, values in chunk.items():
            print(values["messages"])
            
            # Check if the message contains content
            if "messages" in values and values["messages"]:
                final_message = values["messages"][-1]
                if hasattr(final_message, 'content'):
                    final_message_content = final_message.content
                
        print("\n\n")

    if final_message_content:
        print("Final message content from the last chunk:")
        print(final_message_content)
    
    return final_message_content

# Chainlit interaction setup
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Welcome! Please provide your PubMed query and screening criteria.").send()

@cl.on_message
async def main(message):
    # Extract query and screening criteria from the user's message
    user_input = message.content

    # Build inputs for the agent
    # system_instructions = SystemMessage(content="""
    # 1. Use the PubMed search tool to search for papers.
    # 2. Retrieve the abstracts from the search results.
    # 3. Screen the abstracts based on the criteria provided by the user. If error happens,retry by feeding in both 'abstracts' and 'screening criteria' as input.
    #     The 'abstracts' is a list of dictionary with keys as PMID, Title, Abstract (which is extracted from preivous step). For the decisions of include and exclude, give me the similarity score you calculated.
    # 4. Please provide a full summary at the end of the entire flow executed, detailing the whole process/reasoning for each paper. 
    # The user will provide the search query and screening criteria.
    # Make sure you finish everything in one step before moving on to next step.
    # Do not call more than one tool in one action.""")

    system_instructions = SystemMessage(content="""Please execute the following steps in sequence:
    1. Use the PubMed search tool to search for papers.
    2. Retrieve the abstracts from the search results.
    3. Screen the abstracts based on the criteria provided by the user.
    4. Fetch full-text articles for all the papers that pass step 3. Store the full-text articles in the Qdrant vector database, 
        and extract the requested information for each article that passed step 3 from the full-text using the query provided by the user.
    5. Please provide a full summary at the end of the entire flow executed, detailing each paper's title, PMID, and the whole process/screening/reasoning for each paper. 
    The user will provide the search query, screening criteria, and the query for information extraction.
    Make sure you finish everything in one step before moving on to next step.
    Do not call more than one tool in one action.""")
    human_inputs = HumanMessage(content=user_input)
    
    inputs = {
        "messages": [system_instructions, human_inputs],
        "cycle_count": 0,
    }

    # Run the agent flow and capture the response
    response = await run_graph(inputs)
    
    # Display the response in the Chainlit UI
    if response:
        await cl.Message(content=response).send()
    else:
        await cl.Message(content="Sorry, I couldn't process the request.").send()
