from Bio import Entrez
from langchain.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio

from utils.pubmed import *
from utils.models import REASONING_LLM as model
from utils.prompts import PUBMED_SYSTEM_MESSAGE

# Load the pre-trained model for embeddings (you can choose a different model if preferred)

# Set your Entrez email for PubMed queries
Entrez.email = "zhiji.ding@gmail.com"

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
model = model.bind_tools(tool_belt)

tool_node = ToolNode(tool_belt)

def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response], "cycle_count": state["cycle_count"] + 1}  # Increment cycle count

# Create the state graph for managing the flow between the agent and tools
uncompiled_graph = StateGraph(AgentState)
uncompiled_graph.add_node("agent", call_model)
uncompiled_graph.add_node("action", tool_node)

# Set the entry point for the graph
uncompiled_graph.set_entry_point("agent")

# Add conditional edges for the agent to action
uncompiled_graph.add_conditional_edges("agent", should_continue)
uncompiled_graph.add_edge("action", "agent")

# Compile the state graph
compiled_graph = uncompiled_graph.compile()


# streamlit implementation
st.title("🔬💊 PubMed search tool")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    

# async def generate_response(user_input):

#     system_instructions = SystemMessage(content=PUBMED_SYSTEM_MESSAGE)
#     human_inputs = HumanMessage(content=user_input)
    
#     inputs = {
#         "messages": [system_instructions, human_inputs],
#         "cycle_count": 0,
#     }

#     # Run the agent flow and capture the response
#     response = await run_graph(inputs)
    
#     # Display the response in the Chainlit UI
#     if response:
#         st.info(response)
#     else:
#         st.info("Sorry, I couldn't process the request.")

async def run_graph(user_inputs):
    final_message_content = None  # Variable to store the final message content
    system_instructions = SystemMessage(content=PUBMED_SYSTEM_MESSAGE)
    human_inputs = HumanMessage(content=user_inputs)
    
    inputs = {
        "messages": [system_instructions, human_inputs],
        "cycle_count": 0,
    }
    async for chunk in compiled_graph.astream(inputs, stream_mode="updates"):
        for _, values in chunk.items():
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
        
with st.form("my_form"):
    text = st.text_area("Enter text:", "What are causes of diabetes?")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        response = asyncio.run(run_graph(text))
        if response:
            st.info(response)
        else:
            st.info("Sorry, I couldn't process the request.")