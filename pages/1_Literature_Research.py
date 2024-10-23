import streamlit as st
st.set_page_config(layout="wide")

from Bio import Entrez

from langchain_core.messages import SystemMessage, HumanMessage


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio

from utils.pubmed import compile_pubmed_search_graph
from utils.models import REASONING_LLM
from utils.prompts import PUBMED_SYSTEM_MESSAGE


async def run_graph(user_inputs, graph):
    """
    user_input: sample question: 
        {
        "query": "("Impact"[Title/Abstract]) AND (("COVID"[Title/Abstract] OR "pandemic"[Title/Abstract]) AND ("healthcare resource utilization"[Title/Abstract] OR "health resource utilization"[Title/Abstract]))", 
        "screening_criteria": "comparison of healthcare resource utilization before and after the COVID pandemic", 
        "extraction_query": "data source information used in each paper (especially medical records)"
        }
    """
    
    final_message_content = None  # Variable to store the final message content
    system_instructions = SystemMessage(content=PUBMED_SYSTEM_MESSAGE)
    human_inputs = HumanMessage(content=user_inputs)
    
    inputs = {
        "messages": [system_instructions, human_inputs],
        "cycle_count": 0,
    }
    async for chunk in graph.astream(inputs, stream_mode="updates"):
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

###################### streamlit implementation #######################

st.title("🔬💊 PubMed search tool")

with st.sidebar:
    # Set your Entrez email for PubMed queries
    Entrez.email = st.text_input("Enter your Email")

if not st.session_state["openai_api_key"]:
    st.warning("Please go to the Welcome page and enter your OpenAI API key to proceed.")

else:
    graph = compile_pubmed_search_graph(model=REASONING_LLM)    
    with st.form("my_form"):
        text = st.text_area("Enter text:", "What are causes of diabetes?")
        submitted = st.form_submit_button("Submit")
        if submitted:
            response = asyncio.run(run_graph(text, graph))
            if response:
                st.info(response)
            else:
                st.info("Sorry, I couldn't process the request.")
                
