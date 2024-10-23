import streamlit as st
st.set_page_config(layout="wide")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import asyncio

import streamlit as st
from utils.graphs import compile_analysis_graph

async def run_analysis(file_path, question, graph):
    
    analysis = ""
    image_file = None
    references = []
    # Run the compiled graph asynchronously
    async for chunk in graph.astream(question, file_path=file_path):
        
        for role, values in chunk.items():
            # Extract messages from values
            if "messages" in values:
                if role == 'Visualisation':
                    analysis = values["messages"][0].content
                    image_file = values["messages"][0].image_path
                elif role == "Research" or role == "Query":
                    references.append(values["messages"][0].content)
                else:
                    pass
    references = '\n'.join(references)

    # return analysis, '\n'.join(references)
    return f"## Data Analysis:\n{analysis}:\n\n## References:\n{references}", image_file

########### Streamlit ############

st.title("📈 Data analysis 📊")
# check api key
if not st.session_state["openai_api_key"]:
    st.warning("Please go to the Welcome page and enter your OpenAI API key to proceed.")

if 'csv_file' in st.session_state:
    df = pd.read_csv(st.session_state['csv_file']).head()
    st.dataframe(df)

if ('question' in st.session_state) and ('response' in st.session_state):
    with st.chat_message("user"):
        st.markdown(st.session_state['question'])
    with st.chat_message("assistant"):
        st.markdown(st.session_state['response'])

else:
    ## upload file
    with st.form("file_submission_form"):
        file_path = st.text_area("Path to your .csv file")
        submitted = st.form_submit_button("Submit")
    
    if submitted:
        if file_path.endswith('.csv') and os.path.isfile(file_path):
            st.write("Data received")
            df = pd.read_csv(file_path).head()
            st.session_state['csv_file'] = file_path
            st.dataframe(df)
        else:
            st.write('Invalid path.')
                
    ##### build chat
    if question := st.chat_input("Your question or hypothesis here."):

        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            graph = compile_analysis_graph()
            response, image_file = asyncio.run(run_analysis(file_path, question, graph))
            st.markdown(response)
            st.session_state['image_file'] = image_file
            st.session_state['response'] = response
            st.session_state['question'] = question
    