import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import asyncio

import streamlit as st
from utils.graphs import compile_analysis_graph

async def run_analysis(file_path, question, graph):
    
    # Run the compiled graph asynchronously
    async for chunk in graph.astream(question, file_path=file_path):
        
        references = []
        for role, values in chunk.items():
            # Extract messages from values
            if "messages" in values:
                if role == 'Visualisation':
                    analysis = values["messages"][0].content
                else:
                    references.append(values["messages"][0].content)
        references = '\n'.join(references)

    # return analysis, '\n'.join(references)
    return f"## Data Analysis:\n{analysis}:\n\n## References:\n{references}"

########### Streamlit ############
st.title("📈 Data analysis 📊")
# check api key
if not st.session_state["openai_api_key"]:
    st.warning("Please go to the Welcome page and enter your OpenAI API key to proceed.")

else:
    graph = compile_analysis_graph()

## upload file
with st.form("file_submission_form"):
    file_path = st.text_area("Path to your .csv file")
    submitted = st.form_submit_button("Submit")
    if submitted:
        try:
            df = pd.read_csv(file_path).head()
            st.write("Example data")
            st.dataframe(df)
        except:
            st.write('Invalid path.')


##### build chat
### display user message
if question := st.chat_input("What is the cause of diabetes?"):

    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        response = asyncio.run(run_analysis(file_path, question, graph))
        st.markdown(response)
        
        image_folder = 'data/data_visualisation'
        image_path = os.path.join(image_folder, os.listdir(image_folder)[-1])

        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
            st.image(img_bytes, caption="Data Visualization", use_column_width=True)
