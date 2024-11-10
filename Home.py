import os
import streamlit as st

# Sidebar navigation
# st.sidebar.page_link('Home.py', label='Home')
# st.sidebar.page_link('pages/1_Literature_Research.py', label='Literature Research')
# st.sidebar.page_link('pages/2_Data_Analysis.py', label='Data Analysis')

# Global storage for API key using session state
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None

st.title("Welcome to LitPilot")

# API Key input on the welcome page
openai_api_key = st.text_input("Enter your OpenAI API Key to proceed:", type="password")
"[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

if openai_api_key:
    st.session_state["openai_api_key"] = openai_api_key
    os.environ['OPENAI_API_KEY'] = st.session_state["openai_api_key"]
    st.success("API key received!")