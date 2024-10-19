import streamlit as st

# Global storage for API key using session state
if "api_key" not in st.session_state:
    st.session_state["open_api_key"] = None

st.title("Welcome to Research Pilot")

# API Key input on the welcome page
openai_api_key = st.text_input("Enter your OpenAI API Key to proceed:", type="password")
"[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

if openai_api_key:
    st.session_state["open_api_key"] = openai_api_key
    st.success("API key received!")