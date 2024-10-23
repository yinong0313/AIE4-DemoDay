import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd

st.title(f"🧐 {st.session_state['question']}")

st.dataframe(pd.read_csv(st.session_state['csv_file']))

def show_image():           
    if 'image_file' in st.session_state:
        with open(st.session_state['image_file'], "rb") as img_f:
            img_bytes = img_f.read()
            st.image(img_bytes, caption="Data Visualization", use_column_width=True)
    else:
        st.write("No graph available.")

show_image()