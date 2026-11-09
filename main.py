import streamlit as st
from vms.dashboard import app
st.set_page_config(page_title="AI VMS People Counting", layout="wide")
app()
