# app.py
import streamlit as st

from config import BASE_DIR, DATA_RAW_DIR, DATA_CLEAN_DIR, MODELS_DIR
from utils.page_config import PAGE_CONFIG

# -------------------------------------------------------
# Basis-Konfiguration
# -------------------------------------------------------

st.set_page_config(
    page_title="Lyrics Text Analytics",
    page_icon="🎶",
    layout="wide",
)

# -------------------------------------------------------
# UI: Sidebar Navigation
# -------------------------------------------------------

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Ansicht auswählen:",
    list(PAGE_CONFIG.keys()),
)

# -------------------------------------------------------
# Page routing
# -------------------------------------------------------

if page in PAGE_CONFIG:
    PAGE_CONFIG[page]()
else:
    st.error("Ausgewählte Seite ist nicht konfiguriert.")
