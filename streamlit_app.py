import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from feature_engine.selection import *
from FEATURE_SELECTION import *

variables = [
    "Drop features", "Drop Constant Features", "Drop Duplicated Features",
    "Drop Correlated Features", "Smart Correlated Selection", "MRMR",
    "Select By Single Feature Performance", "Recursive Feature Elimination",
    "Recursive Feature Addition", "Drop High PSI Featutres", "Select By Information Value",
    "Select By Shuffling", "Select By Target Mean Performance", "Probe Feature Selection"
]

for i in variables:
    if i not in st.session_state:
        st.session_state[i] = None

class Features:
    def __init__(self, data):
        self.dataset = data

    def display(self):
        with st.sidebar:
            selected_option = option_menu("Select stage", ["See Correlations", "Select Features", "Create Features"])
        
        if selected_option == "See Correlations":
            self.correlations()
        elif selected_option == "Select Features":
            pass
        elif selected_option == "Create Features":
            pass

    def correlations(self):
        tab1, tab2 = st.tabs(["Perform Operations", "View Data"])
        
        with tab1:
            col1, col2 = st.columns([1, 2])
            radio_options = col1.radio("Options Were", ["pearson", "spearman", "kendall", "point", "cramers", "variance_threshold"])
            
            with col2:
                if st.button("Execute Feature Selection", use_container_width=True):
                    feature_selection = FeatureSelection(self.dataset)
                    getattr(feature_selection, radio_options, lambda: st.warning("Invalid Method"))()
        
        with tab2:
            st.write("View Data Placeholder")
