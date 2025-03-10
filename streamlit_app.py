import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from feature_engine.selection import *
from FEATURE_SELECTION import *

variables = [
    "Drop features", "Drop Constant Features", "Drop Duplicated Features",
    "Drop Correlated Features", "Smart Correlated Selection", "MRMR",
    "Select By Single Feature Performance", "Recursive Feature Elimination",
    "Recursive Feature Addition", "Drop High PSI Features", "Select By Information Value",
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
            self.select_features()
        elif selected_option == "Create Features":
            self.create_features()

    def correlations(self):
        tab1, tab2 = st.tabs(["Perform Operations", "View Data"])
        
        with tab1:
            col1, col2 = st.columns([1, 2],border=True)
            radio_options = col1.radio("Options Were", ["pearson", "spearman", "kendall", "point", "cramers"])
            
            with col2:
                select_columns = st.multiselect("Select columns", self.dataset.columns.tolist())
                if select_columns:
                    dataset = self.dataset.copy(deep=True)[select_columns]
                    if st.button("Execute Feature Selection", use_container_width=True):
                        feature_selection = FeatureSelection(dataset)
                        try:
                            getattr(feature_selection, radio_options, lambda: st.warning("Invalid Method"))()
                        except Exception as e:
                            st.error(e)
        
        with tab2:
            st.write("View Data Placeholder")
            
    def select_features(self):
        feature_methods = {
            "Drop features": self.drop_features,
            "Drop Constant Features": self.drop_constant_features,
            "Drop Duplicated Features": self.drop_duplicated_features,
            "Drop Correlated Features": self.drop_correlated_features,
            "Smart Correlated Selection": self.smart_correlated_selection,
            "MRMR": self.mrmr,
            "Select By Single Feature Performance": self.select_by_single_feature_performance,
            "Recursive Feature Elimination": self.recursive_feature_elimination,
            "Recursive Feature Addition": self.recursive_feature_addition,
            "Drop High PSI Features": self.drop_high_psi_features,
            "Select By Information Value": self.select_by_information_value,
            "Select By Shuffling": self.select_by_shuffling,
            "Select By Target Mean Performance": self.select_by_target_mean_performance,
            "Probe Feature Selection": self.probe_feature_selection
        }
        
        tab1, tab2 = st.tabs(["Perform Operations", "View Data"])
        
        with tab1:
            col1, col2 = st.columns([1, 2],border=True)
            radio_options = col1.radio("Options Were", list(feature_methods.keys()))
            
            if radio_options:
                with col2:
                    feature_methods[radio_options](radio_options)
    
    def create_features(self):
        st.write("Feature creation logic goes here.")
    
    def drop_features(self,keyy):
        select_columns = st.multiselect("Select columns", self.dataset.columns.tolist())
        if select_columns:
            dataset = self.dataset.copy(deep=True)
            if st.button("Execute Feature Selection", use_container_width=True):   
                object=DropFeatures(select_columns)
                dataframe=object.fit_transform(dataset)
                st.session_state[keyy]=dataframe
                st.dataframe(dataframe)
    def drop_constant_features(self,option):
        select_columns = st.multiselect("Select columns", self.dataset.columns.tolist())
        tol=int(st.number_input("""Threshold to detect constant/quasi-constant features. Variables showing the same value in a
        percentage of observations greater than tol will be considered 
        constant / quasi-constant and dropped. If tol=1, the transformer 
        removes constant variables. Else, it will remove quasi-constant variables.
        For example, if tol=0.98, the transformer will remove variables that show
        the same value in 98% of the observations.""",1))
        if select_columns:
            dataset = self.dataset.copy(deep=True)
            if st.button("Execute Feature Selection", use_container_width=True):   
                object=DropConstantFeatures(select_columns,tol)
                dataframe=object.fit_transform(dataset)
                st.session_state[option]=dataframe
                st.dataframe(dataframe)
    
    def drop_duplicated_features(self):
        st.write("Dropping duplicated features.")
    
    def drop_correlated_features(self):
        st.write("Dropping correlated features.")
    
    def smart_correlated_selection(self):
        st.write("Performing smart correlated selection.")
    
    def mrmr(self):
        st.write("Executing MRMR feature selection.")
    
    def select_by_single_feature_performance(self):
        st.write("Selecting by single feature performance.")
    
    def recursive_feature_elimination(self):
        st.write("Performing recursive feature elimination.")
    
    def recursive_feature_addition(self):
        st.write("Performing recursive feature addition.")
    
    def drop_high_psi_features(self):
        st.write("Dropping high PSI features.")
    
    def select_by_information_value(self):
        st.write("Selecting by information value.")
    
    def select_by_shuffling(self):
        st.write("Selecting by shuffling.")
    
    def select_by_target_mean_performance(self):
        st.write("Selecting by target mean performance.")
    
    def probe_feature_selection(self):
        st.write("Performing probe feature selection.")

file_uploader = st.sidebar.file_uploader("Upload CSV", type=['csv'])
if file_uploader:
    dataframe = pd.read_csv(file_uploader)
    st.session_state['readed_csv'] = dataframe
    selected_output = st.selectbox("Outputs to select", [x for x in st.session_state.keys() if x])
    if selected_output:
        Features(st.session_state['readed_csv']).display()
