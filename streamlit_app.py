import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature_engine.imputation import MeanMedianImputer
import chardet
from correlation import Correlation

# Initialize session state for storing dataframes
if "allData" not in st.session_state:
    st.session_state["allData"] = {}

def parse_csv(uploaded_file):
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, encoding=encoding)

def parse_excel(uploaded_file):
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, engine='openpyxl',encoding=encoding)

# Sidebar file uploader for CSV and Excel files
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        df = parse_csv(uploaded_file)
    else:
        df = parse_excel(uploaded_file)
    
    st.session_state["allData"]["stage 0 - readed csv"] = df

# Sidebar option menu
with st.sidebar:
    selected_option = option_menu(
        "Feature Engineering Options",
        ["Correlations", "Feature Selection", "Feature Extraction", "Feature Transformation", "Feature Creation"],
        icons=["graph-up", "filter-square", "arrow-down-up", "sliders", "plus-square"],
        menu_icon="tools",
        default_index=0
    )

# Select dataset from session state
if st.session_state["allData"]:
    selected_data = st.selectbox("Select dataset", st.session_state["allData"].keys())
    df = st.session_state["allData"].get(selected_data)

    if df is not None:
        if selected_option == "Correlations":
            object=Correlation(df)
            object.display()

        elif selected_option == "Feature Selection":
            st.subheader("Feature Selection")
            missing_values = df.isnull().sum()
            st.write("Missing Values:", missing_values)
            st.write("Feature Variance:", df.var())
            
        elif selected_option == "Feature Extraction":
            st.subheader("Feature Extraction")
            st.write("Extracting Principal Components (PCA) is an example of feature extraction.")
            
        elif selected_option == "Feature Transformation":
            st.subheader("Feature Transformation")
            scaler = StandardScaler()
            transformed_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))
            st.write(pd.DataFrame(transformed_data, columns=df.select_dtypes(include=[np.number]).columns))
            
        elif selected_option == "Feature Creation":
            st.subheader("Feature Creation")
            df['New_Feature'] = df.select_dtypes(include=[np.number]).mean(axis=1)
            st.write(df.head())
