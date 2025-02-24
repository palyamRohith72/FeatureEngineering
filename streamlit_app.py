import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import chardet
from FEATURE_SELECTION import FeatureSelection, StatisticalFunctions, FinalDataSet

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
    return pd.read_excel(uploaded_file, engine='openpyxl', encoding=encoding)

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
        ["Feature Selection", "Feature Extraction", "Feature Transformation", "Feature Creation"],
        icons=["filter-square", "arrow-down-up", "sliders", "plus-square"],
        menu_icon="tools",
        default_index=0
    )

# Select dataset from session state
if st.session_state["allData"]:
    selected_data = st.selectbox("Select dataset", st.session_state["allData"].keys())
    df = st.session_state["allData"].get(selected_data)

    if df is not None:
        tab1, tab2, tab3 = st.tabs(["Perform Operations", "View Data", "See Documentation"])

        with tab1:
            col1, col2 = st.columns([1, 2])
            
            if selected_option == "Feature Selection":
                feature_selection = FeatureSelection(df)
                method = col1.radio("Select Feature Selection Method", ["pearson", "spearman", "kendall", "point", "cramers", "variance_threshold"])
                if col2.checkbox("Execute Feature Selection"):
                    with col2:
                        getattr(feature_selection, method)()
            
            elif selected_option == "Feature Extraction":
                statistical_functions = StatisticalFunctions(df)
                method = col1.radio("Select Feature Extraction Method", ["generic_univariate_select", "select_fdr", "select_fpr", "select_fwe", "select_k_best", "select_percentile"])
                if col2.checkbox("Execute Feature Extraction"):
                    with col2:
                        getattr(statistical_functions, method)()
            
            elif selected_option == "Feature Transformation":
                final_dataset = FinalDataSet(df)
                method = col1.radio("Select Feature Transformation Method", ["drop_features", "drop_constant_features", "drop_duplicate_features", "drop_correlated_features", "smart_correlated_selection"])
                if col2.checkbox("Execute Feature Transformation"):
                    with col2:
                        transformed_df = getattr(final_dataset, method)()
                        st.session_state["allData"][f"transformed-{method.__name__()}"] = transformed_df
            
            elif selected_option == "Feature Creation":
                final_dataset = FinalDataSet(df)
                method = col1.radio("Select Feature Creation Method", ["select_by_single_feature_performance", "recursive_feature_elimination", "recursive_feature_addition", "select_by_information_value", "select_by_shuffling", "select_by_target_mean_performance", "select_by_mrmr"])
                if col2.button("Execute Feature Creation"):
                    with col2:
                        transformed_df = getattr(final_dataset, method)()
                        st.session_state["allData"]["transformed-{method.__name__()}"] = transformed_df
        
        with tab2:
            st.subheader("Dataset Preview", divider='blue')
            st.dataframe(df.head())
        
        with tab3:
            st.subheader("Documentation", divider='blue')
            st.markdown("""
                **Feature Engineering Options**
                - **Feature Selection**: Methods to filter or rank features based on statistical tests.
                - **Feature Extraction**: Techniques to derive new features from existing ones.
                - **Feature Transformation**: Methods to modify existing features for better model performance.
                - **Feature Creation**: Approaches to generate new features to enhance model learning.
            """)
