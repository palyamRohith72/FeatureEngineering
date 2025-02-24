import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import chardet
from FEATURE_SELECTION import *

# Initialize session state for storing dataframes
if "allData" not in st.session_state:
    st.session_state["allData"] = {}

def parse_file(uploaded_file, file_type):
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, encoding=encoding) if file_type == "csv" else pd.read_excel(uploaded_file, engine='openpyxl', encoding=encoding)

# Sidebar file uploader for CSV and Excel files
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv", "xlsx"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1]
    df = parse_file(uploaded_file, file_extension)
    st.session_state["allData"]["stage_0_readed_csv"] = df

# Sidebar option menu
with st.sidebar:
    selected_option = option_menu(
        "Feature Engineering Options",
        ["Relation Ship Between Features", "Feature Selection", "Feature Transformation", "Feature Creation"],
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
            if selected_option == "Relation Ship Between Features":
                feature_selection = FeatureSelection(df)
                final_dataset = FinalDataSet(df)
                method = st.selectbox("Select Feature Selection Method", [
                    "pearson", "spearman", "kendall", "point", "cramers", "variance_threshold"
                ])
                if st.checkbox("Execute Feature Selection"):
                    if method == "pearson":
                        feature_selection.pearson()
                    elif method == "spearman":
                        feature_selection.spearman()
                    elif method == "kendall":
                        feature_selection.kendall()
                    elif method == "point":
                        feature_selection.point()
                    elif method == "cramers":
                        feature_selection.cramers()

            elif selected_option == "Feature Selection":
                correlation=FeatureSelection(df)
                statistical_functions = StatisticalFunctions(df)
                final_dataset = FinalDataSet(df)
                method = st.selectbox("Select Feature Extraction Method", [
                    "varience threshold","generic_univariate_select", "select_fdr", "select_fpr",
                    "select_fwe", "select_k_best", "select_percentile","drop_features", "drop_constant_features","drop_duplicate_features", "drop_correlated_features", "smart_correlated_selection"
                ])
                if st.checkbox("Execute Feature Extraction"):
                    if method=="varience threshold":
                        dataFrame=correlation.variance_threshold()
                        st.session_state['allData']['Stage 1 - Feature Selection - Varience Threshold']=dataFrame
                    if method == "generic_univariate_select":
                        dataFrame=statistical_functions.generic_univariate_select()
                        st.session_state['allData']['Stage 1 - Feature Selection - Generic Univariate Select']=dataFrame
                    elif method == "select_fdr":
                        dataFrame=statistical_functions.select_fdr()
                        st.session_state['allData']['Stage 1 - Feature Selection - False Density Rate']=dataFrame
                    elif method == "select_fpr":
                        dataFrame=statistical_functions.select_fpr()
                        st.session_state['allData']['Stage 1 - Feature Selection - False Positive Rate']=dataFrame
                    elif method == "select_fwe":
                        dataFrame=statistical_functions.select_fwe()
                        st.session_state['allData']['Stage 1 - Feature Selection - Select FWE']=dataFrame
                    elif method == "select_k_best":
                        dataFrame=statistical_functions.select_k_best()
                        st.session_state['allData']['Stage 1 - Feature Selection - Select K Best']=dataFrame
                    elif method == "select_percentile":
                        dataFrame=statistical_functions.select_percentile()
                        st.session_state['allData']['Stage 1 - Feature Selection - Select Percentile']=dataFrame
                    elif method == "drop_features":
                        transformed_df = final_dataset.drop_features()
                    elif method == "drop_constant_features":
                        transformed_df = final_dataset.drop_constant_features()
                    elif method == "drop_duplicate_features":
                        transformed_df = final_dataset.drop_duplicate_features()
                    elif method == "drop_correlated_features":
                        transformed_df = final_dataset.drop_correlated_features()
                    elif method == "smart_correlated_selection":
                        transformed_df = final_dataset.smart_correlated_selection()
                    st.session_state["allData"][f"transformed_{method}"] = transformed_df

            elif selected_option == "Feature Creation":
                method = col1.radio("Select Feature Creation Method", [
                    "select_by_single_feature_performance", "recursive_feature_elimination", "recursive_feature_addition",
                    "select_by_information_value", "select_by_shuffling", "select_by_target_mean_performance", "select_by_mrmr"
                ])
                if col2.button("Execute Feature Creation"):
                    if method == "select_by_single_feature_performance":
                        transformed_df = final_dataset.select_by_single_feature_performance()
                    elif method == "recursive_feature_elimination":
                        transformed_df = final_dataset.recursive_feature_elimination()
                    elif method == "recursive_feature_addition":
                        transformed_df = final_dataset.recursive_feature_addition()
                    elif method == "select_by_information_value":
                        transformed_df = final_dataset.select_by_information_value()
                    elif method == "select_by_shuffling":
                        transformed_df = final_dataset.select_by_shuffling()
                    elif method == "select_by_target_mean_performance":
                        transformed_df = final_dataset.select_by_target_mean_performance()
                    elif method == "select_by_mrmr":
                        transformed_df = final_dataset.select_by_mrmr()
                    st.session_state["allData"][f"transformed_{method}"] = transformed_df

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
