import streamlit as st
import pandas as pd
from feature_engine.creation import MathFeatures, RelativeFeatures, CyclicalFeatures, DecisionTreeFeatures

def math_features(key, data):
    dataset = data.copy(deep=True)
    columns = st.multiselect("Select the columns", dataset.columns.tolist(), key=f"math_cols_{key}")
    func = st.selectbox("Select mathematical function", ["sum", "diff", "prod", "div", "true_div", "floor_div", "exp", "mod"], key=f"math_func_{key}")
    new_variables_names = st.text_input("Give comma-separated new column names", key=f"math_new_vars_{key}")
    new_variables = new_variables_names.split(",") if new_variables_names else None
    drop_original = st.checkbox("Drop original variables?", key=f"math_drop_{key}")
    
    if st.button("Apply", use_container_width=True, key=f"math_apply_{key}"):
        try:
            transformed_data = MathFeatures(variables=columns, func=func, new_variables_names=new_variables, drop_original=drop_original).fit_transform(dataset)
            st.dataframe(transformed_data)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def relative_features(key, data):
    dataset = data.copy(deep=True)
    columns = st.multiselect("Select the columns", dataset.columns.tolist(), key=f"rel_cols_{key}")
    reference = st.multiselect("Select reference columns", dataset.columns.tolist(), key=f"rel_ref_{key}")
    func = st.selectbox("Select function", ["sum", "diff", "prod", "div", "true_div", "floor_div", "exp", "mod"], key=f"rel_func_{key}")
    drop_original = st.checkbox("Drop original variables?", key=f"rel_drop_{key}")
    
    if st.button("Apply", use_container_width=True, key=f"rel_apply_{key}"):
        try:
            transformed_data = RelativeFeatures(variables=columns, func=func, reference=reference, drop_original=drop_original).fit_transform(dataset)
            st.dataframe(transformed_data)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def cyclical_features(key, data):
    dataset = data.copy(deep=True)
    variables = st.multiselect("Select columns", dataset.columns.tolist(), key=f"cyc_cols_{key}")
    max_values = st.text_area("Enter max values as dictionary", key=f"cyc_max_{key}")
    max_values = eval(max_values) if max_values else None
    drop_original = st.checkbox("Drop original variables?", key=f"cyc_drop_{key}")
    
    if st.button("Apply", use_container_width=True, key=f"cyc_apply_{key}"):
        try:
            transformed_data = CyclicalFeatures(variables=variables, max_values=max_values, drop_original=drop_original).fit_transform(dataset)
            st.dataframe(transformed_data)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def decision_tree_features(key, data):
    dataset = data.copy(deep=True)
    variables = st.multiselect("Select numerical variables", dataset.columns.tolist(), key=f"dtf_vars_{key}")
    features_to_combine = st.text_input("Enter features to combine (integer, list, tuple or None)", key=f"dtf_combine_{key}")
    features_to_combine = eval(features_to_combine) if features_to_combine else None
    precision = st.number_input("Precision (Decimals after comma)", min_value=0, max_value=10, value=3, key=f"dtf_precision_{key}")
    cv = st.number_input("Cross-validation folds", min_value=2, max_value=10, value=3, key=f"dtf_cv_{key}")
    scoring = st.text_input("Scoring metric (default: neg_mean_squared_error)", value='neg_mean_squared_error', key=f"dtf_scoring_{key}")
    param_grid = st.text_area("Enter param_grid dictionary", key=f"dtf_paramgrid_{key}")
    param_grid = eval(param_grid) if param_grid else None
    regression = st.checkbox("Regression model?", value=True, key=f"dtf_reg_{key}")
    random_state = st.number_input("Random State", value=0, key=f"dtf_rand_{key}")
    missing_values = st.selectbox("Handle missing values", ['raise', 'ignore'], key=f"dtf_missing_{key}")
    drop_original = st.checkbox("Drop original variables?", key=f"dtf_drop_{key}")
    
    if st.button("Apply", use_container_width=True, key=f"dtf_apply_{key}"):
        try:
            transformed_data = DecisionTreeFeatures(
                variables=variables,
                features_to_combine=features_to_combine,
                precision=precision,
                cv=cv,
                scoring=scoring,
                param_grid=param_grid,
                regression=regression,
                random_state=random_state,
                missing_values=missing_values,
                drop_original=drop_original
            ).fit_transform(dataset)
            st.dataframe(transformed_data)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def custom_features(key, data):
    dataset = data.copy(deep=True)
    st.write(f"The columns present in the dataset are {dataset.columns}")
    column = st.text_input("Enter Desired Column Name", key=f"custom_col_{key}")
    entered_query = st.text_area("Enter the pandas condition which creates the column, assuming the dataset is named 'dataset'. Example: dataset['column1'] * 2", key=f"custom_query_{key}")
    
    if st.button("Apply", use_container_width=True, key=f"custom_apply_{key}"):
        try:
            dataset[column] = eval(entered_query)
            st.dataframe(dataset)
        except Exception as e:
            st.error(f"An error occurred: {e}")
