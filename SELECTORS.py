# importing libraries
import pandas as pd
import streamlit as st
import numpy as np
from feature_engine.selection import *
def drop_features(keyy,data):
  select_columns = st.multiselect("Select columns", data.columns.tolist())
  if select_columns:
      dataset = data.copy(deep=True)
      if st.button("Execute Feature Selection", use_container_width=True):   
          object=DropFeatures(select_columns)
          dataframe=object.fit_transform(dataset)
          st.session_state[keyy]=dataframe
          st.dataframe(dataframe)
def drop_constant_features(option,data):
    select_columns = st.multiselect("Select columns", data.columns.tolist())
    tol=int(st.number_input("""Threshold to detect constant/quasi-constant features. Variables showing the same value in a
    percentage of observations greater than tol will be considered 
    constant / quasi-constant and dropped. If tol=1, the transformer 
    removes constant variables. Else, it will remove quasi-constant variables.
    For example, if tol=0.98, the transformer will remove variables that show
    the same value in 98% of the observations.""",1))
    dataset = data.copy(deep=True)
    if st.button("Execute Feature Selection", use_container_width=True):   
        try:
            object=DropConstantFeatures(select_columns,tol)
            dataframe=object.fit_transform(dataset)
            st.session_state[option]=dataframe
            st.dataframe(dataframe)
        except Exception as e:
            st.error(e)
def drop_duplicated_features(option,data):
    select_columns = st.multiselect("Select columns", data.columns.tolist())
    dataset = data.copy(deep=True)
    if st.button("Execute Feature Selection", use_container_width=True):   
        try:
            object=DropDuplicateFeatures(select_columns)
            dataframe=object.fit_transform(dataset)
            st.session_state[option]=dataframe
            st.dataframe(dataframe)
        except Exception as e:
            st.error(e)

def drop_correlated_features(option,data):
    select_columns = st.multiselect("Select columns", data.columns.tolist())
    threshold=st.number_input("The correlation threshold above which a feature will be deemed correlated with another one and removed from the dataset.",0.1)
    method=st.selectbox("Correlation method -Can take ‘pearson’, ‘spearman’, ‘kendall’",["pearson","spearman","kendall"])
    dataset = data.copy(deep=True)
    if st.button("Execute Feature Selection", use_container_width=True):   
        try:
            object=DropCorrelatedFeatures(select_columns,method,threshold)
            dataframe=object.fit_transform(dataset)
            st.session_state[option]=dataframe
            st.dataframe(dataframe)
        except Exception as e:
            st.error(e)

def smart_correlated_selection(option,data):
    st.write("Performing smart correlated selection.")

def mrmr(option,data):
    st.write("Executing MRMR feature selection.")

def select_by_single_feature_performance(option,data):
    st.write("Selecting by single feature performance.")

def recursive_feature_elimination(option,data):
    st.write("Performing recursive feature elimination.")

def recursive_feature_addition(option,data):
    st.write("Performing recursive feature addition.")

def select_by_information_value(option,data):
    st.write("Selecting by information value.")

def select_by_shuffling(option,data):
    st.write("Selecting by shuffling.")

def select_by_target_mean_performance(option,data):
    st.write("Selecting by target mean performance.")

def probe_feature_selection(option,data):
    st.write("Performing probe feature selection.")




def drop_high_psi_features(option,df):
    # Clone the dataframe to preserve the original
    df_clone = df.copy()
    
    st.sidebar.header("Drop High PSI Features Configuration")
    
    # Input for split_col
    split_col = st.sidebar.selectbox(
        "Select the column to split the dataset (split_col):",
        options=[None] + list(df_clone.columns),
        index=0,
        help="The variable that will be used to split the dataset into the basis and test sets. If None, the dataframe index will be used."
    )
    
    # Input for split_frac
    split_frac = st.sidebar.slider(
        "Proportion of observations in each dataset (split_frac):",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="The proportion of observations in each of the basis and test dataframes. If 0.6, 60% of the observations will be put in the basis data set."
    )
    
    # Input for split_distinct
    split_distinct = st.sidebar.checkbox(
        "Split based on unique values (split_distinct):",
        value=False,
        help="If True, split_frac is applied to the vector of unique values in split_col instead of being applied to the whole vector of values."
    )
    
    # Input for cut_off
    cut_off = st.sidebar.text_input(
        "Threshold to split the dataset (cut_off):",
        value="",
        help="Threshold to split the dataset based on the split_col variable. If int, float or date, observations where the split_col values are <= threshold will go to the basis data set and the rest to the test set. If a list, observations where the split_col values are within the list will go to the basis data set."
    )
    
    # Input for switch
    switch = st.sidebar.checkbox(
        "Switch the order of basis and test datasets (switch):",
        value=False,
        help="If True, the order of the 2 dataframes used to determine the PSI (basis and test) will be switched."
    )
    
    # Input for threshold
    threshold = st.sidebar.selectbox(
        "Threshold to drop a feature (threshold):",
        options=[0.25, 0.10, 'auto'],
        index=0,
        help="The threshold to drop a feature. If the PSI for a feature is >= threshold, the feature will be dropped. Common values are 0.25 (large shift) and 0.10 (medium shift). If 'auto', the threshold will be calculated based on the size of the basis and test dataset and the number of bins."
    )
    
    # Input for bins
    bins = st.sidebar.slider(
        "Number of bins or intervals (bins):",
        min_value=1,
        max_value=20,
        value=10,
        step=1,
        help="Number of bins or intervals. For continuous features with good value spread, 10 bins is commonly used."
    )
    
    # Input for strategy
    strategy = st.sidebar.selectbox(
        "Strategy for discretization (strategy):",
        options=['equal_frequency', 'equal_width'],
        index=0,
        help="If the intervals into which the features should be discretized are of equal size or equal number of observations. 'equal_width' for equally spaced bins or 'equal_frequency' for bins based on quantiles."
    )
    
    # Input for min_pct_empty_bins
    min_pct_empty_bins = st.sidebar.number_input(
        "Minimum percentage for empty bins (min_pct_empty_bins):",
        min_value=0.0,
        max_value=1.0,
        value=0.0001,
        step=0.0001,
        help="Value to add to empty bins or intervals. If after sorting the variable values into bins, a bin is empty, the PSI cannot be determined. By adding a small number to empty bins, we can avoid this issue."
    )
    
    # Input for missing_values
    missing_values = st.sidebar.selectbox(
        "Handling missing values (missing_values):",
        options=['raise', 'ignore'],
        index=0,
        help="Whether to perform the PSI feature selection on a dataframe with missing values. 'raise' will raise an error, 'ignore' will drop missing values."
    )
    
    # Input for variables
    variables = st.sidebar.multiselect(
        "Variables to evaluate (variables):",
        options=list(df_clone.columns),
        default=None,
        help="The list of variables to evaluate. If None, the transformer will evaluate all numerical variables in the dataset."
    )
    
    # Input for confirm_variables
    confirm_variables = st.sidebar.checkbox(
        "Confirm variables (confirm_variables):",
        value=False,
        help="If set to True, variables that are not present in the input dataframe will be removed from the list of variables."
    )
    
    # Input for p_value
    p_value = st.sidebar.number_input(
        "P-value for auto threshold (p_value):",
        min_value=0.001,
        max_value=0.05,
        value=0.001,
        step=0.001,
        help="The p-value to test the null hypothesis that there is no feature drift. This parameter is used only if threshold is set to 'auto'."
    )
    
    # Initialize DropHighPSIFeatures with user inputs
    psi_selector = DropHighPSIFeatures(
        split_col=split_col,
        split_frac=split_frac,
        split_distinct=split_distinct,
        cut_off=cut_off if cut_off else None,
        switch=switch,
        threshold=threshold,
        bins=bins,
        strategy=strategy,
        min_pct_empty_bins=min_pct_empty_bins,
        missing_values=missing_values,
        variables=variables if variables else None,
        confirm_variables=confirm_variables,
        p_value=p_value
    )
    
    # Fit and transform the dataframe
    df_transformed = psi_selector.fit_transform(df_clone)
    
    st.write("Transformed DataFrame:")
    st.write(df_transformed)
    
    return df_transformed
