import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Correlation:
    def __init__(self, df):
        self.dataset = df
    
    def display(self):
        st.title("Correlation Analysis")
        tab1, tab2, tab3 = st.tabs(["Perform Operations", "View Data", "See Documentation"])
        
        with tab1:
            col1, col2 = st.columns([1, 2],border=True)
            method = col1.radio("Select Correlation Method", ["pearson", "kendall", "spearman"], index=0)
            col2.subheader("Correlation Matrix", divider='blue')
            numeric_only = col2.checkbox("Include only numeric data", value=False)
            if col2.button("Compute Correlation",use_container_width=True,type='primary'):
                corr_matrix = self.dataset.corr(method=method, numeric_only=numeric_only)
                col2.dataframe(corr_matrix)
                
                # Plot heatmap
                fig, ax = plt.subplots()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                col2.pyplot(fig)
        
        with tab2:
            st.subheader("Dataset Preview", divider='blue')
            st.dataframe(self.dataset.head())
        
        with tab3:
            st.subheader("Documentation", divider='blue')
            st.markdown(
                """
                **pandas.DataFrame.corr**
                
                Computes pairwise correlation of columns, excluding NA/null values.
                
                **Parameters:**
                - **method**: {‘pearson’, ‘kendall’, ‘spearman’} - Method of correlation.
                - **numeric_only**: bool, default False - Include only float, int, or boolean data.
                
                **Returns:**
                - DataFrame - Correlation matrix.
                """
            )
