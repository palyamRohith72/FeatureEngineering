import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from feature_engine.selection import *
from FEATURESELECTION import *

varaibles=["Drop features","Drop Constant Features","Drop Duplicated Features",
           "Drop Correlated Features","Smart Correlated Selection","MRMR",
           "Select By Single Feature Performance","Recursive Feature Elimination",
           "Recursive Feature Addition","Drop High PSI Featutres","Select By Information Value",
           "Select By Shuffling","Select By Target Mean Performance","Probe Feature Selection"]

for i in variables:
  if i not in st.session_state:
    st.session_state[i]=None

class Features:
  def __init__(self,data):
    self.dataset=data
  def display(self):
    with st.sidebar:
      option_menu=option_menu("Select stage",["See Correlations","Select Features","Create Features"])
    if option_menu=="See Correlations":
      self.correlations()
    if option_menu=="Select Features":
      pass
    if option_menu=="Create Features":
      pass
  def correlations(self):
    tab1,tab2=st.tabs(["Perform Operations","View Data"])
    with tab1:
      col1,col2=st.columns([1,2],border=True)
      radio_options=col1.radio("Options Were",["pearson", "spearman", "kendall", "point","cramers", "variance_threshold"])
      with col2:
        if st.button("Execute Feature Selection",use_container_width=True):
          getattr(feature_selection, method, lambda: st.warning("Invalid Method"))()
