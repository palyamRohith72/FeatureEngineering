def drop_features(keyy):
  select_columns = st.multiselect("Select columns", self.dataset.columns.tolist())
  if select_columns:
      dataset = self.dataset.copy(deep=True)
      if st.button("Execute Feature Selection", use_container_width=True):   
          object=DropFeatures(select_columns)
          dataframe=object.fit_transform(dataset)
          st.session_state[keyy]=dataframe
          st.dataframe(dataframe)
def drop_constant_features(option):
    select_columns = st.multiselect("Select columns", self.dataset.columns.tolist())
    tol=int(st.number_input("""Threshold to detect constant/quasi-constant features. Variables showing the same value in a
    percentage of observations greater than tol will be considered 
    constant / quasi-constant and dropped. If tol=1, the transformer 
    removes constant variables. Else, it will remove quasi-constant variables.
    For example, if tol=0.98, the transformer will remove variables that show
    the same value in 98% of the observations.""",1))
    dataset = self.dataset.copy(deep=True)
    if st.button("Execute Feature Selection", use_container_width=True):   
        try:
            object=DropConstantFeatures(select_columns,tol)
            dataframe=object.fit_transform(dataset)
            st.session_state[option]=dataframe
            st.dataframe(dataframe)
        except Exception as e:
            st.error(e)
def drop_duplicated_features(option):
    select_columns = st.multiselect("Select columns", self.dataset.columns.tolist())
    dataset = self.dataset.copy(deep=True)
    if st.button("Execute Feature Selection", use_container_width=True):   
        try:
            object=DropDuplicateFeatures(select_columns)
            dataframe=object.fit_transform(dataset)
            st.session_state[option]=dataframe
            st.dataframe(dataframe)
        except Exception as e:
            st.error(e)

def drop_correlated_features(option):
    select_columns = st.multiselect("Select columns", self.dataset.columns.tolist())
    threshold=st.number_input("The correlation threshold above which a feature will be deemed correlated with another one and removed from the dataset.",0.1)
    method=st.selectbox("Correlation method -Can take ‘pearson’, ‘spearman’, ‘kendall’",["pearson","spearman","kendall"])
    dataset = self.dataset.copy(deep=True)
    if st.button("Execute Feature Selection", use_container_width=True):   
        try:
            object=DropCorrelatedFeatures(select_columns,method,threshold)
            dataframe=object.fit_transform(dataset)
            st.session_state[option]=dataframe
            st.dataframe(dataframe)
        except Exception as e:
            st.error(e)

def smart_correlated_selection(option):
    st.write("Performing smart correlated selection.")

def mrmr(self,option):
    st.write("Executing MRMR feature selection.")

def select_by_single_feature_performance(option):
    st.write("Selecting by single feature performance.")

def recursive_feature_elimination(option):
    st.write("Performing recursive feature elimination.")

def recursive_feature_addition(option):
    st.write("Performing recursive feature addition.")

def drop_high_psi_features(option):
    st.write("Dropping high PSI features.")

def select_by_information_value(option):
    st.write("Selecting by information value.")

def select_by_shuffling(option):
    st.write("Selecting by shuffling.")

def select_by_target_mean_performance(option):
    st.write("Selecting by target mean performance.")

def probe_feature_selection(option):
    st.write("Performing probe feature selection.")
