import numpy as np
import pandas as pd
import streamlit
#from streamlit.hashing import _CodeHasher
from sklearn.metrics import r2_score
import streamlit as st
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

#Styling
st.set_page_config(layout="wide", page_title="Linear Model Simulator")



st.title("Linear Model Simulator")
with st.sidebar:
    uploaded_file = st.sidebar.file_uploader("Upload dataset file", type=['csv', 'xlsx'])

    df = pd.DataFrame()
    #st.session_state["df_session"] = df
    #df = pd.get_dummies(df)
    if uploaded_file is not None:
        file_extension =uploaded_file.name.split('.')[-1]
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    #st.write(cat_cols)
    df.to_csv(r"for_page2.csv")
    def get_target():
        target = st.sidebar.selectbox("Choose target variable", df.columns)
        return target
    target = get_target()

    def get_predictors():
        l1 = df.columns
        #select_options = l1.remove(target)
        final_selections = st.sidebar.multiselect("Select options:", l1)
        if target in final_selections:
            st.warning("Cannot select target in predictors")
        return final_selections
    input_vars = get_predictors()

def get_model():
    model_list = ['Linear Regression', 'Lasso', 'Ridge', 'ElasticNet']
    selected_model = st.selectbox("Select linear model", model_list)
    return selected_model
model = get_model()

def hyper_params():
    alpha = 0
    max_iter = 0
    l1_ratio = 0
    if model == 'Lasso':
        alpha = st.slider("alpha", 0.1, 50.0)
        max_iter = st.slider("max_iter", 1, 7000)
    elif model == 'ElasticNet':
        alpha = st.slider("alpha", 0.1, 50.0)
        l1_ratio = st.slider("l1_ratio",0.0, 1.0)
        max_iter = st.slider("max_iter", 1, 7000)
    elif model == 'Ridge':
        alpha = st.slider("alpha", 0.1, 50.0)
        max_iter = st.slider("max_iter", 1, 7000)
    l1 = []
    l1.append(alpha)
    l1.append(max_iter)
    if model == "Lasso" or model == "Ridge":
        return l1
    elif model == "ElasticNet":
        l2 = l1
        l2.append(l1_ratio)
        return l2
hyper_params = hyper_params()



def data_generator():
    if len(cat_cols) > 0 and (set(cat_cols).intersection(set(input_vars))):
        dummy_initial = []
        for i in input_vars:
            if i in cat_cols:
                dummy_cols = pd.get_dummies(df[i])
                dummy_initial.append(i)
        #st.write(dummy_cols)
        X = df[input_vars[0:]]
        X = pd.concat([X, dummy_cols], axis= 1)
        X.drop(columns=dummy_initial, inplace= True)
    else:
        X = df[input_vars[0:]]
    y = df[target]
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.dataframe(X[0:5], )
    with col3:
        (st.dataframe(y[0:5], ))
    y = np.reshape(-1,1)
    temp = pd.get_dummies(X)
if uploaded_file != None and len(input_vars) != 0:
    data_generator()



col1, col2, col3 , col4, col5 = st.columns(5)

with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3:
    b1 = st.button('Generate Model')
if "list_of_results" not in st.session_state:
    st.session_state["list_of_results"] = []
def model_generator():
    if len(cat_cols) > 0 and (set(cat_cols).intersection(set(input_vars))):
        dummy_initial = []
        for i in input_vars:
            if i in cat_cols:
                dummy_cols = pd.get_dummies(df[i])
                dummy_initial.append(i)
        X = df[input_vars[0:]]
        X = pd.concat([X, dummy_cols], axis= 1)
        X.drop(columns=dummy_initial, inplace= True)
    else:
        X = df[input_vars[0:]]
    y = df[target[0:]]
    #y = y.reshape(-1,1)
    #y = np.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    #st.write(X_train)
    if model == "Linear Regression":
        m1 = LinearRegression()
        alpha = 0
        max_iter = 0
        l1_ratio = 0
    if model == "Lasso":
        alpha = hyper_params[0]
        max_iter = hyper_params[1]
        l1_ratio = 0
        m1 = Lasso(alpha = alpha, max_iter = max_iter)
    if model == "Ridge":
        alpha = hyper_params[0]
        max_iter = hyper_params[1]
        l1_ratio = 0
        m1 = Ridge(alpha = alpha, max_iter = max_iter)
    if model == "ElasticNet":
        alpha = hyper_params[0]
        l1_ratio = hyper_params[2]
        max_iter = hyper_params[1]
        m1 = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, max_iter = max_iter)
    m1.fit(X_train, y_train)
    y_pred = m1.predict(X_test)
    r_squared = r2_score(y_test, y_pred)
    st.write("Coefficients: ", m1.coef_)
    st.write("Intercept: ", m1.intercept_)
    st.write("R-squared value: ", r_squared)
    #st.session_state["list_of_results"].append([model,r_squared,alpha,max_iter, l1_ratio])
    st.session_state["list_of_results"].append([m1,r_squared])

if b1:
    model_generator()
b2 = st.button("All results")
if b2:
    st.write(st.session_state["list_of_results"])
