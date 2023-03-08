import pandas as pd
import streamlit as st
st.set_page_config(layout="wide", page_title="MultiCollinearity Checker")
st.title("MultiCollinearity Checker")

df_page2 = pd.read_csv(r"C:\Users\91992\Downloads\archive (5)\for_page2.csv")
col_list_out_form = st.multiselect("Select variables to check for multi collinearity", df_page2.columns)

if len(col_list_out_form) > 0:
    st.write(df_page2[col_list_out_form].corr())