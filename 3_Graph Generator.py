import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
sns.set_style("dark")
st.set_option('deprecation.showPyplotGlobalUse', False)
df_page3 = pd.read_csv(r"C:\Users\91992\Downloads\archive (5)\for_page2.csv")
choice = st.sidebar.selectbox("Choose graph", ["Boxplot", "Scatterplot", "Countplot"])
numeric_cols = [col for col in df_page3.columns if df_page3[col].dtype.kind in ('i', 'f')]
cat_cols = [col for col in df_page3.columns if df_page3[col].dtype == 'object']

if choice == "Scatterplot":
    x_var = st.sidebar.selectbox("Choose x", numeric_cols)
    y_var = st.sidebar.selectbox("Choose y", numeric_cols)
    fig, ax = plt.subplots()
    sns.scatterplot(x = x_var, y = y_var, data = df_page3, ax = ax, palette='PuBu_r')
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(f"Scatterplot of {x_var} vs. {y_var}")
    st.pyplot(fig)
if choice == "Boxplot":
    x_var = st.sidebar.selectbox("Select variable", numeric_cols)
    fig, ax = plt.subplots()
    sns.boxplot(x = df_page3[x_var])
    ax.set_xlabel(x_var)
    ax.set_title(f"Boxplot of {x_var}")
    st.pyplot(fig)
if choice == "Countplot":
    if len(cat_cols) != 0:
        x_var = st.sidebar.selectbox("Select variable", cat_cols)
        l1_counter = range(1,len(df_page3[x_var])+1)
        #counter = st.sidebar.selectbox("Enter count", l1_counter)
        fig, ax = plt.subplots()
        #sns.countplot(x = df_page3[x_var][0:counter], order= df_page3[x_var][0:counter].value_counts().index)
        sns.countplot(x = df_page3[x_var])
        ax.set_xlabel(x_var)
        ax.set_title(f"Countplot of {x_var}")
        plt.xticks(rotation = 90)
        st.pyplot(fig)
    else:
        st.warning("No categorical variables to plot")
