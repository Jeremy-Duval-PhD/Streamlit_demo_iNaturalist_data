import streamlit as st
import numpy as np
import pandas as pd

def __get_1st_cat(df, col):
    return df[col].value_counts(sort='asc').index[0]

def set_title(df):
    cm_name = __get_1st_cat(df, 'common_name')
    sc_name = __get_1st_cat(df, 'scientific_name')
    taxon = __get_1st_cat(df, 'iconic_taxon_name')
    
    st.title(cm_name.capitalize())
    st.header(sc_name.capitalize())
    st.subheader(taxon.capitalize())

if 'data_name' not in st.session_state:
    st.badge("⚠️ Please load a dataset", color="orange")
else:
    df = st.session_state['data']
    set_title(df)

