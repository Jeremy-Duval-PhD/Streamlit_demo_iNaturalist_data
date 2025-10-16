import streamlit as st
import pandas as pd
from pages.Data_loading_functions import *
    

st.title("Load your iNaturalist data")

tab_search, tab_down = st.tabs(["From Search", "From File"])   
get_data_from_api(tab_search)
get_uploaded_data(tab_down)
raw_df, df, data_name = get_session_state_data()

st.divider()

tab1, tab2 = st.tabs(["Cleaned Data", "Raw Data"])       
tab1.dataframe(df, column_config={'image_url':st.column_config.ImageColumn(),\
                                  'url':st.column_config.LinkColumn(display_text="Observation link")},\
               column_order=get_clean_columns_order())    
tab2.dataframe(raw_df)