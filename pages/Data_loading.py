import streamlit as st
import pandas as pd
from pages.Data_loading_functions import *
    

st.title("Load your iNaturalist data")

#get_uploaded_data()
get_data_from_api()
raw_df, df, data_name = get_session_state_data()

tab1, tab2 = st.tabs(["Cleaned Data", "Raw Data"])       
tab1.dataframe(df, column_config={'image_url':st.column_config.ImageColumn(),\
                                  'url':st.column_config.LinkColumn(display_text="Observation link")},\
               column_order=get_clean_columns_order())    
tab2.dataframe(raw_df)