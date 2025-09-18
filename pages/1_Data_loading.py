import streamlit as st
import pandas as pd

@st.cache_data
def upload_file_to_df(uploaded_file):
    with st.spinner("Please wait..."):
        df = pd.read_csv(uploaded_file, index_col='observed_on', parse_dates=True)
    return df

@st.cache_data
def clean_df(df):
    df = df.drop_duplicates()
    col_to_keep = ['time_zone', 'quality_grade', 'license', 'url', 'image_url', 'num_identification_agreements', 
               'latitude', 'longitude', 'public_positional_accuracy', 'scientific_name', 'common_name', 'iconic_taxon_name']
    df = df[col_to_keep]
    
    return df


st.title("Load your iNaturalist data")

uploaded_file = st.file_uploader("Upload your iNaturalist data", type=['csv'])
if uploaded_file is not None:
    raw_df = upload_file_to_df(uploaded_file)
    file_name = uploaded_file.name
    if 'data_name' not in st.session_state \
    or file_name != st.session_state['data_name']:
        st.session_state['data_name'] = file_name
        st.session_state['raw_data'] = raw_df
        df = clean_df(raw_df)
        st.session_state['data'] = df
else:
    df = pd.DataFrame()
    raw_df = pd.DataFrame()

tab1, tab2 = st.tabs(["Cleaned Data", "Raw Data"])       
tab1.dataframe(df)    
tab2.dataframe(raw_df)