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

def init_all_session_state_var(raw_df, df, file_name):
    st.session_state['data_name'] = file_name
    st.session_state['raw_data'] = raw_df
    st.session_state['data'] = df
    # get min and max years and generate a range to avoid missing years
    min_year = min(list(df.index.year))
    max_year = max(list(df.index.year))
    st.session_state['years'] = list(range(min_year, max_year+1,1))
    

st.title("Load your iNaturalist data")

uploaded_file = st.file_uploader("Upload your iNaturalist data", type=['csv'])
if uploaded_file is not None:
    raw_df = upload_file_to_df(uploaded_file)
    file_name = uploaded_file.name
    if 'data_name' not in st.session_state \
    or file_name != st.session_state['data_name']:
        df = clean_df(raw_df)
        init_all_session_state_var(raw_df, df, file_name)
else:
    df = pd.DataFrame()
    raw_df = pd.DataFrame()

tab1, tab2 = st.tabs(["Cleaned Data", "Raw Data"])       
tab1.dataframe(df)    
tab2.dataframe(raw_df)