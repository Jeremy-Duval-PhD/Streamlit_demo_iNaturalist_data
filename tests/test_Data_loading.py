import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pages.Data_loading_functions import *
from Data_for_test import raw_data, clean_data

import pytest 
from streamlit.testing.v1 import AppTest


def test_get_clean_columns_order():
    columns_getted = get_clean_columns_order()
    is_valid = True
    for col in ['url', 'image_url', 'time_zone', 'quality_grade', \
            'latitude', 'longitude', 'public_positional_accuracy', \
            'scientific_name', 'common_name', 'iconic_taxon_name']:
        is_valid = is_valid and (col in columns_getted)
    
    assert is_valid
    
    
def test_clean_df():
    raw_df = pd.DataFrame(raw_data)
    df = clean_df(raw_df)
    
    assert len(df) == 10
    assert 'year' in list(df.columns)
    assert 'uuid' not in list(df.columns)
    
    
def test_init_all_session_state_var():
    at = AppTest.from_file("app.py").run()
    
    raw_df = pd.DataFrame(raw_data)
    df = pd.DataFrame(clean_data)
    file_name = 'filipendula_ulmaria.csv'
    
    init_all_session_state_var(raw_df, df, file_name)
    
    assert st.session_state['data_name'] == file_name
    assert st.session_state['raw_data'].shape == raw_df.shape
    assert st.session_state['data'].shape == df.shape
    
    min_year = min(list(df.index.year))
    max_year = max(list(df.index.year))
    assert st.session_state['years'][0] == min_year
    assert st.session_state['years'][-1] == max_year
    assert len(st.session_state['years']) == (max_year-min_year+1)