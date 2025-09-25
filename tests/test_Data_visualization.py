import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pages.Data_visualization_functions import *
from pages.Data_visualization_functions import _get_1st_cat
from pages.Data_loading_functions import init_all_session_state_var
from Data_for_test import raw_data, clean_data

import pytest 
from streamlit.testing.v1 import AppTest

import pandas as pd




def test_get_1st_cat():
    df = pd.DataFrame(clean_data)
    assert _get_1st_cat(pd.DataFrame(), None) == None
    assert _get_1st_cat(df, 'quality_grade') == 'research'
    assert _get_1st_cat(df, 'common_name') == 'Reine-des-prés'
    assert _get_1st_cat(df, 'year') == 2017
    
    

def test_get_pie_lbl():
    values = ['a'] * 10 + ['b'] * 90
    df = pd.DataFrame({'lbl':values})
    vc = df['lbl'].value_counts(normalize=True)
    res = sorted(get_pie_lbl(vc))
    assert len(res) == 2
    assert res[0] == 'a(10%)'
    assert res[1] == 'b(90%)'
    
    values = ['a'] * 100_000 + ['b']
    df = pd.DataFrame({'lbl':values})
    vc = df['lbl'].value_counts(normalize=True)
    res = sorted(get_pie_lbl(vc))
    assert len(res) == 2
    assert res[0] == 'a(100%)'
    assert res[1] == 'b(0%)'


def test_set_title(monkeypatch):
    df = pd.DataFrame(clean_data)

    calls = {}

    def mock_title(text):
        calls["title"] = text

    def mock_header(text):
        calls["header"] = text

    def mock_subheader(text):
        calls["subheader"] = text
        
    
    monkeypatch.setattr(st, "title", mock_title)
    monkeypatch.setattr(st, "header", mock_header)
    monkeypatch.setattr(st, "subheader", mock_subheader)
    
    set_title(df)

    assert calls["title"] == "Reine-des-prés"
    assert calls["header"] == "Filipendula ulmaria"
    assert calls["subheader"] == "Plantae"
