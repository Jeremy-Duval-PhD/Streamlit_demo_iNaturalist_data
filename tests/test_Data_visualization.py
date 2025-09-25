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
import numpy as np



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
    
    
def test_get_centroids():
    df = pd.DataFrame({
        'year':[2000,2000,2000,2000,2002,2002,2002,2003],
        'latitude':[10,20,10,0,15,25,None,30],
        'longitude':[10,20,10,0,15,25,None,30]
        })
    res = get_centroids(df)
    
    assert len(res.columns) == 5 # function add destinations
    assert 2001 not in res['year'].to_list()
    
    assert res.loc[2000,'latitude'] == 10
    assert res.loc[2002,'latitude'] == 20
    assert res.loc[2003,'latitude'] == 30 
    
    assert res.loc[2000,'longitude'] == 10
    assert res.loc[2002,'longitude'] == 20
    assert res.loc[2003,'longitude'] == 30 
    
    assert res.loc[2000, 'dest_lat'] == res.loc[2002, 'latitude']
    assert res.loc[2002, 'dest_lat'] == res.loc[2003, 'latitude']
    
    assert res.loc[2000, 'dest_lon'] == res.loc[2002, 'longitude']
    assert res.loc[2002, 'dest_lon'] == res.loc[2003, 'longitude']
    
    
def test_get_predictions_with_ci():
    df = pd.DataFrame({
        'year': [2000, 2001, 2002, 2003, 2004],
        'latitude': [45.0, 45.5, 46.0, 46.5, 47.0]
    })


    X = sm.add_constant(df['year'])
    y = df['latitude']

    model = sm.OLS(y, X).fit()

    preds = get_predictions_with_ci(model, X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == len(X)
    for col in ["mean", "mean_ci_lower", "mean_ci_upper"]:
        assert col in preds.columns

    assert (preds["mean_ci_lower"] <= preds["mean"]).all()
    assert (preds["mean_ci_upper"] >= preds["mean"]).all()  
    
    
def test_interv_long_lati_regr_tab(monkeypatch):
    df = pd.DataFrame({
        'year':[2000,2000,2000,2000,2002,2002,2002,2002],
        'latitude':[10,10,10,10,15,25,35,45],
        'longitude':[10,20,30,40,10,20,30,40],
        })
    
    X = sm.add_constant(df['year'])
    
    y = df['latitude']
    model_lat = sm.OLS(y, X).fit()
    
    y = df['longitude']
    model_lon = sm.OLS(y, X).fit()
    
    
    calls = {}
    
    def mock_dataframe(df):
        calls["dataframe"] = df

    def mock_badge(text, icon, color):
        calls[f"badge_{color}_{text}"] = True

    monkeypatch.setattr(st, "dataframe", mock_dataframe)
    monkeypatch.setattr(st, "badge", mock_badge)

    interv_long_lati_regr_tab(model_lat, model_lon)


    assert "dataframe" in calls
    df_result = calls["dataframe"]
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape == (2, 5)
    assert set(df_result.columns) == {
        "Variable", "Intercept", "Slope (year)", "p-value (slope)", "R²"
    }

    assert "badge_green_Year influence latitude" in calls
    assert "badge_red_Year don't influence longitude" in calls
    
    
    
@pytest.mark.parametrize("y_values, expected_color, expected_icon", [
    (
        [1.1, 2.0, 3.1, 4.0, 5.1, 6.2, 6.9, 8.0],  # normal distribution
        "green", "All years have a normal distribution."
    ),
    (
        [1, 10, 1, 10, 1, 10, 1, 10],  # Non-normal distribution
        "red", "Years without a normal distribution"
    ),
])
def test_normality_test(monkeypatch, y_values, expected_color, expected_icon):
    df = pd.DataFrame({
        'x': range(1, len(y_values) + 1),
        'y': y_values,
        'year': [2020]*len(y_values) 
    })

    calls = {}

    def mock_write(msg):
        calls['write'] = msg

    def mock_badge(text, icon=None, color=None):
        calls['badge'] = {
            "text": text,
            "icon": icon,
            "color": color
        }

    def mock_markdown(msg):
        calls['markdown'] = msg

    monkeypatch.setattr(st, "write", mock_write)
    monkeypatch.setattr(st, "badge", mock_badge)
    monkeypatch.setattr(st, "markdown", mock_markdown)

    container = st
    variables = ['y']
    need_normalization, normality_df = normality_test(df, variables, container)

    if expected_color == 'green':
        assert not need_normalization
    else:
        assert need_normalization

    assert 'markdown' in calls
    assert expected_icon in calls['markdown']
    
    
@pytest.mark.parametrize("y_values", [
    [1.1, 2.0, 3.1, 4.0, 5.1, 6.2, 6.9, 8.0],  # Normal distribution
    [1, 10, 1, 10, 1, 10, 1, 10],              # Non-normal distribution
])
def test_normalisation_by_yeo_johnson(y_values):
    # Création du DataFrame
    df = pd.DataFrame({
        'x': range(1, len(y_values) + 1),
        'y': y_values,
        'year': [2020] * len(y_values)
    })

    container = st
    variables = ['y']

    need_normalization_before, _ = normality_test(df, variables, container)

    df_transformed, need_normalization_after, _ = normalisation_by_yeo_johnson(
        df.copy(), variables, _, container
    )

    assert not np.allclose(df['y'].values, df_transformed['y'].values)
    
    
    