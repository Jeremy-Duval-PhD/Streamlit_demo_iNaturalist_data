import streamlit as st

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pages.Data_visualization_functions import *

    
######### Script #########

if 'data_name' not in st.session_state:
    st.warning("⚠️ Please load a dataset")
else:
    if 'important_dialog' not in st.session_state:
        st.session_state['important_dialog'] = True
        dialog_important_info()
    
    df = st.session_state['data']
    set_title(df)
    
    filter_df, year_filter, quality_grade_filter = plot_first_and_filters(df)
    st.write('')
    
    # map
    st.markdown(f'### Observations\' map between {year_filter[0]} ' \
                   +f'and {year_filter[1]}')
    show_map(filter_df)
    
    st.write('')
    
    # figures part
    plot_KDE(filter_df)
    st.write('')
    plot_regr_scatt(filter_df, year_filter)
    st.write('')
    plot_annual_centroids(filter_df)
    st.write('')
    if year_filter[0] != year_filter[1]:
        plot_interv_long_lati_regression(filter_df, year_filter)
        st.write('')
        plot_manova(filter_df, year_filter)
        st.write('')
        permanova_test(filter_df, year_filter)
        st.write('')
    else:
        st.info('For more statistics, please select multiple years.')