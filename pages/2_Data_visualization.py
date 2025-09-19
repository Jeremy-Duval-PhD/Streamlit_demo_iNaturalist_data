import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def __get_1st_cat(df, col):
    return df[col].value_counts(sort='asc').index[0]

def set_title(df):
    cm_name = __get_1st_cat(df, 'common_name')
    sc_name = __get_1st_cat(df, 'scientific_name')
    taxon = __get_1st_cat(df, 'iconic_taxon_name')
    
    st.title(cm_name.capitalize())
    st.header(sc_name.capitalize())
    st.subheader(taxon.capitalize())
    st.write('')
    
def get_pie_lbl(vc):
    return [str(x) + '(' + str(int(round(y*100,0))) + '%)' for x, y in zip(vc.index, vc.values)]




if 'data_name' not in st.session_state:
    st.badge("⚠️ Please load a dataset", color="orange")
else:
    df = st.session_state['data']
    set_title(df)

    
    col0_1, col0_2, col0_3 = st.columns([1,1,1], vertical_alignment='center')
    col0_1.markdown('### Number of observation per year')
    nb_obs_year_fig = plt.figure()
    noy = df['latitude'].resample('YE').count()
    noy.index = noy.index.year
    noy.plot()
    col0_1.pyplot(nb_obs_year_fig)
    
    col0_2.markdown('### Quality grade proportion')
    qg_pie_fig = plt.figure()
    vc = df['quality_grade'].value_counts(normalize=True)
    vc.plot.pie(labels=get_pie_lbl(vc))
    col0_2.pyplot(qg_pie_fig)
    
    #filters part
    col0_3.markdown('### Data filters')
    
    min_year = st.session_state['years'][0]
    max_year = st.session_state['years'][-1]
    year_filter = col0_3.slider('Filter years', \
                              min_year, \
                              max_year, \
                              (min_year,max_year))
    quality_grade_filter = col0_3.segmented_control(
        'Observation quality', 
        ['research','all'], \
        selection_mode='single',\
        default='all')
        
    filter_df = df.loc[(df.index >= f'{year_filter[0]}-01-01')
                      &(df.index <= f'{year_filter[1]}-12-31')]
    if quality_grade_filter != 'all':
        filter_df = filter_df[filter_df['quality_grade'] == quality_grade_filter]

    
    
    st.write('')

    # map
    st.markdown(f'### Observations\' map between {year_filter[0]} ' \
                   +f'and {year_filter[1]}')
    st.map(filter_df)
    
    st.write('')
    
    # figures part
    st.markdown('### Kernel Density Estimator')
    col2_1, col2_2 = st.columns([1,1], vertical_alignment='center') 
    
    sns_kde_lat = sns.jointplot(data=filter_df, x=filter_df.index, \
                            y='latitude', kind="kde", fill=True)
    col2_1.pyplot(sns_kde_lat)
    
    sns_kde_lon = sns.jointplot(data=filter_df, x=filter_df.index, \
                            y='longitude', kind="kde", fill=True)
    col2_2.pyplot(sns_kde_lon)