import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk


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
    
    
@st.dialog('⚠️ Important informations')
def dialog_important_info():
    st.markdown('''
                This application provides simple data visualization tools and 
                allows for preliminary analysis of geospatial changes in 
                observations of a species over time.
                
                It is important to note that we cannot rely on this information 
                alone. It remains generic and is influenced by the observers 
                themselves.
                ''')
    
    
def get_pie_lbl(vc):
    return [str(x) + '(' + str(int(round(y*100,0))) + '%)' for x, y in zip(vc.index, vc.values)]


@st.fragment
def show_map(filter_df):
    st.map(filter_df)
    
    
@st.fragment
def plot_KDE(filter_df):
    st.markdown('''
                ### Kernel Density Estimator
                ''')
    with st.popover("warning"):
        st.badge('''
                ⚠️ *Streamlit may have trouble loading the KDE graph and displaying
                it as lines that is difficult to read.*'''
                , color="orange")
    st.markdown('''
                KDE plots are used to analyze the distribution of data and the 
                relationships between them.
                
                On the left, the relationship between latitude and years.
                On the right, the relationship between longitude and years.
                ''')
    col2_1, col2_2 = st.columns([1,1], vertical_alignment='center') 
    
    sns_kde_lat = sns.jointplot(data=filter_df, x=filter_df.index, \
                            y='latitude', kind="kde", fill=True)
    col2_1.pyplot(sns_kde_lat)
    
    sns_kde_lon = sns.jointplot(data=filter_df, x=filter_df.index, \
                            y='longitude', kind="kde", fill=True)
    col2_2.pyplot(sns_kde_lon)


@st.fragment
def plot_regr_scatt(filter_df):
    st.markdown('''
                ### Regression Plots
                Here, the graphs present spatial observations in the form of 
                scatter plots. A linear regression line (with confidence intervals) 
                attempts to approximate the trend of these scatter plots.
                
                Comparing these scatter plots, as well as the lines if they are 
                relevant, between years makes it possible to potentially see 
                initial geospatial changes over time.
                ''')
    
    filter_df = filter_df.copy()
    st.pyplot(sns.lmplot(data=filter_df, x="longitude", y="latitude", hue="year"))


def get_centroids(filter_df):
    centroids = filter_df.groupby('year')[['year','latitude', 'longitude']].mean()
    
    new_lat = []
    new_lon = []
    i = 0
    for index, row in centroids.iterrows():
        new_lat.append(row['latitude'])
        new_lon.append(row['longitude'])
        if i > 0:
            new_lat[i-1] = row['latitude']
            new_lon[i-1] = row['longitude']
        i+=1
    centroids["dest_lat"] = new_lat
    centroids["dest_lon"] = new_lon
    
    return centroids


def plot_pydeck_map(centroids, container):
    layer1 = pdk.Layer(
        "ScatterplotLayer",
        centroids,
        pickable=True,
        get_position=["longitude", "latitude"],
        get_color=[255, 0, 0, 160],
        get_radius=10000,
    )
    
    layer2 = pdk.Layer(
        "ArcLayer",
        centroids,
        width_scale=10,
        get_source_position=["longitude", "latitude"],
        get_target_position=["dest_lon", "dest_lat"],
        get_source_color=[255, 0, 0, 160],
        get_target_color=[0, 128, 200],
        auto_highlight=True,
    )
    
    # Set the viewport
    view_state = pdk.ViewState(
        latitude=46.75,
        longitude=2,
        zoom=5,
        pitch=0,
    )
        
    deck = pdk.Deck(layers=[layer1, layer2],  
                    initial_view_state=view_state,
                    tooltip={"text": "{year}"})
    container.pydeck_chart(deck)
    
@st.fragment
def plot_centroid_scatter_plot(centroids, container):
    #fig = plt.figure()
    #plt.plot(data=centroids, x='longitude', y='latitude', labels='year')
    
    
    fig, ax = plt.subplots()
    ax.scatter(centroids['longitude'], centroids['latitude'])
    
    for index, row in centroids.iterrows():
        ax.annotate(int(row['year']), (row['longitude'], row['latitude']))
    container.pyplot(fig)


def plot_annual_centroids(filter_df):
    st.markdown('''
                ### Centroids evolution
                ''')
    col1, col2 = st.columns([3,2], vertical_alignment='center')
    
    col1.markdown('''
                The centroids are calculated based on the average longitude and
                latitude for each year. By observing their evolution, we can 
                visualize the potential movement of the species across the territory.
                
                On the right, the centroids are plotted as scatter plots. 
                
                Below, they have been placed on the map and linked together from
                one year to the next.         
                ''')
    
    centroids = get_centroids(filter_df)
    plot_centroid_scatter_plot(centroids, col2)
    plot_pydeck_map(centroids, st)


def plot_movement_analysis(filter_df):
    st.write("TODO")
    
def plot_centroids_regression(filter_df):
    st.write("TODO")


def plot_distance_analysis(filter_df):
    st.write("TODO")
    
    
def plot_manova(filter_df):
    st.write("TODO")
    
    #plot_centroids_regression(filter_df)
    #plot_distance_analysis(filter_df)





if 'data_name' not in st.session_state:
    st.badge("⚠️ Please load a dataset", color="orange")
else:
    if 'important_dialog' not in st.session_state:
        st.session_state['important_dialog'] = True
        dialog_important_info()
    
    df = st.session_state['data']
    set_title(df)

    
    col0_1, col0_2, col0_3 = st.columns([1,1,1], vertical_alignment='top')
    col1_1, col1_2, col1_3 = st.columns([1,1,1], vertical_alignment='center')
    col0_1.markdown('### Number of observation per year')
    nb_obs_year_fig = plt.figure()
    noy = df['latitude'].resample('YE').count()
    noy.index = noy.index.year
    noy.plot()
    col1_1.pyplot(nb_obs_year_fig)
    
    col0_2.markdown('### Quality grade proportion')
    qg_pie_fig = plt.figure()
    vc = df['quality_grade'].value_counts(normalize=True)
    vc.plot.pie(labels=get_pie_lbl(vc))
    col1_2.pyplot(qg_pie_fig)
    
    #filters part
    col0_3.markdown('### Data filters')
    
    min_year = st.session_state['years'][0]
    max_year = st.session_state['years'][-1]
    year_filter = col1_3.slider('Filter years', \
                              min_year, \
                              max_year, \
                              (min_year,max_year))
    quality_grade_filter = col1_3.segmented_control(
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
    show_map(filter_df)
    
    st.write('')
    
    # figures part
    plot_KDE(filter_df)
    st.write('')
    plot_regr_scatt(filter_df)
    st.write('')
    plot_annual_centroids(filter_df)
    st.write('')
    plot_movement_analysis(filter_df)
    st.write('')
    plot_manova(filter_df)
    st.write('')