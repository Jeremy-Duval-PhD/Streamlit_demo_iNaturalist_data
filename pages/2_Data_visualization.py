import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from itertools import combinations

import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from scipy.stats import shapiro, levene
from scipy.spatial.distance import pdist, squareform

from sklearn.preprocessing import PowerTransformer
from skbio.stats.distance import DistanceMatrix, permanova


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
    
    #with st.popover("warning"):
    #    st.badge('''
    #            ⚠️ *Streamlit may have trouble loading the KDE graph and displaying
    #            it as lines that is difficult to read.*'''
    #            , color="orange")
    
    st.markdown('''
                KDE plots are used to analyze the distribution of data and the 
                relationships between them.
                
                On the left, the relationship between latitude and years.
                On the right, the relationship between longitude and years.
                ''')
    col2_1, col2_2 = st.columns([1,1], vertical_alignment='center') 
    
    sns_kde_lat = sns.jointplot(data=filter_df, x=filter_df['year'], \
                            y='latitude', kind="kde", fill=True)
    col2_1.pyplot(sns_kde_lat)
    
    sns_kde_lon = sns.jointplot(data=filter_df, x=filter_df['year'], \
                            y='longitude', kind="kde", fill=True)
    col2_2.pyplot(sns_kde_lon)


@st.fragment
def plot_regr_scatt(filter_df, year_filter):
    st.markdown('''
                ### Regression Plots
                Here, the graphs present spatial observations in the form of 
                scatter plots. A linear regression line (with confidence intervals) 
                attempts to approximate the trend of these scatter plots.
                
                Comparing these scatter plots, as well as the lines if they are 
                relevant, between years makes it possible to potentially see 
                initial geospatial changes over time.
                ''')
    col1, col2 = st.columns([3,1], vertical_alignment='top') 
    
    years_lst = list(range(year_filter[0], year_filter[-1]+1))
    years_selected = col2.multiselect(
        "Years to show :",
        years_lst,
        default=years_lst
        )
    
    nfilter_df = filter_df.copy()
    nfilter_df = nfilter_df[nfilter_df['year'].isin(years_selected)]
    
    col1.pyplot(sns.lmplot(data=nfilter_df, x="longitude", y="latitude", hue="year"))


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


def plot_first_and_filters(df):
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

    return filter_df, year_filter, quality_grade_filter


def interv_long_lati_regr_tab(model_lat, model_lon):
    summary_data = {
        "Variable": ["Latitude", "Longitude"],
        "Intercept": [model_lat.params["const"], model_lon.params["const"]],
        "Slope (year)": [model_lat.params["year"], model_lon.params["year"]],
        "p-value (slope)": [model_lat.pvalues["year"], model_lon.pvalues["year"]],
        "R²": [model_lat.rsquared, model_lon.rsquared]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.round(4)
    
    st.dataframe(summary_df)
    
    if model_lat.pvalues["year"] <= 0.05:
        st.badge("Year influence latitude", icon=":material/check:", color="green")
    else:
        st.badge("Year don't influence latitude", icon=":material/close:", color="red")
    
    if model_lon.pvalues["year"] <= 0.05:
        st.badge("Year influence longitude", icon=":material/check:", color="green")
    else:
        st.badge("Year don't influence longitude", icon=":material/close:", color="red")
    
    

def get_predictions_with_ci(model, X):
    predictions = model.get_prediction(X)
    pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI
    return pred_summary[["mean", "mean_ci_lower", "mean_ci_upper"]]    
    
    
def plot_interv_long_lati_regr(X_sorted, model_lat, lat_pred, model_lon, lon_pred, filter_df):
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Latitude
    sns.scatterplot(x="year", y="latitude", data=filter_df, ax=axs[0], label="Data")
    axs[0].plot(X_sorted["year"], lat_pred["mean"], color="red", label="Regression")
    axs[0].fill_between(X_sorted["year"], lat_pred["mean_ci_lower"], lat_pred["mean_ci_upper"],
                    color="red", alpha=0.2, label="IC 95%")
    axs[0].set_title("Latitude vs Year")
    axs[0].set_xlabel("year")
    axs[0].set_xticks(X_sorted["year"])
    axs[0].legend()
    
    # Longitude
    sns.scatterplot(x="year", y="longitude", data=filter_df, ax=axs[1], label="Data")
    axs[1].plot(X_sorted["year"], lon_pred["mean"], color="red", label="Regression")
    axs[1].fill_between(X_sorted["year"], lon_pred["mean_ci_lower"], lon_pred["mean_ci_upper"],
                    color="red", alpha=0.2, label="IC 95%")
    axs[1].set_title("Longitude vs Year")
    axs[1].set_xlabel("year")
    axs[1].set_xticks(X_sorted["year"])
    axs[1].legend()
    
    # General
    plt.suptitle("Regression of Geographic Coordinates vs. Year", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)


@st.fragment
def plot_interv_long_lati_regression(filter_df, year_filter):

    st.markdown(f'''
                ### Ression on coordinates between {year_filter[0]} and {year_filter[1]}
                
                The following linear regression analyses allow us to determine 
                whether there is a significant impact of the year on latitude 
                (or longitude) with a confidence level of 95%.
                
                If so, the coefficients will allow us to determine whether latitude 
                (or longitude) is increasing (positive sign) or decreasing (negative 
                sign), as well as the rate at which this is happening.         
                ''')
                
    # regressions
    X = sm.add_constant(filter_df["year"])
    model_lat = sm.OLS(filter_df["latitude"], X).fit()
    model_lon = sm.OLS(filter_df["longitude"], X).fit()
    
    X_sorted = X.sort_values(by="year")
    
    lat_pred = get_predictions_with_ci(model_lat, X_sorted)
    lon_pred = get_predictions_with_ci(model_lon, X_sorted)
    
    # plots
    plot_interv_long_lati_regr(X_sorted, model_lat, lat_pred, model_lon, lon_pred, filter_df)
    interv_long_lati_regr_tab(model_lat, model_lon)
    
    
    
    
def levene_test(df, variables, container, step=None):
    # Initialisation d'un tableau de résultats
    levene_results = []
    
    # Test de Levene pour chaque variable
    invalid_var = []
    for var in variables:
        grouped_data = [group[var].values for _, group in df.groupby("year")]
        stat, p = levene(*grouped_data)
        levene_results.append({
            "Variable": var,
            "Stat": round(stat, 4),
            "p-value": round(p, 4)
        })
        if p < 0.05:
            invalid_var.append(var)
    
    # Résultat sous forme de DataFrame
    levene_df = pd.DataFrame(levene_results)
    
    msg=""
    if step != None: 
        msg += f'''
               📌 Step {step} : 
               
               
                '''
    msg += '''
        Variance equality test : 
        
        '''
        
    if len(invalid_var) == 0 :
        msg += 'All group variances are similar.'
    else:
        msg += 'Variances are differents for variable(s) '
        for var in invalid_var:
            msg += f'{var}, '
        msg = msg[:-2] + '.'
        
    container.write(msg)
    container.write(levene_df)
    
    
    
def test_normality(data, group_col, variables):
    results = []
    grouped = data.groupby(group_col)
    for name, group in grouped:
        row = {"year": name}
        for var in variables:
            stat, p = shapiro(group[var])
            row[f"p-value {var}"] = round(p, 4)
        results.append(row)
    return pd.DataFrame(results)


def normality_test(df, variables, container, step=None):
    normality_df = test_normality(df, "year", variables)
    pval_cols = [f"p-value {v}" for v in variables]
    non_normal_years = normality_df[normality_df[pval_cols].lt(0.05).any(axis=1)]
    
    years = non_normal_years['year'].to_list()
    
    msg = ""
    if step != None: 
        msg += f'''
               📌 Step {step} : 
               
               
                '''
    msg += '''
        Distribution test : 
        
        '''
    if len(years) == 0:
        msg += 'All years have a normal distribution.'
    else:
        msg += 'Years without a normal distribution are '
        for year in years:
            msg += f'{year}, '
        msg = msg[:-2] + '.'
    container.markdown(msg)
    container.write(normality_df)
    
    need_normalization = (normality_df[pval_cols] < 0.05).any().any()
    
    return need_normalization, normality_df


def normalisation_by_yeo_johnson(filter_df, variables, normality_df, container):
    container.markdown('''
                📌 Step 3:
                    
                Normality rejected for certain variables. Application of a Yeo-Johnson transformation.
                ''')

    # Yeo-Johnson transformation to approximate a gaussian distribution
    pt = PowerTransformer(method='yeo-johnson')
    df_transformed = filter_df.copy()
    df_transformed[variables] = pt.fit_transform(filter_df[variables])

    # normality test after transformation
    need_normalization, normality_df =  normality_test(df_transformed, variables, \
                                                       container, step=4)
    return df_transformed, need_normalization, normality_df


def posthoc_manova(variables, df_for_manova):
    
    st.markdown('''
                #### MANOVA post hoc
                
                MANOVA allows us to determine whether there is a difference between 
                groups, but not between which groups the difference lies.
                To do this, we will quickly perform an ANOVA test for each variable 
                (longitude and latitude), followed by a pairwise Tukey test. 
                ''')
    
    for var in variables:
        container = st.container()
        expander = st.expander("Details")
        container.write(f"\n🔍 Post hoc for {var}")
        
        # ANOVA
        model = ols(f"{var} ~ C(year)", data=df_for_manova).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        expander.write("ANOVA results :")
        expander.write(anova_table)
        f_stat = anova_table["F"][0]
        p_value_anova = anova_table["PR(>F)"][0]
        if p_value_anova < 0.05:
            container.badge("ANOVA has succeed", icon=":material/check:", color="green")

            # Tukey HSD
            tukey = pairwise_tukeyhsd(endog=df_for_manova[var], \
                                      groups=df_for_manova["year"], alpha=0.05)
            
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], \
                                    columns=tukey._results_table.data[0])
            expander.write("Tukey results :")
            expander.write(tukey_df)
            significatifs = tukey_df[tukey_df["reject"] == True]
            if significatifs.empty:
                container.write("Tukey test results - no significant differencies detected.")
            else: 
                significatifs["comparaison"] = significatifs["group1"].astype(str) \
                    + " - " + significatifs["group2"].astype(str)
                final_df = significatifs[["comparaison", "p-adj"]].rename(columns={"p-adj": "p-value"})

                container.write("Tukey test results - year pairs significantly differents :")
                container.write(final_df)
        else:
            container.badge("ANOVA has failed", icon=":material/close:", color="red")


def show_manova_test_results(res):
    stat = res['year']['stat']['Value']["Pillai's trace"]
    pval = res['year']['stat']['Pr > F']["Pillai's trace"]
    
    summary_data = {
        "Variable": ["year"],
        "Test": ["Pillai's trace"],
        "Result": [stat],
        "p-value": [pval],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.round(4)
    
    st.dataframe(summary_df)

    if pval <= 0.05:
        st.badge("Year influence coordinates", icon=":material/check:", color="green")
    else:
        st.badge("Year don't influence coordinates", icon=":material/close:", color="red")
    


@st.fragment
def manova_test(filter_df):
    variables = ["latitude", "longitude"]
    
    expander = st.expander('Pre-tests')
    
    # Test of variance equality between group
    levene_test(filter_df, variables, expander, step=1)
    
    # Normality test (normality required for MANOVA)
    need_normalization, normality_df =  normality_test(filter_df, variables, \
                                                       expander, step=2)
    
    if need_normalization:
        df_transformed, need_normalization, normality_df = \
            normalisation_by_yeo_johnson(filter_df, variables, normality_df, expander)
        # Utilisation des données transformées
        df_for_manova = df_transformed
    else:
        expander.markdown('''
                    📌 Step 3:
                        
                    Normality validated.
                    ''')
        df_for_manova = filter_df
    
    mtv = MANOVA.from_formula('latitude + longitude ~ year', data=df_for_manova)
    res = mtv.mv_test()
    show_manova_test_results(res)
    
    posthoc_manova(variables, df_for_manova)
    
    
def plot_manova(filter_df, year_filter):
    st.markdown(f'''
                ### MANOVA test on coordinates between {year_filter[0]} and {year_filter[1]}
                
                The ANOVA test is a variance analysis test used to compare a set 
                of groups in order to determine whether they are significantly 
                different.
                The MANOVA test is a variant that can be used to study the impact 
                of variables **x** (in this case, *year*) on several related variables 
                **y** (in this case, *longitude* and *latitude*).
                These tests are based on three assumptions:
                1. Independence of observations
                2. Homogeneity of variances
                3. Normal distribution 
                
                In the pre-test section, we begin by testing the homogeneity of 
                variances with a Levene's test.
                
                Then, we use a Shapiro test to assess whether 
                the distribution of each variable **y** as a function of **x** 
                follows a normal distribution.
                If this is not the case for each year, we perform a Yeo-Johnson 
                transformation to correct the data distribution.
                We then perform a new normality test.
                
                In all cases, we then perform a MANOVA test, here a Pillai's test, 
                which has the advantage of being robust and resistant to imbalances 
                between groups, as well as to the problem of data normality.
                ''')
                
    manova_test(filter_df)
     

@st.cache_data(ttl=3600, show_spinner=False)
def simple_permanova(filter_df, year_filter, grouping, _dm):
    with st.spinner("PERMANOVA test is running...", show_time=True):
        permanova_result = permanova(_dm, grouping=grouping)
        
        stat = permanova_result['test statistic']
        pval = permanova_result['p-value']
    
        summary_data = {
            "Variable": ["year"],
            "Test": ["PERMANOVA, pseudo-F"],
            "Result": [stat],
            "p-value": [pval],
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(4)
    
    return summary_df, pval


@st.cache_data(ttl=3600, show_spinner=False)
def pairwise_permanova(filter_df, year_filter, grouping, _dm):
    n = len(np.unique(grouping))
    nb_combinations = n * (n - 1) // 2
    
    progress_text = "Paiwise permanova in progress. Please wait."
    combi_bar = st.progress(0, text=progress_text)
    
    pairwise_results = []
    i=0
    for g1, g2 in combinations(np.unique(grouping), 2):
        # group index
        mask = (grouping == g1) | (grouping == g2)
        
        # distance sub matrix
        sub_dm = _dm.filter(filter_df.loc[mask, "observation_id"].astype(str).tolist())
        
        # sub groups
        sub_groups = filter_df.loc[mask, "year"].astype(str).values
        
        # PERMANOVA
        res = permanova(sub_dm, grouping=sub_groups)
        pairwise_results.append({
            "Groupe 1": g1,
            "Groupe 2": g2,
            "p-value": res["p-value"],
            "stat": res["test statistic"]
        })
        i += 1
        combi_bar.progress(int(100/nb_combinations*i), text=progress_text)
    
    # Bonnferroni correction
    with st.spinner("Boneferroni correction is running...", show_time=True):
        pairwise_df = pd.DataFrame(pairwise_results)
        pairwise_df["p-ajusted"] = multipletests(pairwise_df["p-value"], method="bonferroni")[1]
        pairwise_df = pairwise_df.sort_values("p-ajusted")
    
        # Results
        significants = pairwise_df[pairwise_df["p-ajusted"] <= 0.05]
        significants["Comparaison"] = significants["Groupe 1"].astype(str) \
            + " - " + significants["Groupe 2"].astype(str)
        significants = significants[['Comparaison', 'p-ajusted']]
        significants = significants.rename(columns={"p-ajusted": "p-value"})
    
    return pairwise_df, significants


@st.fragment
def permanova_test(filter_df, year_filter):
    st.markdown(f'''
                ### PERMANOVA test on coordinates between {year_filter[0]} and {year_filter[1]}
                
                The PERMANOVA test is a nonparametric variant of MANOVA. 
                In this sense, this test is more robust but less accurate.
                ''')
                
    filter_df = filter_df.copy()
    filter_df["observation_id"] = ["obs_" + str(i) for i in range(0, len(filter_df.index))]
    
    # distance matrix calculation
    coords = filter_df[["latitude", "longitude"]].values
    dist_matrix = squareform(pdist(coords, metric='euclidean'))
    dm = DistanceMatrix(dist_matrix, ids=filter_df["observation_id"].astype(str))
    
    # permanova with year as groups
    grouping = filter_df["year"].values
    
    # Test
    summary_df, pval = simple_permanova(filter_df, year_filter, grouping, dm)
    st.dataframe(summary_df)
    if pval <= 0.05:
        st.badge("Year influence coordinates", icon=":material/check:", color="green")
    else:
        st.badge("Year don't influence coordinates", icon=":material/close:", color="red")
    
    
    #post hoc
    st.markdown('''
                #### PERMANOVA post hoc
                
                PERMANOVA allows us to determine whether there is a difference between 
                groups, but not between which groups the difference lies.
                To do this, we will perform a PERMANOVA test for each variable 
                pair of years. 
                ''')
    
    pairwise_df, significants = pairwise_permanova(filter_df, year_filter, grouping, dm)
    st.write("Pairwise PERMANOVA tests results - year pairs significantly differents :")
    st.write(significants.round(4))
    expander = st.expander("Details")
    expander.write(pairwise_df)





    
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
    plot_interv_long_lati_regression(filter_df, year_filter)
    st.write('')
    plot_manova(filter_df, year_filter)
    st.write('')
    permanova_test(filter_df, year_filter)
    st.write('')