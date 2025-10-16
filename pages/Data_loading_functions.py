import streamlit as st
import pandas as pd
from pyinaturalist import get_observations
from pyinaturalist.node_api import get_places_autocomplete
import time
import random


@st.cache_data
def upload_file_to_df(uploaded_file):
    with st.spinner("Please wait..."):
        df = pd.read_csv(uploaded_file, index_col='observed_on', parse_dates=True)
    return df


def get_clean_columns_order():
    return ['url', 'image_url', 'time_zone', 'quality_grade', \
            'latitude', 'longitude', 'public_positional_accuracy', \
            'scientific_name', 'common_name', 'iconic_taxon_name']

@st.cache_data
def clean_df(df):
    df = df.drop_duplicates()
    col_to_keep = get_clean_columns_order()
    df = df[col_to_keep]
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df[df.index.notna()]
    
    df['year'] = df.index.year.astype(int)
    
    return df

def init_all_session_state_var(raw_df, df, file_name):
    st.session_state['data_name'] = file_name
    st.session_state['raw_data'] = raw_df
    st.session_state['data'] = df
    # get min and max years and generate a range to avoid missing years
    min_year = int(min(list(df.index.year)))
    max_year = int(max(list(df.index.year)))
    st.session_state['years'] = list(range(min_year, max_year+1,1))
    
    
def get_session_state_data():
    if 'data_name' in st.session_state:
        data_name = st.session_state['data_name']
        raw_df = st.session_state['raw_data']
        df = st.session_state['data']
    else:
        data_name = ''
        raw_df = pd.DataFrame()
        df = pd.DataFrame()
    
    return raw_df, df, data_name
    
    
def get_uploaded_data(section):
    cntr = section.container(border=True)
    
    help_msg = '''
                 You can export (at csv format) the result of your iNaturalist search 
                 ([link](https://www.inaturalist.org/observations/export)).  
                 Keep the default columns selection, or at least the following columns:
                * url
                * image_url
                * time_zone
                * quality_grade
                * latitude
                * longitude
                * public_positional_accuracy
                * scientific_name
                * common_name
                * iconic_taxon_name
              '''
    uploaded_file = cntr.file_uploader("Upload your iNaturalist data in CSV format", 
                                       type=['csv'],
                                       help=help_msg)
    if uploaded_file is not None:
        raw_df = upload_file_to_df(uploaded_file)
        file_name = uploaded_file.name
        if 'data_name' not in st.session_state \
        or file_name != st.session_state['data_name']:
            df = clean_df(raw_df)
            init_all_session_state_var(raw_df, df, file_name)


def df_col_to_date(df):
    """
    Convert all date columns to pandas datetime without timezone awareness
    (tz-naive), to avoid mix errors between tz-aware and tz-naive values.
    """
    date_cols = [
        'observed_on', 'observed_on_string', 'time_observed_at',
        'created_at', 'updated_at'
    ]
    
    for col in date_cols:
        if col in df.columns:
            try:
                # Convert to datetime with errors='coerce'
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
                # Then remove timezone info to make it tz-naive
                df[col] = df[col].dt.tz_localize(None)
            except Exception as e:
                st.warning(f"⚠️ Failed to parse dates in column '{col}': {e}")
                df[col] = pd.NaT
                
    return df



def extract_coordinates(df_raw):
    """
    Creates the latitude and longitude columns from geojson.coordinates or location.
    Priority given to geojson. If no coordinates are available, returns None.
    """
    def get_lat(row):
        geo = row.get('geojson')
        loc = row.get('location')
        if isinstance(geo, dict) and 'coordinates' in geo:
            # geojson.coordinates = [lon, lat]
            return geo['coordinates'][1]
        elif isinstance(loc, (list, tuple)) and len(loc) == 2:
            # location = [lat, lon]
            return loc[0]
        return None

    def get_lon(row):
        geo = row.get('geojson')
        loc = row.get('location')
        if isinstance(geo, dict) and 'coordinates' in geo:
            return geo['coordinates'][0]
        elif isinstance(loc, (list, tuple)) and len(loc) == 2:
            return loc[1]
        return None

    df_raw['latitude'] = df_raw.apply(get_lat, axis=1)
    df_raw['longitude'] = df_raw.apply(get_lon, axis=1)

    return df_raw




def format_observations_for_export(df_raw):
    """
    Converts the raw DataFrame from the iNaturalist API into a format similar to the site's CSV export.
    Works with df_raw = pd.json_normalize(response['results'])
    """
    df_raw = extract_coordinates(df_raw)
    df = pd.DataFrame()

    # Columns from csv downloaded
    simple_cols = [
        'id', 'uuid', 'observed_on_string', 'observed_on', 'time_zone',
        'created_at', 'updated_at', 'quality_grade', 'license_code', 
        'description', 'num_identification_agreements', 'num_identification_disagreements',
        'captive', 'oauth_application_id', 'place_guess', 'latitude', 'longitude',
        'positional_accuracy', 'public_positional_accuracy', 'geoprivacy', 'taxon_geoprivacy',
        'coordinates_obscured', 'time_observed_at'
    ]
    for col in simple_cols:
        df[col] = df_raw[col] if col in df_raw.columns else None

    # Columns from taxon (flat by json_normalize)
    df['scientific_name'] = df_raw.get('taxon.name')
    df['common_name'] = df_raw.get('taxon.preferred_common_name')
    df['iconic_taxon_name'] = df_raw.get('taxon.iconic_taxon_name')
    df['taxon_id'] = df_raw.get('taxon.id')

    # Media columns
    df['image_url'] = df_raw['photos'].apply(
        lambda photos: photos[0]['url'] if isinstance(photos, list) and len(photos) > 0 and isinstance(photos[0], dict) else None
    ) if 'photos' in df_raw.columns else None

    df['sound_url'] = df_raw['sounds'].apply(
        lambda sounds: sounds[0]['file_url'] if isinstance(sounds, list) and len(sounds) > 0 and isinstance(sounds[0], dict) else None
    ) if 'sounds' in df_raw.columns else None

    # 🩹 FIX : handle 'tags' variations safely
    def safe_tags_extraction(tags):
        if isinstance(tags, list):
            if all(isinstance(t, dict) and 'name' in t for t in tags):
                return ','.join(t['name'] for t in tags)
            elif all(isinstance(t, str) for t in tags):
                return ','.join(tags)
        elif isinstance(tags, str):
            return tags
        return ''

    df['tag_list'] = df_raw['tags'].apply(safe_tags_extraction) if 'tags' in df_raw.columns else ''

    # User columns
    df['user.login'] = df_raw.get('user.login')

    # Link to the observation
    if 'uri' in df_raw.columns:
        df['url'] = df_raw['uri']
    else:
        df['url'] = df_raw['id'].apply(
            lambda x: f"https://www.inaturalist.org/observations/{x}" if pd.notna(x) else None
        )

    # Change df for a Date df
    df = df_col_to_date(df)
    if 'observed_on' in df.columns:
        df.set_index('observed_on', inplace=True, drop=True)

    return df



def on_form_submit():
    species_name = st.session_state["species_name"]
    place_id = st.session_state["place_id"]
    order_param = st.session_state["order_param"]
    max_results = st.session_state["max_results"]
    
    geo_args = {}
    if place_id:
        geo_args["place_id"] = place_id
    
    with st.spinner("Fetching observations..."):
        all_results = []
        page = 1
        per_page = 200
        total_fetched = 0
        progress = st.progress(0, text="Fetching data from iNaturalist...")

        start_time = time.time()
        
        while total_fetched < max_results:
            # --- API call ---
            response = get_observations(
                taxon_name=species_name,
                order_by="observed_on",
                order=order_param,
                **geo_args,
                per_page=per_page,
                page=page
            )
            results = response.get("results", [])
            if not results:
                progress.progress(
                    1.0,
                    text=f"Downloaded {total_fetched:,} / {max_results:,} observations "
                         f"(remaining time: 0m 0s) → no additional observations availables"
                )
                break

            # --- Store results ---
            all_results.extend(results)
            total_fetched += len(results)
            page += 1

            # --- Update progress bar ---
            elapsed = time.time() - start_time
            est_total = (elapsed / total_fetched) * max_results if total_fetched else 0
            eta = est_total - elapsed
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            
            progress.progress(
                min(total_fetched / max_results, 1.0),
                text=f"Downloaded {total_fetched:,} / {max_results:,} observations "
                     f"(remaining time: {eta_min}m {eta_sec}s)"
            )

            # --- Gentle delay (to avoid rate-limiting) ---
            time.sleep(random.uniform(0.3, 0.7))  # between 300 and 700ms per call

        if not all_results:
            st.warning('No observations found.')
        else:
            st.success(f'✅ Retrieved {len(all_results):,} observations successfully.')
            
            raw_df = format_observations_for_export(pd.json_normalize(all_results))
            df = clean_df(raw_df)
            init_all_session_state_var(raw_df, df, species_name)



def get_data_from_api(section):
    inat_form = section.form("inat_form")
    
    # Species field
    help_msg = '''You can use common or scientific names.'''
    species_name = inat_form.text_input(
        "Species:", 
        placeholder="ex: Filipendula ulmaria",
        help=help_msg
        )
    st.session_state["species_name"] = species_name

    # Place field
    help_msg = '''You can type any place name.
                  The best match from iNaturalist will be selected.  
                  Leave it blank to search worldwide.'''
    
    place_query = inat_form.text_input(
        "Enter a place or country name:",
        key="place_query",
        help=help_msg
    )

    # Place ID, default = None
    place_id = None

    # Recherche du meilleur match sur iNaturalist (mais seulement à la soumission)
    if place_query and len(place_query.strip()) >= 2:
        try:
            response = get_places_autocomplete(q=place_query.strip())
            results = response.get("results", [])
            
            if results:
                best_match = results[0]
                place_id = best_match.get("id")
                st.session_state["place_id"] = place_id
                inat_form.markdown(f"Selected country: **{best_match['name']}**")
            else:
                inat_form.warning("No matching place found on iNaturalist.")
                st.session_state["place_id"] = None
        except Exception as e:
            inat_form.error(f"Error from iNaturalist API : {e}")
            st.session_state["place_id"] = None
    else:
        st.session_state["place_id"] = None
        
    # Order selection
    order_choice = inat_form.radio(
        "Order of observations:",
        options=["Newest first", "Oldest first"],
        horizontal=True,
        help="Choose how to sort the observations by observation date."
    )
    
    order_param = "desc" if order_choice == "Newest first" else "asc"
    st.session_state["order_param"] = order_param

    # Limit number of results
    max_results = inat_form.number_input(
        "Maximum number of observations to download",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
        help="⚠️ Large downloads (>50,000) can take several minutes and use a lot of memory.",
    )
    st.session_state["max_results"] = max_results

    submit = inat_form.form_submit_button("🔍 Search")
    if submit:
        on_form_submit()
        
        
        
        
        