import streamlit as st
import pandas as pd
from pyinaturalist import get_observations, get_taxa
from pyinaturalist.node_api import get_places_autocomplete

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
    st.write(str(list(df.columns)))
    col_to_keep = get_clean_columns_order()
    df = df[col_to_keep]
    df = df.dropna(subset=['latitude', 'longitude'])
    
    df['year'] = df.index.year
    
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
    
    
def get_uploaded_data():
    uploaded_file = st.file_uploader("Upload your iNaturalist data in CSV format", type=['csv'])
    if uploaded_file is not None:
        raw_df = upload_file_to_df(uploaded_file)
        file_name = uploaded_file.name
        if 'data_name' not in st.session_state \
        or file_name != st.session_state['data_name']:
            df = clean_df(raw_df)
            init_all_session_state_var(raw_df, df, file_name)


def df_col_to_date(df):
    date_cols = [
        'observed_on', 'observed_on_string', 'time_observed_at',
        'created_at', 'updated_at'
    ]
    
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    return df


def extract_coordinates(df_raw):
    """
    Crée les colonnes latitude et longitude à partir de geojson.coordinates ou location.
    Priorité à geojson. Si aucune coordonnée disponible, retourne None.
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

    # Colonnes simples
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

    # Colonnes dérivées de taxon (aplaties par json_normalize)
    df['scientific_name'] = df_raw['taxon.name'] if 'taxon.name' in df_raw.columns else None
    df['common_name'] = df_raw['taxon.preferred_common_name'] if 'taxon.preferred_common_name' in df_raw.columns else None
    df['iconic_taxon_name'] = df_raw['taxon.iconic_taxon_name'] if 'taxon.iconic_taxon_name' in df_raw.columns else None
    df['taxon_id'] = df_raw['taxon.id'] if 'taxon.id' in df_raw.columns else None

    # Colonnes média
    df['image_url'] = df_raw['photos'].apply(
        lambda photos: photos[0]['url'] if isinstance(photos, list) and len(photos) > 0 else None
    ) if 'photos' in df_raw.columns else None

    df['sound_url'] = df_raw['sounds'].apply(
        lambda sounds: sounds[0]['file_url'] if isinstance(sounds, list) and len(sounds) > 0 else None
    ) if 'sounds' in df_raw.columns else None

    # Colonnes texte / tags
    df['tag_list'] = df_raw['tags'].apply(
        lambda tags: ','.join([t['name'] for t in tags]) if isinstance(tags, list) else ''
    ) if 'tags' in df_raw.columns else ''

    # Colonnes user
    df['user.login'] = df_raw['user.login'] if 'user.login' in df_raw.columns else None

    # Colonnes supplémentaires manquantes dans API brute
    df['url'] = None

    df = df_col_to_date(df)
    df.set_index('observed_on', inplace=True)

    return df


@st.cache_data(ttl=300)
def get_country_suggestions_cached(query: str):
    """Retourne la liste d'objets place (dict) correspondant au query (type Country)."""
    if not query or len(query) < 2:
        return []
    resp = get_places_autocomplete(q=query)
    return [p for p in resp.get("results", []) if p.get("type") == "Country"]

@st.cache_data(ttl=3600)
def get_place_id_by_name_cached(name: str):
    """Retourne le place_id pour un nom de lieu exact (type Country)."""
    if not name:
        return None
    resp = get_places_autocomplete(q=name)
    for p in resp.get("results", []):
        if p.get("type") == "Country" and p.get("name") == name:
            return p.get("id")
    return None


def on_form_submit():
    species_name = st.session_state["species_name"]
    place_id = st.session_state.get("place_id")
    
    geo_args = {}
    if place_id:
        geo_args["place_id"] = place_id
    
    with st.spinner("Please wait..."):
        # call to iNaturalist API
        st.write("geo_args:", geo_args)
        response = get_observations(taxon_name=species_name, **geo_args, per_page=100)
        results = response.get("results", [])
        st.write("results:", bool(results))
        if not results:
            st.warning('No observations found.')
        else:
            st.success('The request succeeded.')
            
            #raw_df = pd.DataFrame(results)
            raw_df = format_observations_for_export(\
                                        pd.json_normalize(response['results']))
            
            df = clean_df(raw_df)
            st.write(df)
            init_all_session_state_var(raw_df, df, species_name)


def get_data_from_api():
    with st.form("inat_form", clear_on_submit=False):
        st.subheader("From iNaturalist website")
        
        species_name = st.text_input("Species:", "Filipendula ulmaria")
        st.session_state["species_name"] = species_name
    
        st.markdown("#### Geographical filter")
        
        # Champ texte pour taper le nom du pays
        st.text_input("Tapez le nom du pays (≥2 lettres) :", key="country_query",
                      help="Ex : France, Germany, United States")

        # Récupérer suggestions (cache) à partir du texte saisi
        query = st.session_state.get("country_query", "")
        suggestions = get_country_suggestions_cached(query)

        # Construire la liste d'affichage (nom) pour le selectbox
        suggestion_names = ["-- Aucun --"] + [p["name"] for p in suggestions]

        # selectbox affiche les suggestions
        selected_name = st.selectbox("Suggestions :", suggestion_names, key="selected_country")

        # Si l'utilisateur sélectionne un nom (pas '-- Aucun --'), récupérer place_id (cache)
        if selected_name and selected_name != "-- Aucun --":
            place_id = get_place_id_by_name_cached(selected_name)
            st.session_state["place_id"] = place_id
            st.markdown(f"Place selected: **{selected_name}** (place_id: {place_id})")
        else:
            st.session_state["place_id"] = None
    
        st.form_submit_button("🔍 Search", on_click=on_form_submit)
        
        
        
        
        