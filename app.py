import streamlit as st

st.logo('assets/logo.png', size='large', link='https://github.com/Jeremy-Duval-PhD')

st.set_page_config(
    page_title="Hello",
    page_icon='assets/logo.png',
)

home_page = st.Page("pages/0_Home.py")
upload_page = st.Page("pages/1_Data_loading.py")
visualization_page = st.Page("pages/2_Data_visualization.py")
pages = [home_page, upload_page, visualization_page]

pg = st.navigation(pages)
pg.run()

