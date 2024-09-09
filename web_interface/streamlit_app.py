import streamlit as st

#region: -- PAGE CONFIG --
st.set_page_config(
    page_title="Breast Cancer Image Prediction",
    page_icon="⚕️"
)
#endregion: -- PAGE CONFIG --

#region: -- PAGE SETUP --
classification_page = st.Page(
    page="views/classification.py",
    title="Classification",
    icon=":material/settings:"
)

gallery_page = st.Page(
    page="views/gallery.py",
    title="Gallery",
    icon=":material/gallery_thumbnail:"
)

page = st.navigation(pages=[classification_page, gallery_page])
page.run()
# endregion: -- PAGE SETUP --


