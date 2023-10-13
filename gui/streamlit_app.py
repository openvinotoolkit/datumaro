import streamlit as st
from streamlit_elements import elements

from datumaro_gui import apps, components
from datumaro_gui.utils.page import page_group

PROJECTS= []

def main():
    page = page_group("p")

    with st.sidebar:
        with elements("new_element"):
            st.title(":wedding: Datumaro")
            # st.write("by Intel:registered: Corporation")

        with st.expander(":house: Home", True):
            page.item(":open_book: Introduction", apps.home, default=True)

        with st.expander(":rocket: Projects", True):
            page.item(":microscope: Single dataset", components.projects)
            page.item(":telescope: Multiple datasets", components.elements)

        with st.expander(":male-astronaut: About us", True):
            page.item(":male-mechanic: Developers", apps.aboutus)

    page.show()

if __name__ == "__main__":
    st.set_page_config(page_title="Welcome To Datumaro", page_icon="⭐", layout="wide")
    main()
