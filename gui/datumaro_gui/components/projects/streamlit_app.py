# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import streamlit as st
import streamlit_antd_components as sac
from streamlit import session_state as state

from . import tabs
from .data_loader import DataRepo, DatasetHelper


def main():
    st.write(
        """
        :factory: Dataset management &nbsp; [![GitHub][github_badge]][github_link] [![PyPI][pypi_badge]][pypi_link]
        =====================

        Import a dataset and manipulate it!

        [github_badge]: https://badgen.net/badge/icon/GitHub?icon=github&color=black&label
        [github_link]: https://github.com/openvinotoolkit/datumaro

        [pypi_badge]: https://badgen.net/pypi/v/streamlit-elements?icon=pypi&color=black&label
        [pypi_link]: https://pypi.org/project/datumaro/
        """
    )

    # Define a custom CSS style
    custom_css = """
    <style>
        .css-q8sbsg p {
            font-size: 16px;
        }
        .container-outline p {
            border: 2px solid #000; /* Adjust the border properties as needed */
            padding: 10px; /* Adjust the padding as needed */
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    keys = ["uploaded_zip", "data_helper", "subset"]
    for k in keys:
        if k not in state:
            state[k] = None

    if state["subset"] is None:
        state["subset"] = 0

    data_repo = DataRepo()

    with st.expander("Import a dataset"):
        uploaded_zip = st.file_uploader("Upload a zip file containing dataset", type=["zip"])

        if uploaded_zip is not None:
            if uploaded_zip != state["uploaded_zip"]:
                # Extract the contents of the uploaded zip file to the temporary directory
                dataset_dir = data_repo.unzip_dataset(uploaded_zip)
                print("dataset_dir: ", dataset_dir)

                state["uploaded_zip"] = uploaded_zip

                data_helper = DatasetHelper(dataset_dir)
                state["data_helper"] = data_helper

            data_helper: DatasetHelper = state["data_helper"]
            # Display the list of image files in the UI
            selected_format = st.selectbox(
                "Select a format to import:", data_helper.detect_format()
            )

            if selected_format is not None:
                if selected_format != data_helper.format():
                    data_helper.import_dataset(selected_format)
        elif state["data_helper"] is not None:
            state["data_helper"] = None

    st.title("")

    if state["data_helper"] is not None:
        selected_tab = sac.tabs(
            [
                sac.TabsItem(label="GENERAL", icon="incognito"),
                sac.TabsItem(label="VALIDATE", icon="incognito"),
                sac.TabsItem(label="VISUALIZE", icon="image"),
                sac.TabsItem(label="EXPLORE", icon="tags"),
                sac.TabsItem(label="ANALYZE", icon="clipboard2-data-fill", disabled=True),
                sac.TabsItem(label="TRANSFORM", icon="tools"),
                sac.TabsItem(label="EXPORT", icon="cloud-arrow-down"),
            ],
            format_func="title",
            align="center",
        )

        tab_funcs = {
            "GENERAL": tabs.call_general,
            "VALIDATE": tabs.call_validate,
            "VISUALIZE": tabs.call_visualize,
            "EXPLORE": tabs.call_explore,
            "ANALYZE": tabs.call_analyze,
            "TRANSFORM": tabs.call_transform,
            "EXPORT": tabs.call_export,
        }
        tab_funcs.get(selected_tab, tabs.call_general)()


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
