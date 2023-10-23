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

    keys = [
        "uploaded_zip_1",
        "data_helper_1",
        "subset_1",
        "uploaded_zip_2",
        "data_helper_2",
        "subset_2",
    ]
    for k in keys:
        if k not in state:
            state[k] = None

    if state["subset_1"] is None:
        state["subset_1"] = 0
    if state["subset_2"] is None:
        state["subset_2"] = 0

    data_repo = DataRepo()

    with st.expander("Import a dataset"):
        uploaded_zips = st.file_uploader(
            "Upload a zip file containing dataset", type=["zip"], accept_multiple_files=True
        )

        if uploaded_zips:
            if len(uploaded_zips) > 2:
                st.error("You could not upload more than 2 datasets in once", icon="🚨")

            uploaded_zip_1 = uploaded_zips[0]
            uploaded_zip_2 = uploaded_zips[1]
            if (
                uploaded_zip_1 != state["uploaded_zip_1"]
                and uploaded_zip_2 != state["uploaded_zip_2"]
            ):
                # Extract the contents of the uploaded zip file to the temporary directory
                dataset_1_dir = data_repo.unzip_dataset(uploaded_zip_1)
                dataset_2_dir = data_repo.unzip_dataset(uploaded_zip_2)
                print("dataset_1_dir: ", dataset_1_dir)
                print("dataset_2_dir: ", dataset_2_dir)

                state["uploaded_zip_1"] = uploaded_zip_1
                state["uploaded_zip_2"] = uploaded_zip_2

                data_helper_1 = DatasetHelper(dataset_1_dir)
                state["data_helper_1"] = data_helper_1
                data_helper_2 = DatasetHelper(dataset_2_dir)
                state["data_helper_2"] = data_helper_2

            data_helper_1: DatasetHelper = state["data_helper_1"]
            # Display the list of image files in the UI
            selected_format_1 = st.selectbox(
                "Select a format to import Dataset 1:", data_helper_1.detect_format()
            )

            if selected_format_1 is not None:
                if selected_format_1 != data_helper_1.format():
                    data_helper_1.import_dataset(selected_format_1)

            data_helper_2: DatasetHelper = state["data_helper_2"]
            # Display the list of image files in the UI
            selected_format_2 = st.selectbox(
                "Select a format to import Dataset 2:", data_helper_2.detect_format()
            )
            if selected_format_2 is not None:
                if selected_format_2 != data_helper_2.format():
                    data_helper_2.import_dataset(selected_format_2)

        elif state["data_helper_1"] is not None and state["data_helper_2"] is not None:
            state["data_helper_1"] = None
            state["data_helper_2"] = None

    st.title("")

    if state["data_helper_1"] is not None and state["data_helper_2"] is not None:
        selected_tab = sac.tabs(
            [
                sac.TabsItem(label="GENERAL", icon="incognito"),
                sac.TabsItem(label="VALIDATE", icon="incognito"),
                sac.TabsItem(label="COMPARE", icon="twisted_rightwards_arrows"),
                sac.TabsItem(label="VISUALIZE", icon="image"),
                sac.TabsItem(label="EXPLORE", icon="tags", disabled=True),
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
            "COMAPRE": tabs.call_compare,
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
