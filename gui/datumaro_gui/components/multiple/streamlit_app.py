# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import streamlit as st
import streamlit_antd_components as sac
from datumaro_gui.utils.dataset.data_loader import DataRepo, MultipleDatasetHelper
from datumaro_gui.utils.drawing.css import custom_css
from datumaro_gui.utils.readme import github_pypi_desc
from streamlit import session_state as state

from . import tabs


def main():
    st.write(github_pypi_desc)
    st.markdown(custom_css, unsafe_allow_html=True)

    keys = [
        "uploaded_zip_1",
        "data_helper_1",
        "subset_1",
        "uploaded_zip_2",
        "data_helper_2",
        "subset_2",
        "mapping",
        "matched",
        "high_level_table",
        "mid_level_table",
        "low_level_table",
    ]
    for k in keys:
        if k not in state:
            state[k] = None

    if state["subset_1"] is None:
        state["subset_1"] = 0
    if state["subset_2"] is None:
        state["subset_2"] = 0

    data_repo = DataRepo()

    with st.expander("Import datasets", expanded=True):
        uploaded_zips = st.file_uploader(
            "Upload two zip files containing dataset", type=["zip"], accept_multiple_files=True
        )

        if uploaded_zips and len(uploaded_zips) > 1:
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

                state["uploaded_zip_1"] = uploaded_zip_1
                state["uploaded_zip_2"] = uploaded_zip_2

                data_helper_1 = MultipleDatasetHelper(dataset_1_dir)
                state["data_helper_1"] = data_helper_1
                data_helper_2 = MultipleDatasetHelper(dataset_2_dir)
                state["data_helper_2"] = data_helper_2

            data_helper_1: MultipleDatasetHelper = state["data_helper_1"]
            # Display the list of image files in the UI
            selected_format_1 = st.selectbox(
                "Select a format to import Dataset 1:", data_helper_1.detect_format()
            )

            if selected_format_1 is not None:
                if selected_format_1 != data_helper_1.format():
                    data_helper_1.import_dataset(selected_format_1)

            data_helper_2: MultipleDatasetHelper = state["data_helper_2"]
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
                sac.TabsItem(label="VALIDATE", icon="graph-up", disabled=True),
                sac.TabsItem(label="COMPARE", icon="arrow-left-right"),
                sac.TabsItem(label="VISUALIZE", icon="image", disabled=True),
                sac.TabsItem(label="TRANSFORM", icon="tools"),
                sac.TabsItem(label="MERGE", icon="union"),
                sac.TabsItem(label="EXPORT", icon="cloud-arrow-down"),
            ],
            format_func="title",
            align="center",
        )

        tab_funcs = {
            "GENERAL": tabs.call_general,
            "VALIDATE": tabs.call_validate,
            "COMPARE": tabs.call_compare,
            "VISUALIZE": tabs.call_visualize,
            "TRANSFORM": tabs.call_transform,
            "MERGE": tabs.call_merge,
            "EXPORT": tabs.call_export,
        }
        tab_funcs.get(selected_tab, tabs.call_general)()


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
