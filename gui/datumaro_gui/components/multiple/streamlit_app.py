# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os

import streamlit as st
import streamlit_antd_components as sac
from datumaro_gui.utils.dataset.data_loader import DataRepo, MultipleDatasetHelper
from datumaro_gui.utils.dataset.state import (
    format_selector,
    get_download_folder_path,
    multiple_file_selector,
    multiple_state_keys,
    reset_state,
)
from datumaro_gui.utils.drawing.css import custom_css
from datumaro_gui.utils.readme import github_pypi_desc
from streamlit import session_state as state

from . import tabs


def main():
    print("__multiple__.main is called")
    st.write(github_pypi_desc)
    st.markdown(custom_css, unsafe_allow_html=True)

    if "multiple_filename" not in state:
        # Do not add `single_filename` to `single_state_keys`
        # This key is for avoiding re-importing
        state["multiple_filename"] = None
        reset_state(multiple_state_keys, state)

    input_path = st.text_input(
        "Select a path to import data:",
        value=get_download_folder_path(),
        key="multiple_input_path",
    )
    filenames = multiple_file_selector(input_path)

    print(f'state["multiple_filename"] = {state["multiple_filename"]}')
    if filenames is not None and len(filenames) > 1:
        # change dataset
        if len(filenames) > 2:
            st.error("You could not upload more than 2 datasets in once", icon="🚨")
        state["multiple_filename"] = filenames
        print(f'state["multiple_filename"] changed to {state["multiple_filename"]}')
        reset_state(multiple_state_keys, state)
        dataset_1_dir, dataset_2_dir = filenames[0], filenames[1]

        if dataset_1_dir.endswith(".zip"):
            data_repo = DataRepo()
            dataset_1_dir = data_repo.unzip_dataset(dataset_1_dir)
        if dataset_2_dir.endswith(".zip"):
            data_repo = DataRepo()
            dataset_2_dir = data_repo.unzip_dataset(dataset_2_dir)

        data_helper_1 = MultipleDatasetHelper(dataset_1_dir)
        state["data_helper_1"] = data_helper_1
        uploaded_file = os.path.basename(dataset_1_dir)
        state["uploaded_file_1"] = uploaded_file

        data_helper_2 = MultipleDatasetHelper(dataset_2_dir)
        state["data_helper_2"] = data_helper_2
        uploaded_file = os.path.basename(dataset_2_dir)
        state["uploaded_file_2"] = uploaded_file

    elif state["data_helper_1"] is not None and state["data_helper_2"] is not None:
        state["data_helper_1"] = None
        state["data_helper_2"] = None

    data_helper_1 = state["data_helper_1"]
    data_helper_2 = state["data_helper_2"]
    if data_helper_1 is not None and data_helper_2 is not None:
        format = data_helper_1.format()
        if format == "":
            selected_format = format_selector(data_helper=data_helper_1)
            if selected_format != "-":
                data_helper_1.import_dataset(selected_format)

        format = data_helper_2.format()
        if format == "":
            selected_format = format_selector(data_helper=data_helper_2)
            if selected_format != "-":
                data_helper_2.import_dataset(selected_format)
        dataset1 = data_helper_1.dataset()
        dataset2 = data_helper_2.dataset()

        st.title("")
        if dataset1 is not None and dataset2 is not None:
            selected_tab = sac.tabs(
                [
                    sac.TabsItem(label="GENERAL", icon="incognito"),
                    sac.TabsItem(label="COMPARE", icon="arrow-left-right"),
                    sac.TabsItem(label="TRANSFORM", icon="tools"),
                    sac.TabsItem(label="MERGE", icon="union"),
                    sac.TabsItem(label="EXPORT", icon="cloud-arrow-down"),
                ],
                format_func="title",
                align="center",
            )

            tab_funcs = {
                "GENERAL": tabs.call_general,
                "COMPARE": tabs.call_compare,
                "TRANSFORM": tabs.call_transform,
                "MERGE": tabs.call_merge,
                "EXPORT": tabs.call_export,
            }
            tab_funcs.get(selected_tab, tabs.call_general)()


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
