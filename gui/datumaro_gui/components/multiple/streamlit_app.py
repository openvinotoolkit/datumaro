# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os

import streamlit as st
import streamlit_antd_components as sac
from datumaro_gui.utils.dataset.data_loader import DataRepo, MultipleDatasetHelper
from datumaro_gui.utils.dataset.state import (
    import_dataset,
    multiple_file_selector,
    multiple_state_keys,
    reset_state,
)
from datumaro_gui.utils.drawing.css import custom_css
from datumaro_gui.utils.readme import github_pypi_desc
from streamlit import session_state as state

from . import tabs


def main():
    st.write(github_pypi_desc)
    st.markdown(custom_css, unsafe_allow_html=True)

    filenames = multiple_file_selector()
    reset_state(multiple_state_keys, state)

    if filenames is not None and len(filenames) > 1:
        if len(filenames) > 2:
            st.error("You could not upload more than 2 datasets in once", icon="🚨")
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
        state["uploaded_file_1"] = os.path.basename(dataset_1_dir)

        data_helper_2 = MultipleDatasetHelper(dataset_2_dir)
        state["data_helper_2"] = data_helper_2
        state["uploaded_file_2"] = os.path.basename(dataset_2_dir)

        import_dataset(data_helper_1, "Dataset 1")
        import_dataset(data_helper_2, "Dataset 2")

    elif state["data_helper_1"] is not None and state["data_helper_2"] is not None:
        state["data_helper_1"] = None
        state["data_helper_2"] = None

    st.title("")

    if state["data_helper_1"] is not None and state["data_helper_2"] is not None:
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
