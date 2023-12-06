# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import os

import streamlit as st
import streamlit_antd_components as sac
from datumaro_gui.utils.dataset.data_loader import DataRepo, SingleDatasetHelper
from datumaro_gui.utils.dataset.state import file_selector, import_dataset, reset_state
from datumaro_gui.utils.drawing.css import custom_css
from datumaro_gui.utils.readme import github_pypi_desc
from streamlit import session_state as state

from . import tabs


def main():
    st.write(github_pypi_desc)
    st.markdown(custom_css, unsafe_allow_html=True)

    keys = [
        "uploaded_file",
        "data_helper",
        "subset",
        "stats_image",
        "stats_anns",
        "image_size_info",
        "cls_summary",
        "cls_anomaly_info",
        "det_summary",
        "det_anomaly_info",
        "seg_summary",
        "seg_anomaly_info",
        "defined_label",
        "undefined_label",
        "defined_attr",
        "undefined_attr",
    ]

    filename = file_selector()
    reset_state(keys, state)

    if filename is not None:
        reset_state(keys, state)
        if filename.endswith(".zip"):
            data_repo = DataRepo()
            filename = data_repo.unzip_dataset(filename)

        data_helper = SingleDatasetHelper(filename)
        uploaded_file = os.path.basename(filename)
        state["uploaded_file"] = uploaded_file
        state["data_helper"] = data_helper

        import_dataset(data_helper)

    elif state["data_helper"] is not None:
        state["data_helper"] = None

    st.title("")

    if state["data_helper"] is not None:
        selected_tab = sac.tabs(
            [
                sac.TabsItem(label="GENERAL", icon="incognito"),
                sac.TabsItem(label="ANALYZE", icon="clipboard2-data-fill"),
                sac.TabsItem(label="VISUALIZE", icon="image"),
                sac.TabsItem(label="EXPLORE", icon="tags"),
                sac.TabsItem(label="TRANSFORM", icon="tools"),
                sac.TabsItem(label="EXPORT", icon="cloud-arrow-down"),
            ],
            format_func="title",
            align="center",
        )

        tab_funcs = {
            "GENERAL": tabs.call_general,
            "ANALYZE": tabs.call_analyze,
            "VISUALIZE": tabs.call_visualize,
            "EXPLORE": tabs.call_explore,
            "TRANSFORM": tabs.call_transform,
            "EXPORT": tabs.call_export,
        }
        tab_funcs.get(selected_tab, tabs.call_general)()


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
