# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import time

import streamlit as st
import streamlit_antd_components as sac
from datumaro_gui.utils.dataset.data_loader import DataRepo, SingleDatasetHelper
from datumaro_gui.utils.dataset.state import reset_state
from datumaro_gui.utils.drawing.css import custom_css
from datumaro_gui.utils.readme import github_pypi_desc
from streamlit import session_state as state

from . import tabs

# from streamlit_file_browser import st_file_browser


def main():
    st.write(github_pypi_desc)
    st.markdown(custom_css, unsafe_allow_html=True)

    keys = [
        "uploaded_zip",
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

    data_repo = DataRepo()

    # event = st_file_browser("C:/Users/sooahlee/workspace/datumaro/data", show_new_folder=True, glob_patterns='*')
    # st.write(event)

    # if event is not None:
    #     uploaded_zip = event['target']['path']
    #     start_time = time.time()
    #     dataset_dir = data_repo.unzip_dataset(uploaded_zip)
    #     print('unzip time : ', time.time())
    #     state["uploaded_zip"] = uploaded_zip
    #     data_helper = SingleDatasetHelper(dataset_dir)
    #     state["data_helper"] = data_helper
    #     data_helper: SingleDatasetHelper = state["data_helper"]
    #     # Display the list of image files in the UI
    #     try:
    #         formats = data_helper.detect_format()
    #     except Exception:
    #         formats = ["-", "datumaro", "voc", "coco"]  # temp
    #     selected_format = st.selectbox("Select a format to import:", formats)
    #     if selected_format != "-" and selected_format != data_helper.format():
    #         data_helper.import_dataset(selected_format)

    # elif state["data_helper"] is not None:
    #     state["data_helper"] = None

    with st.expander("Import a dataset", expanded=True):
        start_time = time.time()
        uploaded_zip = st.file_uploader("Upload a zip file containing dataset", type=["zip"])
        if uploaded_zip is None:
            reset_state(keys, state)

        if uploaded_zip is not None:
            if uploaded_zip != state["uploaded_zip"]:
                # Extract the contents of the uploaded zip file to the temporary directory
                start_time = time.time()
                dataset_dir = data_repo.unzip_dataset(uploaded_zip)
                print("unzip time : ", time.time() - start_time)
                state["uploaded_zip"] = uploaded_zip
                data_helper = SingleDatasetHelper(dataset_dir)
                state["data_helper"] = data_helper

            data_helper: SingleDatasetHelper = state["data_helper"]
            # Display the list of image files in the UI
            try:
                start_time = time.time()
                formats = data_helper.detect_format()
                print("detect format time : ", time.time() - start_time)
            except Exception:
                formats = ["-", "datumaro", "voc", "coco"]  # temp
            selected_format = st.selectbox("Select a format to import:", formats)
            if selected_format != "-" and selected_format != data_helper.format():
                data_helper.import_dataset(selected_format)

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
        start_time = time.time()
        tab_funcs.get(selected_tab, tabs.call_general)()
        print("call tab time : ", time.time() - start_time)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
