# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os

import streamlit as st
import streamlit_antd_components as sac
from datumaro_gui.utils.dataset.data_loader import SingleDatasetHelper
from datumaro_gui.utils.dataset.state import (
    file_selector,
    format_selector,
    get_download_folder_path,
    reset_state,
    save_dataset,
    single_state_keys,
)
from datumaro_gui.utils.drawing.css import box_style, custom_css
from datumaro_gui.utils.page import check_image_backend
from datumaro_gui.utils.readme import github_pypi_desc
from streamlit import session_state as state

from . import tabs


def main():
    check_image_backend(st.session_state.get("IMAGE_BACKEND"))
    log.info("__single__.main is called")
    st.write(github_pypi_desc)
    st.markdown(custom_css, unsafe_allow_html=True)

    if "single_filename" not in state:
        # Do not add `single_filename` to `single_state_keys`
        # This key is for avoiding re-importing
        state["single_filename"] = None
        reset_state(single_state_keys, state)

    input_path = st.text_input(
        "Select a path to import data:",
        value=get_download_folder_path(),
        key="single_input_path",
    )
    filename = file_selector(input_path)

    log.info(f'state["single_filename"] = {state["single_filename"]}')
    if filename != state["single_filename"]:
        # change dataset
        state["single_filename"] = filename
        log.info(f'state["single_filename"] changed to {state["single_filename"]}')
        reset_state(single_state_keys, state)

        if filename is not None:
            data_helper = SingleDatasetHelper(filename)
            state["data_helper"] = data_helper
            uploaded_file = os.path.basename(filename)
            state["uploaded_file"] = uploaded_file
        elif state["data_helper"] is not None:
            state["data_helper"] = None

    data_helper = state["data_helper"]
    if data_helper is not None:
        format = data_helper.format()
        if format == "":
            selected_format = format_selector(data_helper=data_helper, data_num=uploaded_file)
            if selected_format != "-":
                data_helper.import_dataset(selected_format)

        dataset = data_helper.dataset()

        st.title("")
        if dataset is not None:
            st.markdown("<style>{}</style>".format(box_style), unsafe_allow_html=True)
            current_dataset_name = state["uploaded_file"]
            current_data_helper = state["data_helper"]
            cont = st.container()
            _, c2, c3, _ = cont.columns([1, 2, 1, 1])
            with c2:
                c2.markdown(
                    f"<div class='smallbox'>Current Dataset <span class='bold'>{current_dataset_name}</span></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                c3.button(
                    "Save",
                    on_click=save_dataset,
                    args=(current_data_helper, current_dataset_name),
                    key="bt_save_ds",
                    use_container_width=True,
                )

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
