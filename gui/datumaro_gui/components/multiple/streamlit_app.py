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
    save_dataset,
)
from datumaro_gui.utils.drawing.css import box_style, custom_css
from datumaro_gui.utils.readme import github_pypi_desc
from streamlit import session_state as state

from . import tabs


def main():
    print("__multiple__.main is called")
    st.write(github_pypi_desc)
    st.markdown(custom_css, unsafe_allow_html=True)

    if "multiple_filename" not in state:
        # Do not add `multiple_filename` to `multiple_filename`
        # This key is for avoiding re-importing
        state["multiple_filename"] = None
        reset_state(multiple_state_keys, state)

    input_path = st.text_input(
        "Select a path to import data:",
        value=get_download_folder_path(),
        key="multiple_input_path",
    )
    filename = multiple_file_selector(input_path)

    print(f'state["multiple_filename"] = {state["multiple_filename"]}')
    if filename != state["multiple_filename"] and filename is not None and len(filename) > 1:
        if len(filename) > 2:
            st.error("You could not upload more than 2 datasets in once", icon="🚨")

        # change dataset
        state["multiple_filename"] = filename
        print(f'state["multiple_filename"] changed to {state["multiple_filename"]}')
        reset_state(multiple_state_keys, state)

        filename_1, filename_2 = filename
        if filename_1 is not None:
            if filename_1.endswith(".zip"):
                data_repo = DataRepo()
                filename_1 = data_repo.unzip_dataset(filename_1)
                print(filename_1)

            data_helper = MultipleDatasetHelper(filename_1)
            state["data_helper_1"] = data_helper
            uploaded_file = os.path.basename(filename_1)
            state["uploaded_file_1"] = uploaded_file
        elif state["data_helper_1"] is not None:
            state["data_helper_1"] = None

        if filename_2 is not None:
            if filename_2.endswith(".zip"):
                data_repo = DataRepo()
                filename_2 = data_repo.unzip_dataset(filename_2)
                print(filename_2)

            data_helper = MultipleDatasetHelper(filename_2)
            state["data_helper_2"] = data_helper
            uploaded_file = os.path.basename(filename_2)
            state["uploaded_file_2"] = uploaded_file
        elif state["data_helper_2"] is not None:
            state["data_helper_2"] = None

    data_helper_1 = state["data_helper_1"]
    data_helper_2 = state["data_helper_2"]
    uploaded_file_1 = state["uploaded_file_1"]
    uploaded_file_2 = state["uploaded_file_2"]
    if data_helper_1 is not None and data_helper_2 is not None:
        format_1 = data_helper_1.format()
        format_2 = data_helper_2.format()
        if format_1 == "":
            selected_format_1 = format_selector(data_helper=data_helper_1, data_num=uploaded_file_1)
            if selected_format_1 != "-":
                data_helper_1.import_dataset(selected_format_1)
        if format_2 == "":
            selected_format_2 = format_selector(data_helper=data_helper_2, data_num=uploaded_file_2)
            if selected_format_2 != "-":
                data_helper_2.import_dataset(selected_format_2)
        dataset_1 = data_helper_1.dataset()
        dataset_2 = data_helper_2.dataset()

        st.title("")

        if dataset_1 is not None and dataset_2 is not None:
            st.markdown("<style>{}</style>".format(box_style), unsafe_allow_html=True)
            current_dataset_name_1 = state["uploaded_file_1"]
            current_dataset_name_2 = state["uploaded_file_2"]
            current_data_helper_1 = state["data_helper_1"]
            current_data_helper_2 = state["data_helper_2"]
            if current_data_helper_1._dm_dataset != dataset_1:
                st.warning("Dataset 1 is different!!")
            if current_data_helper_2._dm_dataset != dataset_2:
                st.warning("Dataset 2 is different!!")
            cont1 = st.container()
            _, c2, c3, _ = cont1.columns([1, 2, 1, 1])
            with c2:
                c2.markdown(
                    f"<div class='smallbox'>Current Dataset 1 <span class='bold'>{current_dataset_name_1}</span></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                c3.button(
                    "Save",
                    on_click=save_dataset,
                    args=(current_data_helper_1, current_dataset_name_1),
                    key="bt_save_ds_1",
                    use_container_width=True,
                )
            cont2 = st.container()
            _, c2, c3, _ = cont2.columns([1, 2, 1, 1])
            with c2:
                c2.markdown(
                    f"<div class='smallbox'>Current Dataset 2 <span class='bold'>{current_dataset_name_2}</span></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                c3.button(
                    "Save",
                    on_click=save_dataset,
                    args=(current_data_helper_2, current_dataset_name_2),
                    key="bt_save_ds_2",
                    use_container_width=True,
                )

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
