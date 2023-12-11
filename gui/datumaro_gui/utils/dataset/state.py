# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import os
from pathlib import Path

import streamlit as st

single_state_keys = [
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

multiple_state_keys = [
    "uploaded_file_1",
    "data_helper_1",
    "subset_1",
    "uploaded_file_2",
    "data_helper_2",
    "subset_2",
    "mapping",
    "matched",
    "high_level_table",
    "mid_level_table",
    "low_level_table",
]


def get_data_folder_path():
    cwd = os.getcwd()
    if os.path.basename(cwd) == "gui":
        cwd = Path(cwd).parents[0]
    data_folder_path = os.path.join(cwd, "data")
    return data_folder_path


def reset_subset(state):
    subset_list = ["subset", "subset_1", "subset_2"]
    for subset in subset_list:
        if subset in state.keys() and state[subset] is None:
            state[subset] = 0


def reset_state(keys, state):
    for k in keys:
        if k not in state:
            state[k] = None
    reset_subset(state)


def file_selector(folder_path: str = None):
    if not folder_path:
        folder_path = get_data_folder_path()

    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox(
        "Select a file",
        filenames,
        index=None,
    )
    if selected_filename is not None:
        return os.path.join(folder_path, selected_filename)
    return None


def multiple_file_selector(folder_path: str = None):
    if not folder_path:
        folder_path = get_data_folder_path()

    filenames = os.listdir(folder_path)
    selected_filenames = st.multiselect("Select files", filenames)
    if selected_filenames:
        return [
            os.path.join(folder_path, selected_filename) for selected_filename in selected_filenames
        ]
    return None


def import_dataset(data_helper, data_num: str = "Dataset"):
    try:
        formats = data_helper.detect_format()
    except Exception:
        formats = ["-", "datumaro", "voc", "coco"]  # temp
    selected_format = st.selectbox(f"Select a format to import {data_num}:", formats)
    if selected_format != "-" and selected_format != data_helper.format():
        data_helper.import_dataset(selected_format)
