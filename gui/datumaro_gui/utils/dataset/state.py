# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from datetime import datetime

import streamlit as st
from datumaro_gui.utils.dataset.data_loader import DatasetHelper

single_state_keys = [
    "uploaded_file",
    "data_helper",
    "subset",
    # "stats_image",
    # "stats_anns",
    # "image_size_info",
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
    "selected_transform",
    "correct-reports",
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
    "correct_reports_1",
    "correct_reports_2",
]


def get_download_folder_path():
    return os.path.join(os.path.expanduser("~"), "Downloads")


def reset_subset(state):
    # single
    single_subset_key = "subset"
    if single_subset_key in state.keys() and state[single_subset_key] is None:
        state[single_subset_key] = []
    # multi
    multiple_subset_list = ["subset_1", "subset_2"]
    for subset in multiple_subset_list:
        if subset in state.keys() and state[subset] is None:
            state[subset] = []


def reset_state(keys, state):
    for k in keys:
        if k not in state:
            state[k] = None
    reset_subset(state)


def file_selector(folder_path: str = None):
    if not folder_path:
        folder_path = get_download_folder_path()

    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox(
        "Select a file",
        filenames,
        index=None,
        key="single_file_selector",
    )
    if selected_filename is not None:
        return os.path.join(folder_path, selected_filename)
    return None


def multiple_file_selector(folder_path: str = None):
    if not folder_path:
        folder_path = get_download_folder_path()

    filenames = os.listdir(folder_path)
    selected_filenames = st.multiselect("Select files", filenames, key="multiple_file_selector")
    if selected_filenames:
        return [
            os.path.join(folder_path, selected_filename) for selected_filename in selected_filenames
        ]
    return None


def format_selector(data_helper: DatasetHelper, data_num: str = "Dataset"):
    try:
        formats = data_helper.detect_format()
    except Exception:
        formats = ["-", "datumaro", "voc", "coco"]  # temp

    return st.selectbox(f"Select a format to import {data_num}:", formats)


def import_dataset(data_helper, data_num: str = "Dataset"):
    try:
        formats = data_helper.detect_format()
    except Exception:
        formats = ["-", "datumaro", "voc", "coco"]  # temp
    selected_format = st.selectbox(f"Select a format to import {data_num}:", formats)
    if selected_format != "-" and selected_format != data_helper.format():
        data_helper.import_dataset(selected_format)


def save_dataset(data_helper, filename, save_folder: str = None):
    if not save_folder:
        save_folder = get_download_folder_path()
    save_path = os.path.join(save_folder, f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    data_helper.export(save_path, format="datumaro", save_media=True)
