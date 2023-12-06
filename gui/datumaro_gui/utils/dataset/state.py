# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import os
from pathlib import Path

import streamlit as st

cwd = os.getcwd()
datum_path = Path(cwd).parents[0]
data_folder_path = os.path.join(datum_path, "data")


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


def file_selector(folder_path=data_folder_path):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select a file", filenames, index=None)
    if selected_filename is not None:
        return os.path.join(folder_path, selected_filename)
    return None


def multiple_file_selector(folder_path=data_folder_path):
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
