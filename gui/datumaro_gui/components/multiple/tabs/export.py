# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp

import streamlit as st
from datumaro_gui.utils.dataset.data_loader import DataRepo, MultipleDatasetHelper
from datumaro_gui.utils.dataset.state import get_download_folder_path
from streamlit import session_state as state


def main():
    tasks = ["classification", "detection", "instance_segmentation", "segmentation", "landmark"]
    formats = {
        "classification": ["datumaro", "imagenet", "cifar", "mnist", "mnist_csv", "lfw"],
        "detection": [
            "datumaro",
            "coco_instances",
            "voc_detection",
            "yolo",
            "yolo_ultralytics",
            "kitti_detection",
            "tf_detection_api",
            "open_images",
            "segment_anything",
            "mot_seq_gt",
            "wider_face",
        ],
        "instance_segmentation": [
            "datumaro",
            "coco_instances",
            "voc_instance_segmentation",
            "open_images",
            "segment_anything",
        ],
        "segmentation": [
            "datumaro",
            "coco_panoptic",
            "voc_segmentation",
            "kitti_segmentation",
            "cityscapes",
            "camvid",
        ],
        "landmark": ["datumaro", "coco_person_keypoints", "voc_layout", "lfw"],
    }
    data_helper_1: MultipleDatasetHelper = state["data_helper_1"]
    data_helper_2: MultipleDatasetHelper = state["data_helper_2"]
    uploaded_file_1 = state["uploaded_file_1"]
    uploaded_file_2 = state["uploaded_file_2"]
    dataset_names = [uploaded_file_1, uploaded_file_2, "Merged Dataset"]
    selected_dataset = st.selectbox(
        "Select dataset to export : ", dataset_names, index=2, key="sb_export_ds_mult"
    )
    dataset_dict = {uploaded_file_1: data_helper_1, uploaded_file_2: data_helper_2}

    if selected_dataset == "Merged Dataset" and "data_helper_merged" not in state:
        st.error("Please merge dataset first")
    else:
        data_helper_3: MultipleDatasetHelper = state["data_helper_merged"]
        dataset_dict["Merged Dataset"] = data_helper_3

        selected_task = st.selectbox("Select a task to export:", tasks, key="sb_task_export_mult")
        if selected_task:
            selected_format = st.selectbox(
                "Select a format to export:", formats[selected_task], key="sb_format_export_mult"
            )

        if selected_task and selected_format:
            selected_path = st.text_input(
                "Select a path to export:",
                value=osp.join(get_download_folder_path(), "dataset.zip"),
                key="ti_path_export_mult",
            )

        export_btn = st.button("Export", "btn_export_mult")
        if export_btn:
            data_helper = dataset_dict.get(selected_dataset, None)
            data_helper.export(selected_path, format=selected_format, save_media=True)

            zip_path = DataRepo().zip_dataset(selected_path)

            with open(zip_path, "rb") as fp:
                st.download_button(
                    label="Download ZIP",
                    data=fp,
                    file_name=os.path.basename(zip_path),
                    mime="application/zip",
                )
