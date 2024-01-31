# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp

import streamlit as st
from datumaro_gui.utils.dataset.data_loader import DataRepo, SingleDatasetHelper
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
    selected_task = st.selectbox("Select a task to export:", tasks)
    if selected_task:
        selected_format = st.selectbox("Select a format to export:", formats[selected_task])

    if selected_task and selected_format:
        selected_path = st.text_input(
            "Select a path to export:",
            value=osp.join(get_download_folder_path(), "dataset.zip"),
        )

    export_btn = st.button("Export")
    if export_btn:
        data_helper: SingleDatasetHelper = state["data_helper"]
        data_helper.export(selected_path, format=selected_format, save_media=True)

        uploaded_file = state["uploaded_file"]
        zip_path = DataRepo().zip_dataset(selected_path, output_fn=uploaded_file)

        with open(zip_path, "rb") as fp:
            st.download_button(
                label="Download ZIP",
                data=fp,
                file_name=os.path.basename(zip_path),
                mime="application/zip",
            )
