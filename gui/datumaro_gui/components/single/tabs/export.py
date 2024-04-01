# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

import streamlit as st
from datumaro_gui.utils.dataset.data_loader import SingleDatasetHelper
from datumaro_gui.utils.dataset.state import get_download_folder_path
from datumaro_gui.utils.page import check_image_backend
from streamlit import session_state as state


def main():
    check_image_backend(state.get("IMAGE_BACKEND"))

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
    selected_task = st.selectbox("Select a task to export:", tasks, key="sb_task_export_sin")
    if selected_task:
        selected_format = st.selectbox(
            "Select a format to export:", formats[selected_task], key="sb_format_export_sin"
        )

    if selected_task and selected_format:
        selected_path = st.text_input(
            "Select a path to export:",
            value=osp.join(get_download_folder_path(), "exported_dataset"),
            key="ti_path_export_sin",
        )

    export_btn = st.button("Export", key="btn_export_sin")
    if export_btn:
        data_helper: SingleDatasetHelper = state["data_helper"]
        data_helper.export(selected_path, format=selected_format, save_media=True)
