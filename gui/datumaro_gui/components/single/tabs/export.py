# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp

import streamlit as st
from datumaro_gui.utils.dataset.data_loader import SingleDatasetHelper
from streamlit import session_state as state


def main():
    tasks = ["classification", "detection", "instance_segmentation", "segmentation", "landmark"]
    formats = {
        "classification": ["imagenet", "cifar", "mnist", "mnist_csv", "lfw"],
        "detection": [
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
            "coco_instances",
            "voc_instance_segmentation",
            "open_images",
            "segment_anything",
        ],
        "segmentation": [
            "coco_panoptic",
            "voc_segmentation",
            "kitti_segmentation",
            "cityscapes",
            "camvid",
        ],
        "landmark": ["coco_person_keypoints", "voc_layout", "lfw"],
    }
    selected_task = st.selectbox("Select a task to export:", tasks)
    if selected_task:
        selected_format = st.selectbox("Select a format to export:", formats[selected_task])

    if selected_task and selected_format:
        selected_path = st.text_input(
            "Select a path to export:", value=osp.join(osp.expanduser("~"), "Downloads")
        )

    export_btn = st.button("Export")
    if export_btn:
        if not osp.exists(selected_path):
            os.makedirs(selected_path)
        print(osp.abspath(selected_path))
        data_helper: SingleDatasetHelper = state["data_helper"]
        data_helper.export(selected_path, format=selected_format, save_media=True)
