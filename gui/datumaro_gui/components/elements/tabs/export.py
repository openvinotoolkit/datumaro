# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp

import streamlit as st
from streamlit import session_state as state

from ..data_loader import DatasetHelper


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
    data_helper_1: DatasetHelper = state["data_helper_1"]
    data_helper_2: DatasetHelper = state["data_helper_2"]
    uploaded_zip_1 = state["uploaded_zip_1"].name[:-4]
    uploaded_zip_2 = state["uploaded_zip_2"].name[:-4]
    dataset_names = [uploaded_zip_1, uploaded_zip_2, "Merged Dataset"]
    selected_dataset = st.selectbox("Select dataset to export : ", dataset_names, index=2)
    dataset_dict = {uploaded_zip_1: data_helper_1, uploaded_zip_2: data_helper_2}

    if selected_dataset == "Merged Dataset" and "data_helper_merged" not in state:
        st.error("Please merge dataset first")
    else:
        data_helper_3: DatasetHelper = state["data_helper_merged"]
        dataset_dict["Merged Dataset"] = data_helper_3

        selected_task = st.selectbox("Select a task to export:", tasks)
        if selected_task:
            selected_format = st.selectbox("Select a format to export:", formats[selected_task])

        if selected_task and selected_format:
            selected_path = st.text_input(
                "Select a path to export:", value=osp.join(osp.expanduser("~"), "Downloads")
            )

        export_btn = st.button("Export")
        if export_btn:
            data_helper = dataset_dict.get(selected_dataset, None)
            if not osp.exists(selected_path):
                os.makedirs(selected_path)
            print(osp.abspath(selected_path))
            data_helper.export(selected_path, format=selected_format, save_media=True)
