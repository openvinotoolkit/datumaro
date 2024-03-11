# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from unittest import TestCase

from streamlit.testing.v1 import AppTest

from tests.requirements import Requirements, mark_requirement

cwd = os.getcwd()
app_path = os.path.join(cwd, "gui", "streamlit_app.py")

import sys

sys.path.append(os.path.join(cwd, "gui"))

multiple_dataset_key = "datumaro_gui.utils.page_p_:telescope: Multiple datasets"


def run_export():
    import os

    import streamlit as state
    from streamlit import session_state as state

    from gui.datumaro_gui.components.multiple import tabs
    from gui.datumaro_gui.utils.dataset.data_loader import MultipleDatasetHelper
    from gui.datumaro_gui.utils.dataset.state import multiple_state_keys, reset_state
    from gui.datumaro_gui.utils.page import init_func

    from tests.utils.assets import get_test_asset_path

    init_func(state.get("IMAGE_BACKEND", None))
    reset_state(multiple_state_keys, state)

    dataset_1_dir = get_test_asset_path("coco_dataset", "coco_instances")
    data_helper_1 = MultipleDatasetHelper(dataset_1_dir)
    state["data_helper_1"] = data_helper_1
    uploaded_file = os.path.basename(dataset_1_dir)
    state["uploaded_file_1"] = uploaded_file

    dataset_2_dir = get_test_asset_path("voc_dataset", "voc_dataset1")
    data_helper_2 = MultipleDatasetHelper(dataset_2_dir)
    state["data_helper_2"] = data_helper_2
    uploaded_file = os.path.basename(dataset_2_dir)
    state["uploaded_file_2"] = uploaded_file

    data_helper_1 = state["data_helper_1"]
    data_helper_2 = state["data_helper_2"]

    data_helper_1.import_dataset("coco_instances")
    data_helper_2.import_dataset("voc")

    tabs.call_export()


def run_export_merged():
    import os

    import streamlit as state
    from streamlit import session_state as state

    from gui.datumaro_gui.components.multiple import tabs
    from gui.datumaro_gui.utils.dataset.data_loader import MultipleDatasetHelper
    from gui.datumaro_gui.utils.dataset.state import multiple_state_keys, reset_state
    from gui.datumaro_gui.utils.page import init_func

    from tests.utils.assets import get_test_asset_path

    init_func(state.get("IMAGE_BACKEND", None))
    reset_state(multiple_state_keys, state)

    dataset_1_dir = get_test_asset_path("coco_dataset", "coco_instances")
    data_helper_1 = MultipleDatasetHelper(dataset_1_dir)
    state["data_helper_1"] = data_helper_1
    uploaded_file = os.path.basename(dataset_1_dir)
    state["uploaded_file_1"] = uploaded_file

    dataset_2_dir = get_test_asset_path("voc_dataset", "voc_dataset1")
    data_helper_2 = MultipleDatasetHelper(dataset_2_dir)
    state["data_helper_2"] = data_helper_2
    uploaded_file = os.path.basename(dataset_2_dir)
    state["uploaded_file_2"] = uploaded_file

    data_helper_1 = state["data_helper_1"]
    data_helper_2 = state["data_helper_2"]

    data_helper_1.import_dataset("coco_instances")
    data_helper_2.import_dataset("voc")

    data_helper = MultipleDatasetHelper()
    state["data_helper_merged"] = data_helper

    merged_dataset = data_helper.merge([data_helper_1.dataset(), data_helper_2.dataset()], "union")
    data_helper.update_dataset(merged_dataset)
    tabs.call_export()


class ExportTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_export_page_open(self):
        """Test if the session state is initialized correctly."""
        at = AppTest.from_function(run_export, default_timeout=600).run()

        # selectbox
        assert sorted(at.selectbox("sb_export_ds_mult").options) == [
            "Merged Dataset",
            "coco_instances",
            "voc_dataset1",
        ]
        assert at.selectbox("sb_export_ds_mult").value == "Merged Dataset"

        # error
        assert len(at.error) == 1

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_export_merged_page_open(self):
        """Test if the session state is initialized correctly."""
        at = AppTest.from_function(run_export_merged, default_timeout=600).run()

        # selectbox
        assert sorted(at.selectbox("sb_export_ds_mult").options) == [
            "Merged Dataset",
            "coco_instances",
            "voc_dataset1",
        ]
        assert at.selectbox("sb_export_ds_mult").value == "Merged Dataset"

        # error
        assert len(at.error) == 0

        assert at.selectbox("sb_task_export_mult").options == [
            "classification",
            "detection",
            "instance_segmentation",
            "segmentation",
            "landmark",
        ]
        assert at.selectbox("sb_task_export_mult").value == "classification"

        assert at.selectbox("sb_format_export_mult").options == [
            "datumaro",
            "imagenet",
            "cifar",
            "mnist",
            "mnist_csv",
            "lfw",
        ]
        assert at.selectbox("sb_format_export_mult").value == "datumaro"

        assert at.text_input("ti_path_export_mult").value
        assert not at.button("btn_export_mult").value

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_export(self):
        """"""
        at = AppTest.from_function(run_export_merged, default_timeout=600).run()

        # Click export button
        at.button("btn_export_mult").click().run()

        download_folder = os.path.join(os.path.expanduser("~"), "Downloads")
        filenames = os.listdir(download_folder)
        assert "dataset.zip" in filenames
