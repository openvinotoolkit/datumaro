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


def run_export():
    import os

    import streamlit as state
    from streamlit import session_state as state

    from gui.datumaro_gui.components.single import tabs
    from gui.datumaro_gui.utils.dataset.data_loader import SingleDatasetHelper
    from gui.datumaro_gui.utils.dataset.state import reset_state, single_state_keys
    from gui.datumaro_gui.utils.page import init_func

    from tests.utils.assets import get_test_asset_path

    init_func(state.get("IMAGE_BACKEND", None))
    reset_state(single_state_keys, state)

    dataset_dir = get_test_asset_path("datumaro_dataset")
    data_helper = SingleDatasetHelper(dataset_dir)
    state["data_helper"] = data_helper
    uploaded_file = os.path.basename(dataset_dir)
    state["uploaded_file"] = uploaded_file

    data_helper = state["data_helper"]

    data_helper.import_dataset("datumaro")

    tabs.call_export()


class ExportTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_export_page_open(self):
        """Test if the session state is initialized correctly."""
        at = AppTest.from_function(run_export, default_timeout=600).run()

        selectbox_key = "sb_task_export_sin"
        task_list = [
            "classification",
            "detection",
            "instance_segmentation",
            "segmentation",
            "landmark",
        ]
        assert at.selectbox(selectbox_key).label == "Select a task to export:"
        assert at.selectbox(selectbox_key).options == task_list
        assert at.selectbox(selectbox_key).value == "classification"

        selectbox_key = "sb_format_export_sin"
        format_list = ["datumaro", "imagenet", "cifar", "mnist", "mnist_csv", "lfw"]
        assert at.selectbox(selectbox_key).label == "Select a format to export:"
        assert at.selectbox(selectbox_key).options == format_list
        assert at.selectbox(selectbox_key).value == "datumaro"

        textinput_key = "ti_path_export_sin"
        download_folder = os.path.join(os.path.expanduser("~"), "Downloads", "dataset.zip")
        assert at.text_input(textinput_key).value == download_folder
        assert not at.button("btn_export_sin").value

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_export(self):
        """Test if the dataset is exported correctly."""
        at = AppTest.from_function(run_export, default_timeout=600).run()

        # Click export button
        at.button("btn_export_sin").click().run()

        download_folder = os.path.join(os.path.expanduser("~"), "Downloads")
        filenames = os.listdir(download_folder)
        assert "dataset.zip" in filenames
