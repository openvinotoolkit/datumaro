# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from unittest import TestCase

import pytest
from streamlit.testing.v1 import AppTest

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path
from tests.utils.gui import compare_init_state, compare_single_datahelper, compare_single_stats

cwd = os.getcwd()
app_path = os.path.join(cwd, "gui", "streamlit_app.py")

import sys

sys.path.append(os.path.join(cwd, "gui"))

from gui.datumaro_gui.utils.dataset.state import get_download_folder_path, single_state_keys

single_dataset_key = "datumaro_gui.utils.page_p_:microscope: Single dataset"


class SingleDataTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_single_page_open(self):
        """Test if the session state is initialized correctly."""
        at = AppTest.from_file(app_path, default_timeout=600).run()

        at.sidebar.checkbox(single_dataset_key).check()
        at.sidebar.checkbox("datumaro_gui.utils.page_p_:open_book: Introduction").uncheck()
        at.run()

        assert not at.exception
        assert at.sidebar.checkbox(single_dataset_key).value

        # Check state initialized
        assert compare_init_state(single_state_keys, at.session_state.filtered_state.keys())

        # Check text_input
        assert at.text_input.values[0] == get_download_folder_path()

        # Check selectbox value
        assert at.selectbox.values[0] == None

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_test_asset(self):
        """Test if the session state is initialized correctly."""
        at = AppTest.from_file(app_path, default_timeout=600).run()

        at.sidebar.checkbox(single_dataset_key).check()
        at.sidebar.checkbox("datumaro_gui.utils.page_p_:open_book: Introduction").uncheck()
        at.run()

        at.text_input("single_input_path").input(get_test_asset_path()).run()

        at.selectbox("single_file_selector").select("datumaro_dataset")
        at.run(timeout=60)

        assert at.selectbox("single_file_selector").value == "datumaro_dataset"

        # Check state
        assert at.session_state.filtered_state["uploaded_file"] == "datumaro_dataset"
        expected_image_stats = {
            "images count": 4,
            "unique images count": 2,
            "repeated images count": 2,
            "repeated images": [
                [("a", "train"), ("c", "test")],
                [("b", "train"), ("d", "validation")],
            ],
        }
        compare_single_datahelper(
            at.session_state.filtered_state["data_helper"],
            [get_test_asset_path("datumaro_dataset"), ["datumaro"], "datumaro"],
        )
        compare_single_stats(at.session_state.filtered_state["data_helper"], expected_image_stats)
