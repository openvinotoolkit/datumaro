# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from unittest import TestCase

import pytest
from streamlit.testing.v1 import AppTest

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path
from tests.utils.gui import compare_init_state, compare_multiple_datahelper

cwd = os.getcwd()
app_path = os.path.join(cwd, "gui", "streamlit_app.py")

import sys

sys.path.append(os.path.join(cwd, "gui"))

from gui.datumaro_gui.utils.dataset.state import get_download_folder_path, multiple_state_keys

multiple_dataset_key = "datumaro_gui.utils.page_p_:telescope: Multiple datasets"


class MutipleDataTest(TestCase):
    @pytest.mark.xfail(reason="Cannot copy contextvar to thread")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_multiple_page_open(self):
        """Test if the session state is initialized correctly."""
        at = AppTest.from_file(app_path, default_timeout=600).run()

        at.sidebar.checkbox(multiple_dataset_key).check()
        at.sidebar.checkbox("datumaro_gui.utils.page_p_:open_book: Introduction").uncheck()
        at.run()

        assert not at.exception
        assert at.sidebar.checkbox(multiple_dataset_key).value

        # Check state initialized
        assert compare_init_state(multiple_state_keys, at.session_state.filtered_state.keys())

        # Check text_input
        assert at.text_input.values[0] == get_download_folder_path()

        # Check multiselect value
        assert len(at.multiselect.values[0]) == 0

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_test_asset(self):
        """Test if the session state is initialized correctly."""
        at = AppTest.from_file(app_path)
        at.run(timeout=60)

        at.sidebar.checkbox(multiple_dataset_key).check()
        at.sidebar.checkbox("datumaro_gui.utils.page_p_:open_book: Introduction").uncheck()
        at.run()

        at.text_input("multiple_input_path").input(get_test_asset_path()).run()

        at.multiselect("multiple_file_selector").select("imagenet_dataset")
        at.run(timeout=60)

        at.multiselect("multiple_file_selector").select("datumaro_dataset")
        at.run(timeout=60)

        assert at.multiselect("multiple_file_selector").value == [
            "imagenet_dataset",
            "datumaro_dataset",
        ]

        # Check state
        assert at.session_state.filtered_state["uploaded_file_1"] == "imagenet_dataset"
        assert at.session_state.filtered_state["uploaded_file_2"] == "datumaro_dataset"
        compare_multiple_datahelper(
            at.session_state.filtered_state["data_helper_1"],
            [get_test_asset_path("imagenet_dataset"), ["imagenet"], "imagenet"],
        )
        compare_multiple_datahelper(
            at.session_state.filtered_state["data_helper_2"],
            [get_test_asset_path("datumaro_dataset"), ["datumaro"], "datumaro"],
        )
