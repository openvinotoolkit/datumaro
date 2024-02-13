# Copyright (C) 2019-2024 Intel Corporation
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


def run_compare():
    import os

    import streamlit as state
    from streamlit import session_state as state

    from gui.datumaro_gui.components.multiple import tabs
    from gui.datumaro_gui.utils.dataset.data_loader import MultipleDatasetHelper
    from gui.datumaro_gui.utils.dataset.state import multiple_state_keys, reset_state

    from tests.utils.assets import get_test_asset_path

    reset_state(multiple_state_keys, state)

    dataset_1_dir = get_test_asset_path("datumaro_dataset")
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

    data_helper_1.import_dataset("datumaro")
    data_helper_2.import_dataset("voc")

    tabs.call_compare()


class CompareTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_compare_page_open(self):
        """Test if the session state is initialized correctly."""
        at = AppTest.from_function(run_compare, default_timeout=600).run()

        # header
        assert len(at.header.values) == 2

        # subheader
        assert len(at.subheader.values) == 6

        # columns
        assert len(at.columns) == 4

        # caption
        assert len(at.caption.values) == 3

        # state
        assert not at.session_state.high_level_table.empty
        assert not at.session_state.mid_level_table.empty
        assert at.session_state.low_level_table == None
        assert at.session_state.matched == ["bicycle", "car"]
        assert not at.session_state.mapping

        # toggle
        toggle_key = "tg_low_lvl_on_comp_mul"
        assert at.toggle(toggle_key).label == "Show low-level table"
        assert not at.toggle(toggle_key).value

        # dataframe
        assert len(at.dataframe) == 4

        # button
        button_key = "btn_mapping_comp_mul"
        assert at.button(button_key).label == "Finalize mapping"
        assert not at.button(button_key).value

        # slider
        slider_key = "sd_threshold_comp_mul"
        assert at.slider(slider_key).label == "Desired similarity threshold"
        assert at.slider(slider_key).min == 0.0
        assert at.slider(slider_key).max == 1.0

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_compare_low_level_table(self):
        at = AppTest.from_function(run_compare, default_timeout=600).run()

        at.toggle("tg_low_lvl_on_comp_mul").set_value(True).run()

        assert not at.session_state.low_level_table.empty
        assert len(at.caption.values) == 4

    # @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    # def test_compare_finalize_mapping(self):
    # TODO
