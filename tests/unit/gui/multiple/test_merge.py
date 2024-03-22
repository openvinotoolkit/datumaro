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


def run_merge():
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

    tabs.call_merge()


class MergeTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_merge_page_open(self):
        """Test if the page of merge tab is opened correctly."""
        at = AppTest.from_function(run_merge, default_timeout=600).run()

        # selectbox
        assert at.selectbox("sb_merge_method_mult").options == ["union", "intersect", "exact"]

        # button
        assert not at.button("merge_btn_mult").value

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_merge(self):
        """Test if the dataset is merged correctly."""
        at = AppTest.from_function(run_merge, default_timeout=600).run()

        assert "data_helper_merged" not in at.session_state

        # select method
        at.selectbox("sb_merge_method_mult").select("union").run()
        at.button("merge_btn_mult").click().run()

        assert "data_helper_merged" in at.session_state
        expected_num_labels = (
            at.session_state.data_helper_1.num_labels + at.session_state.data_helper_2.num_labels
        )
        assert at.session_state.data_helper_merged.num_labels == expected_num_labels
