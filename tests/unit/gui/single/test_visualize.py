# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from unittest import TestCase

from streamlit.testing.v1 import AppTest

from datumaro import Dataset
from datumaro.components.visualizer import Visualizer

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets

cwd = os.getcwd()
import sys

sys.path.append(os.path.join(cwd, "gui"))


def run_visualize():
    import os

    import streamlit as state
    from streamlit import session_state as state

    from gui.datumaro_gui.components.single import tabs
    from gui.datumaro_gui.utils.dataset.data_loader import SingleDatasetHelper
    from gui.datumaro_gui.utils.dataset.state import reset_state, single_state_keys

    from tests.utils.assets import get_test_asset_path

    reset_state(single_state_keys, state)

    dataset_dir = get_test_asset_path("datumaro_dataset")
    data_helper = SingleDatasetHelper(dataset_dir)
    state["data_helper"] = data_helper
    uploaded_file = os.path.basename(dataset_dir)
    state["uploaded_file"] = uploaded_file

    data_helper = state["data_helper"]

    data_helper.import_dataset("datumaro")

    tabs.call_visualize()


class VisualizeTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_visualize_page_open(self):
        """Test if the page of visualize tab is opened correctly."""
        at = AppTest.from_function(run_visualize, default_timeout=600).run()

        # subheader
        assert len(at.subheader.values) == 2

        # columns
        assert len(at.columns) == 2

        # selectbox
        selectbox_key = "sb_select_subset_viz"
        subset_list = ["test", "train", "validation"]
        assert at.selectbox(selectbox_key).label == "Select a subset:"
        assert sorted(at.selectbox(selectbox_key).options) == subset_list
        assert at.selectbox(selectbox_key).value == subset_list[0]

        selectbox_key = "sb_select_id_viz"
        ids = ["c"]
        assert at.selectbox(selectbox_key).label == "Select a dataset item:"
        assert at.selectbox(selectbox_key).options == ids
        assert at.selectbox(selectbox_key).value == ids[0]

        selectbox_key = "sb_select_ann_id_viz"
        options = ["All", "0"]
        assert at.selectbox(selectbox_key).label == "Select annotation:"
        assert at.selectbox(selectbox_key).options == options
        assert at.selectbox(selectbox_key).value == options[0]

        # select_slider
        select_slider_key = "ss_select_alpha_viz"
        assert at.select_slider(select_slider_key).label == "Choose a transparency of annotations"
        assert at.select_slider(select_slider_key).options == list(map(str, range(0, 110, 10)))
        assert at.select_slider(select_slider_key).value == 20

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_visualize_check_api(self):
        """Test if apis are worked out correctly."""
        at = AppTest.from_function(run_visualize, default_timeout=600).run()

        # selectbox
        selectbox_key = "sb_select_subset_viz"
        at.selectbox(selectbox_key).select("train").run()
        assert at.selectbox(selectbox_key).value == "train"

        selectbox_key = "sb_select_id_viz"
        train_ids = ["a", "b"]
        assert at.selectbox(selectbox_key).options == train_ids

        at.selectbox(selectbox_key).select_index(1).run()
        assert at.selectbox(selectbox_key).value == "b"

        selectbox_key = "sb_select_ann_id_viz"
        at.selectbox(selectbox_key).select_index(1).run()
        assert at.selectbox(selectbox_key).value == 0

        # select_slider
        select_slider_key = "ss_select_alpha_viz"
        at.select_slider(select_slider_key).set_value(50)
        assert at.select_slider(select_slider_key).value == 50

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_visualizer(self):
        """Test if the dataset is visualized correctly."""
        at = AppTest.from_function(run_visualize, default_timeout=600).run()

        # Check visualizer
        dataset = at.session_state.data_helper.dataset()
        visualizer = Visualizer(dataset, figsize=(8, 8), alpha=0.2, show_plot_title=False)
        expected_dataset = Dataset.import_from(get_test_asset_path("datumaro_dataset"), "datumaro")
        compare_datasets(self, visualizer.dataset, expected_dataset)
        assert visualizer.figsize == (8, 8)
        assert visualizer.alpha == 0.2
        assert not visualizer.show_plot_title

        # Check vis_one_sample
        selected_id = at.selectbox("sb_select_id_viz").value
        selected_subset = at.selectbox("sb_select_subset_viz").value
        selected_ann_id = at.selectbox("sb_select_ann_id_viz").value
        fig = visualizer.vis_one_sample(selected_id, selected_subset, ann_id=selected_ann_id)
        assert fig is not None
