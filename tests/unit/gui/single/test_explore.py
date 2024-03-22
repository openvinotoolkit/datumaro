# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from unittest import TestCase

from streamlit.testing.v1 import AppTest

from datumaro.components.visualizer import Visualizer

from tests.requirements import Requirements, mark_requirement

cwd = os.getcwd()
import sys

sys.path.append(os.path.join(cwd, "gui"))


def run_explore():
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

    dataset_dir = get_test_asset_path("explore_dataset")
    data_helper = SingleDatasetHelper(dataset_dir)
    state["data_helper"] = data_helper
    uploaded_file = os.path.basename(dataset_dir)
    state["uploaded_file"] = uploaded_file

    data_helper = state["data_helper"]

    data_helper.import_dataset("imagenet")

    tabs.call_explore()


class ExploreTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_explore_page_open(self):
        """Test if the page of explore tab is opened correctly."""
        at = AppTest.from_function(run_explore, default_timeout=600).run()

        # subheader
        assert len(at.subheader.values) == 2

        # columns
        assert len(at.columns) == 2

        # radio
        radio_key = "rd_query_type"
        query_types = ["Label", "Dataset Image", "User Image", "Text"]
        assert at.radio(radio_key).label == "Query Type"
        assert at.radio(radio_key).options == query_types
        assert at.radio(radio_key).value == query_types[0]

        # multiselect
        multiselect_key = "ms_select_label"
        options = ["bird", "cat", "dog", "monkey"]
        assert at.multiselect(multiselect_key).label == "Select Label(s)"
        assert sorted(at.multiselect(multiselect_key).options) == options
        assert not at.multiselect(multiselect_key).value

        # button
        button_key = "btn_add_query"
        assert at.button(button_key).label == "Add to Query List"
        assert not at.session_state.explore_queries

        button_key = "btn_search"
        assert at.button(button_key).label == "Search"
        assert at.button(button_key).disabled

        # number_input
        number_input_key = "ni_topk"
        state_explore_topk = at.session_state.explore_topk
        assert at.number_input(number_input_key).value == state_explore_topk

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_explore_check_label(self):
        """Test if label method of explore are worked out correctly."""
        at = AppTest.from_function(run_explore, default_timeout=600).run()

        # Select label to explore
        multiselect_key = "ms_select_label"
        at.multiselect(multiselect_key).select("cat").run()
        at.multiselect(multiselect_key).select("dog").run()
        selected_labels = ["cat", "dog"]
        assert at.multiselect(multiselect_key).value == selected_labels

        # Check Empty query
        assert not at.session_state.explore_queries

        # Click add query button
        button_key = "btn_add_query"
        at.button(button_key).click().run()

        # Check query added
        assert len(at.session_state.explore_queries) == 1
        assert at.session_state.explore_queries[0].labels == selected_labels
        assert at.checkbox("query_0").label == ":label: cat,dog"

        # Check Remove button disabled
        button_key = "btn_remove_query"
        assert at.button(button_key).disabled

        # Set topk as 1
        number_input_key = "ni_topk"
        at.number_input(number_input_key).set_value(1).run()
        assert at.number_input(number_input_key).value == 1

        # Check Empty explore results
        assert not at.session_state.explore_results

        # Click search
        button_key = "btn_search"
        at.button(button_key).click().run()
        assert at.session_state.explore_results

        checkbox_key = "result_0"
        expected_result = ["cat", "dog"]
        assert any(map(lambda v: v in at.checkbox(checkbox_key).label, expected_result))

        # Click result
        at.checkbox(checkbox_key).check().run()
        select_slider_key = "ss_selected_alpha_for_result"
        assert at.select_slider(select_slider_key)
        assert at.select_slider(select_slider_key).options == list(map(str, range(0, 110, 10)))
        assert at.select_slider(select_slider_key).value == 20

        # Set slider value as 50
        at.select_slider(select_slider_key).set_value(50).run()
        assert at.select_slider(select_slider_key).value == 50

        # Check visualizer
        dataset = at.session_state.data_helper.dataset()
        visualizer = Visualizer(dataset, figsize=(8, 8), alpha=0.2, show_plot_title=False)
        selected_item = at.session_state.explore_results[0]
        fig = visualizer.vis_one_sample(selected_item.id, selected_item.subset)
        assert fig is not None

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_explore_check_label_remove_query(self):
        """Test if label method of explore are worked out correctly."""
        at = AppTest.from_function(run_explore, default_timeout=600).run()

        # Add unnecessary label to query
        multiselect_key = "ms_select_label"
        at.multiselect(multiselect_key).select("bird").run()

        # Click add query button
        button_key = "btn_add_query"
        at.button(button_key).click().run()

        # Add real label to query
        at.multiselect(multiselect_key).unselect("bird").run()
        at.multiselect(multiselect_key).select("cat").run()
        at.multiselect(multiselect_key).select("dog").run()
        at.button(button_key).click().run()

        # Check query status
        assert len(at.session_state.explore_queries) == 2
        assert at.checkbox("query_0").label == ":label: bird"
        assert at.checkbox("query_1").label == ":label: cat,dog"

        # Remove unnecessary label from query
        at.checkbox("query_0").check().run()
        button_key = "btn_remove_query"
        at.button(button_key).click().run()

        # Check updated query status
        assert len(at.session_state.explore_queries) == 1

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_explore_check_dataset_image(self):
        """Test if label method of explore are worked out correctly."""
        at = AppTest.from_function(run_explore, default_timeout=600).run()

        # Select explore method
        radio_key = "rd_query_type"
        at.radio(radio_key).set_value("Dataset Image").run()

        # Select subset to explore
        selectbox_key = "sb_select_subset_exp"
        assert at.selectbox(selectbox_key).label == "Select a subset:"
        assert at.selectbox(selectbox_key).options == ["default"]

        selected_datasetitem_subset = "default"
        at.selectbox(selectbox_key).select(selected_datasetitem_subset).run()
        assert at.selectbox(selectbox_key).value == selected_datasetitem_subset

        # Select id to explore
        selectbox_key = "sb_select_id_exp"
        assert at.selectbox(selectbox_key).label == "Select a dataset item:"
        assert len(at.selectbox(selectbox_key).options) == 24
        selected_datasetitem_id = "bird:ILSVRC2012_val_00001563"
        at.selectbox(selectbox_key).select(selected_datasetitem_id).run()

        # Check Empty query
        assert not at.session_state.explore_queries

        # Click add query button
        button_key = "btn_add_query"
        at.button(button_key).click().run()

        # Check query added
        assert len(at.session_state.explore_queries) == 1
        assert at.session_state.explore_queries[0].item.id == selected_datasetitem_id
        assert at.session_state.explore_queries[0].item.subset == selected_datasetitem_subset
        assert (
            at.checkbox("query_0").label
            == ":frame_with_picture: default-bird:ILSVRC2012_val_00001563"
        )

        # Check Remove button disabled
        button_key = "btn_remove_query"
        assert at.button(button_key).disabled

        # Set topk as 1
        number_input_key = "ni_topk"
        at.number_input(number_input_key).set_value(1).run()
        assert at.number_input(number_input_key).value == 1

        # Check Empty explore results
        assert not at.session_state.explore_results

        # Click search
        button_key = "btn_search"
        at.button(button_key).click().run()
        assert at.session_state.explore_results

        checkbox_key = "result_0"
        assert at.checkbox(checkbox_key).label == "default-bird:ILSVRC2012_val_00001563"

        # Click result
        at.checkbox(checkbox_key).check().run()
        select_slider_key = "ss_selected_alpha_for_result"
        assert at.select_slider(select_slider_key)
        assert at.select_slider(select_slider_key).options == list(map(str, range(0, 110, 10)))
        assert at.select_slider(select_slider_key).value == 20

        # Set slider value as 50
        at.select_slider(select_slider_key).set_value(50).run()
        assert at.select_slider(select_slider_key).value == 50

        # Check visualizer
        dataset = at.session_state.data_helper.dataset()
        visualizer = Visualizer(dataset, figsize=(8, 8), alpha=0.2, show_plot_title=False)
        selected_item = at.session_state.explore_results[0]
        fig = visualizer.vis_one_sample(selected_item.id, selected_item.subset)
        assert fig is not None

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_explore_check_user_image(self):
        # TODO
        """Test if user image method of explore are worked out correctly."""

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_explore_check_text(self):
        """Test if text method of explore are worked out correctly."""
        at = AppTest.from_function(run_explore, default_timeout=600).run()

        # Select explore method
        radio_key = "rd_query_type"
        at.radio(radio_key).set_value("Text").run()

        # Set text input to explore
        text_input_key = "ti_query_text_exp"
        assert at.text_input(text_input_key).label == "Input text query:"

        selected_text = "bird"
        at.text_input(text_input_key).input(selected_text).run()
        assert at.text_input(text_input_key).value == selected_text

        # Check Empty query
        assert not at.session_state.explore_queries

        # Click add query button
        button_key = "btn_add_query"
        at.button(button_key).click().run()

        # Check query added
        assert len(at.session_state.explore_queries) == 1
        assert at.session_state.explore_queries[0].text == selected_text
        assert at.checkbox("query_0").label == ":speech_balloon: bird"

        # Check Remove button disabled
        button_key = "btn_remove_query"
        assert at.button(button_key).disabled

        # Set topk as 1
        number_input_key = "ni_topk"
        at.number_input(number_input_key).set_value(1).run()
        assert at.number_input(number_input_key).value == 1

        # Check Empty explore results
        assert not at.session_state.explore_results

        # Click search
        button_key = "btn_search"
        at.button(button_key).click().run()
        assert at.session_state.explore_results

        checkbox_key = "result_0"
        assert at.checkbox(checkbox_key).label == "default-bird:ILSVRC2012_val_00001563"

        # Click result
        at.checkbox(checkbox_key).check().run()
        select_slider_key = "ss_selected_alpha_for_result"
        assert at.select_slider(select_slider_key)
        assert at.select_slider(select_slider_key).options == list(map(str, range(0, 110, 10)))
        assert at.select_slider(select_slider_key).value == 20

        # Set slider value as 50
        at.select_slider(select_slider_key).set_value(50).run()
        assert at.select_slider(select_slider_key).value == 50

        # Check visualizer
        dataset = at.session_state.data_helper.dataset()
        visualizer = Visualizer(dataset, figsize=(8, 8), alpha=0.2, show_plot_title=False)
        selected_item = at.session_state.explore_results[0]
        fig = visualizer.vis_one_sample(selected_item.id, selected_item.subset)
        assert fig is not None
