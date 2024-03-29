# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from typing import NamedTuple
from unittest import TestCase

import pandas as pd
from streamlit.testing.v1 import AppTest

from tests.requirements import Requirements, mark_requirement

cwd = os.getcwd()
import sys

sys.path.append(os.path.join(cwd, "gui"))

from gui.datumaro_gui.components.multiple.tabs.transform import (
    TransformAggregation,
    TransformAutoCorrection,
    TransformFiltration,
    TransformLabelRemap,
    TransformReindexing,
    TransformRemove,
    TransformSplit,
    TransformSubsetRename,
)


def run_transform():
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

    tabs.call_transform()


def run_transform_imagenet():
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

    dataset_1_dir = get_test_asset_path("imagenet_dataset")
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

    data_helper_1.import_dataset("imagenet")
    data_helper_2.import_dataset("voc")

    tabs.call_transform()


def run_transform_remapped():
    import os

    import pandas as pd
    import streamlit as state
    from streamlit import session_state as state

    from datumaro.components.annotation import AnnotationType
    from gui.datumaro_gui.components.multiple import tabs
    from gui.datumaro_gui.utils.dataset.data_loader import MultipleDatasetHelper
    from gui.datumaro_gui.utils.dataset.info import return_matches
    from gui.datumaro_gui.utils.dataset.state import multiple_state_keys, reset_state
    from gui.datumaro_gui.utils.page import init_func

    from tests.utils.assets import get_test_asset_path

    init_func(state.get("IMAGE_BACKEND", None))
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

    categories_1 = state.data_helper_1._dm_dataset.categories()[
        AnnotationType.label
    ]._indices.keys()
    categories_2 = state.data_helper_2._dm_dataset.categories()[
        AnnotationType.label
    ]._indices.keys()
    uploaded_file_1 = state["uploaded_file_1"]
    uploaded_file_2 = state["uploaded_file_2"]
    matches, _ = return_matches(categories_1, categories_2, uploaded_file_1, uploaded_file_2)
    state.matched = matches
    data_dict = {uploaded_file_1: ["tom", "mary"], uploaded_file_2: ["person", "person"]}
    state.mapping = pd.DataFrame(data_dict)

    tabs.call_transform()


class Split(NamedTuple):
    subset: str
    ratio: float


class TransformTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_page_open(self):
        """Test if the page of transform tab is opened correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # select dataset
        assert at.selectbox(key="trans_selectbox_d1").index == 0
        assert at.selectbox(key="trans_selectbox_d2").index == 1

        # Remap
        assert not at.session_state.matched
        assert not at.session_state.mapping

        button_key = "btn_remap_c1"
        assert at.button(key=button_key)
        assert at.button(key=button_key).label == "Do Label Remap"
        assert not at.button(key=button_key).value

        toggle_key = "tg_del_unselected_remap_mul_c1"
        assert at.toggle(key=toggle_key)
        assert at.toggle(key=toggle_key).label == "Delete unselected labels"
        assert not at.toggle(key=toggle_key).value

        button_key = "btn_remap_c2"
        assert at.button(key=button_key)
        assert at.button(key=button_key).label == "Do Label Remap"
        assert not at.button(key=button_key).value

        toggle_key = "tg_del_unselected_remap_mul_c2"
        assert at.toggle(key=toggle_key)
        assert at.toggle(key=toggle_key).label == "Delete unselected labels"
        assert not at.toggle(key=toggle_key).value
        assert len(at.warning) == 2

        # Aggregation
        multiselect_key = "ms_selected_subsets_agg_mul_c1"
        assert at.multiselect(key=multiselect_key)
        subset_list = ["test", "train", "validation"]
        assert at.multiselect(key=multiselect_key).label == "Select subsets to be aggregated"
        assert sorted(at.multiselect(key=multiselect_key).options) == subset_list
        assert sorted(at.multiselect(key=multiselect_key).value) == subset_list

        text_input_key = "ti_dst_subset_name_agg_mul_c1"
        assert at.text_input(key=text_input_key)
        assert at.text_input(key=text_input_key).value == "default"
        assert at.text_input(text_input_key).label == "Aggregated Subset Name"

        button_key = "btn_agg_mul_c1"
        assert at.button(key=button_key)
        assert at.button(key=button_key).label == "Do aggregation"
        assert not at.button(key=button_key).value

        multiselect_key = "ms_selected_subsets_agg_mul_c2"
        assert at.multiselect(key=multiselect_key)
        subset_list = ["test", "train"]
        assert at.multiselect(key=multiselect_key).label == "Select subsets to be aggregated"
        assert sorted(at.multiselect(key=multiselect_key).options) == subset_list
        assert sorted(at.multiselect(key=multiselect_key).value) == subset_list

        text_input_key = "ti_dst_subset_name_agg_mul_c2"
        assert at.text_input(key=text_input_key)
        assert at.text_input(key=text_input_key).value == "default"
        assert at.text_input(text_input_key).label == "Aggregated Subset Name"

        button_key = "btn_agg_mul_c2"
        assert at.button(key=button_key)
        assert at.button(key=button_key).label == "Do aggregation"
        assert not at.button(key=button_key).value

        # Split
        button_key = "btn_add_subset_split_mul_c1"
        assert at.button(key=button_key)
        assert not at.button(key=button_key).value

        button_key = "btn_do_split_mul_c1"
        assert at.button(key=button_key)
        assert not at.button(key=button_key).value

        button_key = "btn_add_subset_split_mul_c2"
        assert at.button(key=button_key)
        assert not at.button(key=button_key).value

        button_key = "btn_do_split_mul_c2"
        assert at.button(key=button_key)
        assert not at.button(key=button_key).value

        # Subset Rename
        selectbox_key = "sb_subset_rename_mul_c1"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select a subset to rename"
        assert at.selectbox(selectbox_key).options == list(
            at.session_state.data_helper_1.dataset().subsets().keys()
        )

        text_input_key = "ti_subset_rename_mul_c1"
        assert at.text_input(text_input_key)
        assert at.text_input(text_input_key).label == "New subset name"

        button_key = "btn_subset_rename_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Do Subset Rename"

        selectbox_key = "sb_subset_rename_mul_c2"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select a subset to rename"
        assert at.selectbox(selectbox_key).options == list(
            at.session_state.data_helper_2.dataset().subsets().keys()
        )

        text_input_key = "ti_subset_rename_mul_c2"
        assert at.text_input(text_input_key)
        assert at.text_input(text_input_key).label == "New subset name"

        button_key = "btn_subset_rename_mul_c2"
        assert at.button(button_key)
        assert at.button(button_key).label == "Do Subset Rename"

        # Reindex
        number_input_key = "ni_start_idx_reindex_mul_c1"
        assert at.number_input(number_input_key)
        assert at.number_input(number_input_key).min == 0

        button_key = "btn_start_idx_reindex_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Set IDs from 0"

        button_key = "btn_start_media_name_reindex_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Set IDs with media name"

        number_input_key = "ni_start_idx_reindex_mul_c2"
        assert at.number_input(number_input_key)
        assert at.number_input(number_input_key).min == 0

        button_key = "btn_start_idx_reindex_mul_c2"
        assert at.button(button_key)
        assert at.button(button_key).label == "Set IDs from 0"

        button_key = "btn_start_media_name_reindex_mul_c2"
        assert at.button(button_key)
        assert at.button(button_key).label == "Set IDs with media name"

        # Filter
        selectbox_key = "sb_selected_mode_filt_mul_c1"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select filtering mode"
        assert at.selectbox(selectbox_key).options == ["items", "annotations", "items+annotations"]

        text_input_key = "ti_filter_expr_filt_mul_c1"
        assert at.text_input(text_input_key)
        assert (
            at.text_input(text_input_key).label
            == "Enter XML filter expression ([XPATH](https://devhints.io/xpath))"
        )
        assert at.text_input(text_input_key).placeholder == 'Eg. /item[subset="train"]'
        assert at.text_input(text_input_key).value == None

        button_key = "btn_filter_filt_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Filter dataset"

        toggle_key = "tg_show_xml_filt_mul_c1"
        assert at.toggle(toggle_key)
        assert not at.toggle(toggle_key).value

        # Remove
        selectbox_key = "sb_select_subset_remove_mul_c1"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select a subset"
        assert sorted(at.selectbox(selectbox_key).options) == sorted(
            list(at.session_state.data_helper_1.subset_to_ids().keys())
        )
        selected_subset = at.selectbox(selectbox_key).value

        selectbox_key = "sb_select_id_remove_mul_c1"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select an item"
        assert (
            at.selectbox(selectbox_key).options
            == at.session_state.data_helper_1.subset_to_ids()[selected_subset]
        )
        selected_id = at.selectbox(selectbox_key).value
        selected_item = at.session_state.data_helper_1.dataset().get(selected_id, selected_subset)

        selectbox_key = "sb_select_ann_id_mul_c1"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select an annotation"
        assert at.selectbox(selectbox_key).options == [
            "All",
        ] + sorted(list({str(ann.id) for ann in selected_item.annotations}))

        button_key = "btn_remove_item_remove_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Remove item"

        button_key = "btn_remove_annot_remove_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Remove annotation"

        selectbox_key = "sb_select_subset_remove_mul_c2"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select a subset"
        assert sorted(at.selectbox(selectbox_key).options) == sorted(
            list(at.session_state.data_helper_2.subset_to_ids().keys())
        )
        selected_subset = at.selectbox(selectbox_key).value

        selectbox_key = "sb_select_id_remove_mul_c2"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select an item"
        assert (
            at.selectbox(selectbox_key).options
            == at.session_state.data_helper_2.subset_to_ids()[selected_subset]
        )
        selected_id = at.selectbox(selectbox_key).value
        selected_item = at.session_state.data_helper_2.dataset().get(selected_id, selected_subset)

        selectbox_key = "sb_select_ann_id_mul_c2"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select an annotation"
        assert at.selectbox(selectbox_key).options == [
            "All",
        ] + sorted(list({str(ann.id) for ann in selected_item.annotations}))

        button_key = "btn_remove_item_remove_mul_c2"
        assert at.button(button_key)
        assert at.button(button_key).label == "Remove item"

        button_key = "btn_remove_annot_remove_mul_c2"
        assert at.button(button_key)
        assert at.button(button_key).label == "Remove annotation"

        assert len(at.metric) == 4

        # Auto correct
        selectbox_key = "sb_select_task_auto_corr_mul_c1"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select a task"
        assert at.selectbox(selectbox_key).options == [
            "Classification",
            "Detection",
            "Segmentation",
        ]

        button_key = "btn_auto_corr_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Correct a dataset"

        selectbox_key = "sb_select_task_auto_corr_mul_c2"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select a task"
        assert at.selectbox(selectbox_key).options == [
            "Classification",
            "Detection",
            "Segmentation",
        ]

        button_key = "btn_auto_corr_mul_c2"
        assert at.button(button_key)
        assert at.button(button_key).label == "Correct a dataset"


class TransformLabelRemapTest(TestCase):
    def test_name(self):
        transform = TransformLabelRemap()
        assert transform.name == "Label Remapping"

    def test_info(self):
        transform = TransformLabelRemap()
        assert transform.info == "This helps to remap labels of dataset."

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_label_remap_page_open(self):
        """Test if label remap method of transform in multiple dataset page is opened correctly."""
        at = AppTest.from_function(run_transform_remapped, default_timeout=600).run()

        expected_matches = ["bicycle", "car"]

        # mapping
        assert not at.session_state.mapping.empty
        assert (
            at.session_state.mapping.columns
            == [at.session_state.uploaded_file_1, at.session_state.uploaded_file_2]
        ).all()

        # matched
        assert at.session_state.matched
        assert at.session_state.matched == expected_matches

        # toggle
        toggle_key = "tg_del_unselected_remap_mul_c1"
        assert at.toggle(toggle_key)
        assert at.toggle(toggle_key).label == "Delete unselected labels"
        assert not at.toggle(toggle_key).value

        # button
        button_key = "btn_remap_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Do Label Remap"

        # warning
        assert len(at.warning) == 0

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_do_label_remap(self):
        """Test if _do_label_remap function of TransformLabelRemap in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform_remapped, default_timeout=600).run()

        # Before
        assert at.session_state.data_helper_1.dataset().get_label_cat_names() == [
            "car",
            "bicycle",
            "tom",
            "mary",
        ]
        mapping = at.session_state.mapping
        uploaded_file_1 = at.session_state.uploaded_file_1
        uploaded_file_2 = at.session_state.uploaded_file_2
        mode = "default"

        # Call the _do_label_remap() method
        TransformLabelRemap._do_label_remap(
            at.session_state.data_helper_1,
            mapping,
            uploaded_file_2,
            uploaded_file_1,
            mode,
            delete_unselected=False,
        )

        # After
        expected_results = ["background", "person"]
        assert (
            sorted(at.session_state.data_helper_1.dataset().get_label_cat_names())
            == expected_results
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_label_remap_check_toggle(self):
        """Test if toggle api for label remap method of TransformLabelRemap in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform_remapped, default_timeout=600).run()

        toggle_key = "tg_del_unselected_remap_mul_c1"

        # Before
        assert not at.toggle(toggle_key).value

        at.toggle(toggle_key).set_value(True).run()

        # After
        assert at.toggle(toggle_key).value

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_label_remap_check_button(self):
        """Test if toggle api for label remap method of TransformLabelRemap in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform_remapped, default_timeout=600).run()

        # Before
        assert at.session_state.data_helper_1.dataset().get_label_cat_names() == [
            "car",
            "bicycle",
            "tom",
            "mary",
        ]

        # Click Do Label Remap button
        button_key = "btn_remap_c1"
        at.button(button_key).click().run()
        expected_remapped_labels = ["background", "person"]

        # Before remap : ['car', 'bicycle', 'tom', 'mary']
        # After reamp : ['background', 'person']
        assert (
            sorted(at.session_state.data_helper_1.dataset().get_label_cat_names())
            == expected_remapped_labels
        )


class TransformAggregationTest(TestCase):
    def test_name(self):
        transform = TransformAggregation()
        assert transform.name == "Aggregation"

    def test_info(self):
        transform = TransformAggregation()
        assert (
            transform.info == "This helps to merge subsets within a dataset into a single subset."
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_do_aggregation(self):
        """Test if _do_aggregation function of TransformAggregation in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Before
        before_subsets = list(at.session_state.data_helper_1.dataset().subsets().keys())
        assert sorted(before_subsets) == ["test", "train", "validation"]

        # Call the _do_aggregation() method
        TransformAggregation._do_aggregation(
            at.session_state.data_helper_1,
            selected_subsets=["test", "train", "validation"],
            dst_subset_name="aggregated",
        )

        # After
        expected_subsets = ["aggregated"]
        assert list(at.session_state.data_helper_1.dataset().subsets().keys()) == expected_subsets

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_aggregation_check_multiselect(self):
        """Test if multiselect api for aggregation method of TransformAggregation in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # multiselect
        multiselect_key = "ms_selected_subsets_agg_mul_c1"

        # Before
        subset_list = ["test", "train", "validation"]
        assert sorted(at.multiselect(multiselect_key).value) == subset_list

        # Unselect train
        at.multiselect(multiselect_key).unselect("train").run()

        # After
        assert sorted(at.multiselect(multiselect_key).value) == ["test", "validation"]

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_aggregation_check_text_input(self):
        """Test if text_input api for aggregation method of TransformAggregation in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # text_input
        text_input_key = "ti_dst_subset_name_agg_mul_c1"

        # Before
        assert at.text_input(text_input_key).value == "default"

        # After
        dst_text_input = "test"
        at.text_input(text_input_key).input(dst_text_input).run()
        assert at.text_input(text_input_key).value == dst_text_input

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_aggregation_check_button(self):
        """Test if button api for aggregation method of TransformAggregation in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Before
        before_subsets = sorted(list(at.session_state.data_helper_1.dataset().subsets().keys()))
        assert before_subsets == ["test", "train", "validation"]

        # Click aggregation button
        button_key = "btn_agg_mul_c1"
        at.button(button_key).click().run()

        # After
        expected_subsets = ["default"]
        assert list(at.session_state.data_helper_1.dataset().subsets().keys()) == expected_subsets

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_aggregation_page_open(self):
        """Test if aggregation method of transform in multiple dataset page is opened correctly."""
        at = AppTest.from_function(run_transform_remapped, default_timeout=600).run()

        # multiselect
        multiselect_key = "ms_selected_subsets_agg_mul_c1"
        assert at.multiselect(multiselect_key)
        subset_list = ["test", "train", "validation"]
        assert at.multiselect(multiselect_key).label == "Select subsets to be aggregated"
        assert sorted(at.multiselect(multiselect_key).options) == subset_list
        assert sorted(at.multiselect(multiselect_key).value) == subset_list

        # text_input
        text_input_key = "ti_dst_subset_name_agg_mul_c1"
        assert at.text_input(text_input_key)
        assert at.text_input(text_input_key).label == "Aggregated Subset Name"
        assert at.text_input(text_input_key).value == "default"

        # button
        button_key = "btn_agg_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Do aggregation"


class TransformSplitTest(TestCase):
    def test_name(self):
        transform = TransformSplit()
        assert transform.name == "Split"

    def test_info(self):
        transform = TransformSplit()
        assert (
            transform.info
            == "This helps to divide a dataset into multiple subsets with a given ratio."
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_split_page_open(self):
        """Test if split method of transform in multiple dataset page is opened correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Check page open correctly
        # button
        button_key = "btn_add_subset_split_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Add subset"

        button_key = "btn_do_split_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Do split"

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_add_subset_default_splits(self):
        """Test if _add_subset with default splits in split method of TransformSplit is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        button_key = "btn_add_subset_split_mul_c1"
        at.button(button_key).click().run()
        at.button(button_key).click().run()
        at.button(button_key).click().run()

        assert len(at.session_state["subset_1"]) == 3
        assert at.session_state["subset_1"] == [
            TransformSplit.Split("train", 0.5),
            TransformSplit.Split("val", 0.2),
            TransformSplit.Split("test", 0.3),
        ]

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_split_delete_subset(self):
        """Test if _delete_subset in split method of TransformSplit is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        at.session_state["subset_1"] = [
            TransformSplit.Split("train", 0.5),
            TransformSplit.Split("val", 0.2),
            TransformSplit.Split("test", 0.3),
        ]
        at.run()
        assert len(at.session_state["subset_1"]) == 3

        # Assuming "remove" button key
        at.button(f"btn_subset_remove_split_mul_0_c1").click().run()
        assert len(at.session_state["subset_1"]) == 2
        assert at.session_state["subset_1"] == [
            TransformSplit.Split("val", 0.2),
            TransformSplit.Split("test", 0.3),
        ]

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_do_split(self):
        """Test if _do_split with valid ratios in split method of TransformSplit is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # before
        before_subsets = sorted(list(at.session_state.data_helper_1.dataset().subsets().keys()))
        assert before_subsets == ["test", "train", "validation"]
        assert len(at.session_state.subset_1) == 0

        # Set up initial splits with valid ratios
        at.session_state["subset_1"] = [
            TransformSplit.Split("train", 1),
        ]
        at.run()
        assert len(at.session_state.subset_1) == 1

        # Call the _do_split() method
        TransformSplit._do_split(at.session_state.data_helper_1, at.session_state.subset_1)
        expected_subsets = ["train"]
        assert list(at.session_state.data_helper_1.dataset().subsets().keys()) == expected_subsets

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_do_split_invalid_ratios(self):
        """Test if _do_split with invalid ratios in split method of TransformSplit is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Set up initial splits with invalid ratios (sum not equal to 1)
        at.session_state["subset_1"] = [
            TransformSplit.Split("train", 0.5),
            TransformSplit.Split("val", 0.6),
        ]
        at.run()

        # Call the _do_split() method
        TransformSplit._do_split(at.session_state.data_helper_1, at.session_state.subset_1)

        # Assert that the toast message is displayed
        assert at.toast.values == ["Sum of ratios is expected to be 1!"]

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_split_check_text_input(self):
        """Test if split method of TransformSplit in multiple dataset page is opened correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        button_key = "btn_add_subset_split_mul_c1"
        at.button(button_key).click().run()

        text_input_key = "ti_subset_name_split_mul_0_c1"
        assert at.text_input(text_input_key)
        assert at.text_input(text_input_key).label == "name"
        assert at.text_input(text_input_key).value == "train"

        at.text_input(text_input_key).input("default").run()
        assert at.text_input(text_input_key).value == "default"

        text_input_key = "ti_subset_ratio_split_mul_0_c1"
        assert at.text_input(text_input_key)
        assert at.text_input(text_input_key).label == "ratio"
        assert at.text_input(text_input_key).value == "0.5"

        at.text_input(text_input_key).input("1").run()
        assert at.text_input(text_input_key).value == "1"

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_split_check_button(self):
        """Test if split method of TransformSplit in multiple dataset page is opened correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # before
        before_subsets = sorted(list(at.session_state.data_helper_1.dataset().subsets().keys()))
        assert before_subsets == ["test", "train", "validation"]

        button_key = "btn_add_subset_split_mul_c1"
        at.button(button_key).click().run()

        assert at.session_state.subset_1 == [Split("train", 0.5)]

        text_input_key = "ti_subset_ratio_split_mul_0_c1"
        at.text_input(text_input_key).input("1").run()

        assert at.session_state.subset_1 == [Split("train", 1)]

        button_key = "btn_do_split_mul_c1"
        at.button(button_key).click().run()

        assert list(at.session_state.data_helper_1._dm_dataset.subsets().keys()) == ["train"]


class TransformSubsetRenameTest(TestCase):
    def test_name(self):
        transform = TransformSubsetRename()
        assert transform.name == "Subset Rename"

    def test_info(self):
        transform = TransformSubsetRename()
        assert transform.info == "This helps to rename subset in dataset"

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_subset_rename_page_open(self):
        """Test if subset rename method of TransformSubsetRename in multiple dataset page is opened correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # before
        before_subsets = sorted(list(at.session_state.data_helper_1.dataset().subsets().keys()))
        assert before_subsets == ["test", "train", "validation"]

        # Check page open correctly
        # selectbox
        selectbox_key = "sb_subset_rename_mul_c1"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select a subset to rename"
        assert at.selectbox(selectbox_key).options == list(
            at.session_state.data_helper_1.dataset().subsets().keys()
        )

        # text input
        text_input_key = "ti_subset_rename_mul_c1"
        assert at.text_input(text_input_key)
        assert at.text_input(text_input_key).label == "New subset name"

        # button
        button_key = "btn_subset_rename_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Do Subset Rename"

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_remap_subset(self):
        """Test if _remap_subset function of TransformSubsetRename in multiple dataset page is opened correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # before
        before_subsets = sorted(list(at.session_state.data_helper_1.dataset().subsets().keys()))
        assert before_subsets == ["test", "train", "validation"]

        # Call the _remap_subset() method
        TransformSubsetRename._remap_subset(
            at.session_state.data_helper_1, target_subset="validation", target_name="val"
        )

        expected_subsets = ["test", "train", "val"]
        assert (
            sorted(list(at.session_state.data_helper_1.dataset().subsets().keys()))
            == expected_subsets
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_subset_rename_check_select(self):
        """Test if select api for subset rename method of TransformSubsetRename in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        selectbox_key = "sb_subset_rename_mul_c1"

        # Before
        assert at.selectbox(selectbox_key).value != "train"

        # Unselect train
        at.selectbox(selectbox_key).select("train").run()

        # After
        assert at.selectbox(selectbox_key).value == "train"

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_subset_rename_check_text_input(self):
        """Test if text_input api for subset rename method pf TransformSubsetRename in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # text input
        text_input_key = "ti_subset_rename_mul_c1"

        # Before
        assert not at.text_input(text_input_key).value

        # Unselect train
        at.text_input(text_input_key).input("subset").run()

        # After
        assert at.text_input(text_input_key).value == "subset"

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_subset_rename_check_button(self):
        """Test if button api for subset rename method of TransformSubsetRename in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Set key for each api
        selectbox_key = "sb_subset_rename_mul_c1"
        text_input_key = "ti_subset_rename_mul_c1"
        button_key = "btn_subset_rename_mul_c1"

        # Before
        before_subsets = sorted(list(at.session_state.data_helper_1.dataset().subsets().keys()))
        assert before_subsets == ["test", "train", "validation"]

        # Unselect train
        at.selectbox(selectbox_key).select("validation").run()
        at.text_input(text_input_key).input("val").run()
        at.button(button_key).click().run()

        # After
        expected_subsets = ["test", "train", "val"]
        assert (
            sorted(list(at.session_state.data_helper_1.dataset().subsets().keys()))
            == expected_subsets
        )


class TransformReindexTest(TestCase):
    def test_name(self):
        transform = TransformReindexing()
        assert transform.name == "Reindexing"

    def test_info(self):
        transform = TransformReindexing()
        assert transform.info == "This helps to reidentify all items."

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_reindex_page_open(self):
        """Test if reindexing method of TransformReindexing in multiple dataset page is opened correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Check page open correctly
        # number_input
        number_input_key = "ni_start_idx_reindex_mul_c1"
        assert at.number_input(number_input_key)
        assert at.number_input(number_input_key).min == 0

        # button
        button_key = "btn_start_idx_reindex_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Set IDs from 0"

        button_key = "btn_start_media_name_reindex_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Set IDs with media name"

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex_check_number_input(self):
        """Test if number_input api for reindexing method of TransformReindexing in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # number_input
        number_input_key = "ni_start_idx_reindex_mul_c1"

        # Before
        assert at.number_input(number_input_key).value == 0

        # Set value
        at.number_input(number_input_key).increment().run()

        # After
        assert at.number_input(number_input_key).value == 1

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex_with_index(self):
        """Test if _reindex_with_index function of TransformReindexing in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Before
        assert at.session_state.data_helper_1.dataset().__getitem__(0).id != "10"

        # Call the _reindex_with_index() method
        TransformReindexing._reindex_with_index(at.session_state.data_helper_1, start_index=10)

        # Assert that the IDs have been updated
        assert at.session_state.data_helper_1.dataset().__getitem__(0).id == "10"

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex_check_reindex_idx_button(self):
        """Test if button api for reindexing method based on index of TransformReindexing in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        button_key = "btn_start_idx_reindex_mul_c1"

        # Before
        assert at.session_state.data_helper_1.dataset().__getitem__(0).id != "0"

        # Click button
        at.button(button_key).click().run()

        # After
        assert at.session_state.data_helper_1.dataset().__getitem__(0).id == "0"

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex_with_media_name(self):
        """Test if _reindex_with_image function of TransformReindexing in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform_imagenet, default_timeout=600).run()

        # Before
        item_ids = sorted([item.id for item in at.session_state.data_helper_1.dataset()])
        assert item_ids[0] == "label_0:label_0_1"

        # Call the _reindex_with_image() method
        TransformReindexing._reindex_with_image(at.session_state.data_helper_1)

        # Assert that the IDs have been updated
        item_ids = sorted([item.id for item in at.session_state.data_helper_1.dataset()])
        assert item_ids[0] == "label_0_1"

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex_check_reindex_media_name_button(self):
        """Test if button api for reindexing method based on media name of TransformReindexing in single dataset is worked out correctly."""
        at = AppTest.from_function(run_transform_imagenet, default_timeout=600).run()

        button_key = "btn_start_media_name_reindex_mul_c1"

        # Before
        item_ids = sorted([item.id for item in at.session_state.data_helper_1.dataset()])
        assert item_ids[0] == "label_0:label_0_1"

        # Click button
        at.button(button_key).click().run()

        # After
        item_ids = sorted([item.id for item in at.session_state.data_helper_1.dataset()])
        assert item_ids[0] == "label_0_1"


class TransformFiltrationTest(TestCase):
    def test_name(self):
        transform = TransformFiltration()
        assert transform.name == "Filtration"

    def test_info(self):
        transform = TransformFiltration()
        assert transform.info == "This helps to filter some items or annotations within a dataset."

    def test_link(self):
        # Assuming `self._datumaro_doc` is set correctly
        transform = TransformFiltration()
        assert (
            transform.link
            == f"{transform._datumaro_doc}/command-reference/context_free/filter.html"
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_filtration_page_open(self):
        """Test if filtration method of TransformFiltration in multiple dataset page is opened correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Check page open correctly
        # selectbox
        selectbox_key = "sb_selected_mode_filt_mul_c1"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select filtering mode"
        assert at.selectbox(selectbox_key).options == ["items", "annotations", "items+annotations"]

        # text input
        text_input_key = "ti_filter_expr_filt_mul_c1"
        assert at.text_input(text_input_key)
        assert (
            at.text_input(text_input_key).label
            == "Enter XML filter expression ([XPATH](https://devhints.io/xpath))"
        )
        assert at.text_input(text_input_key).placeholder == 'Eg. /item[subset="train"]'
        assert at.text_input(text_input_key).value == None

        # button
        button_key = "btn_filter_filt_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Filter dataset"

        # toggle
        toggle_key = "tg_show_xml_filt_mul_c1"
        assert at.toggle(toggle_key)
        assert not at.toggle(toggle_key).value

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_filtration_check_selectbox(self):
        """Test if selectbox api of TransformFiltration in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # selectbox
        selectbox_key = "sb_selected_mode_filt_mul_c1"

        # Before
        assert at.selectbox(selectbox_key).value == "items"

        # Select another option
        expected = "annotations"
        at.selectbox(selectbox_key).select(expected).run()

        # After
        assert at.selectbox(selectbox_key).value == expected

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_filtration_check_text_input(self):
        """Test if text_input api of TransformFiltration in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # text input
        text_input_key = "ti_filter_expr_filt_mul_c1"

        # Before
        assert at.text_input(text_input_key).value == None

        # Set text input
        expected = "/item[label='bicycle']"
        at.text_input(text_input_key).input(expected).run()

        # After
        assert at.text_input(text_input_key).value == expected

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_filtration_check_toggle(self):
        """Test if toggle api of TransformFiltration in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # toggle
        toggle_key = "tg_show_xml_filt_mul_c1"

        # Before
        assert not at.toggle(toggle_key).value

        # Turn on toggle
        at.toggle(toggle_key).set_value(True).run()

        # After
        assert at.toggle(toggle_key).value

        # selectbox
        selectbox_key = "sb_selected_subset_filt_mul_c1"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select a subset"
        subset_options = sorted(at.selectbox(selectbox_key).options)
        assert subset_options == sorted(list(at.session_state.data_helper_1.subset_to_ids().keys()))
        selected_subset = subset_options[0]

        selectbox_key = "sb_selected_id_filt_mul_c1"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select an item"
        assert (
            at.selectbox(selectbox_key).options
            == at.session_state.data_helper_1.subset_to_ids()[selected_subset]
        )

        # code
        assert len(at.code) == 1

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_filtration_check_button(self):
        """Test if button api of TransformFiltration in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # button
        button_key = "btn_filter_filt_mul_c1"

        # Before
        assert len(at.session_state.data_helper_1.dataset()) == 4

        text_input_key = "ti_filter_expr_filt_mul_c1"
        at.text_input(text_input_key).input("/item[label='bicycle']").run()

        # Click button
        at.button(button_key).click().run()

        # After
        assert len(at.session_state.data_helper_1.dataset()) == 0

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_filter_dataset(self):
        """Test if _filter_dataset function of TransformFiltration in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Before
        assert len(at.session_state.data_helper_1.dataset()) == 4

        # Call the _filter_dataset() method with valid filter expression and mode
        TransformFiltration._filter_dataset(
            at.session_state.data_helper_1,
            filter_expr="/item[label='bicycle']",
            selected_mode="items",
        )

        # Check filtered dataset
        assert len(at.session_state.data_helper_1.dataset()) == 0


class TransformRemoveTest(TestCase):
    def test_name(self):
        transform = TransformRemove()
        assert transform.name == "Remove"

    def test_info(self):
        transform = TransformRemove()
        assert transform.info == "This helps to remove some items or annotations within a dataset."

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_remove_page_open(self):
        """Test if remove method of transform in multiple dataset page is opened correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # selectbox
        selectbox_key = "sb_select_subset_remove_mul_c1"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select a subset"
        assert sorted(at.selectbox(selectbox_key).options) == sorted(
            list(at.session_state.data_helper_1.subset_to_ids().keys())
        )
        selected_subset = at.selectbox(selectbox_key).value

        selectbox_key = "sb_select_id_remove_mul_c1"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select an item"
        assert (
            at.selectbox(selectbox_key).options
            == at.session_state.data_helper_1.subset_to_ids()[selected_subset]
        )
        selected_id = at.selectbox(selectbox_key).value
        selected_item = at.session_state.data_helper_1.dataset().get(selected_id, selected_subset)

        selectbox_key = "sb_select_ann_id_mul_c1"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select an annotation"
        assert at.selectbox(selectbox_key).options == [
            "All",
        ] + sorted(list({str(ann.id) for ann in selected_item.annotations}))

        # button
        button_key = "btn_remove_item_remove_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Remove item"

        button_key = "btn_remove_annot_remove_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Remove annotation"

        # metric
        assert at.metric[0].label == "Items in the dataset"
        assert at.metric[0].value == "4"
        assert at.metric[1].label == "Annotation in the item"
        assert at.metric[1].value == "2"

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_remove_check_selectbox(self):
        """Test if selectbox api of TransformRemove in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Before
        selectbox_subset_key = "sb_select_subset_remove_mul_c1"
        selected_subset = at.selectbox(selectbox_subset_key).value
        assert selected_subset == "test"

        selectbox_id_key = "sb_select_id_remove_mul_c1"
        selected_id = at.selectbox(selectbox_id_key).value
        assert selected_id == "c"

        selectbox_ann_id_key = "sb_select_ann_id_mul_c1"
        selected_ann_id = at.selectbox(selectbox_ann_id_key).value
        assert selected_ann_id == "All"

        # Set subset as train
        at.selectbox(selectbox_subset_key).select("train").run()

        # After
        assert at.selectbox(selectbox_subset_key).value == "train"
        assert at.selectbox(selectbox_id_key).options == ["a", "b"]
        assert at.selectbox(selectbox_id_key).value == "a"

        # Set item as b
        at.selectbox(selectbox_id_key).select("b").run()

        assert at.selectbox(selectbox_id_key).value == "b"
        assert at.selectbox(selectbox_ann_id_key).options == ["All", "0", "1", "2", "3"]

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_remove_item(self):
        """Test if _remove_item function of TransformRemove in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Before
        assert len(at.session_state.data_helper_1.dataset()) == 4
        assert at.session_state.data_helper_1.dataset().get_annotations() == 7

        # Call the _remove_item() method
        TransformRemove._remove_item(
            at.session_state.data_helper_1, selected_id="c", selected_subset="test"
        )

        # After
        assert len(at.session_state.data_helper_1.dataset()) == 3
        assert at.session_state.data_helper_1.dataset().get_annotations() == 5

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_remove_check_remove_item_button(self):
        """Test if button api for removing item of TransformRemove in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Before
        assert len(at.session_state.data_helper_1.dataset()) == 4
        assert at.session_state.data_helper_1.dataset().get_annotations() == 7

        # Click Remove item button
        button_key = "btn_remove_item_remove_mul_c1"
        at.button(button_key).click().run()

        # After
        assert len(at.session_state.data_helper_1.dataset()) == 3
        assert at.session_state.data_helper_1.dataset().get_annotations() == 5

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_remove_annotation_all(self):
        """Test if _remove_annotation function of TransformRemove in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Before
        assert len(at.session_state.data_helper_1.dataset()) == 4
        assert at.session_state.data_helper_1.dataset().get_annotations() == 7

        # Call the _remove_annotation() method with "All"
        TransformRemove._remove_annotation(
            at.session_state.data_helper_1,
            selected_id="c",
            selected_subset="test",
            selected_ann_id="All",
        )

        # After
        assert len(at.session_state.data_helper_1.dataset()) == 4
        assert at.session_state.data_helper_1.dataset().get_annotations() == 5

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_remove_annotation_specific(self):
        """Test if _remove_annotation function with specific annotation id of TransformRemove in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Before
        assert len(at.session_state.data_helper_1.dataset()) == 4
        assert at.session_state.data_helper_1.dataset().get_annotations() == 7

        # Call the _remove_annotation() method with a specific annotation ID
        TransformRemove._remove_annotation(
            at.session_state.data_helper_1,
            selected_id="b",
            selected_subset="train",
            selected_ann_id=0,
        )

        # After
        assert len(at.session_state.data_helper_1.dataset()) == 4
        assert at.session_state.data_helper_1.dataset().get_annotations() == 3

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_remove_check_remove_annotation_button(self):
        """Test if button api for removing annotation of TransformRemove in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Before
        assert len(at.session_state.data_helper_1.dataset()) == 4
        assert at.session_state.data_helper_1.dataset().get_annotations() == 7

        # Click Remove annotation button
        button_key = "btn_remove_annot_remove_mul_c1"
        at.button(button_key).click().run()

        # After
        assert len(at.session_state.data_helper_1.dataset()) == 4
        assert at.session_state.data_helper_1.dataset().get_annotations() == 5


class TransformAutoCorrectionTest(TestCase):
    def test_name(self):
        transform = TransformAutoCorrection()
        assert transform.name == "Auto-correction"

    def test_info(self):
        transform = TransformAutoCorrection()
        assert transform.info == "This helps to correct a dataset and clean up validation report."

    def test_link(self):
        # Assuming `self._datumaro_doc` is set correctly
        transform = TransformAutoCorrection()
        assert (
            transform.link
            == f"{transform._datumaro_doc}/jupyter_notebook_examples/notebooks/12_correct_dataset.html"
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_auto_correction_page_open(self):
        """Test if auto-correction method of transform in multiple dataset page is opened correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # selectbox
        selectbox_key = "sb_select_task_auto_corr_mul_c1"
        assert at.selectbox(selectbox_key)
        assert at.selectbox(selectbox_key).label == "Select a task"
        assert at.selectbox(selectbox_key).options == [
            "Classification",
            "Detection",
            "Segmentation",
        ]

        # button
        button_key = "btn_auto_corr_mul_c1"
        assert at.button(button_key)
        assert at.button(button_key).label == "Correct a dataset"

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_recommend_task(self):
        """Test if _recommend_task function of TransformAutoCorrection in multiple dataset is worked out correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        assert (
            TransformAutoCorrection._recommend_task(at.session_state.data_helper_1.get_ann_stats())
            == "Classification"
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_validation_summary_with_reports(self):
        """Test if _get_validation_summary function of TransformAutoCorrection in multiple dataset is worked out correctly."""

        validation_reports = {
            "validation_reports": [
                {
                    "anomaly_type": "MissingAnnotation",
                    "description": "Item needs 'label' annotation(s), but not found.",
                    "severity": "warning",
                    "item_id": "0",
                    "subset": "train",
                },
            ]
        }
        summary = TransformAutoCorrection._get_validation_summary(validation_reports)
        assert summary == {
            "warning": {"MissingAnnotation": 1},
        }

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_validation_summary_empty_reports(self):
        """Test if _get_validation_summary function with empty reports of TransformAutoCorrection in multiple dataset is worked out correctly."""
        reports = {"validation_reports": []}
        summary = TransformAutoCorrection._get_validation_summary(reports)
        assert summary is None

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_df(self):
        """Test if _get_df function of TransformAutoCorrection in multiple dataset is worked out correctly."""

        summary = {
            "error": {"UndefinedAttribute": 1},
            "warning": {"MissingAnnotation": 2},
        }
        result = TransformAutoCorrection._get_df(summary)
        expected = pd.DataFrame(
            {
                "severity": ["error", "warning"],
                "anomaly_type": ["UndefinedAttribute", "MissingAnnotation"],
                "count": [1, 2],
            }
        )

        # Assert expected DataFrame contents
        assert expected.equals(result)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_compared_df(self):
        """Test if _get_compared_df function of TransformAutoCorrection in multiple dataset is worked out correctly."""

        summary1 = {
            "error": {"UndefinedAttribute": 2, "MultiLabelAnnotations": 1},
            "warning": {"MissingAnnotation": 3},
        }
        summary2 = {
            "error": {"UndefinedAttribute": 1, "MultiLabelAnnotations": 1},
            "warning": {"MissingAnnotation": 0},
            "info": {"FewSamplesInLabel": 4},
        }

        result = TransformAutoCorrection._get_compared_df(summary1, summary2)
        expected = pd.DataFrame(
            {
                "severity": ["error", "error", "warning", "info"],
                "anomaly_type": [
                    "MultiLabelAnnotations",
                    "UndefinedAttribute",
                    "MissingAnnotation",
                    "FewSamplesInLabel",
                ],
                "count(src)": [1, 2, 3, 0],
                "count(dst)": [1, 1, 0, 4],
            }
        )

        # Assert expected DataFrame contents
        assert expected.equals(result)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_auto_correct_check_seleckbox(self):
        """Test if selectbox api of TransformAutoCorrection in multiple dataset is worked out correctly."""

        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Before
        selectbox_key = "sb_select_task_auto_corr_mul_c1"
        selected_task = at.selectbox(selectbox_key).value
        assert selected_task == "Classification"
        assert list(at.dataframe.values[1]["severity"]) == ["error", "error", "warning", "info"]
        assert sorted(list(at.dataframe.values[1]["anomaly_type"])) == [
            "FewSamplesInLabel",
            "MissingAnnotation",
            "MultiLabelAnnotations",
            "UndefinedAttribute",
        ]
        assert list(at.dataframe.values[1]["count"]) == [3, 1, 1, 1]

        # Set task as Detection
        at.selectbox(selectbox_key).select("Detection").run()

        # After
        assert at.selectbox(selectbox_key).value == "Detection"
        assert list(at.dataframe.values[1]["severity"]) == ["warning", "warning", "info"]
        assert sorted(list(at.dataframe.values[1]["anomaly_type"])) == [
            "ImbalancedLabels",
            "LabelDefinedButNotFound",
            "MissingAnnotation",
        ]
        assert list(at.dataframe.values[1]["count"]) == [4, 4, 1]

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_auto_correct_check_button(self):
        """Test if button api of TransformAutoCorrection in multiple dataset is worked out correctly."""

        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Before
        assert list(at.dataframe.values[1].keys()) == ["severity", "anomaly_type", "count"]
        assert list(at.dataframe.values[1]["severity"]) == ["error", "error", "warning", "info"]
        assert sorted(list(at.dataframe.values[1]["anomaly_type"])) == [
            "FewSamplesInLabel",
            "MissingAnnotation",
            "MultiLabelAnnotations",
            "UndefinedAttribute",
        ]
        assert list(at.dataframe.values[1]["count"]) == [3, 1, 1, 1]

        # Set task as Detection
        button_key = "btn_auto_corr_mul_c1"
        at.button(button_key).click().run()

        # After
        assert list(at.dataframe.values[1].keys()) == [
            "severity",
            "anomaly_type",
            "count(src)",
            "count(dst)",
        ]
        assert list(at.dataframe.values[1]["severity"]) == [
            "error",
            "error",
            "warning",
            "warning",
            "info",
            "info",
            "info",
            "info",
        ]
        assert sorted(list(at.dataframe.values[1]["anomaly_type"])) == [
            "FewSamplesInAttribute",
            "FewSamplesInLabel",
            "ImbalancedLabels",
            "LabelDefinedButNotFound",
            "MissingAnnotation",
            "MultiLabelAnnotations",
            "OnlyOneAttributeValue",
            "UndefinedAttribute",
        ]
        assert list(at.dataframe.values[1]["count(src)"]) == [1, 3, 0, 1, 0, 1, 0, 0]
        assert list(at.dataframe.values[1]["count(dst)"]) == [0, 0, 1, 0, 3, 3, 1, 3]
