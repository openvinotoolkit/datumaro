# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from typing import NamedTuple
from unittest import TestCase

from streamlit.testing.v1 import AppTest

from datumaro.components.annotation import AnnotationType

from tests.requirements import Requirements, mark_requirement

cwd = os.getcwd()
app_path = os.path.join(cwd, "gui", "streamlit_app.py")

import sys

sys.path.append(os.path.join(cwd, "gui"))

multiple_dataset_key = "datumaro_gui.utils.page_p_:telescope: Multiple datasets"


def run_transform():
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
    data_dict = {uploaded_file_1: matches, uploaded_file_2: matches}
    state.mapping = pd.DataFrame(data_dict)

    tabs.call_transform()


class Split(NamedTuple):
    subset: str
    ratio: float


class TransformTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_page_open(self):
        """Test if the session state is initialized correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # select dataset
        assert at.selectbox(key="trans_selectbox_d1").index == 0
        assert at.selectbox(key="trans_selectbox_d2").index == 1

        # Remap
        assert not at.session_state.matched
        assert not at.session_state.mapping
        assert not at.button(key="remap_btn_c1").value
        assert not at.toggle(key="remap_del_tog_c1").value
        assert not at.button(key="remap_btn_c2").value
        assert not at.toggle(key="remap_del_tog_c2").value
        assert len(at.warning) == 2

        # Aggregation
        assert at.multiselect(key="aggre_subset_list_c1").value == ["test", "train", "validation"]
        assert at.text_input(key="aggre_subset_name_c1").value == "default"
        assert not at.button(key="aggre_subset_btn_c1").value
        assert at.multiselect(key="aggre_subset_list_c2").value == ["test", "train"]
        assert at.text_input(key="aggre_subset_name_c2").value == "default"
        assert not at.button(key="aggre_subset_btn_c2").value

        # Split
        assert not at.button(key="add_subset_btn_c1").value
        assert not at.button(key="split_btn_c1").value
        assert not at.button(key="add_subset_btn_c2").value
        assert not at.button(key="split_btn_c2").value

        # Reindex
        assert at.number_input(key="item_reindex_input_c1").value == 0
        assert not at.button(key="item_reindex_btn_c1").value
        assert not at.button(key="item_media_name_btn_c1").value
        assert at.number_input(key="item_reindex_input_c2").value == 0
        assert not at.button(key="item_reindex_btn_c2").value
        assert not at.button(key="item_media_name_btn_c2").value

        # Filter
        assert at.selectbox(key="filter_selected_mode_c1").index == 0
        assert not at.text_input(key="filter_expr_c1").value
        assert not at.button(key="filter_btn_c1").value
        assert not at.toggle(key="show_xml_c1").value
        assert at.selectbox(key="filter_selected_mode_c2").index == 0
        assert not at.text_input(key="filter_expr_c2").value
        assert not at.button(key="filter_btn_c2").value
        assert not at.toggle(key="show_xml_c2").value

        # Remove
        assert at.selectbox(key="rm_selected_subset_c1").index == 0
        assert at.selectbox(key="rm_selected_id_c1").index == 0
        assert at.selectbox(key="rm_selected_ann_id_c1").index == 0
        assert not at.button(key="rm_item_btn_c1").value
        assert not at.button(key="rm_ann_btn_c1").value
        assert at.selectbox(key="rm_selected_subset_c2").index == 0
        assert at.selectbox(key="rm_selected_id_c2").index == 0
        assert at.selectbox(key="rm_selected_ann_id_c2").index == 0
        assert not at.button(key="rm_item_btn_c2").value
        assert not at.button(key="rm_ann_btn_c2").value
        assert len(at.metric) == 4

        # Auto correct
        assert at.selectbox(key="correct_task_c1").value
        assert not at.button(key="correct_btn_c1").value
        assert at.selectbox(key="correct_task_c2").value
        assert not at.button(key="correct_btn_c2").value

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_remap(self):
        """Test if label of dataset is remapped correctly."""
        at = AppTest.from_function(run_transform_remapped, default_timeout=600).run()

        expected_matches = ["bicycle", "car"]

        # mapping
        assert not at.session_state.mapping.empty
        assert (
            at.session_state.mapping.columns
            == [at.session_state.uploaded_file_1, at.session_state.uploaded_file_2]
        ).all()
        assert (
            at.session_state.mapping[at.session_state.uploaded_file_1].to_numpy()
            == expected_matches
        ).all()
        assert (
            at.session_state.mapping[at.session_state.uploaded_file_2].to_numpy()
            == expected_matches
        ).all()

        # matched
        assert at.session_state.matched
        assert at.session_state.matched == expected_matches

        # Click Do Label Remap button
        at.button("remap_btn_c1").click().run(timeout=180)
        expected_remapped_labels = ["background", "bicycle", "car"]

        # Before remap : ['car', 'bicycle', 'tom', 'mary']
        # After reamp : ['car', 'bicycle', 'background']
        assert (
            sorted(
                at.session_state.data_helper_1._dm_dataset.categories()[
                    AnnotationType.label
                ]._indices.keys()
            )
            == expected_remapped_labels
        )

        # Check toast

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_aggregation(self):
        """Test if subsets of dataset are aggregated correctly."""
        at = AppTest.from_function(run_transform_remapped, default_timeout=600).run()

        # Select only "test" and "validation"
        at.multiselect("aggre_subset_list_c1").unselect("train").run()
        assert at.multiselect(key="aggre_subset_list_c1").value == ["test", "validation"]

        # Set subset name as "test"
        at.text_input("aggre_subset_name_c1").input("test").run()
        assert at.text_input(key="aggre_subset_name_c1").value == "test"

        # Before subset list : ["test", "train", "validation"]
        assert sorted(at.session_state.data_helper_1._dm_dataset.subsets().keys()) == [
            "test",
            "train",
            "validation",
        ]

        # Click Do Aggregation button
        at.button("aggre_subset_btn_c1").click().run()

        # After subset list : ["test", "train"]
        assert sorted(at.session_state.data_helper_1._dm_dataset.subsets().keys()) == [
            "test",
            "train",
        ]

        # Check toast

    def test_transform_split(self):
        """Test if the dataset is split into subsets correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Check st.subset empty
        assert not at.session_state.subset_1

        # Click Add subset button
        at.button("add_subset_btn_c1").click().run()

        # Check st.subset
        expected_subset = [Split("train", 0.5)]
        assert at.session_state.subset_1 == expected_subset

        # Check subset name text_input
        assert at.text_input("subset_name_0_c1")
        at.text_input("subset_name_0_c1").input("train").run()

        # Check subset ratio text_input
        assert at.text_input("subset_ratio_0_c1")
        at.text_input("subset_ratio_0_c1").input("1").run()

        # Check st.subset
        expected_subset = [Split("train", 1)]
        assert at.session_state.subset_1 == expected_subset

        # Before split : ['train']
        before_subsets = ["test", "train", "validation"]
        assert list(at.session_state.data_helper_1._dm_dataset.subsets().keys()) == before_subsets

        # Click Do Split button
        at.button("split_btn_c1").click().run()

        # After split : ['train']
        assert list(at.session_state.data_helper_1._dm_dataset.subsets().keys()) == ["train"]

        # Check toast

    def test_transform_subset_rename(self):
        """Test if a subset of the dataset is renamed correctly."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        # Before subset rename : ['test', 'train', 'validation']
        before_subsets = ["test", "train", "validation"]
        assert list(at.session_state.data_helper_1.dataset().subsets().keys()) == before_subsets

        # Rename subset 'validation' to 'val'
        at.selectbox("subset_rename_sb_c1").select_index(2).run()
        at.text_input("subset_rename_ti_c1").input("val").run

        # Click Do Subset Rename button
        at.button("subset_rename_btn_c1").click().run()

        # After subset rename : ['test', 'train', 'val']
        after_subsets = ["test", "train", "val"]
        assert list(at.session_state.data_helper_1.dataset().subsets().keys()) == after_subsets

        # Check toast

    def test_transform_reindex(self):
        """Test if the dataset item is reindexed correctly using each method."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        assert not next(iter(at.session_state.data_helper_1._dm_dataset)).id == "0"

        # Click Set IDs from number
        at.button("item_reindex_btn_c1").click().run()
        assert next(iter(at.session_state.data_helper_1._dm_dataset)).id == "0"

        # Click Set IDs with media name
        at.button("item_media_name_btn_c1").click().run()
        assert next(iter(at.session_state.data_helper_1._dm_dataset)).id == "c"

        # Check toast

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_filter(self):
        """Test if the dataset item is filtered correcly with xml."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        assert len(at.session_state.data_helper_1._dm_dataset) == 4

        # Select items mode
        at.selectbox("filter_selected_mode_c1").select_index(0).run()
        at.text_input("filter_expr_c1").input("/item[subset='train']").run()
        at.button("filter_btn_c1").click().run()

        assert len(at.session_state.data_helper_1._dm_dataset) == 2

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_remove_item(self):
        """Test if the dataset item is removed correcly using each method."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        assert len(at.session_state.data_helper_1._dm_dataset) == 4

        # Select annotation mode
        at.selectbox("rm_selected_subset_c1").select_index(0).run()
        at.selectbox("rm_selected_id_c1").select_index(0).run()
        at.selectbox("rm_selected_ann_id_c1").select_index(0).run()
        at.button("rm_item_btn_c1").click().run()

        assert len(at.session_state.data_helper_1._dm_dataset) == 3

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_remove_annotation(self):
        """Test if the annotation of the dataset item is removed correcly using each method."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        assert len(at.session_state.data_helper_1._dm_dataset) == 4
        assert at.session_state.data_helper_1._dm_dataset.get_annotations() == 7

        # Select annotation mode
        at.selectbox("rm_selected_subset_c1").select_index(0).run()
        at.selectbox("rm_selected_id_c1").select_index(0).run()
        at.selectbox("rm_selected_ann_id_c1").select_index(0).run()
        at.button("rm_ann_btn_c1").click().run()

        assert len(at.session_state.data_helper_1._dm_dataset) == 4
        assert at.session_state.data_helper_1._dm_dataset.get_annotations() == 5

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_auto_correct(self):
        """Test if the dataset is automatically corrected based on the validation result."""
        at = AppTest.from_function(run_transform, default_timeout=600).run()

        assert not at.session_state.correct_reports_1
        assert len(at.session_state.data_helper_1._dm_dataset) == 4
        assert at.session_state.data_helper_1._dm_dataset.get_annotated_items() == 3
        assert at.session_state.data_helper_1._dm_dataset.get_annotations() == 7
        assert list(at.session_state.data_helper_1._dm_dataset.subsets().keys()) == [
            "test",
            "train",
            "validation",
        ]
        assert (
            at.session_state.data_helper_1._dm_dataset.get_subset("train").get_annotated_items()
            == 2
        )
        assert at.session_state.data_helper_1._dm_dataset.get_subset("train").get_annotations() == 5

        # Select task
        at.selectbox("correct_task_c1").select_index(0).run()
        at.button("correct_btn_c1").click().run()

        # assert at.session_state.correct_reports_1
        assert len(at.session_state.data_helper_1._dm_dataset) == 2
        assert at.session_state.data_helper_1._dm_dataset.get_annotated_items() == 2
        assert at.session_state.data_helper_1._dm_dataset.get_annotations() == 3
        assert list(at.session_state.data_helper_1._dm_dataset.subsets().keys()) == [
            "test",
            "train",
        ]
        assert (
            at.session_state.data_helper_1._dm_dataset.get_subset("train").get_annotated_items()
            == 1
        )
        assert at.session_state.data_helper_1._dm_dataset.get_subset("train").get_annotations() == 1
