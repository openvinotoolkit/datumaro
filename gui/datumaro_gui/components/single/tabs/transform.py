# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import streamlit as st
import streamlit_antd_components as sac
from datumaro_gui.utils.dataset.data_loader import SingleDatasetHelper
from streamlit import session_state as state
from streamlit_elements import elements


def main():
    data_helper: SingleDatasetHelper = state["data_helper"]
    dataset = data_helper.dataset()
    with elements("transform"):
        with st.expander("Category Management"):
            sac.divider(label="Label remapping", icon="map", align="center", bold=False)
            st.info("This helps to remap labels of dataset.")
            st.write("Check mapping")
            st.warning("Please mapping category first")

        with st.expander("Subset Management"):
            sac.divider(label="Aggregation", icon="columns", align="center", bold=False)
            st.info(
                "This helps to merge all subsets within a dataset into a single **default** subset."
            )
            aggre_subset_btn = st.button("Do aggregation")
            if aggre_subset_btn:
                dataset = data_helper.aggregate(from_subsets=dataset.subsets(), to_subset="default")
                st.toast("Aggregation Success!", icon="🎉")

            sac.divider(label="Split", icon="columns-gap", align="center", bold=False)
            st.info("This helps to divide a dataset into multiple subsets with a given ratio.")
            col1, col2 = st.columns(9)[:2]
            with col1:
                add_subset_btn = st.button("Add subset", use_container_width=True)
            with col2:
                split_btn = st.button("Do split", use_container_width=True)

            if add_subset_btn:
                state["subset"] += 1

            name, ratio = st.columns(5)[:2]
            splits = []
            for idx in range(state["subset"]):
                with name:
                    subset_name = st.text_input(
                        key=f"subset_name_{idx}", label="Enter subset name", value="train"
                    )
                with ratio:
                    subset_ratio = st.text_input(
                        key=f"subset_ratio_{idx}", label="Enter subset ratio", value=0.5
                    )
                splits.append((subset_name, float(subset_ratio)))

            if split_btn:
                dataset = data_helper.transform("random_split", splits=splits)
                state["subset"] = 1
                st.toast("Split Success!", icon="🎉")

        with st.expander("Item Management"):
            sac.divider(label="Reindexing", icon="stickies-fill", align="center", bold=False)
            st.info("This helps to reidentify all items.")
            col1, col2 = st.columns(6)[:2]
            with col1:
                item_reindex_btn = st.button("Set IDs from 0", use_container_width=True)

            with col2:
                item_media_name_btn = st.button("Set IDs with media name", use_container_width=True)

            if item_reindex_btn:
                dataset = data_helper.transform("reindex", start=0)
                st.toast("Reindex Success!", icon="🎉")

            if item_media_name_btn:
                dataset = data_helper.transform("id_from_image_name")
                st.toast("Reindex Success!", icon="🎉")

            sac.divider(label="Filtration", icon="funnel-fill", align="center", bold=False)
            st.info("This helps to filter some items or annotations within a dataset.")
            mode, filter_ = st.columns(2)
            with mode:
                selected_mode = st.selectbox(
                    "Select filtering mode",
                    ["items", "annotations", "items+annotations", "annotations+items"],
                )
            with filter_:
                filter_expr = st.text_input(
                    "Enter XML filter expression",
                    disabled=False,
                    placeholder='Eg. /item[subset="train"]',
                    value=None,
                )
            col1 = st.columns(6)[0]
            filter_btn = col1.button("Filter dataset")
            if selected_mode and filter_expr and filter_btn:
                filter_args_dict = {
                    "items": {},
                    "annotations": {"filter_annotations": True},
                    "items+annotations": {
                        "filter_annotations": True,
                        "remove_empty": True,
                    },
                    "annotations+items": {
                        "filter_annotations": True,
                        "remove_empty": True,
                    },
                }
                filter_args = filter_args_dict.get(selected_mode, None)
                dataset = data_helper.filter(
                    filter_expr, filter_args
                )  # dataset.filter(filter_expr, **filter_args)
                st.toast("Filter Success!", icon="🎉")

            sac.divider(label="Remove", icon="eraser-fill", align="center", bold=False)
            st.info("This helps to remove some items or annotations within a dataset.")
            subset, item, annotation = st.columns(5)[:3]

            with subset:
                selected_subset = st.selectbox("Select a subset:", dataset.subsets())

            with item:
                ids = [item.id for item in dataset.get_subset(selected_subset)]
                selected_id = st.selectbox("Select a subset item:", ids)

            with annotation:
                item = dataset.get(selected_id, selected_subset)
                ann_ids = [
                    "All",
                ] + [ann.id for ann in item.annotations]
                selected_ann_id = st.selectbox("Select a item annotation:", ann_ids)

            col1, col2 = st.columns(6)[:2]
            with col1:
                rm_item_btn = st.button("Remove item", use_container_width=True)

            with col2:
                rm_ann_btn = st.button("Remove annotation", use_container_width=True)

            if rm_item_btn:
                dataset = data_helper.transform(
                    "remove_items", ids=[(selected_id, selected_subset)]
                )
                st.toast("Remove Success!", icon="🎉")

            if rm_ann_btn:
                if selected_ann_id == "All":
                    dataset = data_helper.transform(
                        "remove_annotations", ids=[(selected_id, selected_subset)]
                    )
                else:
                    dataset = data_helper.transform(
                        "remove_annotations", ids=[(selected_id, selected_subset, selected_ann_id)]
                    )
                st.toast("Success!", icon="🎉")

            sac.divider(label="Auto-correction", icon="hammer", align="center", bold=False)
            st.info("This helps to correct a dataset and clean up validation report.")

            col1, col2 = st.columns(6)[:2]
            with col1:
                correct_btn = st.button("Correct a dataset", use_container_width=True)

            if correct_btn:
                dataset = data_helper.transform("correct", reports=state["reports"])
                st.toast("Correction Success!", icon="🎉")
