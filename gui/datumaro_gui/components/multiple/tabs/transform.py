# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
from types import SimpleNamespace

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from datumaro_gui.utils.dataset.data_loader import MultipleDatasetHelper
from datumaro_gui.utils.dataset.info import get_category_info, get_subset_info
from datumaro_gui.utils.drawing import Dashboard, Pie, Radar
from datumaro_gui.utils.drawing.css import box_style, btn_style
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.annotation import AnnotationType


def render_dataset_management_section(
    col_name,
    data_helper,
    uploaded_file_1,
    uploaded_file_2,
    mapping,
    identicals,
    state_subset_key,
    state_report_key,
):
    dataset = data_helper.dataset()
    with st.expander("Category Management"):
        ######## Remap
        sac.divider(
            label="Label remapping", icon="map", align="center", bold=False, key=f"remap_{col_name}"
        )
        st.info("This helps to remap labels of dataset.")
        st.write("Check mapping")
        st.warning("Please mapping category first") if mapping.empty else st.dataframe(
            mapping, use_container_width=True
        )
        remap_btn = st.button(
            "Do Label Remap", use_container_width=True, key=f"remap_btn_{col_name}"
        )
        if remap_btn:
            mapping_dict = (
                dict(zip(mapping[uploaded_file_2], mapping[uploaded_file_1]))
                if col_name == "c2"
                else dict(zip(mapping[uploaded_file_1], mapping[uploaded_file_2]))
            )
            for label in dataset.categories()[AnnotationType.label]:
                if label.name in identicals or label.name in mapping_dict:
                    continue
                mapping_dict.update({label.name: "background"})
            dataset = dataset.transform("remap_labels", mapping=mapping_dict)
            st.toast("Remap Success!", icon="🎉")

    with st.expander("Subset Management"):
        ######## Aggregation
        sac.divider(
            label="Aggregation", icon="columns", align="center", bold=False, key=f"aggre_{col_name}"
        )
        st.info(
            "This helps to merge all subsets within a dataset into a single **default** subset."
        )
        aggre_subset_btn = st.button(
            "Do aggregation", use_container_width=True, key=f"aggre_subset_btn_{col_name}"
        )
        if aggre_subset_btn:
            dataset = data_helper.aggregate(from_subsets=dataset.subsets(), to_subset="default")
            st.toast("Aggregation Success!", icon="🎉")

        ######## Split
        sac.divider(
            label="Split", icon="columns-gap", align="center", bold=False, key=f"split_{col_name}"
        )
        st.info("This helps to divide a dataset into multiple subsets with a given ratio.")
        col1, col2 = st.columns(2)
        with col1:
            add_subset_btn = st.button(
                "Add subset", use_container_width=True, key=f"add_subset_btn_{col_name}"
            )
        with col2:
            split_btn = st.button("Do split", use_container_width=True, key=f"split_btn_{col_name}")

        if add_subset_btn:
            state[state_subset_key] += 1

        name, ratio = st.columns(2)
        splits = []
        for idx in range(state[state_subset_key]):
            with name:
                subset_name = st.text_input(
                    key=f"subset_name_{idx}_{col_name}",
                    label="Enter subset name",
                    placeholder="train",
                    value=None,
                )
            with ratio:
                subset_ratio = st.text_input(
                    key=f"subset_ratio_{idx}_{col_name}",
                    label="Enter subset ratio",
                    placeholder=0.5,
                    value=None,
                )
            if subset_name is not None and subset_ratio is not None:
                splits.append((subset_name, float(subset_ratio)))

        if split_btn:
            dataset = data_helper.transform("random_split", splits=splits)
            state[state_subset_key] = 1
            st.toast("Split Success!", icon="🎉")

    with st.expander("Item Management"):
        ######## Reindex
        sac.divider(
            label="Reindexing",
            icon="stickies-fill",
            align="center",
            bold=False,
            key=f"reindex_{col_name}",
        )
        st.info("This helps to reidentify all items.")
        col1, col2 = st.columns(2)
        with col1:
            item_reindex_input = st.text_input(
                label="Enter number starts with",
                key=f"item_reindex_input_{col_name}",
                placeholder=0,
                value=0,
            )
        with col2:
            st.markdown("<style>{}</style>".format(btn_style), unsafe_allow_html=True)
            item_reindex_btn = st.button(
                "Set IDs from number", use_container_width=True, key=f"item_reindex_btn_{col_name}"
            )
        item_media_name_btn = st.button(
            "Set IDs with media name",
            use_container_width=True,
            key=f"item_media_name_btn_{col_name}",
        )

        if item_reindex_btn:
            dataset = data_helper.transform("reindex", start=int(item_reindex_input))
            st.toast("Reindex Success!", icon="🎉")

        if item_media_name_btn:
            dataset = data_helper.transform("id_from_image_name")
            st.toast("Reindex Success!", icon="🎉")

        ######## Filter
        sac.divider(
            label="Filtration",
            icon="funnel-fill",
            align="center",
            bold=False,
            key=f"filter_{col_name}",
        )
        st.info("This helps to filter some items or annotations within a dataset.")

        mode, filter_ = st.columns(2)
        with mode:
            selected_mode = st.selectbox(
                "Select filtering mode",
                ["items", "annotations", "items+annotations", "annotations+items"],
                key=f"selected_mode_{col_name}",
            )
        with filter_:
            filter_expr = st.text_input(
                "Enter XML filter expression",
                disabled=False,
                placeholder='Eg. /item[subset="train"]',
                value=None,
                key=f"filter_expr_{col_name}",
            )
        filter_btn = st.button(
            "Filter dataset", use_container_width=True, key=f"filter_btn_{col_name}"
        )
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
            dataset = dataset.filter(filter_expr, **filter_args)
            st.toast("Filter Success!", icon="🎉")

        ######## Remove
        sac.divider(
            label="Remove", icon="eraser-fill", align="center", bold=False, key=f"remove_{col_name}"
        )
        st.info("This helps to remove some items or annotations within a dataset.")
        subset, item, annotation = st.columns(3)

        with subset:
            selected_subset = st.selectbox(
                "Select a subset:", dataset.subsets(), key=f"selected_subset_{col_name}"
            )
        with item:
            ids = [item.id for item in dataset.get_subset(selected_subset)]
            selected_id = st.selectbox("Select a subset item:", ids, key=f"selected_id_{col_name}")

        with annotation:
            item = dataset.get(selected_id, selected_subset)
            ann_ids = [
                "All",
            ] + [ann.id for ann in item.annotations]
            selected_ann_id = st.selectbox(
                "Select a item annotation:", ann_ids, key=f"selected_ann_id_{col_name}"
            )

        col1, col2 = st.columns(2)
        with col1:
            rm_item_btn = st.button(
                "Remove item", use_container_width=True, key=f"rm_item_btn_{col_name}"
            )

        with col2:
            rm_ann_btn = st.button(
                "Remove annotation", use_container_width=True, key=f"rm_ann_btn_{col_name}"
            )

        if rm_item_btn:
            dataset = data_helper.transform("remove_items", ids=[(selected_id, selected_subset)])
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
            st.toast("Remove Success!", icon="🎉")

        ######## Auto correction
        sac.divider(
            label="Auto-correction",
            icon="hammer",
            align="center",
            bold=False,
            key=f"auto_correct_{col_name}",
        )
        st.info("This helps to correct a dataset and clean up validation report.")

        correct_btn = st.button(
            "Correct a dataset", use_container_width=True, key=f"correct_btn_{col_name}"
        )
        if correct_btn:
            dataset = data_helper.transform("correct", reports=state[state_report_key])
            st.toast("Correction Success!", icon="🎉")

    return dataset


def main():
    data_helper_1: MultipleDatasetHelper = state["data_helper_1"]
    data_helper_2: MultipleDatasetHelper = state["data_helper_2"]
    uploaded_file_1 = state["uploaded_file_1"]
    uploaded_file_2 = state["uploaded_file_2"]

    dataset_dict = {uploaded_file_1: data_helper_1, uploaded_file_2: data_helper_2}
    subset_key_dict = {uploaded_file_1: "subset_1", uploaded_file_2: "subset_2"}
    report_key_dict = {uploaded_file_1: "report_1", uploaded_file_2: "report_2"}

    dataset_names = [uploaded_file_1, uploaded_file_2]
    if "data_helper_merged" in state:
        data_helper_3: MultipleDatasetHelper = state["data_helper_merged"]
        dataset_names.append("Merged Dataset")
        dataset_dict["Merged Dataset"] = data_helper_3
        subset_key_dict["Merged Dataset"] = "subset_merged"
        report_key_dict["Merged Dataset"] = "report_merged"

    mapping = (
        pd.DataFrame(columns=[uploaded_file_1, uploaded_file_2])
        if state.mapping is None or state.mapping.empty
        else state.mapping
    )
    identicals = state.matched

    with elements("transform"):
        c1, c2 = st.columns(2)
        st.markdown("<style>{}</style>".format(box_style), unsafe_allow_html=True)
        with c1:
            selected_dataset_1 = st.selectbox(
                "Select dataset to transform : ", dataset_names, index=0
            )
            data_helper_1 = dataset_dict.get(selected_dataset_1, None)
            state_subset_key_1 = subset_key_dict.get(selected_dataset_1, None)
            state_report_key_1 = report_key_dict.get(selected_dataset_1, None)

            transform_dataset = render_dataset_management_section(
                "c1",
                data_helper_1,
                uploaded_file_1,
                uploaded_file_2,
                mapping,
                identicals,
                state_subset_key_1,
                state_report_key_1,
            )
            data_helper_1.update_dataset(transform_dataset)
            c1.markdown(
                f"<div class='highlight blue box'><span class='bold'>{uploaded_file_1}</span></div>",
                unsafe_allow_html=True,
            )

        with c2:
            selected_dataset_2 = st.selectbox(
                "Select dataset to transform : ", dataset_names, index=1
            )
            data_helper_2 = dataset_dict.get(selected_dataset_2, None)
            state_subset_key_2 = subset_key_dict.get(selected_dataset_1, None)
            state_report_key_2 = report_key_dict.get(selected_dataset_1, None)

            transform_dataset = render_dataset_management_section(
                "c2",
                data_helper_2,
                uploaded_file_1,
                uploaded_file_2,
                mapping,
                identicals,
                state_subset_key_2,
                state_report_key_2,
            )
            data_helper_2.update_dataset(transform_dataset)
            c2.markdown(
                f"<div class='highlight red box'><span class='bold'>{selected_dataset_2}</span></div>",
                unsafe_allow_html=True,
            )

        dataset_1 = data_helper_1.dataset()
        subset_info_dict_1 = get_subset_info(dataset_1)
        categories_1 = dataset_1.categories()[AnnotationType.label]
        cat_info_dict_1 = get_category_info(dataset_1, categories_1)

        dataset_2 = data_helper_2.dataset()
        subset_info_dict_2 = get_subset_info(dataset_2)
        categories_2 = dataset_2.categories()[AnnotationType.label]
        cat_info_dict_2 = get_category_info(dataset_2, categories_2)

        board = Dashboard()
        w = SimpleNamespace(
            dashboard=board,
            subset_info_1=Pie(
                name="Subset info of Dataset 1",
                **{"board": board, "x": 0, "y": 0, "w": 3, "h": 6, "minW": 2, "minH": 4},
            ),
            cat_info_1=Radar(
                name="Category info of Dataset 1",
                indexBy="subset",
                keys=[cat.name for cat in categories_1.items],
                **{"board": board, "x": 3, "y": 0, "w": 3, "h": 6, "minW": 2, "minH": 4},
            ),
            subset_info_2=Pie(
                name="Subset info of Dataset 2",
                **{"board": board, "x": 6, "y": 0, "w": 3, "h": 6, "minW": 2, "minH": 4},
            ),
            cat_info_2=Radar(
                name="Category info of Dataset 2",
                indexBy="subset",
                keys=[cat.name for cat in categories_2.items],
                **{"board": board, "x": 9, "y": 0, "w": 3, "h": 6, "minW": 2, "minH": 4},
            ),
        )

        with w.dashboard(rowHeight=50):
            w.subset_info_1(subset_info_dict_1)
            w.subset_info_2(subset_info_dict_2)
            w.cat_info_1(cat_info_dict_1)
            w.cat_info_2(cat_info_dict_2)
