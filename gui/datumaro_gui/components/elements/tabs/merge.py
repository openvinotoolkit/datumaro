# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import streamlit as st
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.annotation import AnnotationType

from ..dashboard import Dashboard, Pie, Radar
from ..data_loader import DatasetHelper
from ..utils import box_style, get_category_info, get_subset_info


def main():
    data_helper_1: DatasetHelper = state["data_helper_1"]
    data_helper_2: DatasetHelper = state["data_helper_2"]
    first_dataset = data_helper_1.dataset()
    second_dataset = data_helper_2.dataset()

    with elements("merge"):
        c1, c2 = st.columns([1, 2])
        with c1:
            merge_methods = ["union", "intersect", "exact"]
            selected_method = st.selectbox("Select a task to export:", merge_methods)
            if st.button("Merged dataset", use_container_width=True):
                data_helper = DatasetHelper()
                state["data_helper_merged"] = data_helper

                merged_dataset = data_helper.merge([first_dataset, second_dataset], selected_method)
                state["subset_merged"] = None
                if state["subset_merged"] is None:
                    state["subset_merged"] = 0
                data_helper.update_dataset(merged_dataset)
                st.toast("Merge Success!", icon="🎉")

        # TODO : Merge report
        with c2:
            st.markdown("<style>{}</style>".format(box_style), unsafe_allow_html=True)
            c2.markdown(
                "<div class='highlight blue box'><span class='bold'>Merged Dataset</span></div>",
                unsafe_allow_html=True,
            )
            if "data_helper_merged" in state:
                merged_data_helper: DatasetHelper = state["data_helper_merged"]
                dataset = merged_data_helper.dataset()

                c2.markdown(
                    "<div class='highlight lightgrayish box'><span class='bold'>Split Dataset</span></div>",
                    unsafe_allow_html=True,
                )
                col1, col2 = c2.columns(2)
                add_subset_btn = col1.button(
                    "Add subset", use_container_width=True, key="add_subset_btn_merged"
                )
                split_btn = col2.button(
                    "Do Split", use_container_width=True, key="split_btn_merged"
                )

                if add_subset_btn:
                    state["subset_merged"] += 1

                name, ratio = c2.columns(2)
                splits = []
                for idx in range(state["subset_merged"]):
                    with name:
                        subset_name = st.text_input(
                            key=f"subset_name_{idx}_merged",
                            label="Enter subset name",
                            placeholder="train",
                            value=None,
                        )
                    with ratio:
                        subset_ratio = st.text_input(
                            key=f"subset_ratio{idx}_merged",
                            label="Enter subset ratio",
                            placeholder=0.5,
                            value=None,
                        )
                    if subset_name is not None and subset_ratio is not None:
                        splits.append((subset_name, float(subset_ratio)))

                if split_btn:
                    dataset = data_helper.transform("random_split", splits=splits)
                    state["subset_merged"] = 1
                    st.toast("Split Success!", icon="🎉")

                subset_info_dict = get_subset_info(dataset)
                categories = dataset.categories()[AnnotationType.label]
                cat_info_dict = get_category_info(dataset, categories)

                board = Dashboard()
                w = SimpleNamespace(
                    dashboard=board,
                    subset_info=Pie(
                        name="Subset info of Merged dataset",
                        **{"board": board, "x": 0, "y": 0, "w": 3, "h": 6, "minW": 2, "minH": 4},
                    ),
                    cat_info=Radar(
                        name="Category info of Merged dataset",
                        indexBy="subset",
                        keys=[cat.name for cat in categories.items],
                        **{"board": board, "x": 3, "y": 0, "w": 3, "h": 6, "minW": 2, "minH": 4},
                    ),
                )

                with w.dashboard(rowHeight=50):
                    w.subset_info(subset_info_dict)
                    w.cat_info(cat_info_dict)
