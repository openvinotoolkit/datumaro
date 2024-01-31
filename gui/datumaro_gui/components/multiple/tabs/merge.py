# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import streamlit as st
from datumaro_gui.utils.dataset.data_loader import MultipleDatasetHelper
from datumaro_gui.utils.dataset.info import get_category_info, get_subset_info
from datumaro_gui.utils.drawing import Dashboard, Pie, Radar
from datumaro_gui.utils.drawing.css import box_style
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.annotation import AnnotationType


def main():
    data_helper_1: MultipleDatasetHelper = state["data_helper_1"]
    data_helper_2: MultipleDatasetHelper = state["data_helper_2"]
    first_dataset = data_helper_1.dataset()
    second_dataset = data_helper_2.dataset()

    c1, c2 = st.columns([1, 2])
    with c1:
        merge_methods = ["union", "intersect", "exact"]
        selected_method = st.selectbox(
            "Select a method to merge:", merge_methods, key="sb_merge_method_mult"
        )
        merge_btn = st.button("Merge dataset", use_container_width=True, key="merge_btn_mult")
        if merge_btn:
            data_helper = MultipleDatasetHelper()
            state["data_helper_merged"] = data_helper

            merged_dataset = data_helper.merge([first_dataset, second_dataset], selected_method)
            data_helper.update_dataset(merged_dataset)
            st.toast("Merge Success!", icon="🎉")

    # TODO : Merge report
    st.markdown("<style>{}</style>".format(box_style), unsafe_allow_html=True)
    if "data_helper_merged" in state:
        merged_data_helper: MultipleDatasetHelper = state["data_helper_merged"]
        dataset = merged_data_helper.dataset()

        subset_info_dict = get_subset_info(dataset)
        categories = dataset.categories()[AnnotationType.label]
        cat_info_dict = get_category_info(dataset, categories)

        with c2:
            c2.markdown(
                "<div class='highlight blue box'><span class='bold'>Merged Dataset</span></div>",
                unsafe_allow_html=True,
            )
            with elements("merge"):
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
