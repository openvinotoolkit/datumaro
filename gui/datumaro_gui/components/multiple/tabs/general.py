# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import streamlit as st
from datumaro_gui.utils.dataset.data_loader import MultipleDatasetHelper
from datumaro_gui.utils.dataset.info import get_category_info, get_subset_info
from datumaro_gui.utils.drawing import Dashboard, Gallery, Pie, Radar
from datumaro_gui.utils.drawing.css import box_style
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.annotation import AnnotationType


def main():
    data_helper_1: MultipleDatasetHelper = state["data_helper_1"]
    data_helper_2: MultipleDatasetHelper = state["data_helper_2"]
    dataset_1 = data_helper_1.dataset()
    dataset_2 = data_helper_2.dataset()
    uploaded_file_1 = state["uploaded_file_1"]
    uploaded_file_2 = state["uploaded_file_2"]

    with elements("general"):
        container = st.container()
        c1, c2 = container.columns(2)
        st.markdown("<style>{}</style>".format(box_style), unsafe_allow_html=True)

        with c1:
            container1 = c1.container()
            container1.subheader("First Dataset Description")
            col1, col2 = container1.columns(2)
            col1.markdown(
                f"<div class='highlight blue box'>Path <span class='bold'>{uploaded_file_1}</span></div>",
                unsafe_allow_html=True,
            )
            col2.markdown(
                f"<div class='highlight red box'>Format <span class='bold'>{dataset_1.format}</span></div>",
                unsafe_allow_html=True,
            )

        with c2:
            container2 = c2.container()
            container2.subheader("Second Dataset Description")
            col1, col2 = container2.columns(2)
            col1.markdown(
                f"<div class='highlight blue box'>Path <span class='bold'>{uploaded_file_2}</span></div>",
                unsafe_allow_html=True,
            )
            col2.markdown(
                f"<div class='highlight red box'>Format <span class='bold'>{dataset_2.format}</span></div>",
                unsafe_allow_html=True,
            )

        subset_info_dict_1 = get_subset_info(dataset_1)
        categories_1 = dataset_1.categories()[AnnotationType.label]
        cat_info_dict_1 = get_category_info(dataset_1, categories_1)

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
            player_1=Gallery(board, 0, 3, 6, 12, minH=4),
            player_2=Gallery(board, 6, 3, 6, 12, minH=4),
        )

        with w.dashboard(rowHeight=50):
            w.subset_info_1(subset_info_dict_1)
            w.subset_info_2(subset_info_dict_2)
            w.cat_info_1(cat_info_dict_1)
            w.cat_info_2(cat_info_dict_2)
            w.player_1(dataset_1)
            w.player_2(dataset_2)
