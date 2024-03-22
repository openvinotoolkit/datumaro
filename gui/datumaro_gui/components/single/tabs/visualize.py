# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import streamlit as st
from datumaro_gui.utils.dataset.data_loader import SingleDatasetHelper
from datumaro_gui.utils.page import check_image_backend
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.visualizer import Visualizer


def main():
    check_image_backend(state.get("IMAGE_BACKEND"))
    data_helper: SingleDatasetHelper = state["data_helper"]
    dataset = data_helper.dataset()
    with elements("visualize"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Parameters")
            selected_subset = st.selectbox(
                "Select a subset:", dataset.subsets(), key="sb_select_subset_viz"
            )
            if selected_subset:
                ids = [item.id for item in dataset.get_subset(selected_subset)]
                selected_id = st.selectbox("Select a dataset item:", ids, key="sb_select_id_viz")

            if selected_id:
                item = dataset.get(selected_id, selected_subset)
                ann_ids = set()
                for ann in item.annotations:
                    ann_ids.add(ann.id)
                options = [
                    "All",
                ] + sorted(list(ann_ids))
                selected_ann_id = st.selectbox(
                    "Select annotation:", options, key="sb_select_ann_id_viz"
                )

            selected_alpha = st.select_slider(
                "Choose a transparency of annotations",
                options=np.arange(0, 110, 10),
                value=20,
                key="ss_select_alpha_viz",
            )

            visualizer = Visualizer(
                dataset, figsize=(8, 8), alpha=selected_alpha * 0.01, show_plot_title=False
            )

        with c2:
            st.subheader("Item")
            if selected_ann_id == "All":
                fig = visualizer.vis_one_sample(selected_id, selected_subset)
            else:
                fig = visualizer.vis_one_sample(
                    selected_id, selected_subset, ann_id=selected_ann_id
                )
            fig.set_facecolor("none")
            st.pyplot(fig, use_container_width=True)
