# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import numpy as np
import streamlit as st
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.visualizer import Visualizer

from ..data_loader import DatasetHelper


def main():
    data_helper: DatasetHelper = state["data_helper"]
    dataset = data_helper.dataset()
    with elements("visualize"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Parameters")
            selected_subset = st.selectbox("Select a subset:", dataset.subsets())
            if selected_subset:
                ids = [item.id for item in dataset.get_subset(selected_subset)]
                selected_id = st.selectbox("Select a dataset item:", ids)

            if selected_id:
                item = dataset.get(selected_id, selected_subset)
                ann_ids = [
                    "All",
                ] + [f"{ann.type.name} - {ann.id}" for ann in item.annotations]
                selected_ann_id = st.selectbox("Select annotation:", ann_ids)

            selected_alpha = st.select_slider(
                "Choose a transparency of annotations",
                options=np.arange(0.0, 1.1, 0.1, dtype=np.float16),
                value=1.0,
            )

            visualizer = Visualizer(dataset, figsize=(8, 8), alpha=selected_alpha)

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
