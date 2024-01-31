# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import streamlit as st
from datumaro_gui.utils.dataset.data_loader import SingleDatasetHelper
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.visualizer import Visualizer


def main():
    data_helper: SingleDatasetHelper = state["data_helper"]
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
                ann_ids = set()
                for ann in item.annotations:
                    ann_ids.add(ann.id)
                options = [
                    "All",
                ] + sorted(list(ann_ids))
                selected_ann_id = st.selectbox("Select annotation:", options)

            selected_alpha = st.select_slider(
                "Choose a transparency of annotations",
                options=np.arange(0, 110, 10),
                value=20,
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
