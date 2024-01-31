# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import io

import numpy as np
import streamlit as st
from datumaro_gui.utils.dataset.data_loader import MultipleDatasetHelper
from matplotlib import pyplot as plt
from PIL import Image
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.visualizer import Visualizer


def main():
    data_helper_1: MultipleDatasetHelper = state["data_helper_1"]
    data_helper_2: MultipleDatasetHelper = state["data_helper_2"]
    uploaded_file_1 = state["uploaded_file_1"]
    uploaded_file_2 = state["uploaded_file_2"]

    dataset_dict = {uploaded_file_1: data_helper_1, uploaded_file_2: data_helper_2}
    dataset_names = [uploaded_file_1, uploaded_file_2]
    with elements("visualize"):
        selected_dataset = st.selectbox("Select dataset to transform : ", dataset_names, index=0)
        dataset = dataset_dict.get(selected_dataset, None).dataset()
        with st.container():
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
                    ] + [ann.id for ann in item.annotations]
                    selected_ann_id = st.selectbox("Select a dataset item:", ann_ids)

                selected_alpha = st.select_slider(
                    "Choose a transparency of annotations",
                    options=np.arange(0.0, 1.1, 0.1, dtype=np.float16),
                )

                visualizer = Visualizer(
                    dataset, figsize=(8, 8), alpha=selected_alpha, show_plot_title=False
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

                # Save the Matplotlib figure to a BytesIO buffer as PNG
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png")
                plt.close(fig)
                buffer.seek(0)
                img = Image.open(buffer)

                st.image(img, use_column_width=True)
