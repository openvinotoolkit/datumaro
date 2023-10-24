# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import streamlit as st
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.visualizer import Visualizer

from ..data_loader import DatasetHelper


def query_list():
    ## query list
    with st.expander("", expanded=True):
        if state["explore_queries"]:
            checkbox_labels = []
            checkbox_keys = []
            for idx, item in enumerate(state["explore_queries"]):
                checkbox_keys.append(f"query_{idx}")
                if isinstance(item, str):
                    checkbox_labels.append(f":speech_balloon: {item}")
                else:  # assume it is dataset item
                    checkbox_labels.append(f":frame_with_picture: {item.subset}-{item.id}")

            def uncheck_others(*selected_key):
                for key in checkbox_keys:
                    if key != selected_key[0]:
                        state[key] = False

            checkboxes = []
            for label, key in zip(checkbox_labels, checkbox_keys):
                cb = st.checkbox(label, key=key, on_change=uncheck_others, args=(key,))
                checkboxes.append(cb)

            current_selected = None
            for idx, cb in enumerate(checkboxes):
                if cb is True:
                    current_selected = idx

            if st.button("Remove from Query List", disabled=current_selected is None):
                state["explore_queries"].pop(current_selected)
                st.rerun()

        else:
            st.write("No query selected")


def main():
    data_helper: DatasetHelper = state["data_helper"]
    dataset = data_helper.dataset()

    if "explore_queries" not in state:
        state["explore_queries"] = []
    if "explore_topk" not in state:
        state["explore_topk"] = 2
    if "explore_results" not in state:
        state["explore_results"] = None

    with elements("explore"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Query")
            ## query form
            query_type = st.radio("Query Type", options=["Image", "Text"], horizontal=True)
            if query_type == "Image":
                selected_subset = st.selectbox("Select a subset:", dataset.subsets())
                if selected_subset:
                    ids = [item.id for item in dataset.get_subset(selected_subset)]
                    selected_id = st.selectbox("Select a dataset item:", ids)

                query = dataset.get(selected_id, selected_subset)
                if query is not None:
                    # visualizer = Visualizer(dataset, figsize=(8, 8), alpha=0)
                    # fig = visualizer.vis_one_sample(selected_id, selected_subset)
                    # st.pyplot(fig)
                    st.image(query.media.path)
            else:
                query = st.text_input("Input text query:")

            if st.button("Add to Query List"):
                if query and query not in state["explore_queries"]:
                    state["explore_queries"].append(query)
        with c2:
            st.subheader("Query List")
            query_list()
            topk = st.number_input("Top K:", value=state["explore_topk"])
            if topk != state["explore_topk"]:
                state["explore_topk"] = topk

            search = st.button("Search", disabled=not state["explore_queries"])

            st.subheader("Result")
            if search is True:
                try:
                    explorer = data_helper.explorer()
                    results = explorer.explore_topk(state["explore_queries"], topk=topk)
                except Exception as e:
                    st.write(
                        "An error occur while searching. Please re-import dataset and try again."
                    )
                    st.write(f"Error: {e}")
                    results = []

                subsets = []
                ids = []
                for result in results:
                    subsets.append(result.subset)
                    ids.append(result.id)
                    state["explore_results"] = {"subsets": subsets, "ids": ids}

            if state["explore_results"] is not None:
                selected_alpha_for_result = st.select_slider(
                    "Choose a transparency of annotations",
                    options=np.arange(0.0, 1.1, 0.1, dtype=np.float16),
                    value=1.0,
                )
                try:
                    visualizer = Visualizer(
                        dataset, figsize=(8, 8), alpha=selected_alpha_for_result
                    )
                    fig = visualizer.vis_gallery(
                        state["explore_results"]["ids"], state["explore_results"]["subsets"]
                    )
                    st.pyplot(fig)
                except Exception as e:
                    st.write(f"Error: {e}")
            else:
                st.write(
                    "Add an image or text query to the 'Query List' and press the 'Search' button."
                )
