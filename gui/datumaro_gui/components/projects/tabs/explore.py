# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import streamlit as st
from streamlit import session_state as state
from streamlit_elements import elements

import datumaro as dm
from datumaro.components.visualizer import Visualizer

from ..data_loader import DataRepo, DatasetHelper

USER_UPLOADED_SUBSET = "__user_uploaded__"


def query_list(title=""):
    ## query list
    with st.expander(title, expanded=True):
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
            if current_selected is not None:
                selected_item = state["explore_queries"][current_selected]
                if not isinstance(selected_item, str):
                    st.image(selected_item.media.path)
            if st.button("Remove from Query List", disabled=current_selected is None):
                item = state["explore_queries"].pop(current_selected)
                if isinstance(item, dm.DatasetItem) and item.subset == USER_UPLOADED_SUBSET:
                    file_id = state["explore_user_uploaded_images"].get(item.media.path)
                    if (
                        state["explore_user_uploaded_file"] is None
                        or state["explore_user_uploaded_file"].file_id != file_id
                    ):
                        DataRepo().delete_by_id(file_id)
                st.rerun()

        else:
            st.write("No query selected")


def result_list(
    dataset,
    title="",
):
    ## query list
    if state["explore_results"]:
        with st.expander(title, expanded=True):
            checkbox_labels = []
            checkbox_keys = []
            for idx, item in enumerate(state["explore_results"]):
                checkbox_keys.append(f"result_{idx}")
                checkbox_labels.append(f"{item.subset}-{item.id}")

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

        if current_selected is not None:
            selected_alpha_for_result = st.select_slider(
                "Choose a transparency of annotations",
                options=np.arange(0, 110, 10),
                value=20,
            )
            try:
                visualizer = Visualizer(
                    dataset, figsize=(8, 8), alpha=selected_alpha_for_result * 0.01
                )
                selected_item = state["explore_results"][current_selected]
                fig = visualizer.vis_one_sample(selected_item.id, selected_item.subset)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Exception: {e}")

    else:
        st.write("There're no matched images")
        st.write("Check the 'Query List' and press the 'Search' button again.")


def uploader_cb():
    user_paths = []
    for item in state["explore_queries"]:
        if isinstance(item, dm.DatasetItem) and item.subset == USER_UPLOADED_SUBSET:
            user_paths.append(item.media.path)

    deletable = []
    for path in state["explore_user_uploaded_images"]:
        if path not in user_paths:
            deletable.append(path)

    for path in deletable:
        file_id = state["explore_user_uploaded_images"].pop(path)
        DataRepo().delete_by_id(file_id)


def main():
    data_repo = DataRepo()
    data_helper: DatasetHelper = state["data_helper"]
    dataset = data_helper.dataset()

    if "explore_queries" not in state:
        state["explore_queries"] = []
    if "explore_topk" not in state:
        state["explore_topk"] = 2
    if "explore_results" not in state:
        state["explore_results"] = None
    if "explore_user_uploaded_images" not in state:
        state["explore_user_uploaded_images"] = {}

    with elements("explore"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Query Parameters")
            ## query form
            query_type = st.radio(
                "Query Type", options=["User Image", "Dataset Image", "Text"], horizontal=True
            )
            if query_type == "User Image":
                uploaded_img = st.file_uploader(
                    "Upload an image file",
                    type=["jpg", "png", "jpeg"],
                    key="explore_user_uploaded_file",
                    on_change=uploader_cb,
                )
                if uploaded_img is not None:
                    path = data_repo.save_file(uploaded_img)
                    state["explore_user_uploaded_images"][path] = uploaded_img.file_id
                    st.image(path)
                    query = dm.DatasetItem(
                        id=uploaded_img.name,
                        subset=USER_UPLOADED_SUBSET,
                        media=dm.Image.from_file(path),
                    )
                else:
                    query = None
            elif query_type == "Dataset Image":
                selected_subset = st.selectbox("Select a subset:", dataset.subsets())
                if selected_subset:
                    ids = [item.id for item in dataset.get_subset(selected_subset)]
                    selected_id = st.selectbox("Select a dataset item:", ids)

                query = dataset.get(selected_id, selected_subset)
                if query is not None:
                    st.image(query.media.path)
            else:
                query = st.text_input("Input text query:")

            if st.button("Add to Query List"):
                if query and query not in state["explore_queries"]:
                    state["explore_queries"].append(query)

            query_list("Query List")
            topk = st.number_input("Top K:", value=state["explore_topk"])
            if topk != state["explore_topk"]:
                state["explore_topk"] = topk
            search = st.button(
                "Search", disabled=not state["explore_queries"], use_container_width=True
            )

        with c2:
            st.subheader("Results")
            if search is True:
                try:
                    progress_bar = st.progress(0, text="Initializing...")
                    for percentage_complete in range(100):
                        explorer = data_helper.explorer()
                        progress_bar.progress(
                            percentage_complete + 1,
                            text=f"Initializing Explorer [{percentage_complete+1}/100]",
                        )
                    progress_bar.empty()

                    progress_bar = st.progress(0, text="Searching...")
                    for percentage_complete in range(100):
                        results = explorer.explore_topk(state["explore_queries"], topk=topk)
                        progress_bar.progress(
                            percentage_complete + 1,
                            text=f"Searching top-{topk} examples [{percentage_complete+1}/100]",
                        )
                    progress_bar.empty()
                except Exception as e:
                    st.write(
                        "An error occur while searching. Please re-import dataset and try again."
                    )
                    st.write(f"Error: {e}")
                    results = []

                if len(results) > 0:
                    state["explore_results"] = list(results)
                else:
                    state["explore_results"] = []

            if state["explore_results"] is not None:
                result_list(dataset, "subset-id")

            else:
                st.write(
                    "Add an image or text query to the 'Query List' and press the 'Search' button."
                )
