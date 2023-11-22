# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import copy
from abc import ABCMeta, abstractmethod

import numpy as np
import streamlit as st
from datumaro_gui.utils.dataset.data_loader import DataRepo, SingleDatasetHelper
from streamlit import session_state as state
from streamlit_elements import elements

import datumaro as dm
from datumaro.components.algorithms.hash_key_inference.explorer import Explorer
from datumaro.components.visualizer import Visualizer

USER_UPLOADED_SUBSET = "__user_uploaded__"


class Query(metaclass=ABCMeta):
    @abstractmethod
    def label(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def query(self) -> dm.DatasetItem | str | None:
        raise NotImplementedError


class QueryImage(Query):
    def __init__(self, item: dm.DatasetItem):
        self.item = item

    def label(self) -> str:
        return f":frame_with_picture: {self.item.subset}-{self.item.id}"

    def query(self) -> dm.DatasetItem | str | None:
        return self.item

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.item == other.item
        return False

    @property
    def path(self) -> str:
        return self.item.media.path


class QueryUplodedImage(QueryImage):
    def __init__(self, id: str, path: str):
        super().__init__(
            dm.DatasetItem(
                id=id,
                subset=USER_UPLOADED_SUBSET,
                media=dm.Image.from_file(path),
            )
        )


class QueryText(Query):
    def __init__(self, text: str):
        self.text = text

    def label(self) -> str:
        return f":speech_balloon: {self.text}"

    def query(self) -> dm.DatasetItem | str | None:
        return self.text

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.text == other.text
        return False


class QueryLabel(Query):
    def __init__(self, labels: list[str]):
        self.labels = labels  # assume it is sorted.

    def label(self) -> str:
        return f":label: {','.join(self.labels)}"

    def query(self) -> dm.DatasetItem | str | None:
        return None

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.labels == other.labels  # assume it is sorted.
        return False


def query_list(title=""):
    ## query list
    with st.expander(title, expanded=True):
        if state["explore_queries"]:
            checkbox_labels = []
            checkbox_keys = []
            for idx, item in enumerate(state["explore_queries"]):
                checkbox_keys.append(f"query_{idx}")
                checkbox_labels.append(item.label())

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
                if isinstance(selected_item, QueryImage):
                    st.image(selected_item.path)
            if st.button("Remove from Query List", disabled=current_selected is None):
                item = state["explore_queries"].pop(current_selected)
                if isinstance(item, QueryUplodedImage):
                    file_id = state["explore_user_uploaded_images"].get(item.path)
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


def explore_topk(dataset: dm.Dataset, topk: int) -> list:
    labels = None
    queries = []
    for item in state["explore_queries"]:
        query = item.query()
        if isinstance(item, QueryLabel):
            if labels is None:
                labels = set(item.labels)
            else:
                labels = labels & set(item.labels)
        elif query is not None:
            queries.append(query)

    if labels is not None:
        if len(labels) == 0:
            return []

        target_dataset = copy.deepcopy(dataset)
        filter_str = '/item/annotation[label="'
        filter_str += '" or label="'.join(labels)
        filter_str += '"]'
        target_dataset.filter(filter_str, filter_annotations=True, remove_empty=True)

        if len(target_dataset) == 0:
            return []
    else:
        target_dataset = dataset

    if not queries:  # just select first {topk} images from target_dataset
        results = []
        for item in target_dataset:
            results.append(item)
            if len(results) == topk:
                break
    else:
        explorer = Explorer(target_dataset)
        results = explorer.explore_topk(queries, topk=topk)

    return results


def uploader_cb():
    user_paths = []
    for item in state["explore_queries"]:
        if isinstance(item, QueryUplodedImage):
            user_paths.append(item.path)

    deletable = []
    for path in state["explore_user_uploaded_images"]:
        if path not in user_paths:
            deletable.append(path)

    for path in deletable:
        file_id = state["explore_user_uploaded_images"].pop(path)
        DataRepo().delete_by_id(file_id)


def main():
    data_repo = DataRepo()
    data_helper: SingleDatasetHelper = state["data_helper"]
    dataset = data_helper.dataset()

    if "explore_queries" not in state:
        state["explore_queries"] = []
    if "explore_topk" not in state:
        state["explore_topk"] = 2
    if "explore_results" not in state:
        state["explore_results"] = None
    if "explore_user_uploaded_images" not in state:
        state["explore_user_uploaded_images"] = {}

    label_cat: dm.LabelCategories = dataset.categories().get(dm.AnnotationType.label, None)
    query_types = ["User Image", "Dataset Image", "Text"]
    if label_cat is not None:
        query_types.append("Label")

    with elements("explore"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Query Parameters")
            query = None
            ## query form
            query_type = st.radio("Query Type", options=query_types, horizontal=True)
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
                    query = QueryUplodedImage(id=uploaded_img.name, path=path)
            elif query_type == "Dataset Image":
                selected_subset = st.selectbox("Select a subset:", dataset.subsets())
                if selected_subset:
                    ids = [item.id for item in dataset.get_subset(selected_subset)]
                    selected_id = st.selectbox("Select a dataset item:", ids)

                query = QueryImage(dataset.get(selected_id, selected_subset))
                if query is not None:
                    st.image(query.path)
            elif query_type == "Text":
                query = QueryText(st.text_input("Input text query:"))
            elif query_type == "Label":
                if len(label_cat.label_groups) > 0:
                    label_groups = {group.name: group.labels for group in label_cat.label_groups}
                    selected_group = st.selectbox("Select Label Group", [label_groups.keys()])
                    if selected_group:
                        labels = label_groups[selected_group]
                    else:
                        labels = []
                else:
                    labels = [item.name for item in label_cat.items]

                if len(labels) > 0:
                    labels.sort()
                    selected_labels = st.multiselect("Select Label(s)", labels)
                    if len(selected_labels) > 0:
                        query = QueryLabel(selected_labels)
            else:
                raise NotImplementedError

            if st.button("Add to Query List", use_container_width=True):
                if query is not None and query not in state["explore_queries"]:
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
                    results = explore_topk(dataset, topk)
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
