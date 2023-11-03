# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import difflib

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.annotation import AnnotationType
from datumaro.components.comparator import TableComparator

from ..data_loader import DatasetHelper


def return_matches(first_labels, second_labels, first_name, second_name):
    # Find common elements between the lists
    matches = list(set(first_labels) & set(second_labels))

    # Find unmatched elements for each list
    unmatched_a = [item for item in first_labels if item not in matches]
    unmatched_b = [item for item in second_labels if item not in matches]

    return matches, {first_name: unmatched_a, second_name: unmatched_b}


def main():
    data_helper_1: DatasetHelper = state["data_helper_1"]
    data_helper_2: DatasetHelper = state["data_helper_2"]
    first_dataset = data_helper_1.dataset()
    second_dataset = data_helper_2.dataset()
    uploaded_zip_1 = state["uploaded_zip_1"].name[:-4]
    uploaded_zip_2 = state["uploaded_zip_2"].name[:-4]

    # Initialize state
    if not "mapping":
        state.mapping = pd.DataFrame(columns=[uploaded_zip_1, uploaded_zip_2])
    if not "matched":
        state.matched = []

    with elements("compare"):
        comparator = TableComparator()
        (
            high_level_table,
            _,
            _,
            _,
        ) = comparator.compare_datasets(first_dataset, second_dataset, mode="high")
        (
            _,
            mid_level_table,
            _,
            _,
        ) = comparator.compare_datasets(first_dataset, second_dataset, mode="mid")

        ### high level
        # Split the string into rows and extract data
        high_level_rows = high_level_table.strip().split("\n")
        high_level_header = [col.strip() for col in high_level_rows[1].split("|")[1:-1]]
        high_level_data = []

        for row in high_level_rows[3::2]:
            values = [col.strip() for col in row.split("|")[1:-1]]
            high_level_data.append(values)

        high_level_df = pd.DataFrame(high_level_data, columns=high_level_header)

        ### mid level
        mid_level_rows = mid_level_table.strip().split("\n")
        mid_level_header = [col.strip() for col in mid_level_rows[1].split("|")[1:-1]]
        mid_level_data = []

        for row in mid_level_rows[3::2]:
            values = [col.strip() for col in row.split("|")[1:-1]]
            mid_level_data.append(values)

        mid_level_df = pd.DataFrame(mid_level_data, columns=mid_level_header)

        container = st.container()
        c1, c2 = container.columns([1, 2])
        with c1:
            st.header("Comparison result")
            c1.subheader("High Level Table")
            c1.dataframe(high_level_df, use_container_width=True)
            c1.subheader("Mid Level Table")
            c1.dataframe(mid_level_df, use_container_width=True)

        with c2:
            st.header("Compare Categories")
            categories_1 = first_dataset.categories()[AnnotationType.label]._indices.keys()
            categories_2 = second_dataset.categories()[AnnotationType.label]._indices.keys()

            col1, col2 = st.columns(2)
            col1.subheader("Matched Labels")
            matches, unmatches = return_matches(
                categories_1, categories_2, uploaded_zip_1, uploaded_zip_2
            )
            matched_label_df = pd.DataFrame(
                {uploaded_zip_1: pd.Series(matches), uploaded_zip_2: pd.Series(matches)}
            )

            def cooling_highlight(val):
                return "background-color: rgba(172, 229, 238, 0.5)" if val in matches else ""

            col1.dataframe(
                matched_label_df.style.applymap(
                    cooling_highlight, subset=[uploaded_zip_1, uploaded_zip_2]
                ),
                use_container_width=True,
            )

            unmatched_label_df = pd.DataFrame(
                {
                    uploaded_zip_1: pd.Series(unmatches[uploaded_zip_1]),
                    uploaded_zip_2: pd.Series(unmatches[uploaded_zip_2]),
                }
            )
            col1.subheader("Unmatched Labels")
            col1.dataframe(unmatched_label_df, use_container_width=True)

            state.matched = matches
            with col2:
                col2.subheader("Suggest label mapping")
                mappings = {}  # Initialize an empty dictionary for storing mappings

                # Function to normalize a string
                def normalize_string(s):
                    s = s.lower()  # Convert the string to lowercase
                    s = s.replace(" ", "")  # Remove spaces from the string
                    return s

                threshold = st.slider("Desired similarity threshold", 0.0, 1.0, 0.7, step=0.1)
                # Iterate over items in the first list
                for item1 in unmatches[uploaded_zip_1]:
                    normalized_item1 = normalize_string(item1)

                    # Iterate over items in the second list
                    for item2 in unmatches[uploaded_zip_2]:
                        normalized_item2 = normalize_string(item2)

                        # Compare the similarity and check if it's above a threshold
                        similarity = difflib.SequenceMatcher(
                            None, normalized_item1, normalized_item2
                        ).ratio()
                        if similarity > threshold:  # Set your desired similarity threshold
                            # Add the mapping to the dictionary
                            mappings[item1] = item2

                # Convert the mappings dictionary to a DataFrame
                selected_df = pd.DataFrame(
                    mappings.items(), columns=[uploaded_zip_1, uploaded_zip_2]
                )

                gb = GridOptionsBuilder.from_dataframe(selected_df)
                gb.configure_pagination(enabled=True)
                gb.configure_selection(selection_mode="multiple", use_checkbox=True)
                gb.configure_column(
                    uploaded_zip_1,
                    editable=True,
                    cellEditor="agSelectCellEditor",
                    cellEditorParams={"values": sorted(unmatches[uploaded_zip_1])},
                )
                gb.configure_column(
                    uploaded_zip_2,
                    editable=True,
                    cellEditor="agSelectCellEditor",
                    cellEditorParams={"values": sorted(unmatches[uploaded_zip_2])},
                )
                gb.configure_grid_options(domLayout="normal")

                gridoptions = gb.build()
                grid_table = AgGrid(
                    selected_df,
                    gridOptions=gridoptions,
                    height=300,
                    width="100%",
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    theme="streamlit",
                )

                if st.button("Finalize mapping") and grid_table["selected_rows"] is not None:
                    sel_row = grid_table["selected_rows"]
                    data_dict = {
                        uploaded_zip_1: [item[uploaded_zip_1] for item in sel_row],
                        uploaded_zip_2: [item[uploaded_zip_2] for item in sel_row],
                    }
                    mapping_df = pd.DataFrame(data_dict)
                    st.dataframe(mapping_df, use_container_width=True)
                    state.mapping = mapping_df
                    st.info(
                        "Use the generated mapping to transform your dataset by updating the labels."
                        "\n\nYou can continue the process within the 'transform' tab.",
                        icon="ℹ",
                    )
