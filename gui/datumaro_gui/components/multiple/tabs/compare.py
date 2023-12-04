# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import difflib
import re

import pandas as pd
import streamlit as st
from datumaro_gui.utils.dataset.data_loader import MultipleDatasetHelper
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.annotation import AnnotationType
from datumaro.components.comparator import TableComparator


def return_matches(first_labels, second_labels, first_name, second_name):
    # Find common elements between the lists
    matches = list(set(first_labels) & set(second_labels))

    # Find unmatched elements for each list
    unmatched_a = [item for item in first_labels if item not in matches]
    unmatched_b = [item for item in second_labels if item not in matches]

    return matches, {first_name: unmatched_a, second_name: unmatched_b}


def main():
    data_helper_1: MultipleDatasetHelper = state["data_helper_1"]
    data_helper_2: MultipleDatasetHelper = state["data_helper_2"]
    first_dataset = data_helper_1.dataset()
    second_dataset = data_helper_2.dataset()
    uploaded_zip_1 = state["uploaded_zip_1"].name[:-4]
    uploaded_zip_2 = state["uploaded_zip_2"].name[:-4]
    high_level_df = state["high_level_table"]
    mid_level_df = state["mid_level_table"]
    low_level_df = state["low_level_table"]

    # Initialize state
    if not "mapping":
        state.mapping = pd.DataFrame(columns=[uploaded_zip_1, uploaded_zip_2])
    if not "matched":
        state.matched = []

    with elements("compare"):
        comparator = TableComparator()

        if high_level_df is None and mid_level_df is None:
            high_level_table, _, _, _ = comparator.compare_datasets(
                first_dataset, second_dataset, "high"
            )
            _, mid_level_table, _, _ = comparator.compare_datasets(
                first_dataset, second_dataset, "mid"
            )

            ### high level
            # Split the string into rows and extract data
            high_level_lines = high_level_table.split("\n")
            high_level_data_lines = [
                re.split(r"\s*[|]\s*", line.strip("|")) for line in high_level_lines if "|" in line
            ]
            high_level_header = [
                header.strip() for header in high_level_data_lines[0] if header.strip()
            ]
            high_level_df = pd.DataFrame(high_level_data_lines[1:], columns=high_level_header)
            high_level_df = high_level_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            ### mid level
            mid_level_lines = mid_level_table.split("\n")
            mid_level_data_lines = [
                re.split(r"\s*[|]\s*", line.strip("|")) for line in mid_level_lines if "|" in line
            ]
            mid_level_header = [
                header.strip() for header in mid_level_data_lines[0] if header.strip()
            ]
            mid_level_df = pd.DataFrame(mid_level_data_lines[1:], columns=mid_level_header)
            mid_level_df = mid_level_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            state["high_level_table"] = high_level_df
            state["mid_level_table"] = mid_level_df

        container = st.container()
        c1, c2 = container.columns([1, 2])
        with c1:
            st.header("Comparison result")
            c1.subheader("High Level Table")
            c1.caption(
                "High-level overview provides information on dataset composition, class counts, common classes, "
                "image counts, unique or repeated images, and annotation counts for each dataset.",
            )
            c1.dataframe(high_level_df, use_container_width=True)
            c1.subheader("Mid Level Table")
            c1.caption(
                "Mid-level overview provides insights into subsets of each dataset, showcasing image characteristics. "
                "As subsets do not overlap, details are presented for each dataset, including label-specific image "
                "counts and ratios.",
            )
            c1.dataframe(mid_level_df, use_container_width=True)
            c1.subheader("Low Level Table")

            ### low level
            on = st.toggle("Show low-level table")
            if on:
                if low_level_df is None:
                    _, _, low_level_table, _ = comparator.compare_datasets(
                        first_dataset, second_dataset, "low"
                    )
                    low_level_lines = low_level_table.split("\n")
                    low_level_data_lines = [
                        re.split(r"\s*[|]\s*", line.strip("|"))
                        for line in low_level_lines
                        if "|" in line
                    ]
                    low_level_header = [
                        header.strip() for header in low_level_data_lines[0] if header.strip()
                    ]
                    low_level_df = pd.DataFrame(low_level_data_lines[1:], columns=low_level_header)
                    low_level_df = low_level_df.applymap(
                        lambda x: x.strip() if isinstance(x, str) else x
                    )
                    state["low_level_table"] = low_level_df

                c1.dataframe(low_level_df, use_container_width=True)

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
