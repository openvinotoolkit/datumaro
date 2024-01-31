# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import difflib
import re

import pandas as pd
import streamlit as st
from datumaro_gui.utils.dataset.data_loader import MultipleDatasetHelper
from datumaro_gui.utils.dataset.info import return_matches
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.annotation import AnnotationType
from datumaro.components.comparator import TableComparator


def normalize_string(s):
    s = s.lower()
    s = s.replace(" ", "")
    return s


def get_dataframe(table):
    lines = table.split("\n")
    data_lines = [re.split(r"\s*[|]\s*", line.strip("|")) for line in lines if "|" in line]
    header = [header.strip() for header in data_lines[0] if header.strip()]

    df = pd.DataFrame(data_lines[1:], columns=header)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    return df


def main():
    data_helper_1: MultipleDatasetHelper = state["data_helper_1"]
    data_helper_2: MultipleDatasetHelper = state["data_helper_2"]
    first_dataset = data_helper_1.dataset()
    second_dataset = data_helper_2.dataset()
    uploaded_file_1 = state["uploaded_file_1"]
    uploaded_file_2 = state["uploaded_file_2"]
    high_level_df = state["high_level_table"]
    mid_level_df = state["mid_level_table"]
    low_level_df = state["low_level_table"]

    # Initialize state
    if not "mapping":
        state.mapping = pd.DataFrame(columns=[uploaded_file_1, uploaded_file_2])
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

            high_level_df = get_dataframe(high_level_table)
            mid_level_df = get_dataframe(mid_level_table)

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

            ### low level
            c1.subheader("Low Level Table")
            c1.caption(
                "Low-level overview uses Shift Analyzer to demonstrate covariate shift and label shift between two "
                "datasets.\n\nBy the way, the low-level analysis takes a bit of time to compute. Please bear with us "
                "for a moment; your patience is much appreciated!",
            )
            on = st.toggle("Show low-level table", key="low_lvl_tb_toggle")
            if on:
                if low_level_df is None:
                    _, _, low_level_table, _ = comparator.compare_datasets(
                        first_dataset, second_dataset, "low"
                    )
                    low_level_df = get_dataframe(low_level_table)
                    state["low_level_table"] = low_level_df

                c1.dataframe(low_level_df, use_container_width=True)
                c1.caption(
                    "In Datumaro, covariate shift is supported with two methods, with the default being Frechet "
                    "Inception Distance. This metric, measuring the distance of variances, indicates dataset "
                    "similarity as smaller values suggest likeness. Label shift employs Anderson-Darling test "
                    "from SciPy, revealing how well the data follows a normal distribution."
                )

        with c2:
            st.header("Compare Categories")
            categories_1 = first_dataset.categories()[AnnotationType.label]._indices.keys()
            categories_2 = second_dataset.categories()[AnnotationType.label]._indices.keys()

            col1, col2 = st.columns(2)
            col1.subheader("Matched Labels")
            matches, unmatches = return_matches(
                categories_1, categories_2, uploaded_file_1, uploaded_file_2
            )
            matched_label_df = pd.DataFrame(
                {uploaded_file_1: pd.Series(matches), uploaded_file_2: pd.Series(matches)}
            )

            def cooling_highlight(val):
                return "background-color: rgba(172, 229, 238, 0.5)" if val in matches else ""

            col1.dataframe(
                matched_label_df.style.applymap(
                    cooling_highlight, subset=[uploaded_file_1, uploaded_file_2]
                ),
                use_container_width=True,
            )

            unmatched_label_df = pd.DataFrame(
                {
                    uploaded_file_1: pd.Series(unmatches[uploaded_file_1]),
                    uploaded_file_2: pd.Series(unmatches[uploaded_file_2]),
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

                threshold = st.slider(
                    "Desired similarity threshold", 0.0, 1.0, 0.7, step=0.1, key="sim_slider"
                )
                # Iterate over items in the first list
                for item1 in unmatches[uploaded_file_1]:
                    normalized_item1 = normalize_string(item1)

                    # Iterate over items in the second list
                    for item2 in unmatches[uploaded_file_2]:
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
                    mappings.items(), columns=[uploaded_file_1, uploaded_file_2]
                )

                gb = GridOptionsBuilder.from_dataframe(selected_df)
                gb.configure_pagination(enabled=True)
                gb.configure_selection(selection_mode="multiple", use_checkbox=True)
                gb.configure_column(
                    uploaded_file_1,
                    editable=True,
                    cellEditor="agSelectCellEditor",
                    cellEditorParams={"values": sorted(unmatches[uploaded_file_1])},
                )
                gb.configure_column(
                    uploaded_file_2,
                    editable=True,
                    cellEditor="agSelectCellEditor",
                    cellEditorParams={"values": sorted(unmatches[uploaded_file_2])},
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

                if (
                    st.button("Finalize mapping", key="mapping_btn")
                    and grid_table["selected_rows"] is not None
                ):
                    sel_row = grid_table["selected_rows"]
                    data_dict = {
                        uploaded_file_1: [item[uploaded_file_1] for item in sel_row],
                        uploaded_file_2: [item[uploaded_file_2] for item in sel_row],
                    }
                    mapping_df = pd.DataFrame(data_dict)
                    st.dataframe(mapping_df, use_container_width=True)
                    state.mapping = mapping_df
                    st.info(
                        "Use the generated mapping to transform your dataset by updating the labels."
                        "\n\nYou can continue the process within the 'transform' tab.",
                        icon="ℹ",
                    )
