# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import streamlit as st
import streamlit_antd_components as sac
from streamlit import session_state as state
from streamlit_elements import elements

from ..data_loader import DatasetHelper


def main():
    data_helper: DatasetHelper = state["data_helper"]
    dataset = data_helper.dataset()
    with elements("transform"):
        with st.expander("Category Management"):
            sac.divider(label="Label remapping", icon="map", align="center", bold=False)

        with st.expander("Subset Management"):
            sac.divider(label="Aggregation", icon="columns", align="center", bold=False)
            st.info(
                "This helps to merge all subsets within a dataset into a single **default** subset."
            )
            aggre_subset_btn = st.button("Do aggregation")
            if aggre_subset_btn:
                dataset = data_helper.aggregate(from_subsets=dataset.subsets(), to_subset="default")
                st.toast("Success!", icon="🎉")
                # success = st.success("Success!")
                # time.sleep(1)
                # success.empty()

            sac.divider(label="Split", icon="columns-gap", align="center", bold=False)
            st.info("This helps to divide a dataset into multiple subsets with a given ratio.")
            col1, col2 = st.columns(9)[:2]
            with col1:
                add_subset_btn = st.button("Add subset", use_container_width=True)
            with col2:
                split_btn = st.button("Do split", use_container_width=True)

            if add_subset_btn:
                state["subset"] += 1

            name, ratio = st.columns(5)[:2]
            splits = []
            for idx in range(state["subset"]):
                with name:
                    subset_name = st.text_input(
                        key=f"subset_name_{idx}", label="Enter subset name", value="train"
                    )
                with ratio:
                    subset_ratio = st.text_input(
                        key=f"subset_ratio_{idx}", label="Enter subset ratio", value=0.5
                    )
                splits.append((subset_name, float(subset_ratio)))

            if split_btn:
                dataset = data_helper.transform("random_split", splits=splits)
                state["subset"] = 1
                st.toast("Success!", icon="🎉")

        with st.expander("Item Management"):
            sac.divider(label="Reindexing", icon="stickies-fill", align="center", bold=False)
            st.info("This helps to reidentify all items.")
            col1, col2 = st.columns(6)[:2]
            with col1:
                item_reindex_btn = st.button("Set IDs from 0", use_container_width=True)

            with col2:
                item_media_name_btn = st.button("Set IDs with media name", use_container_width=True)

            if item_reindex_btn:
                dataset = data_helper.transform("reindex", start=0)
                st.toast("Success!", icon="🎉")

            if item_media_name_btn:
                dataset = data_helper.transform("id_from_image_name")
                st.toast("Success!", icon="🎉")

            sac.divider(label="Filtration", icon="funnel-fill", align="center", bold=False)
            sac.divider(label="Remove", icon="eraser-fill", align="center", bold=False)
            st.info("This helps to remove some items or annotations within a dataset.")
            subset, item, annotation = st.columns(5)[:3]

            with subset:
                selected_subset = st.selectbox("Select a subset:", dataset.subsets())

            with item:
                ids = [item.id for item in dataset.get_subset(selected_subset)]
                selected_id = st.selectbox("Select a subset item:", ids)

            with annotation:
                item = dataset.get(selected_id, selected_subset)
                ann_ids = [
                    "All",
                ] + [ann.id for ann in item.annotations]
                selected_ann_id = st.selectbox("Select a item annotation:", ann_ids)

            col1, col2 = st.columns(6)[:2]
            with col1:
                rm_item_btn = st.button("Remove item", use_container_width=True)

            with col2:
                rm_ann_btn = st.button("Remove annotation", use_container_width=True)

            if rm_item_btn:
                dataset = data_helper.transform(
                    "remove_items", ids=[(selected_id, selected_subset)]
                )
                st.toast("Success!", icon="🎉")

            if rm_ann_btn:
                if selected_ann_id == "All":
                    dataset = data_helper.transform(
                        "remove_annotations", ids=[(selected_id, selected_subset)]
                    )
                else:
                    dataset = data_helper.transform(
                        "remove_annotations", ids=[(selected_id, selected_subset, selected_ann_id)]
                    )
                st.toast("Success!", icon="🎉")

            sac.divider(label="Auto-correction", icon="hammer", align="center", bold=False)
            st.info("This helps to correct a dataset and clean up validation report.")

            col1, col2 = st.columns(6)[:2]
            with col1:
                correct_btn = st.button("Correct a dataset", use_container_width=True)

            if correct_btn:
                dataset = data_helper.transform("correct", reports=state["reports"])
                st.toast("Success!", icon="🎉")

            # sac.divider(label='SUBSET MAPPING', icon='code-square', align='center', bold=False)
            # with st.container():
            #     rename_subset_btn = st.button("Rename subsets")
            #     name1, emo, name2 = st.columns(8)[:3]
            #     for idx, subset in enumerate(dataset.subsets()):
            #         with name1:
            #             st.write("")
            #             st.write("\n\n\n\n\n\n")
            #             st.write(str(subset))
            #         with emo:
            #             st.write("")
            #             st.write("\n\n\n\n\n\n")
            #             st.write(":arrow_right:")
            #         with name2:
            #             subset_new_name = st.text_input(
            #                 key=f"subset_new_name_{idx}", label="New subset name", value="train"
            #             )

            #     if rename_subset_btn:
            #         dataset = data_helper.transform("random_split", splits=splits)

        # with mui.Accordion:
        #     with mui.AccordionSummary(expandIcon=mui.icon.ExpandMore, sx={'background': 'grey[900]', 'border': 1}):
        #         mui.Typography("Category Management", fontSize=16)
        #     with mui.AccordionDetails():
        #         with mui.Box(sx={
        #             "flex": 1, "borderTop": 1, "borderBottom": 1,
        #             "borderColor": "divider", "paddingTop": 1, "paddingBottom": 2}
        #         ):
        #             mui.Typography(mui.icon.LooksOne, " Label remapping", fontSize=16)
        #             columns = [
        #                 { "field": 'id', "headerName": 'Label ID', "width": 100},
        #                 { "field": 'cur_name', "headerName": 'Label Name', "width": 200, "editable": False},
        #                 { "field": 'new_name', "headerName": 'New Label Name', "width": 200, "editable": True},
        #                 { "field": 'num_ann', "headerName": 'Number of annotations', "width": 200, "editable": False},
        #             ]

        #             categories = dataset.categories()[AnnotationType.label]
        #             labels = []
        #             for cat_name, idx in categories._indices.items():
        #                 labels.append({'id': idx, 'cur_name': cat_name, 'new_name': cat_name, 'num_ann': None})

        #             board = Dashboard()
        #             w = SimpleNamespace(
        #                 dashboard=board,
        #                 data_grid=DataGrid(board, 0, 0, 8, 7, minH=len(labels)),
        #             )

        #             with w.dashboard(rowHeight=50):
        #                 w.data_grid(data=labels, grid_name="Label schema", columns=columns)

        #             mui.Button("Do label remapping", sx={'border': 1, 'color': 'white', 'background': 'black'})

        # with mui.Accordion:
        #     with mui.AccordionSummary(expandIcon=mui.icon.ExpandMore, sx={'background': 'grey[900]', 'border': 1}):
        #         mui.Typography("Subset Management")
        #     with mui.AccordionDetails():
        #         with mui.Box(sx={
        #             "flex": 1, "borderTop": 1, "borderBottom": 1,
        #             "borderColor": "divider", "paddingTop": 1, "paddingBottom": 2
        #         }):
        #             mui.Typography(mui.icon.LooksOne, " Aggregation", fontSize=16, sx={"paddingBottom": 2})

        #         with mui.Box(sx={
        #             "flex": 1, "borderTop": 1, "borderBottom": 1,
        #             "borderColor": "divider", "paddingTop": 1, "paddingBottom": 2
        #         }):
        #             mui.Typography(mui.icon.LooksTwo, " Split", fontSize=16, sx={"paddingBottom": 2})

        #             with mui.ButtonGroup(sx={"aria-label": "split button"}):
        #                 add_subset_btn = mui.Button("Add subset",
        #                     sx={'border': 1, 'color': 'white', 'background': 'black'}
        #                 )
        #                 split_subset_btn = mui.Button("Do split",
        #                     sx={'border': 1, 'color': 'white', 'background': 'black'}
        #                 )

        #             if add_subset_btn:
        #                 state['subset'] += 1

        #             splits = []
        #             for idx in range(state['subset']):
        #                 mui.Typography("")
        #                 subset_name = mui.TextField(
        #                     id=f"name_{idx}",
        #                     label="subset name", defaultValue="default", variant="outlined", size="small"
        #                 )
        #                 subset_ratio = mui.TextField(
        #                     id=f"ratio_{idx}",
        #                     label="subset ratio", defaultValue="1.0", variant="outlined", size="small"
        #                 )
        #                 print(subset_name, subset_ratio)
        #                 splits.append((subset_name, float(subset_ratio)))

        #             if split_subset_btn:
        #                 dataset = data_helper.transform("random_split", splits=splits)
        #                 print(dataset)
        #                 state['subset'] = 0

        #                 success = st.success("Success!")
        #                 time.sleep(1)
        #                 success.empty()

        # with mui.Accordion:
        #     with mui.AccordionSummary(expandIcon=mui.icon.ExpandMore, sx={'background': 'grey[900]', 'border': 1}):
        #         mui.Typography("Item Management", fontSize=16) #, fontFamily='Helvetica Neue'
        #     with mui.AccordionDetails():
        #         with mui.Box(sx={
        #             "flex": 1, "borderTop": 1, "borderBottom": 1,
        #             "borderColor": "divider", "paddingTop": 1, "paddingBottom": 2
        #         }):
        #             mui.Typography(mui.icon.LooksOne, " ID reindexing ", fontSize=16, sx={"paddingBottom": 1})
        #             with mui.ButtonGroup(sx={"aria-label": "split button"}):
        #                 mui.Button("Set IDs from 0", sx={'border': 1, 'color': 'white', 'background': 'black'})
        #                 mui.Button(
        #                     "Set IDs from media name", sx={'border': 1, 'color': 'white', 'background': 'black'}
        #                 )

        #         with mui.Box(sx={
        #             "flex": 1, "borderTop": 1, "borderBottom": 1,
        #             "borderColor": "divider", "paddingTop": 1, "paddingBottom": 2
        #         }):
        #             mui.Typography("Filtration")
