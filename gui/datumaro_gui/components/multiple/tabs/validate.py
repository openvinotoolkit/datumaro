# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import streamlit as st
from datumaro_gui.utils.dataset.data_loader import MultipleDatasetHelper
from datumaro_gui.utils.drawing import Bar, Dashboard, DataGrid
from streamlit import session_state as state
from streamlit_elements import elements


def main():
    data_helper_1: MultipleDatasetHelper = state["data_helper_1"]
    data_helper_2: MultipleDatasetHelper = state["data_helper_2"]

    tasks = ["classification", "detection", "segmentation"]
    if "task_1" not in state:
        state["task_1"] = tasks[0]
    if "task_2" not in state:
        state["task_2"] = tasks[0]

    with elements("general"):
        c1, c2 = st.columns(2)
        with c1:
            ## Dataset 1
            selected_task = st.selectbox(
                "Select a task for validation of dataset 1:",
                tasks,
                index=tasks.index(state["task_1"]),
            )
            if selected_task != state["task_1"]:
                state["task_1"] = selected_task
            reports_1 = data_helper_1.validate(selected_task)

            val_info_1 = {}
            anomaly_info_1 = []
            for report in reports_1["validation_reports"]:
                anomaly_type = report["anomaly_type"]
                val_info_1[anomaly_type] = val_info_1.get(anomaly_type, 0) + 1

                anomaly_info_1.append(
                    {
                        "anomaly": anomaly_type,
                        "subset": report.get("subset", "None"),
                        "id": report.get("item_id", "None"),
                        "description": report["description"],
                    }
                )
            val_info_dict_1 = []
            for type, cnt in val_info_1.items():
                temp_dict = {
                    "id": type,
                    "label": type,
                    "value": cnt,
                }
                val_info_dict_1.append(temp_dict)

        with c2:
            ## Dataset 2
            selected_task = st.selectbox(
                "Select a task for validation of dataset 2:",
                tasks,
                index=tasks.index(state["task_2"]),
            )
            if selected_task != state["task_2"]:
                state["task_2"] = selected_task
            reports_2 = data_helper_2.validate(selected_task)

            val_info_2 = {}
            anomaly_info_2 = []
            for report in reports_2["validation_reports"]:
                anomaly_type = report["anomaly_type"]
                val_info_2[anomaly_type] = val_info_2.get(anomaly_type, 0) + 1

                anomaly_info_2.append(
                    {
                        "anomaly": anomaly_type,
                        "subset": report.get("subset", "None"),
                        "id": report.get("item_id", "None"),
                        "description": report["description"],
                    }
                )
            val_info_dict_2 = []
            for type, cnt in val_info_2.items():
                temp_dict = {
                    "id": type,
                    "label": type,
                    "value": cnt,
                }
                val_info_dict_2.append(temp_dict)

            board = Dashboard()
            w = SimpleNamespace(
                dashboard=board,
                val_info_1=Bar(
                    name="Validation info of dataset 1",
                    **{"board": board, "x": 0, "y": 0, "w": 6, "h": 6, "minW": 2, "minH": 4},
                ),
                val_info_2=Bar(
                    name="Validation info",
                    **{"board": board, "x": 6, "y": 0, "w": 6, "h": 6, "minW": 2, "minH": 4},
                ),
                data_grid_1=DataGrid(board, 0, 0, 6, 12, minH=4),
                data_grid_2=DataGrid(board, 6, 0, 6, 12, minH=4),
            )

            with w.dashboard(rowHeight=50):
                w.val_info_1(val_info_dict_1)
                w.val_info_2(val_info_dict_2)
                w.data_grid_1(data=anomaly_info_1, grid_name="Validation report of Dataset 1")
                w.data_grid_2(data=anomaly_info_2, grid_name="Validation report of Dataset 2")
