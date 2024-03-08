# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import streamlit as st
from datumaro_gui.utils.dataset.data_loader import SingleDatasetHelper
from datumaro_gui.utils.drawing import Bar, Dashboard, DataGrid
from streamlit import session_state as state
from streamlit_elements import elements


def main():
    data_helper: SingleDatasetHelper = state["data_helper"]
    tasks = ["classification", "detection", "segmentation"]
    if "task" not in state:
        state["task"] = tasks[0]

    with elements("general"):
        selected_task = st.selectbox(
            "Select a task for validation:", tasks, index=tasks.index(state["task"])
        )
        if selected_task != state["task"]:
            state["task"] = selected_task
        reports = data_helper.validate(selected_task)

        val_info = {}
        anomaly_info = []
        for idx, report in enumerate(reports["validation_reports"]):
            anomaly_type = report["anomaly_type"]
            val_info[anomaly_type] = val_info.get(anomaly_type, 0) + 1

            anomaly_info.append(
                {
                    "id": idx,
                    "anomaly": anomaly_type,
                    "subset": report.get("subset", "None"),
                    "item_id": report.get("item_id", "None"),
                    "description": report["description"],
                }
            )

        val_info_dict = []
        for type, cnt in val_info.items():
            temp_dict = {
                "id": type,
                "label": type,
                "value": cnt,
            }
            val_info_dict.append(temp_dict)

        board = Dashboard()
        w = SimpleNamespace(
            dashboard=board,
            val_info=Bar(
                name="Validation info",
                **{"board": board, "x": 0, "y": 0, "w": 3, "h": 6, "minW": 2, "minH": 4},
            ),
            data_grid=DataGrid(board, 3, 0, 6, 12, minH=4),
        )

        with w.dashboard(rowHeight=50):
            w.val_info(val_info_dict)
            w.data_grid(data=anomaly_info, grid_name="Validation report")
