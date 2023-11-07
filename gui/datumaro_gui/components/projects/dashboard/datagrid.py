# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
from streamlit_elements import mui

from .dashboard import Dashboard


class DataGrid(Dashboard.Item):
    DEFAULT_COLUMNS = [
        {"field": "anomaly", "headerName": "Anomaly", "width": 200},
        {
            "field": "subset",
            "headerName": "Subset",
            "width": 100,
            "editable": True,
        },
        {
            "field": "item_id",
            "headerName": "ID",
            "width": 300,
            "editable": True,
        },
        {
            "field": "description",
            "headerName": "Discription",
            "width": 500,
            "editable": True,
        },
    ]

    def _handle_edit(self, params):
        print(params)

    def __call__(
        self,
        data,
        grid_icon: Dashboard.Icon = None,
        grid_name=None,
        columns=None,
        checkbox_selection=True,
    ):
        with mui.Paper(
            key=self._key,
            sx={
                "display": "flex",
                "flexDirection": "column",
                "borderRadius": 3,
                "overflow": "hidden",
            },
            elevation=1,
        ):
            with self.title_bar(padding="10px 15px 10px 15px", dark_switcher=False):
                if grid_icon is not None:
                    grid_icon.mui()
                else:
                    mui.icon.ViewCompact()
                mui.Typography(grid_name) if grid_name else mui.Typograph("Data grid")

            with mui.Box(sx={"flex": 1, "minHeight": 0}):
                mui.DataGrid(
                    columns=columns if columns else self.DEFAULT_COLUMNS,
                    rows=data,
                    pageSize=100,
                    rowsPerPageOptions=[10],
                    checkboxSelection=checkbox_selection,
                    disableSelectionOnClick=True,
                    onCellEditCommit=self._handle_edit,
                )
