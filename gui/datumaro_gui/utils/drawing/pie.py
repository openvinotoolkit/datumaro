# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from streamlit_elements import mui, nivo

from .dashboard import Dashboard


class Pie(Dashboard.Item):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def __call__(self, json_data, legend=True):
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
            with self.title_bar():
                mui.icon.PieChart()
                mui.Typography(self.name, sx={"flex": 1})

            with mui.Box(sx={"flex": 1, "minHeight": 0}):
                kwargs = self._get_default_kwargs_for_nivo_chart(
                    json_data, Dashboard.Chart.Pie, legends=legend
                )
                nivo.Pie(**kwargs)
