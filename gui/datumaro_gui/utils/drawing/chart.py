# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from streamlit_elements import mui, nivo

from .dashboard import Dashboard


class Chart(Dashboard.Item):
    def __init__(self, chart: Dashboard.Chart, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chart = chart
        self._icon = chart.icon

    def __call__(
        self, title, data, icon: Dashboard.Icon = None, legends=True, chart_kwargs: dict = None
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
            with self.title_bar():
                if icon is not None:
                    icon.mui()
                else:
                    self._icon()
                mui.Typography(title, sx={"flex": 1})

            with mui.Box(sx={"flex": 1, "minHeight": 0}):
                if data is None:
                    mui.Typography(
                        "No data",
                        sx={
                            "padding": "15px 15px 5px 40px",
                        },
                    )
                else:
                    kwargs = self._get_default_kwargs_for_nivo_chart(
                        data, self._chart, legends=legends
                    )
                    if chart_kwargs is not None:
                        for key, val in chart_kwargs.items():
                            kwargs[key] = val
                    if self._chart == Dashboard.Chart.Bar:
                        nivo.Bar(**kwargs)
                    elif self._chart == Dashboard.Chart.Pie:
                        nivo.Pie(**kwargs)
                    elif self._chart == Dashboard.Chart.Radar:
                        nivo.Radar(**kwargs)
                    elif self._chart == Dashboard.Chart.Sunburst:
                        nivo.Sunburst(**kwargs)
                    elif self._chart == Dashboard.Chart.ScatterPlot:
                        nivo.ScatterPlot(**kwargs)
