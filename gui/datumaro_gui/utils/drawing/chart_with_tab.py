# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from streamlit import session_state as state
from streamlit_elements import mui, nivo

from .dashboard import Dashboard


class ChartWithTab(Dashboard.Item):
    def __init__(
        self,
        icon: Dashboard.Icon,
        title: str,
        tabs: list,
        tab_state_key: str = "ChartWithTab.index",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if tab_state_key not in state:
            valid = 0
            for idx, tab in enumerate(tabs):
                if tab["data"] is not None:
                    valid = idx
                    break
            state[tab_state_key] = valid
        self._tab_state_key = tab_state_key
        self._title = title
        self._icon = icon.mui
        self._tabs = (
            tabs  # list of dict. dict should have keys, "title", "data", "chart_type:Chart"
        )

    def _change_tab(self, _, index):
        state[self._tab_state_key] = index

    def __call__(self, legends=True):
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
                if self._icon:
                    self._icon()
                mui.Typography(self._title, sx={"flex": 1})

            with mui.Stack(
                className=self._draggable_class,
                alignItems="center",
                direction="row",
                spacing=1,
                sx={
                    "padding": "0px 15px 0px 15px",
                    "borderBottom": 1,
                    "borderColor": "divider",
                },
            ):
                with mui.Tabs(
                    value=state[self._tab_state_key],
                    onChange=self._change_tab,
                    scrollButtons=True,
                    variant="scrollable",
                    sx={"flex": 1},
                ):
                    for tab in self._tabs:
                        mui.Tab(label=tab["title"], disabled=tab["data"] is None)

            for index, tab in enumerate(self._tabs):
                with mui.Box(sx=self._box_style, hidden=(index != state[self._tab_state_key])):
                    data = tab["data"]
                    if data is None:
                        pass
                    chart = tab["chart_type"]
                    kwargs = self._get_default_kwargs_for_nivo_chart(data, chart, legends=legends)
                    if "chart_kwargs" in tab:
                        for key, val in tab["chart_kwargs"].items():
                            kwargs[key] = val
                    if chart == Dashboard.Chart.Bar:
                        nivo.Bar(**kwargs)
                    elif chart == Dashboard.Chart.Pie:
                        nivo.Pie(**kwargs)
                    elif chart == Dashboard.Chart.Radar:
                        nivo.Radar(**kwargs)
                    elif chart == Dashboard.Chart.Sunburst:
                        nivo.Sunburst(**kwargs)
                    elif chart == Dashboard.Chart.ScatterPlot:
                        nivo.ScatterPlot(**kwargs)
                    else:
                        pass
