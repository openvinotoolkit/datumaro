# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import IntEnum
from types import SimpleNamespace
from uuid import uuid4

from datumaro_gui.utils.drawing.css import apply_css_styles, box_style
from streamlit_elements import dashboard, mui


class Dashboard:
    DRAGGABLE_CLASS = "draggable"

    def __init__(self):
        self._layout = []

    def _register(self, item):
        self._layout.append(item)

    @contextmanager
    def __call__(self, title=None, **props):
        # Draggable classname query selector.
        props["draggableHandle"] = f".{Dashboard.DRAGGABLE_CLASS}"
        css_dict = apply_css_styles(box_style, "yellow", "stat_highlight")
        if title is not None:
            mui.Typography(title, style=css_dict, variant="h5")
        with dashboard.Grid(self._layout, **props):
            yield

    class Icon(IntEnum):
        Label, Info, Warning, PieChart, Radar, Collections, ScatterPlot = range(7)

        @property
        def mui(self):
            return {
                self.Label: mui.icon.Label,
                self.Info: mui.icon.Info,
                self.Warning: mui.icon.Warning,
                self.PieChart: mui.icon.PieChart,
                self.Radar: mui.icon.Radar,
                self.Collections: mui.icon.Collections,
                self.ScatterPlot: mui.icon.ScatterPlot,
            }.get(self)

    class Chart(IntEnum):
        Bar, Pie, Radar, Sunburst, ScatterPlot = range(5)

        @property
        def icon(self):
            return {
                self.Bar: mui.icon.BarChart,
                self.Pie: mui.icon.PieChart,
                self.Radar: mui.icon.Radar,
                self.Sunburst: mui.icon.DonutLarge,
                self.ScatterPlot: mui.icon.ScatterPlot,
            }.get(self)

    class Item(ABC):
        def __init__(self, board, x, y, w, h, **item_props):
            self._key = str(uuid4())
            self._draggable_class = Dashboard.DRAGGABLE_CLASS
            self._dark_mode = True
            self._box_style = {
                "flex": 1,
                "minHeight": 0,
                "borderBottom": 1,
                "borderTop": 1,
                "borderColor": "divider",
            }
            self._theme = {
                "dark": {
                    "background": "#252526",
                    "textColor": "#FAFAFA",
                    "tooltip": {
                        "container": {
                            "background": "#3F3F3F",
                            "color": "FAFAFA",
                        }
                    },
                },
                "light": {
                    "background": "#FFFFFF",
                    "textColor": "#31333F",
                    "tooltip": {
                        "container": {
                            "background": "#FFFFFF",
                            "color": "#31333F",
                        }
                    },
                },
            }
            board._register(dashboard.Item(self._key, x, y, w, h, **item_props))

        def _switch_theme(self):
            self._dark_mode = not self._dark_mode

        def _get_default_kwargs_for_nivo_chart(self, data, chart, legends=True):
            sn = SimpleNamespace(
                data=data,
                theme=self._theme["dark" if self._dark_mode else "light"],
                valueFormat=" >-,",
            )
            sn.borderColor = {"from": "color"}
            sn.defs = [
                {
                    "id": "dots",
                    "type": "patternDots",
                    "background": "inherit",
                    "color": "rgba(255, 255, 255, 0.3)",
                    "size": 4,
                    "padding": 1,
                    "stagger": True,
                },
                {
                    "id": "lines",
                    "type": "patternLines",
                    "background": "inherit",
                    "color": "rgba(255, 255, 255, 0.3)",
                    "rotation": -45,
                    "lineWidth": 6,
                    "spacing": 10,
                },
            ]
            sn.fill = [
                {"match": {"id": "ruby"}, "id": "dots"},
                {"match": {"id": "c"}, "id": "dots"},
                {"match": {"id": "go"}, "id": "dots"},
                {"match": {"id": "python"}, "id": "dots"},
                {"match": {"id": "scala"}, "id": "lines"},
                {"match": {"id": "lisp"}, "id": "lines"},
                {"match": {"id": "elixir"}, "id": "lines"},
                {"match": {"id": "javascript"}, "id": "lines"},
            ]

            if chart == Dashboard.Chart.Radar:
                sn.margin = {"top": 70, "right": 80, "bottom": 40, "left": 80}
            else:
                sn.margin = {"top": 40, "right": 80, "bottom": 80, "left": 80}

            if chart == Dashboard.Chart.Pie:
                sn.innerRadius = 0.5
                sn.padAngle = 0.7
                sn.cornerRadius = 3
                sn.activeOuterRadiusOffset = 8
                sn.borderWidth = 1
                sn.borderColor = {
                    "from": "color",
                    "modifiers": [
                        [
                            "darker",
                            0.2,
                        ]
                    ],
                }
                sn.arcLinkLabelsSkipAngle = 10
                sn.arcLinkLabelsTextColor = "grey"
                sn.arcLinkLabelsThickness = 2
                sn.arcLinkLabelsColo = {"from": "color"}
                sn.arcLabelsSkipAngle = (10,)
                sn.arcLabelsTextColor = ({"from": "color", "modifiers": [["darker", 2]]},)
            elif chart == Dashboard.Chart.Bar:
                sn.padding = 0.3
                sn.innerPadding = 0
                sn.labelSkipWidth = (10,)
                sn.labelSkipHeight = (10,)
                sn.labelTextColor = {"from": "color", "modifiers": [["darker", 2]]}
            elif chart == Dashboard.Chart.Radar:
                sn.gridLabelOffset = 36
                sn.dotSize = 10
                sn.dotColor = {"theme": "background"}
                sn.dotBorderWidth = 2
                sn.motionConfig = "wobbly"
                sn.blendMode = "multiply"
            # elif chart == Dashboard.Chart.Sunburst:
            #     sn.borderColor = { "theme": "background" }

            if legends is True:
                if chart == Dashboard.Chart.Radar:
                    sn.legends = [
                        {
                            "anchor": "top-left",
                            "direction": "column",
                            "translateX": -50,
                            "translateY": -40,
                            "itemWidth": 80,
                            "itemHeight": 20,
                            "itemTextColor": "#999",
                            "symbolSize": 12,
                            "symbolShape": "circle",
                            "effects": [{"on": "hover", "style": {"itemTextColor": "#000"}}],
                        }
                    ]
                else:
                    sn.legends = [
                        {
                            "anchor": "bottom",
                            "direction": "row",
                            "justify": False,
                            "translateX": 0,
                            "translateY": 56,
                            "itemsSpacing": 0,
                            "itemWidth": 100,
                            "itemHeight": 18,
                            "itemTextColor": "#999",
                            "itemDirection": "left-to-right",
                            "itemOpacity": 1,
                            "symbolSize": 18,
                            "symbolShape": "circle",
                            "effects": [{"on": "hover", "style": {"itemTextColor": "#000"}}],
                        }
                    ]
            return vars(sn)

        @contextmanager
        def title_bar(self, padding="5px 15px 5px 15px", dark_switcher=True):
            with mui.Stack(
                className=self._draggable_class,
                alignItems="center",
                direction="row",
                spacing=1,
                sx={
                    "padding": padding,
                    "borderBottom": 1,
                    "borderColor": "divider",
                },
            ):
                yield

                if dark_switcher:
                    if self._dark_mode:
                        mui.IconButton(mui.icon.DarkMode, onClick=self._switch_theme)
                    else:
                        mui.IconButton(
                            mui.icon.LightMode, sx={"color": "#ffc107"}, onClick=self._switch_theme
                        )

        @abstractmethod
        def __call__(self):
            """Show elements."""
            raise NotImplementedError
