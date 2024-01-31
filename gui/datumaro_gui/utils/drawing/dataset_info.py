# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from streamlit_elements import mui

from .dashboard import Dashboard


class DatasetInfoBox(Dashboard.Item):
    def _draw_text_with_icon(self, icon, text, title=False):
        padding = {"top": 5, "right": 15, "bottom": 5, "left": 15}
        if title:
            variant = "h6"
            padding["top"] = 15
        else:
            variant = "body2"
            padding["left"] = 40

        with mui.Stack(
            alignItems="center",
            direction="row",
            spacing=1,
            sx={
                "padding": f"{padding['top']}px {padding['right']}px {padding['bottom']}px {padding['left']}px",
                "borderBottom": 0,
                "borderColor": "divider",
            },
        ):
            if icon:
                icon()
            mui.Typography(text, sx={"flex": 1}, variant=variant)

    def __call__(self, title, data):
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
                mui.icon.Info()  # Collections()
                mui.Typography(title, sx={"flex": 1})

            with mui.Box(
                sx={
                    "flex": 1,
                    "flexDirection": "column",
                    "overflow": "hidden",
                    "overflowY": "scroll",
                    "minHeight": 0,
                    "padding": "5px 15px 5px 15px",
                }
            ):
                n_images = data["n_images"]
                self._draw_text_with_icon(mui.icon.Collections, f"{n_images:,} Images", title=True)
                self._draw_text_with_icon(
                    mui.icon.InfoOutlined, f"{data['n_unique']:,} Unique Images"
                )
                if data["n_repeated"] > 0:
                    self._draw_text_with_icon(
                        mui.icon.ReportProblemOutlined, f"{data['n_repeated']:,} Repeated Image(s)"
                    )
                else:
                    self._draw_text_with_icon(mui.icon.InfoOutlined, "No Repeated Images")
                self._draw_text_with_icon(
                    mui.icon.InfoOutlined,
                    f"Average image size: {data['avg_w']:,.2f} x {data['avg_h']:,.2f}",
                )

                mui.Divider(sx={"padding": "5px"})
                self._draw_text_with_icon(
                    mui.icon.FolderCopy, f"{data['n_subsets']} Subsets", title=True
                )

                mui.Divider(sx={"padding": "5px"})
                n_anns = data["n_anns"]
                n_unannotated = data["n_unannotated"]
                n_labels = data["n_labels"]
                self._draw_text_with_icon(
                    mui.icon.Label, f"{n_anns:,} Annotations with {n_labels:,} Labels", title=True
                )
                if n_unannotated > 0:
                    self._draw_text_with_icon(
                        mui.icon.ReportProblemOutlined, f"{n_unannotated:,} Unannotated Image(s)"
                    )
                else:
                    self._draw_text_with_icon(mui.icon.InfoOutlined, "All images are annotated")
                if n_images > 0:
                    self._draw_text_with_icon(
                        mui.icon.InfoOutlined, f"{n_anns / n_images:,.1f} Annotations / Image"
                    )
                self._draw_text_with_icon(
                    mui.icon.InfoOutlined, f"{n_anns / n_labels:,.1f} Annotations / Label"
                )
