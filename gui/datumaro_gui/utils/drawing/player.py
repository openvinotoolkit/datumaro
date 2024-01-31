# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from streamlit_elements import lazy, media, mui, sync

from .dashboard import Dashboard


class Player(Dashboard.Item):
    YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=CmSKVW1v0xM"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._url = self.YOUTUBE_VIDEO_URL

    def _set_url(self, event):
        self._url = event.target.value

    def __call__(self):
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
                mui.icon.OndemandVideo()
                mui.Typography("Media player")

            with mui.Stack(
                direction="row",
                spacing=2,
                justifyContent="space-evenly",
                alignItems="center",
                sx={"padding": "10px"},
            ):
                mui.TextField(
                    defaultValue=self._url,
                    label="URL",
                    variant="standard",
                    sx={"flex": 0.97},
                    onSubmit=lazy(self._set_url),
                )
                mui.IconButton(
                    mui.icon.PlayCircleFilled,
                    onClick=sync(
                        lambda: media.Player(self._url, controls=True, width="100%", height="100%")
                    ),
                    sx={"color": "primary.main"},
                )
