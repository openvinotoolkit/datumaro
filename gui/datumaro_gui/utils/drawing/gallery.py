# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import base64
import itertools
import os

import cv2
from streamlit_elements import html, mui

from datumaro.components.media import ImageFromNumpy

from .dashboard import Dashboard


class Gallery(Dashboard.Item):
    IMAGE_FORMATS = {
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".png": "png",
        ".gif": "gif",
        ".bmp": "bmp",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _process_image(self, item):
        if isinstance(item.media, ImageFromNumpy):
            data = item.media.data
            cv2_image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            _, encoded_image = cv2.imencode(".jpg", cv2_image)
            img_format = ".jpg"
            bin_str = base64.b64encode(encoded_image).decode("utf-8")
        else:
            data_path = item.media.path
            img_format = os.path.splitext(data_path)[-1].lower()
            with open(data_path, "rb") as f:
                data = f.read()
                bin_str = base64.b64encode(data).decode()

        return f"data:image/{self.IMAGE_FORMATS.get(img_format, 'jpeg')};base64,{bin_str}"

    def __call__(self, dataset, max_number: int = 100, title="Gallery"):
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
                mui.Typography(title)

            # Create a Streamlit Material-UI Box
            with mui.Box(sx={"flex": 1, "minHeight": 0, "overflow": "auto"}):
                image = [
                    self._process_image(item) for item in itertools.islice(dataset, max_number)
                ]

                with mui.ImageList(cols=4):
                    for im in image:
                        mui.ImageListItem(html.img(src=im))
