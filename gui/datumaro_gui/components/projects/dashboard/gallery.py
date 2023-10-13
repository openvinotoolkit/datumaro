import base64
import os

import streamlit as st
from streamlit_elements import mui, html

from .dashboard import Dashboard

class Gallery(Dashboard.Item):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, dataset):
        with mui.Paper(key=self._key, sx={"display": "flex", "flexDirection": "column", "borderRadius": 3, "overflow": "hidden"}, elevation=1):
            with self.title_bar(padding="10px 15px 10px 15px", dark_switcher=False):
                mui.icon.OndemandVideo()
                mui.Typography("Gallery")

            # Create a Streamlit Material-UI Box
            with mui.Box(sx={"flex": 1, "minHeight": 0, "overflow": "auto"}):
                image = []
                for item in dataset:
                    data_path = item.media.path
                    img_format = os.path.splitext(data_path)[-1].replace(".", "")
                    with open(data_path, "rb") as f:
                        data = f.read()
                    bin_str = base64.b64encode(data).decode()
                    html_code = f"data:image/{img_format};base64,{bin_str}"
                    image.append(html_code)

                with mui.ImageList(cols=4):
                    for im in image:
                        mui.ImageListItem(html.img(src=im))
