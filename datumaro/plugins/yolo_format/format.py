# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List

import numpy as np
from attr import attrs, field

from datumaro.components.annotation import Bbox
from datumaro.components.media import Image


class YoloPath:
    DEFAULT_SUBSET_NAME = "train"
    SUBSET_NAMES = ["train", "valid"]
    RESERVED_CONFIG_KEYS = ["backup", "classes", "names"]


@attrs(slots=True, init=False, order=False, eq=False)
class RelativeCoordBbox(Bbox):
    points = field(init=False)
    _points: List[float] = field()
    _image: Image = field()

    def __eq__(self, other):
        if not isinstance(other, Bbox):
            return False

        return (
            np.array_equal(self.points, other.points)
            and self.label == other.label
            and self.z_order == other.z_order
            and self.id == other.id
            and self.group == other.group
            and self.attributes == other.attributes
        )


@property
def _rbbox_points(self):
    img_h, img_w = self._image.size
    return [
        self._points[0] * img_w,
        self._points[1] * img_h,
        self._points[2] * img_w,
        self._points[3] * img_h,
    ]


@_rbbox_points.setter
def _rbbox_points(self, value):
    self._points = value


RelativeCoordBbox.points = _rbbox_points
