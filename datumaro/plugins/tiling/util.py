# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.components.media import BboxIntCoords


def x1y1x2y2_to_cxcywh(x1: int, y1: int, x2: int, y2: int) -> BboxIntCoords:
    assert x1 <= x2
    assert y1 <= y2

    c_x = int((x1 + x2) * 0.5)
    c_y = int((y1 + y2) * 0.5)

    w = x2 - x1
    h = y2 - y1

    return c_x, c_y, w, h


def cxcywh_to_x1y1x2y2(c_x: int, c_y: int, w: int, h: int) -> BboxIntCoords:
    half_w = int(w * 0.5)
    half_h = int(h * 0.5)

    x1 = c_x - half_w
    y1 = c_y - half_h
    x2 = c_x + (w - half_w)
    y2 = c_y + (h - half_h)

    return x1, y1, x2, y2


def clip_x1y1x2y2(
    x1: int, y1: int, x2: int, y2: int, max_x: int, max_y: int, min_x: int = 0, min_y: int = 0
) -> BboxIntCoords:
    assert x1 <= x2
    assert y1 <= y2

    x1 = max(min_x, x1)
    y1 = max(min_y, y1)
    x2 = min(x2, max_x)
    y2 = min(y2, max_y)
    return x1, y1, x2, y2


def x1y1x2y2_to_xywh(x1: int, y1: int, x2: int, y2: int) -> BboxIntCoords:
    return x1, y1, x2 - x1, y2 - y1


def xywh_to_x1y1x2y2(x: int, y: int, w: int, h: int) -> BboxIntCoords:
    return x, y, x + w, y + h
