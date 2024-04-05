# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from datumaro.components.annotation import Bbox, Mask, Polygon
from datumaro.util.annotation_util import SpatialAnnotation, get_bbox, segment_iou

from .requirements import Requirements, mark_requirement


class SegmentIouTest:
    @pytest.mark.parametrize(
        "a, b, expected_iou",
        [
            (Bbox(0, 0, 2, 2), Bbox(0, 0, 2, 1), 0.5),  # nested
            (Bbox(0, 0, 2, 2), Bbox(1, 0, 2, 2), 1 / 3),  # partially intersecting
            (Bbox(0, 0, 2, 2), Polygon([0, 0, 0, 1, 1, 1, 1, 0]), 0.25),
            (Polygon([0, 0, 0, 2, 2, 2, 2, 0]), Polygon([1, 0, 3, 0, 3, 2, 1, 2]), 1 / 3),
            (Bbox(0, 0, 2, 2), Mask(np.array([[0, 1, 1], [0, 1, 1]])), 1 / 3),
            (Mask(np.array([[1, 1, 0], [1, 1, 0]])), Mask(np.array([[0, 1, 1], [0, 1, 1]])), 1 / 3),
            (Polygon([0, 0, 0, 2, 2, 2, 2, 0]), Mask(np.array([[0, 1, 1], [0, 1, 1]])), 1 / 3),
        ],
    )
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_segment_iou_can_match_shapes(
        self, a: SpatialAnnotation, b: SpatialAnnotation, expected_iou: float
    ):
        assert expected_iou == segment_iou(a, b)

    @pytest.mark.parametrize(
        "a, b, expected_iou",
        [
            (Bbox(0, 0, 2, 2), Bbox(0, 0, 2, 1), 0.5),  # nested
            (Bbox(0, 0, 2, 2), Bbox(1, 0, 2, 2), 0.5),  # partially intersecting
            (Bbox(0, 0, 2, 2), Polygon([0, 0, 0, 1, 1, 1, 1, 0]), 0.25),
            (Polygon([0, 0, 0, 2, 2, 2, 2, 0]), Polygon([1, 0, 3, 0, 3, 2, 1, 2]), 0.5),
            (Bbox(0, 0, 2, 2), Mask(np.array([[0, 1, 1], [0, 1, 1]])), 0.5),
            (Mask(np.array([[1, 1, 0], [1, 1, 0]])), Mask(np.array([[0, 1, 1], [0, 1, 1]])), 0.5),
            (Polygon([0, 0, 0, 2, 2, 2, 2, 0]), Mask(np.array([[0, 1, 1], [0, 1, 1]])), 0.5),
        ],
    )
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_segment_iou_can_match_shapes_as_crowd(
        self, a: SpatialAnnotation, b: SpatialAnnotation, expected_iou: float
    ):
        # In this mode, intersection is divided by the GT object area
        assert expected_iou == segment_iou(a, b, is_crowd=True)

    @pytest.mark.parametrize(
        "a, b, expected_iou",
        [
            (Bbox(0, 0, 2, 2, attributes={"is_crowd": True}), Bbox(1, 0, 2, 2), 0.5),
        ],
    )
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_segment_iou_can_get_is_crowd_from_attribute(
        self, a: SpatialAnnotation, b: SpatialAnnotation, expected_iou: float
    ):
        # In this mode, intersection is divided by the GT object area
        assert expected_iou == segment_iou(a, b, is_crowd="is_crowd")


@pytest.mark.parametrize(
    "obj, expected_bbox",
    [
        ((0, 1, 3, 4), (0, 1, 3, 4)),
        (Bbox(0, 0, 2, 2), (0, 0, 2, 2)),
        (Polygon([0, 0, 0, 1, 1, 1, 1, 0]), (0, 0, 1, 1)),  # polygons don't include the last pixel
        (Polygon([1, 0, 3, 0, 3, 2, 1, 2]), (1, 0, 2, 2)),
        (Mask(np.array([[0, 1, 1], [0, 1, 1]])), (1, 0, 2, 2)),
    ],
)
@mark_requirement(Requirements.DATUM_GENERAL_REQ)
def test_can_get_bbox(obj, expected_bbox):
    assert expected_bbox == tuple(get_bbox(obj))
