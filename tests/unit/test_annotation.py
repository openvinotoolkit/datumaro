# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import List
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import shapely.geometry as sg

from datumaro.components.annotation import (
    Annotations,
    Ellipse,
    ExtractedMask,
    HashKey,
    Mask,
    RotatedBbox,
)
from datumaro.util.image import lazy_image


class EllipseTest:
    @pytest.fixture
    def fxt_ellipses(self) -> List[Ellipse]:
        np.random.seed(3003)
        size = 5
        x1x2 = size * np.random.random([10, 2])
        y1y2 = size * np.random.random([10, 2])

        x1x2.sort(axis=1)
        y1y2.sort(axis=1)

        return [Ellipse(x1, y1, x2, y2) for (x1, x2), (y1, y2) in zip(x1x2, y1y2)]

    def test_get_points(self, fxt_ellipses: List[Ellipse]):
        for ellipse in fxt_ellipses:
            analytical_area = ellipse.get_area()
            numerical_area = sg.Polygon(ellipse.get_points(num_points=360 * 10)).area
            assert np.abs(analytical_area - numerical_area) < 1e-6


class HashKeyTest:
    @pytest.fixture
    def fxt_hashkeys_same(self):
        hash_key = np.random.randint(0, 256, size=(64,), dtype=np.uint8)
        hashkey1 = HashKey(hash_key=hash_key)
        hashkey2 = HashKey(hash_key=hash_key)
        return hashkey1, hashkey2

    @pytest.fixture
    def fxt_hashkeys_diff(self):
        np.random.seed(3003)
        hashkey1 = HashKey(hash_key=np.random.randint(0, 256, size=(64,), dtype=np.uint8))
        hashkey2 = HashKey(hash_key=np.random.randint(0, 256, size=(64,), dtype=np.uint8))
        return hashkey1, hashkey2

    @pytest.mark.parametrize(
        "fxt_hashkeys,expected", [("fxt_hashkeys_same", True), ("fxt_hashkeys_diff", False)]
    )
    def test_compare_hashkey(self, fxt_hashkeys, expected, request):
        hashkey1, hashkey2 = request.getfixturevalue(fxt_hashkeys)
        assert (expected, hashkey1 == hashkey2)


class RotatedBboxTest:
    @pytest.fixture
    def fxt_rot_bbox(self):
        coords = np.random.randint(0, 180, size=(5,), dtype=np.uint8)
        return RotatedBbox(coords[0], coords[1], coords[2], coords[3], coords[4])

    def test_create_polygon(self, fxt_rot_bbox):
        polygon = fxt_rot_bbox.as_polygon()

        expected = RotatedBbox.from_rectangle(polygon)
        assert fxt_rot_bbox == expected


@pytest.fixture
def fxt_index_mask():
    return np.random.randint(0, 10, size=(10, 10))


@pytest.fixture
def fxt_index_mask_file(fxt_index_mask, tmpdir):
    fpath = Path(tmpdir, "mask.png")
    cv2.imwrite(str(fpath), fxt_index_mask)
    yield fpath


class ExtractedMaskTest:
    def test_extracted_mask(self, fxt_index_mask, fxt_index_mask_file):
        index_mask = lazy_image(path=str(fxt_index_mask_file), dtype=np.uint8)
        for index in range(10):
            mask = ExtractedMask(index_mask=index_mask, index=index)
            assert np.allclose(mask.image, (fxt_index_mask == index))


class AnnotationsTest:
    @pytest.mark.parametrize("dtype", [np.uint8, np.int32])
    def test_get_semantic_seg_mask_extracted_mask(self, fxt_index_mask_file, fxt_index_mask, dtype):
        index_mask = lazy_image(path=str(fxt_index_mask_file), dtype=np.uint8)
        annotations = Annotations(
            ExtractedMask(index_mask=index_mask, index=index, label=index) for index in range(10)
        )
        with patch("datumaro.components.annotation.Mask.as_class_mask") as mock_as_class_mask:
            semantic_seg_mask = annotations.get_semantic_seg_mask(ignore_index=255, dtype=dtype)

        assert np.allclose(semantic_seg_mask, fxt_index_mask)
        # It should directly look up index_mask and there is no calling as_class_mask()
        mock_as_class_mask.assert_not_called()

    @pytest.mark.parametrize("dtype", [np.uint8, np.int32])
    def test_get_semantic_seg_mask_extracted_mask_remapping_label(
        self, fxt_index_mask_file, fxt_index_mask, dtype
    ):
        index_mask = lazy_image(path=str(fxt_index_mask_file), dtype=np.uint8)
        annotations = Annotations(
            ExtractedMask(
                index_mask=index_mask,
                index=index,
                label=index % 5,  # Remapping label
            )
            for index in range(10)
        )
        semantic_seg_mask = annotations.get_semantic_seg_mask(ignore_index=255, dtype=dtype)

        # fxt_index_mask % 5 is label-remapped ground truth
        assert np.allclose(semantic_seg_mask, fxt_index_mask % 5)

    @pytest.mark.parametrize("dtype", [np.uint8, np.int32])
    def test_get_semantic_seg_mask_binary_mask(self, fxt_index_mask, dtype):
        annotations = Annotations(
            Mask(
                image=fxt_index_mask == index,
                label=index,
            )
            for index in range(10)
        )
        semantic_seg_mask = annotations.get_semantic_seg_mask(ignore_index=255, dtype=dtype)

        assert np.allclose(semantic_seg_mask, fxt_index_mask)
