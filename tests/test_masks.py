from unittest import TestCase

import numpy as np
import pytest

import datumaro.util.mask_tools as mask_tools
from datumaro.components.annotation import CompiledMask
from datumaro.util.annotation_util import BboxCoords

from .requirements import Requirements, mark_requirement


def _compare_polygons(a: mask_tools.Polygon, b: mask_tools.Polygon) -> bool:
    a = mask_tools.close_polygon(mask_tools.simplify_polygon(a))[:-2]
    b = mask_tools.close_polygon(mask_tools.simplify_polygon(b))[:-2]
    if len(a) != len(b):
        return False

    a_points = np.reshape(a, (-1, 2))
    b_points = np.reshape(b, (-1, 2))
    for b_direction in [1, -1]:
        # Polygons can be reversed, need to check both directions
        b_ordered = b_points[::b_direction]

        for b_pos in range(len(b_ordered)):
            b_current = b_ordered
            if b_pos > 0:
                b_current = np.roll(b_current, b_pos, axis=0)

            if np.array_equal(a_points, b_current):
                return True

    return False


def _compare_polygon_groups(a: mask_tools.PolygonGroup, b: mask_tools.PolygonGroup) -> bool:
    def _deduplicate(group: mask_tools.PolygonGroup) -> mask_tools.PolygonGroup:
        unique = list()

        for polygon in group:
            found = False
            for existing_polygon in unique:
                if _compare_polygons(polygon, existing_polygon):
                    found = True
                    break

            if not found:
                unique.append(polygon)

        return unique

    a = _deduplicate(a)
    b = _deduplicate(b)

    return len(a) == len(b) and len(a) == len(_deduplicate(a + b))


class PolygonConversionsTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_mask_can_be_converted_to_polygon(self):
        mask = np.array(
            [
                [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        expected = [
            [1, 0, 3, 0, 3, 2, 1, 0],
            [5, 0, 8, 0, 5, 3],
        ]

        computed = mask_tools.mask_to_polygons(mask)

        self.assertTrue(_compare_polygon_groups(expected, computed))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_crop_covered_segments(self):
        image_size = [7, 7]
        initial = [
            [1, 1, 6, 1, 6, 6, 1, 6],  # rectangle polygon
            mask_tools.mask_to_rle(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 1, 0],
                        [0, 1, 1, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 1, 1, 0, 0, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                    ]
                )
            ),  # compressed RLE
            mask_tools.to_uncompressed_rle(
                mask_tools.mask_to_rle(
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                        ]
                    )
                ),
                width=image_size[1],
                height=image_size[0],
            ),  # uncompressed RLE
            [1, 1, 6, 6, 1, 6],  # lower-left triangle polygon
        ]
        expected = [
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),  # half-covered
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),  # half-covered
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),  # unchanged
            mask_tools.rles_to_mask([initial[3]], *image_size),  # unchanged
        ]

        computed = mask_tools.crop_covered_segments(
            initial, *image_size, ratio_tolerance=0, return_masks=True
        )

        self.assertEqual(len(initial), len(computed))
        for i, (e_mask, c_mask) in enumerate(zip(expected, computed)):
            self.assertTrue(np.array_equal(e_mask, c_mask), "#%s: %s\n%s\n" % (i, e_mask, c_mask))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_crop_covered_segments_and_avoid_holes_from_objects_inside_background_object(self):
        image_size = [7, 7]
        initial = [
            [1, 1, 6, 1, 6, 6, 1, 6],
            mask_tools.mask_to_rle(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                    ]
                )
            ),
        ]
        expected = [
            # no changes expected
            mask_tools.rles_to_mask([initial[0]], *image_size),
            mask_tools.rles_to_mask([initial[1]], *image_size),
        ]

        computed = mask_tools.crop_covered_segments(
            initial, *image_size, ratio_tolerance=0.1, return_masks=True
        )

        self.assertEqual(len(initial), len(computed))
        for i, (e_mask, c_mask) in enumerate(zip(expected, computed)):
            self.assertTrue(np.array_equal(e_mask, c_mask), "#%s: %s\n%s\n" % (i, e_mask, c_mask))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_crop_covered_segments_and_crop_fully_covered_background_segments(self):
        image_size = [7, 7]
        initial = [
            mask_tools.mask_to_rle(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                    ]
                )
            ),
            [1, 1, 6, 1, 6, 6, 1, 6],
        ]
        expected = [
            # The fully-covered background mask must be cropped
            None,
            mask_tools.rles_to_mask([initial[1]], *image_size),
        ]

        computed = mask_tools.crop_covered_segments(
            initial, *image_size, ratio_tolerance=0.1, return_masks=True
        )

        self.assertEqual(len(initial), len(computed))
        for i, (e_mask, c_mask) in enumerate(zip(expected, computed)):
            self.assertTrue(np.array_equal(e_mask, c_mask), "#%s: %s\n%s\n" % (i, e_mask, c_mask))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_crop_covered_segments_and_keep_input_shape_type(self):
        image_size = [7, 7]
        initial = [
            [1, 1, 6, 1, 6, 6, 1, 6],
            mask_tools.mask_to_rle(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                    ]
                )
            ),
        ]
        expected = [
            # no changes expected
            [initial[0]],
            initial[1],
        ]

        computed = mask_tools.crop_covered_segments(
            initial, *image_size, ratio_tolerance=0.1, return_masks=False
        )

        self.assertEqual(len(initial), len(computed))
        for i, (e_segm, c_segm) in enumerate(zip(expected, computed)):
            if mask_tools.is_polygon_group(e_segm):
                self.assertTrue(
                    _compare_polygon_groups(e_segm, c_segm), "#%s: %s\n%s\n" % (i, e_segm, c_segm)
                )
            else:
                e_segm = mask_tools.rles_to_mask([e_segm], *image_size)
                self.assertTrue(
                    np.array_equal(e_segm, c_segm), "#%s: %s\n%s\n" % (i, e_segm, c_segm)
                )

    def _test_mask_to_rle(self, source_mask):
        rle_uncompressed = mask_tools.mask_to_rle(source_mask)

        from pycocotools import mask as mask_utils

        resulting_mask = mask_utils.frPyObjects(rle_uncompressed, *rle_uncompressed["size"])
        resulting_mask = mask_utils.decode(resulting_mask)

        self.assertTrue(
            np.array_equal(source_mask, resulting_mask), "%s\n%s\n" % (source_mask, resulting_mask)
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_mask_to_rle_multi(self):
        cases = [
            np.array(
                [
                    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            np.array([[0]]),
            np.array([[1]]),
            np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                    [1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                    [1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                    [1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
                ]
            ),
        ]

        for case in cases:
            self._test_mask_to_rle(case)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_close_open_polygon(self):
        source = [1, 1, 2, 3, 4, 5]
        expected = [1, 1, 2, 3, 4, 5, 1, 1]

        actual = mask_tools.close_polygon(source)

        self.assertListEqual(expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_close_closed_polygon(self):
        source = [1, 1, 2, 3, 4, 5, 1, 1]
        expected = [1, 1, 2, 3, 4, 5, 1, 1]

        actual = mask_tools.close_polygon(source)

        self.assertListEqual(expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_close_polygon_with_no_points(self):
        source = []
        expected = []

        actual = mask_tools.close_polygon(source)

        self.assertListEqual(expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_simplify_polygon(self):
        source = [1, 1, 1, 1, 2, 3, 4, 5, 4, 5]
        expected = [1, 1, 2, 3, 4, 5]

        actual = mask_tools.simplify_polygon(source)

        self.assertListEqual(expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_simplify_polygon_with_less_3_points(self):
        source = [1, 1]
        expected = [1, 1, 1, 1, 1, 1]

        actual = mask_tools.simplify_polygon(source)

        self.assertListEqual(expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_compare_polygons(self):
        a = [1, 1, 2, 3, 4, 4, 5, 6, 1, 1]
        b_variants = [
            [2, 3, 4, 4, 5, 6, 1, 1],
            [4, 4, 5, 6, 1, 1, 2, 3],
            [5, 6, 1, 1, 2, 3, 4, 4],
            [1, 1, 2, 3, 4, 4, 5, 6],
        ]

        for b in b_variants:
            self.assertTrue(_compare_polygons(a, b), b)


class ColormapOperationsTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_paint_mask(self):
        mask = np.zeros((1, 3), dtype=np.uint8)
        mask[:, 0] = 0
        mask[:, 1] = 1
        mask[:, 2] = 2

        colormap = mask_tools.generate_colormap(3)

        expected = np.zeros((*mask.shape, 3), dtype=np.uint8)
        expected[:, 0] = colormap[0][::-1]
        expected[:, 1] = colormap[1][::-1]
        expected[:, 2] = colormap[2][::-1]

        actual = mask_tools.paint_mask(mask, colormap)

        self.assertTrue(np.array_equal(expected, actual), "%s\nvs.\n%s" % (expected, actual))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_unpaint_mask(self):
        colormap = mask_tools.generate_colormap(3)
        inverse_colormap = mask_tools.invert_colormap(colormap)

        mask = np.zeros((1, 3, 3), dtype=np.uint8)
        mask[:, 0] = colormap[0][::-1]
        mask[:, 1] = colormap[1][::-1]
        mask[:, 2] = colormap[2][::-1]

        expected = np.zeros((1, 3), dtype=np.uint8)
        expected[:, 0] = 0
        expected[:, 1] = 1
        expected[:, 2] = 2

        actual = mask_tools.unpaint_mask(mask, inverse_colormap)

        self.assertTrue(np.array_equal(expected, actual), "%s\nvs.\n%s" % (expected, actual))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_unpaint_incorrect_mask(self):
        colormap = mask_tools.generate_colormap(3)
        inverse_colormap = mask_tools.invert_colormap(colormap)

        mask = np.zeros((1, 3, 3), dtype=np.uint8)
        mask[:, 0] = colormap[0][::-1]
        mask[:, 1] = colormap[1][::-1]
        mask[:, 2] = (255, 255, 255)

        with self.assertRaises(KeyError):
            mask_tools.unpaint_mask(mask, inverse_colormap)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_remap_mask(self):
        class_count = 10
        remap_fn = lambda c: class_count - c

        src = np.empty((class_count, class_count), dtype=np.uint8)
        for c in range(class_count):
            src[c:, c:] = c

        expected = np.empty_like(src)
        for c in range(class_count):
            expected[c:, c:] = remap_fn(c)

        actual = mask_tools.remap_mask(src, remap_fn)

        self.assertTrue(np.array_equal(expected, actual), "%s\nvs.\n%s" % (expected, actual))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_masks(self):
        masks = [
            np.array([0, 2, 4, 0, 0, 1]),
            np.array([0, 1, 1, 0, 2, 0]),
            np.array([0, 0, 2, 3, 0, 0]),
        ]
        expected = np.array([0, 1, 2, 3, 2, 1])

        actual = mask_tools.merge_masks(masks)

        self.assertTrue(np.array_equal(expected, actual), "%s\nvs.\n%s" % (expected, actual))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_decode_compiled_mask(self):
        class_idx = 1000
        instance_idx = 10000
        mask = np.array([1])
        compiled_mask = CompiledMask(mask * class_idx, mask * instance_idx)

        labels = compiled_mask.get_instance_labels()

        self.assertEqual({instance_idx: class_idx}, labels)


class MaskTest:
    @pytest.mark.parametrize(
        "mask, expected_bbox",
        [
            (np.array([[0, 1, 1], [0, 1, 1]]), [1, 0, 2, 2]),
            (np.array([[0, 0, 0], [0, 0, 0]]), [0, 0, 0, 0]),
        ],
    )
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_find_mask_bbox(self, mask: mask_tools.BinaryMask, expected_bbox: BboxCoords):
        assert tuple(expected_bbox) == mask_tools.find_mask_bbox(mask)
