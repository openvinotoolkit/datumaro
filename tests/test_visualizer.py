from dataclasses import dataclass
from unittest import TestCase, mock

import numpy as np
from matplotlib.figure import Figure

from datumaro.components.annotation import Bbox, Caption, Label, Points, PolyLine, Polygon
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image
from datumaro.components.visualizer import Visualizer

from .requirements import Requirements, mark_requirement


@dataclass
class GridSizeTestCase:
    infer_grid_size: tuple
    expected_grid_size: tuple


class VisualizerTestBase:
    DEFAULT_GRID_SIZE_TEST_CASES = [
        GridSizeTestCase((None, None), (3, 2)),
        GridSizeTestCase((3, None), (3, 2)),
        GridSizeTestCase((None, 2), (3, 2)),
        GridSizeTestCase((3, 2), (3, 2)),
        GridSizeTestCase((None, 5), (1, 5)),
        GridSizeTestCase((5, 1), (5, 1)),
        GridSizeTestCase((5, 1), (5, 1)),
    ]

    def test_vis_one_sample(self, mocked: mock.MagicMock, check_z_order: bool = True):
        visualizer = Visualizer(self.dataset)

        for item in self.items:
            visualizer.vis_one_sample(item.id, self.subset)

            # Check count
            assert mocked.call_count == len(item.annotations)

            # Check z-order
            if check_z_order:
                called_z_order = [call[0][0].z_order for call in mocked.call_args_list]
                assert sorted(called_z_order) == called_z_order

            mocked.reset_mock()

        # Unknown id
        with self.assertRaises(Exception):
            visualizer.vis_one_sample("unknown", self.subset)

        # Unknown subset
        for item in self.items:
            with self.assertRaises(Exception):
                visualizer.vis_one_sample(item.id, "unknown")

    def test_vis_gallery(self, test_cases: GridSizeTestCase):
        visualizer = Visualizer(self.dataset)

        # Too small nrows and ncols
        ids = [item.id for item in self.items]

        cnt = 1
        while cnt * cnt < len(ids):
            cnt += 1

        with self.assertRaises(Exception):
            small_grid_size = (cnt - 1, cnt - 1)
            visualizer.vis_gallery(ids, self.subset, small_grid_size)

        # Infer grid size for 5 items
        def _check(infer_grid_size, expected_grid_size):
            expected_nrows, expected_ncols = expected_grid_size
            fig = visualizer.vis_gallery(ids, self.subset, infer_grid_size)
            self.assertIsInstance(fig, Figure)
            grid_spec = fig.axes[0].get_gridspec()
            self.assertEqual(grid_spec.nrows, expected_nrows)
            self.assertEqual(grid_spec.ncols, expected_ncols)

        for test_case in test_cases:
            _check(test_case.infer_grid_size, test_case.expected_grid_size)


class LabelVisualizerTest(TestCase, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image(data=np.ones((4, 6, 3))),
                annotations=[
                    Label(
                        label=label_idx,
                        id=img_idx * img_idx + label_idx,
                        group=1,
                        attributes={},
                    )
                    for label_idx in range(img_idx)
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @mock.patch("datumaro.components.visualizer.Visualizer._draw_label")
    def test_vis_one_sample(self, mocked: mock.MagicMock):
        super().test_vis_one_sample(mocked, check_z_order=False)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        super().test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class PointsVisualizerTest(TestCase, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image(data=np.ones((4, 6, 3))),
                annotations=[
                    Points(
                        np.random.random_integers(0, 6, size=10 * 2),
                        label=label_idx,
                        id=img_idx * img_idx + label_idx,
                        group=1,
                        z_order=label_idx,
                        attributes={},
                    )
                    for label_idx in range(img_idx)
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @mock.patch("datumaro.components.visualizer.Visualizer._draw_points")
    def test_vis_one_sample(self, mocked: mock.MagicMock):
        super().test_vis_one_sample(mocked, check_z_order=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        super().test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class PolygonVisualizerTest(TestCase, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image(data=np.ones((4, 6, 3))),
                annotations=[
                    Polygon(
                        np.random.random_integers(0, 6, size=10 * 2),
                        label=label_idx,
                        id=img_idx * img_idx + label_idx,
                        group=1,
                        z_order=label_idx,
                        attributes={},
                    )
                    for label_idx in range(img_idx)
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @mock.patch("datumaro.components.visualizer.Visualizer._draw_polygon")
    def test_vis_one_sample(self, mocked: mock.MagicMock):
        super().test_vis_one_sample(mocked, check_z_order=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        super().test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class PolyLineVisualizerTest(TestCase, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image(data=np.ones((4, 6, 3))),
                annotations=[
                    PolyLine(
                        np.random.random_integers(0, 6, size=10 * 2),
                        label=label_idx,
                        id=img_idx * img_idx + label_idx,
                        group=1,
                        z_order=label_idx,
                        attributes={},
                    )
                    for label_idx in range(img_idx)
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @mock.patch("datumaro.components.visualizer.Visualizer._draw_polygon")
    def test_vis_one_sample(self, mocked: mock.MagicMock):
        super().test_vis_one_sample(mocked, check_z_order=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        super().test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class BboxVisualizerTest(TestCase, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image(data=np.ones((4, 6, 3))),
                annotations=[
                    Bbox(
                        0,
                        0,
                        1,
                        1,
                        label=label_idx,
                        id=img_idx * img_idx + label_idx,
                        group=1,
                        z_order=label_idx,
                        attributes={},
                    )
                    for label_idx in range(img_idx)
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @mock.patch("datumaro.components.visualizer.Visualizer._draw_bbox")
    def test_vis_one_sample(self, mocked: mock.MagicMock):
        super().test_vis_one_sample(mocked)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        super().test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)


class CaptionVisualizerTest(TestCase, VisualizerTestBase):
    @classmethod
    def setUpClass(cls):
        cls.subset = "train"
        cls.items = [
            DatasetItem(
                "image_%03d" % img_idx,
                subset=cls.subset,
                media=Image(data=np.ones((4, 6, 3))),
                annotations=[
                    Caption(
                        caption="".join([str(label_idx)] * 10),
                        id=img_idx * img_idx + label_idx,
                        attributes={},
                    )
                    for label_idx in range(img_idx)
                ],
            )
            for img_idx in range(1, 6)
        ]
        cls.dataset = Dataset.from_iterable(cls.items)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @mock.patch("datumaro.components.visualizer.Visualizer._draw_caption")
    def test_vis_one_sample(self, mocked: mock.MagicMock):
        super().test_vis_one_sample(mocked, check_z_order=False)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        super().test_vis_gallery(self.DEFAULT_GRID_SIZE_TEST_CASES)
