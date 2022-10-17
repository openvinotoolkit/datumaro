from unittest import TestCase, mock

import numpy as np

from datumaro.components.annotation import Bbox
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image
from datumaro.components.visualizer import Visualizer

from .requirements import Requirements, mark_requirement


class BboxVisualizerTest(TestCase):
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
    def test_vis_one_sample(self):
        visualizer = Visualizer(self.dataset)

        for item in self.items:
            with mock.patch("datumaro.components.visualizer.Visualizer._draw_bbox") as mocked:
                visualizer.vis_one_sample(item.id, self.subset)

                # Check count
                assert mocked.call_count == len(item.annotations)

                # Check z-order
                called_z_order = [call.args[0].z_order for call in mocked.call_args_list]
                assert sorted(called_z_order) == called_z_order

        # Unknown id
        with self.assertRaises(Exception):
            visualizer.vis_one_sample("unknown", self.subset)

        # Unknown subset
        for item in self.items:
            with self.assertRaises(Exception):
                visualizer.vis_one_sample(item.id, "unknown")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_vis_gallery(self):
        visualizer = Visualizer(self.dataset)

        # Too small nrows and ncols
        ids = [item.id for item in self.items]

        cnt = 1
        while cnt * cnt < len(ids):
            cnt += 1

        with self.assertRaises(Exception):
            visualizer.vis_gallery(ids, cnt - 1, cnt - 1)
