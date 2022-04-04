import os.path as osp
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import ImageAnnotation
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.nyu_depth_v2_format import NyuDepthV2Importer
from datumaro.util.test_utils import compare_datasets

from .requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), "assets", "nyu_depth_v2_dataset")


class NyuDepthV2ImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_497)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([NyuDepthV2Importer.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_497)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    media=Image(data=np.ones((6, 4, 3))),
                    annotations=[ImageAnnotation(Image(data=np.ones((6, 4))))],
                ),
                DatasetItem(
                    id="2",
                    media=Image(data=np.ones((4, 3, 3))),
                    annotations=[ImageAnnotation(Image(data=np.ones((4, 3))))],
                ),
            ]
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "nyu_depth_v2")

        compare_datasets(self, expected_dataset, dataset, require_media=True)
