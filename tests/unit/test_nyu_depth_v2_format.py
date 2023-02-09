from unittest import TestCase

import numpy as np

from datumaro.components.annotation import DepthAnnotation
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.nyu_depth_v2 import NyuDepthV2Importer
from datumaro.util.test_utils import compare_datasets

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path

DUMMY_DATASET_DIR = get_test_asset_path("nyu_depth_v2_dataset")


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
                    annotations=[DepthAnnotation(Image(data=np.ones((6, 4))))],
                ),
                DatasetItem(
                    id="2",
                    media=Image(data=np.ones((4, 3, 3))),
                    annotations=[DepthAnnotation(Image(data=np.ones((4, 3))))],
                ),
            ]
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "nyu_depth_v2")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_497)
    def test_save_hash(self):
        imported_dataset = Dataset.import_from(DUMMY_DATASET_DIR, "nyu_depth_v2", save_hash=True)
        for item in imported_dataset:
            self.assertTrue(bool(item.hash_key))
