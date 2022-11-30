from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Mask
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import MultiframeImage
from datumaro.plugins.data_formats.brats import BratsImporter
from datumaro.util.test_utils import compare_datasets

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path

DUMMY_DATASET_DIR = get_test_asset_path("brats_dataset")


class BratsImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_616)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([BratsImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_616)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="BRATS_001",
                    subset="train",
                    media=MultiframeImage(np.ones((2, 1, 5, 4))),
                    annotations=[
                        Mask(np.array([[0, 0, 1, 1, 1]]), label=0, attributes={"image_id": 0}),
                        Mask(np.array([[1, 1, 0, 0, 0]]), label=1, attributes={"image_id": 0}),
                        Mask(np.array([[0, 1, 1, 0, 0]]), label=0, attributes={"image_id": 1}),
                        Mask(np.array([[1, 0, 0, 0, 0]]), label=1, attributes={"image_id": 1}),
                        Mask(np.array([[0, 0, 0, 1, 1]]), label=2, attributes={"image_id": 1}),
                    ],
                ),
                DatasetItem(
                    id="BRATS_002", subset="test", media=MultiframeImage(np.ones((2, 1, 5, 4)))
                ),
            ],
            categories=["overall tumor", "edema", "non-enhancing tumor", "enhancing tumor"],
            media_type=MultiframeImage,
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "brats")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_save_hash(self):
        with self.assertRaises(Exception):
            Dataset.import_from(DUMMY_DATASET_DIR, "brats", save_hash=True)
