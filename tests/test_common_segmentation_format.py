import os.path as osp
from collections import OrderedDict
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Mask
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.common_segmentation_format import CommonSegmentationImporter, make_categories
from datumaro.util.test_utils import compare_datasets

from .requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(
    osp.dirname(__file__), "assets", "common_segmentation_dataset", "dataset"
)

DUMMY_DATASET_WITH_FILE_PREFIXES_DIR = osp.join(
    osp.dirname(__file__), "assets", "common_segmentation_dataset", "dataset_with_file_prefixes"
)


class CommonSegmentationImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([CommonSegmentationImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_with_file_prefixes(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_WITH_FILE_PREFIXES_DIR)
        self.assertEqual([CommonSegmentationImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0001",
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 1, 1]]), label=3),
                        Mask(image=np.array([[0, 0, 1, 0, 0]]), label=5),
                    ],
                ),
                DatasetItem(
                    id="0002",
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(image=np.array([[1, 1, 1, 0, 0]]), label=1),
                        Mask(image=np.array([[0, 0, 0, 1, 1]]), label=4),
                    ],
                ),
            ],
            categories=make_categories(
                OrderedDict(
                    [
                        ("Void", (0, 0, 0)),
                        ("Animal", (64, 128, 64)),
                        ("Archway", (192, 0, 128)),
                        ("Bicyclist", (0, 128, 192)),
                        ("Child", (192, 128, 64)),
                        ("Road", (128, 64, 128)),
                    ]
                )
            ),
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "common_segmentation")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_dataset(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0001",
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 1, 1]]), label=3),
                        Mask(image=np.array([[0, 0, 1, 0, 0]]), label=5),
                    ],
                ),
                DatasetItem(
                    id="0002",
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(image=np.array([[1, 1, 1, 0, 0]]), label=1),
                        Mask(image=np.array([[0, 0, 0, 1, 1]]), label=4),
                    ],
                ),
            ],
            categories=make_categories(
                OrderedDict(
                    [
                        ("Void", (0, 0, 0)),
                        ("Animal", (64, 128, 64)),
                        ("Archway", (192, 0, 128)),
                        ("Bicyclist", (0, 128, 192)),
                        ("Child", (192, 128, 64)),
                        ("Road", (128, 64, 128)),
                    ]
                )
            ),
        )

        dataset = Dataset.import_from(
            DUMMY_DATASET_WITH_FILE_PREFIXES_DIR,
            "common_segmentation",
            image_prefix="image_",
            mask_prefix="gt_",
        )

        compare_datasets(self, expected_dataset, dataset, require_media=True)
