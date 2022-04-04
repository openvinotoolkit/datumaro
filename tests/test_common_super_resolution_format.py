import os.path as osp
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import ImageAnnotation
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.common_super_resolution_format import CommonSuperResolutionImporter
from datumaro.util.test_utils import compare_datasets

from .requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), "assets", "common_super_resolution_dataset")


class CommonSuperResolutionImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([CommonSuperResolutionImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    media=Image(data=np.ones((3, 4, 3))),
                    annotations=[
                        ImageAnnotation(Image(data=np.ones((10, 20, 3)))),
                    ],
                    attributes={"upsampled": Image(data=np.ones((10, 20, 3)))},
                ),
                DatasetItem(
                    id="2",
                    media=Image(data=np.ones((3, 4, 3))),
                    annotations=[
                        ImageAnnotation(Image(data=np.ones((10, 20, 3)))),
                    ],
                    attributes={"upsampled": Image(data=np.ones((10, 20, 3)))},
                ),
            ]
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "common_super_resolution")

        compare_datasets(self, expected_dataset, dataset, require_media=True)
