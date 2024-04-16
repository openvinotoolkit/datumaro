from unittest import TestCase

import numpy as np

from datumaro.components.annotation import SuperResolutionAnnotation
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.components.task import TaskType
from datumaro.plugins.data_formats.common_super_resolution import CommonSuperResolutionImporter

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets

DUMMY_DATASET_DIR = get_test_asset_path("common_super_resolution_dataset")


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
                    media=Image.from_numpy(data=np.ones((3, 4, 3))),
                    annotations=[
                        SuperResolutionAnnotation(Image.from_numpy(data=np.ones((10, 20, 3)))),
                    ],
                    attributes={"upsampled": Image.from_numpy(data=np.ones((10, 20, 3)))},
                ),
                DatasetItem(
                    id="2",
                    media=Image.from_numpy(data=np.ones((3, 4, 3))),
                    annotations=[
                        SuperResolutionAnnotation(Image.from_numpy(data=np.ones((10, 20, 3)))),
                    ],
                    attributes={"upsampled": Image.from_numpy(data=np.ones((10, 20, 3)))},
                ),
            ],
            task_type=TaskType.super_resolution,
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "common_super_resolution")

        compare_datasets(self, expected_dataset, dataset, require_media=True)
