from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.components.annotation import Cuboid3d, Mask
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import MultiframeImage
from datumaro.plugins.brats_numpy_format import BratsNumpyImporter
from datumaro.util.test_utils import compare_datasets

from .requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'brats_numpy_dataset')

class BratsNumpyImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_616)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([BratsNumpyImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_616)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='HGG_val0',
                media=MultiframeImage([np.ones((1, 5, 3))] * 2),
                annotations=[
                    Mask(np.array([[[0, 0], [0, 1], [1, 1], [1, 0], [1, 0]]]), label=0),
                    Mask(np.array([[[1, 1], [1, 0], [0, 0], [0, 0], [0, 0]]]), label=1),
                    Mask(np.array([[[0, 0], [0, 0], [0, 0], [0, 1], [0, 1]]]), label=2),
                    Cuboid3d(position=[1, 1, 1], rotation=[2, 2, 2])
                ],
            ),
            DatasetItem(id='HGG_val1',
                media=MultiframeImage([np.ones((1, 5, 3))] * 2),
                annotations=[
                    Mask(np.array([[[0, 0], [1, 0], [1, 1], [1, 1], [0, 0]]]), label=0),
                    Mask(np.array([[[1, 1], [0, 1], [0, 0], [0, 0], [1, 0]]]), label=1),
                    Mask(np.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]]]), label=3),
                    Cuboid3d(position=[0, 0, 0], rotation=[1, 1, 1])
                ],
            )
        ], categories=[
            'overall tumor', 'necrotic and non-enhancing tumor', 'edema', 'enhancing tumor'
        ], media_type=MultiframeImage)

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'brats_numpy')

        compare_datasets(self, expected_dataset, dataset, require_media=True)
