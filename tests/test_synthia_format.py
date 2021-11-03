
from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.components.annotation import (
    AnnotationType, Bbox, Cuboid3d, Label, LabelCategories, MaskCategories, Points, PointsCategories,
)
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem
import datumaro.plugins.synthia_format as Synthia
from datumaro.plugins.synthia_format import SynthiaImporter
from datumaro.util.test_utils import compare_datasets

from .requirements import Requirements, mark_requirement
from datumaro.util.mask_tools import generate_colormap

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets',
    'synthia_dataset')

class SynthiaImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='19-10-2018_12-47-37/000101', subset='train',
                image=np.ones((5, 5, 3)),
                annotations=[
                    Bbox(59.13, 228.35, 71.78, 265.96, label=10, group=1),
                    Cuboid3d(position=[1.73, 0.54, 0.39],
                        scale=[-11.79, 1.19, 41.49],
                        rotation=[0.11, 0.11, 0.11], group=1),
                    Bbox(311.52, 155.15, 317.13, 167.96, label=13, group=2),
                    Cuboid3d(position=[0.64, 0.23, 0.25],
                        scale=[-0.29, -3.72, 46.09],
                        rotation=[-2.94, -2.94, -2.94], group=2)
                ],
            ),
            DatasetItem(id='19-10-2018_12-47-37/000102', subset='train',
                image=np.ones((5, 5, 3)),
                annotations=[
                    Bbox(311.46, 154.88, 317.08, 167.72, label=13, group=1),
                    Cuboid3d(position=[0.64, 0.23, 0.25],
                        scale=[-0.29, -3.72, 45.95],
                        rotation=[-2.94, -2.94, -2.94], group=1),
                    Bbox(0.0, 234.16, 49.46, 263.56, label=8, group=2),
                    Cuboid3d(position=[1.73, 2.09, 4.5],
                        scale=[-18.95, 1.4, 54.22],
                        rotation=[-3.14, -3.14, -3.14], group=2)
                ],
            ),
            DatasetItem(id='19-10-2018_12-47-37/000103', subset='train',
                image=np.ones((5, 5, 3)),
                annotations=[
                    Bbox(0.0, 234.13, 45.54, 263.64, label=8, group=1),
                    Cuboid3d(position=[1.73, 2.09, 4.5],
                        scale=[-19.13, 1.4, 54.03],
                        rotation=[-3.14, -3.14, -3.14], group=1)
                ],
            )
        ], categories=Synthia.make_cityscapes_categories())

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'synthia')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        self.assertTrue(SynthiaImporter.detect(DUMMY_DATASET_DIR))