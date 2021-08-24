# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from unittest.case import TestCase
import os.path as osp

import numpy as np

from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.extractor import AnnotationType, LabelCategories, Mask
from datumaro.plugins.ade20k2017_format import Ade20k2017Importer
from datumaro.util.test_utils import compare_datasets

from tests.requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'ade20k2017_dataset')

class Ade20k2017ImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_399)
    def test_can_detect(self):
        self.assertTrue(Ade20k2017Importer.detect(DUMMY_DATASET_DIR))

    @mark_requirement(Requirements.DATUM_399)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(id='street/1', subset='training',
                    image=np.ones((3, 4, 3)),
                    annotations=[
                        Mask(image=np.array([[0, 1, 0, 0]] * 3), label=0,
                            group=1, z_order=0, id=1),
                        Mask(image=np.array([[0, 0, 0, 1]] * 3), label=2,
                            group=1, z_order=1, id=1),
                        Mask(image=np.array([[0, 0, 1, 1]] * 3),
                            group=2, label=1, z_order=0, id=2,
                            attributes={'walkin': True})
                    ]),
                DatasetItem(id='2', subset='validation',
                    image=np.ones((3, 4, 3)),
                    annotations=[
                        Mask(image=np.array([[0, 1, 0, 1]] * 3), label=0, id=1,
                            z_order=0, group=1),
                        Mask(image=np.array([[0, 0, 1, 0]] * 3), label=1, id=2,
                            z_order=0, group=2),
                        Mask(image=np.array([[0, 1, 0, 1]] * 3), label=2, id=1,
                            z_order=1, group=1),
                        Mask(image=np.array([[0, 1, 0, 0]] * 3), label=3, id=1,
                            z_order=2, group=1)
                    ])
            ], categories={AnnotationType.label: LabelCategories.from_iterable([
                    'sky', 'person', 'license plate', 'rim'])
                }
        )

        imported_dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'ade20k2017')
        compare_datasets(self, expected_dataset, imported_dataset,
            require_images=True)
