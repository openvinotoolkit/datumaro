# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

from unittest.case import TestCase

import numpy as np

from datumaro.plugins.open_images_format import OpenImagesImporter
from datumaro.util.test_utils import compare_datasets_strict
from datumaro.components.extractor import AnnotationType, DatasetItem, Label, LabelCategories
from datumaro.components.dataset import Dataset
from tests.requirements import Requirements, mark_requirement

ASSETS_DIR = osp.join(osp.dirname(__file__), 'assets')

DUMMY_DATASET_DIR_V6 = osp.join(ASSETS_DIR, 'open_images_dataset_v6')
DUMMY_DATASET_DIR_V5 = osp.join(ASSETS_DIR, 'open_images_dataset_v5')

class OpenImagesImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_v6(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(id='a', subset='train', image=np.zeros((8, 6, 3)),
                    annotations=[Label(label=0, attributes={'score': 1})]),
                DatasetItem(id='b', subset='train', image=np.zeros((2, 8, 3)),
                    annotations=[Label(label=0, attributes={'score': 0})]),
                DatasetItem(id='c', subset='test', image=np.ones((10, 5, 3)),
                    annotations=[
                        Label(label=1, attributes={'score': 1}),
                        Label(label=3, attributes={'score': 1}),
                    ]),
                DatasetItem(id='d', subset='validation', image=np.ones((1, 5, 3)),
                    annotations=[]),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable([
                    # The hierarchy file in the test dataset also includes a fake label
                    # /m/x that is set to be /m/0's parent. This is to mimic the real
                    # Open Images dataset, that assigns a nonexistent label as a parent
                    # to all labels that don't have one.
                    '/m/0',
                    ('/m/1', '/m/0'),
                    '/m/2',
                    '/m/3',
                ]),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR_V6, 'open_images')

        compare_datasets_strict(self, expected_dataset, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_v5(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(id='aa', subset='train', image=np.zeros((8, 6, 3))),
                DatasetItem(id='cc', subset='test', image=np.ones((10, 5, 3))),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable([
                    '/m/0',
                    '/m/1',
                ]),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR_V5, 'open_images')

        compare_datasets_strict(self, expected_dataset, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        self.assertTrue(OpenImagesImporter.detect(DUMMY_DATASET_DIR_V6))
        self.assertTrue(OpenImagesImporter.detect(DUMMY_DATASET_DIR_V5))
