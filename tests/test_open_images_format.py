# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

from unittest.case import TestCase

import numpy as np

from datumaro.components.dataset import Dataset
from datumaro.components.extractor import AnnotationType, DatasetItem, Label, LabelCategories
from datumaro.plugins.open_images_format import OpenImagesConverter, OpenImagesImporter
from datumaro.util.image import Image
from datumaro.util.test_utils import TestDir, compare_datasets_strict

from tests.requirements import Requirements, mark_requirement

class OpenImagesFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id='a', subset='train',
                    annotations=[Label(0, attributes={'score': 0.7})]
                ),
                DatasetItem(id='b', subset='train', image=np.zeros((8, 8, 3)),
                    annotations=[Label(1), Label(2, attributes={'score': 0})]
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable([
                    '/m/0',
                    ('/m/1', '/m/0'),
                    '/m/2',
                ]),
            },
        )

        with TestDir() as test_dir:
            OpenImagesConverter.convert(source_dataset, test_dir,
                save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'open_images')

            # the converter assumes that labels without a score have a score of 100%
            source_dataset.get('b', subset='train').annotations[0].attributes['score'] = 1

            compare_datasets_strict(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_no_subsets(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id='a',
                    annotations=[Label(0, attributes={'score': 0.7})]
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(['/m/0']),
            },
        )

        with TestDir() as test_dir:
            OpenImagesConverter.convert(source_dataset, test_dir,
                save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'open_images')

            compare_datasets_strict(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='a/1', image=Image(path='a/1.JPEG',
                data=np.zeros((4, 3, 3)))),
            DatasetItem(id='b/c/d/2', image=Image(path='b/c/d/2.bmp',
                data=np.zeros((3, 4, 3)))),
        ], categories=[])

        with TestDir() as test_dir:
            OpenImagesConverter.convert(dataset, test_dir, save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'open_images')

            compare_datasets_strict(self, dataset, parsed_dataset)

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
