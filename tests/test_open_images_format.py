# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from unittest.case import TestCase
import os
import os.path as osp

import numpy as np

from datumaro.components.dataset import Dataset
from datumaro.components.extractor import (
    AnnotationType, Bbox, DatasetItem, Label, LabelCategories,
)
from datumaro.plugins.open_images_format import (
    OpenImagesConverter, OpenImagesImporter,
)
from datumaro.util.image import Image
from datumaro.util.test_utils import TestDir, compare_datasets

from tests.requirements import Requirements, mark_requirement


class OpenImagesFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_274)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id='a', subset='train',
                    annotations=[Label(0, attributes={'score': 0.7})]
                ),
                DatasetItem(id='b', subset='train', image=np.zeros((8, 8, 3)),
                    annotations=[
                        Label(1),
                        Label(2, attributes={'score': 0}),
                        Bbox(label=0, x=4, y=3, w=2, h=3),
                        Bbox(label=1, x=2, y=3, w=6, h=1, attributes={
                            'score': 0.7,
                            'occluded': True, 'truncated': False,
                            'is_group_of': True, 'is_depiction': False,
                            'is_inside': False,
                        }),
                    ]
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

        expected_dataset = Dataset.from_extractors(source_dataset)
        expected_dataset.put(
            DatasetItem(id='b', subset='train', image=np.zeros((8, 8, 3)),
                annotations=[
                    # the converter assumes that annotations without a score
                    # have a score of 100%
                    Label(1, attributes={'score': 1}),
                    Label(2, attributes={'score': 0}),
                    Bbox(label=0, x=4, y=3, w=2, h=3, attributes={'score': 1}),
                    Bbox(label=1, x=2, y=3, w=6, h=1, attributes={
                        'score': 0.7,
                        'occluded': True, 'truncated': False,
                        'is_group_of': True, 'is_depiction': False,
                        'is_inside': False,
                    }),
                ]
            ),
        )

        with TestDir() as test_dir:
            OpenImagesConverter.convert(source_dataset, test_dir,
                save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'open_images')

            compare_datasets(self, expected_dataset, parsed_dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_274)
    def test_can_save_and_load_with_no_subsets(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a',
                annotations=[Label(0, attributes={'score': 0.7})]
            ),
        ], categories=['/m/0'])

        with TestDir() as test_dir:
            OpenImagesConverter.convert(source_dataset, test_dir,
                save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'open_images')

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_274)
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

            compare_datasets(self, dataset, parsed_dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_274)
    def test_inplace_save_writes_only_updated_data(self):
        dataset = Dataset.from_iterable([
            DatasetItem('a', subset='modified', image=np.ones((2, 1, 3)),
                annotations=[
                    Label(0, attributes={'score': 1}),
                    Bbox(0, 0, 1, 2, label=0),
                ]),
            DatasetItem('b', subset='modified', image=np.ones((2, 1, 3)),
                annotations=[
                    Label(1, attributes={'score': 1}),
                ]),
            DatasetItem('c', subset='removed', image=np.ones((3, 2, 3)),
                annotations=[Label(2, attributes={'score': 1})]),
            DatasetItem('d', subset='unmodified', image=np.ones((4, 3, 3)),
                annotations=[Label(3, attributes={'score': 1})]),
        ], categories=['/m/0', '/m/1', '/m/2', '/m/3'])

        with TestDir() as path:
            dataset.export(path, 'open_images', save_images=True)

            dataset.put(DatasetItem('e', subset='new', image=np.ones((5, 4, 3)),
                annotations=[Label(1, attributes={'score': 1})]))
            dataset.remove('c', subset='removed')
            del dataset.get('a', subset='modified').annotations[1]
            dataset.save(save_images=True)

            self.assertEqual(
                {
                    'bbox_labels_600_hierarchy.json',
                    'class-descriptions.csv',
                    'modified-annotations-human-imagelabels.csv',
                    'modified-images-with-rotation.csv',
                    'new-annotations-human-imagelabels.csv',
                    'new-images-with-rotation.csv',
                    'unmodified-annotations-human-imagelabels.csv',
                    'unmodified-images-with-rotation.csv',
                },
                set(os.listdir(osp.join(path, 'annotations'))),
            )

            expected_images = {f'{id}.jpg' for id in ['a', 'b', 'd', 'e']}

            actual_images = {file_name
                for _, _, file_names in os.walk(osp.join(path, 'images'))
                for file_name in file_names
            }

            self.assertEqual(actual_images, expected_images)

            dataset_reloaded = Dataset.import_from(path, 'open_images')
            compare_datasets(self, dataset, dataset_reloaded, require_images=True)


ASSETS_DIR = osp.join(osp.dirname(__file__), 'assets')

DUMMY_DATASET_DIR_V6 = osp.join(ASSETS_DIR, 'open_images_dataset_v6')
DUMMY_DATASET_DIR_V5 = osp.join(ASSETS_DIR, 'open_images_dataset_v5')

class OpenImagesImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_274)
    def test_can_import_v6(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(id='a', subset='train', image=np.zeros((8, 6, 3)),
                    annotations=[Label(label=0, attributes={'score': 1})]),
                DatasetItem(id='b', subset='train', image=np.zeros((2, 8, 3)),
                    annotations=[
                        Label(label=0, attributes={'score': 0}),
                        Bbox(label=0, x=1.6, y=0.6, w=6.4, h=0.4, attributes={'score': 1}),
                    ]),
                DatasetItem(id='c', subset='test', image=np.ones((10, 5, 3)),
                    annotations=[
                        Label(label=1, attributes={'score': 1}),
                        Label(label=3, attributes={'score': 1}),
                        Bbox(label=3, x=3.5, y=0, w=0.5, h=5, attributes={
                            'score': 0.7,
                            'occluded': True, 'truncated': False,
                            'is_group_of': True, 'is_depiction': False,
                            'is_inside': False,
                        }),
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

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_274)
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

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_274)
    def test_can_detect(self):
        self.assertTrue(OpenImagesImporter.detect(DUMMY_DATASET_DIR_V6))
        self.assertTrue(OpenImagesImporter.detect(DUMMY_DATASET_DIR_V5))
