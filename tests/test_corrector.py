from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Label, Bbox
from datumaro.components.extractor import DatasetItem
from datumaro.components.dataset import Dataset
from datumaro.plugins.corrector import (
    DeleteImage, DeleteAnnotation, DeleteAttribute,
)

from .requirements import Requirements, mark_requirement


class DeleteImageTest(TestCase):
    @mark_requirement(Requirements.DATUM_API)
    def test_delete_dataset_items(self):
        id_valid = '1'
        id_invalid = '2'
        dataset = Dataset.from_iterable([
            DatasetItem(id=id_valid, subset='train',
                        image=np.ones((10, 20, 3))),
        ], categories=[('cat', '', ['truncated', 'difficult']),
                       ('dog', '', ['truncated', 'difficult']),
                       ('person', '', ['truncated', 'difficult']),
                       ('car', '', ['truncated', 'difficult']), ])

        dataset_fixed = DeleteImage.delete_dataset_items(None, id_valid)
        self.assertEqual(dataset_fixed, None)
        dataset_fixed = DeleteImage.delete_dataset_items(dataset, id_invalid)
        self.assertEqual(len(dataset_fixed), 1)
        dataset_fixed = DeleteImage.delete_dataset_items(dataset, id_valid)
        self.assertEqual(len(dataset_fixed), 0)


class DeleteAnnotationTest(TestCase):
    @mark_requirement(Requirements.DATUM_API)
    def test_delete_dataset_annotations(self):
        id_valid = '1'
        id_invalid = '2'
        dataset = Dataset.from_iterable([
            DatasetItem(id=id_valid, subset='test',
                        image=np.ones((10, 20, 3)),
                        annotations=[
                            Label(0),
                            Bbox(100, 200, 100, 150, label=0),
                        ]),
        ])

        dataset_fixed = DeleteAnnotation.delete_dataset_annotations(
            None, id_valid)
        self.assertEqual(dataset_fixed, None)
        dataset_fixed = DeleteAnnotation.delete_dataset_annotations(
            dataset, id_invalid)
        self.assertEqual(len(dataset_fixed), 1)
        for item in dataset_fixed:
            self.assertEqual(len(item.annotations), 2)
        dataset_fixed = DeleteAnnotation.delete_dataset_annotations(
            dataset, id_valid)
        self.assertEqual(len(dataset_fixed), 1)
        for item in dataset_fixed:
            self.assertEqual(len(item.annotations), 0)


class DeleteAttributeTest(TestCase):
    @mark_requirement(Requirements.DATUM_API)
    def test_delete_dataset_attributes(self):
        id_valid = '1'
        id_invalid = '2'
        dataset = Dataset.from_iterable([
            DatasetItem(id=id_valid, subset='val',
                        image=np.ones((10, 20, 3)),
                        attributes={'qq': 1},
                        annotations=[
                            Label(0, id=0, attributes={
                                'truncated': 1, 'difficult': 2}),
                            Label(1, id=1, attributes={
                                'truncated': 1, 'difficult': 2}),
                            Label(2, id=2, attributes={
                                'truncated': 1, 'difficult': 2}),
                            Label(3, id=3, attributes={
                                'truncated': 1, 'difficult': 2}),
                        ]),
        ])

        dataset_fixed = DeleteAttribute.delete_dataset_attributes(
            None, id_valid)
        self.assertEqual(dataset_fixed, None)
        dataset_fixed = DeleteAttribute.delete_dataset_attributes(
            dataset, id_invalid)
        self.assertEqual(len(dataset_fixed), 1)
        for item in dataset_fixed:
            self.assertEqual(len(item.attributes), 1)
            for anno in item.annotations:
                self.assertEqual(len(anno.attributes), 2)
        dataset_fixed = DeleteAttribute.delete_dataset_attributes(
            dataset, id_valid)
        self.assertEqual(len(dataset_fixed), 1)
        for item in dataset_fixed:
            self.assertEqual(len(item.attributes), 0)
            for anno in item.annotations:
                self.assertEqual(len(anno.attributes), 0)
