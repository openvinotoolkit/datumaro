from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Label, Bbox
from datumaro.components.extractor import DatasetItem
from datumaro.components.dataset import Dataset
from datumaro.plugins.delete_transforms import (
    DeleteImageTransform, DeleteAnnotationTransform, DeleteAttributeTransform,
)

from .requirements import Requirements, mark_requirement


class DeleteImageTransformTest(TestCase):
    @classmethod
    def setUpClass(self):
        self.source = Dataset.from_iterable([
            DatasetItem(id='1', subset='train',
                        image=np.ones((10, 20, 3))),
        ], categories=[('cat', '', ['truncated', 'difficult']),
                       ('dog', '', ['truncated', 'difficult']),
                       ('person', '', ['truncated', 'difficult']),
                       ('car', '', ['truncated', 'difficult']), ])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_item_valid_id(self):
        delete_transform = DeleteImageTransform(self.source, [('1', 'train')])
        for item in self.source:
            actual = delete_transform.transform_item(item)
            print(f'actual: {actual}')
            self.assertEqual(actual, None)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_item_invalid_id(self):
        delete_transform = DeleteImageTransform(self.source, [('2', 'train')])
        for item in self.source:
            actual = delete_transform.transform_item(item)
            self.assertNotEqual(actual, None)


class DeleteAnnotationTransformTest(TestCase):
    @classmethod
    def setUpClass(self):
        self.source = Dataset.from_iterable([
            DatasetItem(id='1', subset='test',
                        image=np.ones((10, 20, 3)),
                        annotations=[
                            Label(0),
                            Bbox(100, 200, 100, 150, label=0),
                        ]),
        ])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_item_valid_id(self):
        delete_transform = DeleteAnnotationTransform(
            self.source, [('1', 'test')])
        for item in self.source:
            actual = delete_transform.transform_item(item)
            self.assertNotEqual(actual, None)
            self.assertEqual(len(actual.annotations), 0)

    def test_transform_item_invalid_id(self):
        delete_transform = DeleteAnnotationTransform(
            self.source, [('2', 'test')])
        for item in self.source:
            actual = delete_transform.transform_item(item)
            self.assertNotEqual(actual, None)
            self.assertEqual(len(actual.annotations), 2)


class DeleteAttributeTransformTest(TestCase):
    @classmethod
    def setUpClass(self):
        self.source = Dataset.from_iterable([
            DatasetItem(id='1', subset='val',
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

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_item_valid_id(self):
        delete_transform = DeleteAttributeTransform(
            self.source, [('1', 'val')])
        for item in self.source:
            actual = delete_transform.transform_item(item)
            for anno in actual.annotations:
                self.assertEqual(len(anno.attributes), 0)

    def test_transform_item_invalid_id(self):
        delete_transform = DeleteAttributeTransform(
            self.source, [('2', 'val')], 'truncated')
        for item in self.source:
            actual = delete_transform.transform_item(item)
            for anno in actual.annotations:
                self.assertEqual(len(anno.attributes), 2)
