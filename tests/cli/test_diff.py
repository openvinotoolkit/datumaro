from datumaro.components.operations import DistanceComparator
from functools import partial
import os
import os.path as osp

import numpy as np

from unittest import TestCase
from datumaro.cli.contexts.project.diff import DatasetDiffVisualizer
from datumaro.components.project import Dataset
from datumaro.components.extractor import (DatasetItem,
    AnnotationType, Label, Mask, Points, Polygon,
    PolyLine, Bbox, Caption,
    LabelCategories, MaskCategories, PointsCategories
)
from datumaro.util.image import Image
from datumaro.util.test_utils import TestDir


class DiffTest(TestCase):
    def test_can_compare_projects(self): # just a smoke test
        label_categories1 = LabelCategories.from_iterable(['x', 'a', 'b', 'y'])
        mask_categories1 = MaskCategories.make_default(len(label_categories1))

        point_categories1 = PointsCategories()
        for index, _ in enumerate(label_categories1.items):
            point_categories1.add(index, ['cat1', 'cat2'], joints=[[0, 1]])

        dataset1 = Dataset.from_iterable([
            DatasetItem(id=100, subset='train', image=np.ones((10, 6, 3)),
                annotations=[
                    Caption('hello', id=1),
                    Caption('world', id=2, group=5),
                    Label(2, id=3, attributes={
                        'x': 1,
                        'y': '2',
                    }),
                    Bbox(1, 2, 3, 4, label=0, id=4, z_order=1, attributes={
                        'score': 1.0,
                    }),
                    Bbox(5, 6, 7, 8, id=5, group=5),
                    Points([1, 2, 2, 0, 1, 1], label=0, id=5, z_order=4),
                    Mask(label=3, id=5, z_order=2, image=np.ones((2, 3))),
                ]),
            DatasetItem(id=21, subset='train',
                annotations=[
                    Caption('test'),
                    Label(2),
                    Bbox(1, 2, 3, 4, label=2, id=42, group=42)
                ]),

            DatasetItem(id=2, subset='val',
                annotations=[
                    PolyLine([1, 2, 3, 4, 5, 6, 7, 8], id=11, z_order=1),
                    Polygon([1, 2, 3, 4, 5, 6, 7, 8], id=12, z_order=4),
                ]),

            DatasetItem(id=42, subset='test',
                attributes={'a1': 5, 'a2': '42'}),

            DatasetItem(id=42),
            DatasetItem(id=43, image=Image(path='1/b/c.qq', size=(2, 4))),
        ], categories={
            AnnotationType.label: label_categories1,
            AnnotationType.mask: mask_categories1,
            AnnotationType.points: point_categories1,
        })


        label_categories2 = LabelCategories.from_iterable(['a', 'b', 'x', 'y'])
        mask_categories2 = MaskCategories.make_default(len(label_categories2))

        point_categories2 = PointsCategories()
        for index, _ in enumerate(label_categories2.items):
            point_categories2.add(index, ['cat1', 'cat2'], joints=[[0, 1]])

        dataset2 = Dataset.from_iterable([
            DatasetItem(id=100, subset='train', image=np.ones((10, 6, 3)),
                annotations=[
                    Caption('hello', id=1),
                    Caption('world', id=2, group=5),
                    Label(2, id=3, attributes={
                        'x': 1,
                        'y': '2',
                    }),
                    Bbox(1, 2, 3, 4, label=1, id=4, z_order=1, attributes={
                        'score': 1.0,
                    }),
                    Bbox(5, 6, 7, 8, id=5, group=5),
                    Points([1, 2, 2, 0, 1, 1], label=0, id=5, z_order=4),
                    Mask(label=3, id=5, z_order=2, image=np.ones((2, 3))),
                ]),
            DatasetItem(id=21, subset='train',
                annotations=[
                    Caption('test'),
                    Label(2),
                    Bbox(1, 2, 3, 4, label=3, id=42, group=42)
                ]),

            DatasetItem(id=2, subset='val',
                annotations=[
                    PolyLine([1, 2, 3, 4, 5, 6, 7, 8], id=11, z_order=1),
                    Polygon([1, 2, 3, 4, 5, 6, 7, 8], id=12, z_order=4),
                ]),

            DatasetItem(id=42, subset='test',
                attributes={'a1': 5, 'a2': '42'}),

            DatasetItem(id=42),
            DatasetItem(id=43, image=Image(path='1/b/c.qq', size=(2, 4))),
        ], categories={
            AnnotationType.label: label_categories2,
            AnnotationType.mask: mask_categories2,
            AnnotationType.points: point_categories2,
        })

        with TestDir() as test_dir:
            with DatasetDiffVisualizer(save_dir=test_dir,
                        comparator=DistanceComparator(iou_threshold=0.8),
                    ) as visualizer:
                visualizer.save(dataset1, dataset2)

            self.assertNotEqual(0, os.listdir(osp.join(test_dir)))