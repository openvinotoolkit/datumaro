from unittest import TestCase
import os
import os.path as osp

import numpy as np

from datumaro.cli.commands.diff import WrongRevspecError, parse_revspec
from datumaro.cli.contexts.project.diff import DatasetDiffVisualizer
from datumaro.components.errors import MultipleFormatsMatchError, \
    UnknownTargetError
from datumaro.components.dataset import DEFAULT_FORMAT, IDataset
from datumaro.components.operations import DistanceComparator
from datumaro.components.project import Dataset, Project
from datumaro.components.extractor import (DatasetItem,
    AnnotationType, Label, Mask, Points, Polygon,
    PolyLine, Bbox, Caption,
    LabelCategories, MaskCategories, PointsCategories
)
from datumaro.util.image import Image
from datumaro.util.test_utils import TestDir
from ..requirements import Requirements, mark_requirement


class DiffTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
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

    def test_can_parse_revspec(self):
        with TestDir() as test_dir:
            dataset_url = osp.join(test_dir, 'source')
            dataset = Dataset.from_iterable([DatasetItem(1)])
            dataset.save(dataset_url)

            proj_dir = osp.join(test_dir, 'proj')
            proj = Project.init(proj_dir)
            proj.import_source('source-1', dataset_url, format=DEFAULT_FORMAT)
            ref = proj.commit("second commit", allow_empty=True)


            with self.subTest("project"):
                self.assertTrue(isinstance(parse_revspec(proj_dir, None),
                    IDataset))

            with self.subTest("project ref"):
                self.assertTrue(isinstance(
                    parse_revspec(proj_dir + "@" + ref, None),
                    IDataset))

            with self.subTest("project ref source"):
                self.assertTrue(isinstance(
                    parse_revspec(proj_dir + "@" + ref + ":source-1", None),
                    IDataset))

            with self.subTest("ref"):
                self.assertTrue(isinstance(
                    parse_revspec(ref, proj),
                    IDataset))

            with self.subTest("ref source"):
                self.assertTrue(isinstance(
                    parse_revspec(ref + ":source-1", proj),
                    IDataset))

            with self.subTest("source"):
                self.assertTrue(isinstance(
                    parse_revspec("source-1", proj),
                    IDataset))

            with self.subTest("dataset (in context)"):
                with self.assertRaises(WrongRevspecError) as cm:
                    parse_revspec(dataset_url, proj)
                self.assertEqual(
                    {UnknownTargetError, MultipleFormatsMatchError},
                    set(type(e) for e in cm.exception.problems)
                )

            with self.subTest("dataset format (in context)"):
                self.assertTrue(isinstance(
                    parse_revspec(dataset_url + ":datumaro", proj),
                    IDataset))

            with self.subTest("dataset (no context)"):
                with self.assertRaises(WrongRevspecError) as cm:
                    parse_revspec(dataset_url, None)
                self.assertEqual(
                    {FileNotFoundError, MultipleFormatsMatchError},
                    set(type(e) for e in cm.exception.problems)
                )

            with self.subTest("dataset format (no context)"):
                self.assertTrue(isinstance(
                    parse_revspec(dataset_url + ":datumaro", None),
                    IDataset))