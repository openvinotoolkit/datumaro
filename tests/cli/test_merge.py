from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.components.annotation import (
    AnnotationType, Bbox, LabelCategories, MaskCategories,
)
from datumaro.components.extractor import DatasetItem
from datumaro.components.project import Dataset, Project
from datumaro.util.test_utils import TestDir, compare_datasets
from datumaro.util.test_utils import run_datum as run
import datumaro.plugins.voc_format.format as VOC

from ..requirements import Requirements, mark_requirement


class MergeTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_run_self_merge(self):
        dataset1 = Dataset.from_iterable([
            DatasetItem(id=100, subset='train', image=np.ones((10, 6, 3)),
                annotations=[
                    Bbox(1, 2, 3, 3, label=0),
                ]),
        ], categories=['a', 'b'])

        dataset2 = Dataset.from_iterable([
            DatasetItem(id=100, subset='train', image=np.ones((10, 6, 3)),
                annotations=[
                    Bbox(1, 2, 3, 4, label=1),
                    Bbox(5, 6, 2, 3, label=2),
                ]),
        ], categories=['a', 'b', 'c'])

        expected =  Dataset.from_iterable([
            DatasetItem(id=100, subset='train', image=np.ones((10, 6, 3)),
                annotations=[
                    Bbox(1, 2, 3, 4, label=2, id=1, group=1,
                        attributes={'score': 0.5, 'occluded': False,
                            'difficult': False, 'truncated': False}),
                    Bbox(5, 6, 2, 3, label=3, id=2, group=2,
                        attributes={'score': 0.5, 'occluded': False,
                            'difficult': False, 'truncated': False}),
                    Bbox(1, 2, 3, 3, label=1, id=1, group=1,
                        attributes={'score': 0.5, 'is_crowd': False}),
                ]),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                ['background', 'a', 'b', 'c']),
            AnnotationType.mask: MaskCategories(VOC.generate_colormap(4))
        })

        with TestDir() as test_dir:
            dataset1_url = osp.join(test_dir, 'dataset1')
            dataset2_url = osp.join(test_dir, 'dataset2')

            dataset1.export(dataset1_url, 'coco', save_images=True)
            dataset2.export(dataset2_url, 'voc', save_images=True)

            proj_dir = osp.join(test_dir, 'proj')
            with Project.init(proj_dir) as project:
                project.import_source('source', dataset2_url, 'voc')

            result_dir = osp.join(test_dir, 'cmp_result')
            run(self, 'merge', dataset1_url + ':coco', '-o', result_dir,
                '-p', proj_dir)

            compare_datasets(self, expected, Dataset.load(result_dir),
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_run_multimerge(self):
        dataset1 = Dataset.from_iterable([
            DatasetItem(id=100, subset='train', image=np.ones((10, 6, 3)),
                annotations=[
                    Bbox(1, 2, 3, 3, label=0),
                ]),
        ], categories=['a', 'b'])

        dataset2 = Dataset.from_iterable([
            DatasetItem(id=100, subset='train', image=np.ones((10, 6, 3)),
                annotations=[
                    Bbox(1, 2, 3, 4, label=1),
                    Bbox(5, 6, 2, 3, label=2),
                ]),
        ], categories=['a', 'b', 'c'])

        expected =  Dataset.from_iterable([
            DatasetItem(id=100, subset='train', image=np.ones((10, 6, 3)),
                annotations=[
                    Bbox(1, 2, 3, 4, label=2, id=1, group=1,
                        attributes={'score': 0.5, 'occluded': False,
                            'difficult': False, 'truncated': False}),
                    Bbox(5, 6, 2, 3, label=3, id=2, group=2,
                        attributes={'score': 0.5, 'occluded': False,
                            'difficult': False, 'truncated': False}),
                    Bbox(1, 2, 3, 3, label=1, id=1, group=1,
                        attributes={'score': 0.5, 'is_crowd': False}),
                ]),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                ['background', 'a', 'b', 'c']),
            AnnotationType.mask: MaskCategories(VOC.generate_colormap(4))
        })

        with TestDir() as test_dir:
            dataset1_url = osp.join(test_dir, 'dataset1')
            dataset2_url = osp.join(test_dir, 'dataset2')

            dataset1.export(dataset1_url, 'coco', save_images=True)
            dataset2.export(dataset2_url, 'voc', save_images=True)

            result_dir = osp.join(test_dir, 'cmp_result')
            run(self, 'merge', dataset2_url + ':voc', dataset1_url + ':coco',
                '-o', result_dir)

            compare_datasets(self, expected, Dataset.load(result_dir),
                require_images=True)
