from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.components.annotation import Bbox
from datumaro.components.errors import ReadonlyDatasetError
from datumaro.components.extractor import DatasetItem
from datumaro.components.project import Dataset, Project
from datumaro.util.test_utils import TestDir, compare_datasets
from datumaro.util.test_utils import run_datum as run

from ..requirements import Requirements, mark_requirement


class PatchTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_run_patch(self):
        dataset = Dataset.from_iterable([
            # Must be overridden
            DatasetItem(id=100, subset='train', image=np.ones((10, 6, 3)),
                annotations=[
                    Bbox(1, 2, 3, 3, label=0),
                ]),

            # Must be kept
            DatasetItem(id=1, image=np.ones((5, 4, 3)),
                annotations=[
                    Bbox(1, 2, 3, 4, label=1)
                ]),
        ], categories=['a', 'b'])

        patch = Dataset.from_iterable([
            # Must override
            DatasetItem(id=100, subset='train', image=np.ones((10, 6, 3)),
                annotations=[
                    Bbox(1, 2, 3, 4, label=0), # Label must be remapped
                    Bbox(5, 6, 2, 3, label=1), # Label must be remapped
                    Bbox(2, 2, 2, 3, label=2), # Will be dropped due to label
                ]),

            # Must be added
            DatasetItem(id=2, image=np.ones((5, 4, 3)),
                annotations=[
                    Bbox(1, 2, 3, 2, label=1) # Label must be remapped
                ]),
        ], categories=['b', 'a', 'c'])

        expected =  Dataset.from_iterable([
            DatasetItem(id=100, subset='train', image=np.ones((10, 6, 3)),
                annotations=[
                    Bbox(1, 2, 3, 4, label=1, id=1, group=1),
                    Bbox(5, 6, 2, 3, label=0, id=2, group=2),
                ]),

            DatasetItem(id=1, image=np.ones((5, 4, 3)),
                annotations=[
                    Bbox(1, 2, 3, 4, label=1, id=1, group=1)
                ]),

            DatasetItem(id=2, image=np.ones((5, 4, 3)),
                annotations=[
                    Bbox(1, 2, 3, 2, label=0, id=2, group=2)
                ]),
        ], categories=['a', 'b'])

        with TestDir() as test_dir:
            dataset_url = osp.join(test_dir, 'dataset1')
            patch_url = osp.join(test_dir, 'dataset2')

            dataset.export(dataset_url, 'coco', save_images=True)
            patch.export(patch_url, 'voc', save_images=True)

            run(self, 'patch', '--overwrite',
                dataset_url + ':coco', patch_url + ':voc',
                '--', '--reindex=1', '--save-images')

            compare_datasets(self, expected,
                Dataset.import_from(dataset_url, format='coco'),
                require_images=True, ignored_attrs='*')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_patch_fails_on_inplace_update_without_overwrite(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.zeros((3, 5, 3)),
                annotations=[ Bbox(1, 2, 3, 4, label=1) ]),
        ], categories=['a', 'b'])

        patch = Dataset.from_iterable([
            DatasetItem(id=2, image=np.zeros((3, 4, 3)),
                annotations=[ Bbox(1, 2, 3, 2, label=1) ]),
        ], categories=['b', 'a', 'c'])

        with TestDir() as test_dir:
            dataset_url = osp.join(test_dir, 'dataset1')
            patch_url = osp.join(test_dir, 'dataset2')

            dataset.export(dataset_url, 'coco', save_images=True)
            patch.export(patch_url, 'coco', save_images=True)

            run(self, 'patch', dataset_url + ':coco', patch_url + ':coco',
                expected_code=1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_patch_fails_on_inplace_update_of_stage(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.zeros((3, 5, 3)),
                annotations=[ Bbox(1, 2, 3, 4, label=1) ]),
        ], categories=['a', 'b'])

        patch = Dataset.from_iterable([
            DatasetItem(id=2, image=np.zeros((3, 4, 3)),
                annotations=[ Bbox(1, 2, 3, 2, label=1) ]),
        ], categories=['b', 'a', 'c'])

        with TestDir() as test_dir:
            dataset_url = osp.join(test_dir, 'dataset1')
            patch_url = osp.join(test_dir, 'dataset2')

            dataset.export(dataset_url, 'coco', save_images=True)
            patch.export(patch_url, 'coco', save_images=True)

            project_dir = osp.join(test_dir, 'proj')
            with Project.init(project_dir) as project:
                project.import_source('source-1', dataset_url, 'coco',
                    no_cache=True)
                project.commit('first commit')

            with self.subTest('without overwrite'):
                run(self, 'patch', '-p', project_dir,
                    'HEAD:source-1', patch_url + ':coco',
                    expected_code=1)

            with self.subTest('with overwrite'):
                with self.assertRaises(ReadonlyDatasetError):
                    run(self, 'patch', '-p', project_dir, '--overwrite',
                        'HEAD:source-1', patch_url + ':coco')
