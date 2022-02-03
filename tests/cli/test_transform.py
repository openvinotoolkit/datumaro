from unittest import TestCase
import os.path as osp

from datumaro.components.annotation import Bbox, Label
from datumaro.components.dataset import Dataset
from datumaro.components.errors import ReadonlyDatasetError
from datumaro.components.extractor import DatasetItem
from datumaro.components.project import Project
from datumaro.util.scope import scope_add, scoped
from datumaro.util.test_utils import TestDir, compare_datasets
from datumaro.util.test_utils import run_datum as run

from ..requirements import Requirements, mark_requirement


class TransformTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_transform_dataset_inplace(self):
        test_dir = scope_add(TestDir())
        Dataset.from_iterable([
            DatasetItem(1, annotations=[Label(0)]),
            DatasetItem(2, annotations=[Label(1)]),
        ], categories=['a', 'b']).export(test_dir, 'coco')

        run(self, 'transform', '-t', 'remap_labels', '--overwrite',
            test_dir + ':coco', '--', '-l', 'a:cat', '-l', 'b:dog')

        expected_dataset = Dataset.from_iterable([
            DatasetItem(1, annotations=[Label(0, id=1, group=1)]),
            DatasetItem(2, annotations=[Label(1, id=2, group=2)]),
        ], categories=['cat', 'dog'])
        compare_datasets(self, expected_dataset,
            Dataset.import_from(test_dir, 'coco'), ignored_attrs='*')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_fails_on_inplace_update_without_overwrite(self):
        with TestDir() as test_dir:
            Dataset.from_iterable([
                DatasetItem(id=1, annotations=[ Bbox(1, 2, 3, 4, label=1) ]),
            ], categories=['a', 'b']).export(test_dir, 'coco')

            run(self, 'transform', '-t', 'random_split', test_dir + ':coco',
                expected_code=1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_transform_fails_on_inplace_update_of_stage(self):
        with TestDir() as test_dir:
            dataset_url = osp.join(test_dir, 'dataset')
            dataset = Dataset.from_iterable([
                DatasetItem(id=1, annotations=[ Bbox(1, 2, 3, 4, label=1) ]),
            ], categories=['a', 'b'])
            dataset.export(dataset_url, 'coco', save_media=True)

            project_dir = osp.join(test_dir, 'proj')
            with Project.init(project_dir) as project:
                project.import_source('source-1', dataset_url, 'coco',
                    no_cache=True)
                project.commit('first commit')

            with self.subTest('without overwrite'):
                run(self, 'transform', '-p', project_dir,
                    '-t', 'random_split', 'HEAD:source-1',
                    expected_code=1)

            with self.subTest('with overwrite'):
                with self.assertRaises(ReadonlyDatasetError):
                    run(self, 'transform', '-p', project_dir, '--overwrite',
                        '-t', 'random_split', 'HEAD:source-1')
