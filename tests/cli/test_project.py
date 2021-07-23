from unittest import TestCase
import os.path as osp
import shutil

import numpy as np

from datumaro.components.dataset import DEFAULT_FORMAT, Dataset
from datumaro.components.extractor import Bbox, DatasetItem, Label
from datumaro.components.project import Project
from datumaro.util.test_utils import TestDir, compare_datasets
from datumaro.util.test_utils import run_datum as run

from ..requirements import Requirements, mark_requirement


class ProjectIntegrationScenarios(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_convert_voc_as_coco(self):
        voc_dir = osp.join(__file__[:__file__.rfind(osp.join('tests', ''))],
            'tests', 'assets', 'voc_dataset', 'voc_dataset1')

        with TestDir() as test_dir:
            result_dir = osp.join(test_dir, 'coco_export')

            run(self, 'convert',
                '-if', 'voc', '-i', voc_dir,
                '-f', 'coco', '-o', result_dir,
                '--', '--save-images', '--reindex', '1')

            self.assertTrue(osp.isdir(result_dir))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_export_coco_as_voc(self):
        # TODO: use subformats once importers are removed
        coco_dir = osp.join(__file__[:__file__.rfind(osp.join('tests', ''))],
            'tests', 'assets', 'coco_dataset', 'coco_instances')

        with TestDir() as test_dir:
            run(self, 'create', '-o', test_dir)
            run(self, 'add', '-f', 'coco', '-p', test_dir, coco_dir)

            result_dir = osp.join(test_dir, 'voc_export')
            run(self, 'export', '-f', 'voc', '-p', test_dir, '-o', result_dir,
                '--', '--save-images')

            self.assertTrue(osp.isdir(result_dir))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_list_info(self):
        # TODO: use subformats once importers are removed
        coco_dir = osp.join(__file__[:__file__.rfind(osp.join('tests', ''))],
            'tests', 'assets', 'coco_dataset', 'coco_instances')

        with TestDir() as test_dir:
            run(self, 'create', '-o', test_dir)
            run(self, 'add', '-f', 'coco', '-p', test_dir, coco_dir)

            run(self, 'info', '-p', test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_use_vcs(self):
        with TestDir() as test_dir:
            dataset_dir = osp.join(test_dir, 'dataset')
            project_dir = osp.join(test_dir, 'proj')
            result_dir = osp.join(project_dir, 'result')

            Dataset.from_iterable([
                DatasetItem(0, image=np.ones((1, 2, 3)), annotations=[
                    Bbox(1, 1, 1, 1, label=0),
                    Bbox(2, 2, 2, 2, label=1),
                ])
            ], categories=['a', 'b']).save(dataset_dir, save_images=True)

            run(self, 'create', '-o', project_dir)
            run(self, 'add', '-p', project_dir, '-f', 'datumaro', dataset_dir)
            run(self, 'commit', '-p', project_dir, '-m', 'Add data')

            run(self, 'transform', '-p', project_dir,
                '-t', 'remap_labels', 'source-1', '--', '-l', 'b:cat')
            run(self, 'commit', '-p', project_dir, '-m', 'Add transform')

            run(self, 'filter', '-p', project_dir,
                '-e', '/item/annotation[label="cat"]', '-m', 'i+a')
            run(self, 'commit', '-p', project_dir, '-m', 'Add filter')

            run(self, 'export', '-p', project_dir, '-f', 'coco',
                '-o', result_dir, 'source-1', '--', '--save-images')
            parsed = Dataset.import_from(result_dir, 'coco')
            compare_datasets(self, Dataset.from_iterable([
                DatasetItem(0, image=np.ones((1, 2, 3)),
                    annotations=[
                        Bbox(2, 2, 2, 2, label=1,
                            group=1, id=1, attributes={'is_crowd': False}),
                ], attributes={ 'id': 1 })
            ], categories=['a', 'cat']), parsed, require_images=True)

            shutil.rmtree(result_dir, ignore_errors=True)
            run(self, 'checkout', '-p', project_dir, 'HEAD~1')
            run(self, 'export', '-p', project_dir, '-f', 'coco',
                '-o', result_dir, '--', '--save-images')
            parsed = Dataset.import_from(result_dir, 'coco')
            compare_datasets(self, Dataset.from_iterable([
                DatasetItem(0, image=np.ones((1, 2, 3)), annotations=[
                    Bbox(1, 1, 1, 1, label=0,
                        group=1, id=1, attributes={'is_crowd': False}),
                    Bbox(2, 2, 2, 2, label=1,
                        group=2, id=2, attributes={'is_crowd': False}),
                ], attributes={ 'id': 1 })
            ], categories=['a', 'cat']), parsed, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_chain_transforms_in_working_tree(self):
        with TestDir() as test_dir:
            source_url = osp.join(test_dir, 'test_repo')
            dataset = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0)]),
                DatasetItem(2, annotations=[Label(1)]),
            ], categories=['a', 'b'])
            dataset.save(source_url)

            project_dir = osp.join(test_dir, 'proj')
            run(self, 'create', '-o', project_dir)
            run(self, 'add', '-p', project_dir,
                '--format', DEFAULT_FORMAT, source_url)
            run(self, 'filter', '-p', project_dir,
                '-e', '/item/annotation[label="b"]')
            run(self, 'transform', '-p', project_dir,
                '-t', 'rename', '--', '-e', '|2|qq|')
            run(self, 'transform', '-p', project_dir,
                '-t', 'remap_labels', '--', '-l', 'a:cat', '-l', 'b:dog')

            built_dataset = Project(project_dir).working_tree.make_dataset()

            expected_dataset = Dataset.from_iterable([
                DatasetItem('qq', annotations=[Label(1)]),
            ], categories=['cat', 'dog'])
            compare_datasets(self, expected_dataset, built_dataset)
