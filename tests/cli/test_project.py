import numpy as np
import os.path as osp
import shutil

from unittest import TestCase

from datumaro.components.dataset import Dataset
from datumaro.components.extractor import Bbox, DatasetItem, Label
from datumaro.cli.__main__ import main
from datumaro.util.test_utils import TestDir, compare_datasets


def run(test, *args, expected_code=0):
    test.assertEqual(expected_code, main(args), str(args))

class ProjectIntegrationScenarios(TestCase):

    def test_can_export_voc_as_coco(self):
        voc_dir = osp.join(__file__[:__file__.rfind(osp.join('tests', ''))],
            'tests', 'assets', 'voc_dataset')

        with TestDir() as test_dir:
            run(self, 'create', '-o', test_dir)
            run(self, 'add', '-f', 'voc', '-p', test_dir, voc_dir)

            result_dir = osp.join(test_dir, 'coco_export')
            run(self, 'export', '-f', 'coco', '-p', test_dir, '-o', result_dir,
                '--', '--save-images')

            self.assertTrue(osp.isdir(result_dir))

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
            run(self, 'commit', '-p', project_dir, '-m', 'Initial commit')

            run(self, 'add', '-p', project_dir, '-f', 'datumaro', dataset_dir)
            run(self, 'commit', '-p', project_dir, '-m', 'Add data')

            run(self, 'transform', '-p', project_dir,
                '-t', 'remap_labels', 'source-1', '--', '-l', 'b:cat')
            run(self, 'commit', '-p', project_dir, '-m', 'Add transform')

            run(self, 'filter', '-p', project_dir,
                '-e', '/item/annotation[label="cat"]', '-m', 'i+a', 'source-1')
            run(self, 'commit', '-p', project_dir, '-m', 'Add filter')

            run(self, 'export', '-p', project_dir, '-f', 'coco', '-o', result_dir)
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
            run(self, 'export', '-p', project_dir, '-f', 'coco', '-o', result_dir)
            parsed = Dataset.import_from(result_dir, 'coco')
            compare_datasets(self, Dataset.from_iterable([
                DatasetItem(0, image=np.ones((1, 2, 3)), annotations=[
                    Bbox(1, 1, 1, 1, label=0,
                        group=1, id=1, attributes={'is_crowd': False}),
                    Bbox(2, 2, 2, 2, label=1,
                        group=2, id=2, attributes={'is_crowd': False}),
                ], attributes={ 'id': 1 })
            ], categories=['a', 'cat']), parsed, require_images=True)
