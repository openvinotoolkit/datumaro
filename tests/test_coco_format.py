from functools import partial
from itertools import product
from unittest import TestCase
import os
import os.path as osp
import pickle

import numpy as np

from datumaro.components.annotation import (
    AnnotationType, Bbox, Caption, Label, LabelCategories, Mask, Points,
    PointsCategories, Polygon,
)
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.coco_format.converter import (
    CocoCaptionsConverter, CocoConverter, CocoImageInfoConverter,
    CocoInstancesConverter, CocoLabelsConverter, CocoPanopticConverter,
    CocoPersonKeypointsConverter, CocoStuffConverter,
)
from datumaro.plugins.coco_format.importer import CocoImporter
from datumaro.util.test_utils import (
    TestDir, check_save_and_load, compare_datasets, compare_datasets_strict,
)

from .requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'coco_dataset')


class CocoImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_instances(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='train', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Bbox(2, 2, 3, 1, label=1,
                        group=1, id=1, attributes={'is_crowd': False})
                ]
            ),

            DatasetItem(id='b', subset='val', image=np.ones((10, 5, 3)),
                attributes={'id': 40},
                annotations=[
                    Polygon([0, 0, 1, 0, 1, 2, 0, 2], label=0,
                        id=1, group=1, attributes={'is_crowd': False,
                            'x': 1, 'y': 'hello'}),
                    Mask(np.array( [[1, 1, 0, 0, 0]] * 10 ), label=1,
                        id=2, group=2, attributes={'is_crowd': True}),
                ]
            ),
        ], categories=['a', 'b', 'c'])

        formats = ['coco', 'coco_instances']
        paths = [
            ('', osp.join(DUMMY_DATASET_DIR, 'coco_instances')),
            ('train', osp.join(DUMMY_DATASET_DIR, 'coco_instances',
                'annotations', 'instances_train.json')),
            ('val', osp.join(DUMMY_DATASET_DIR, 'coco_instances',
                'annotations', 'instances_val.json')),
        ]
        for format, (subset, path) in product(formats, paths):
            if subset:
                expected = expected_dataset.get_subset(subset)
            else:
                expected = expected_dataset

            with self.subTest(path=path, format=format, subset=subset):
                dataset = Dataset.import_from(path, format)
                compare_datasets(self, expected, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_instances_with_any_annotation_filename(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='default', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Bbox(2, 2, 3, 1, label=1,
                        group=1, id=1, attributes={'is_crowd': False})
                ]
            ),

        ], categories=['a', 'b', 'c'])

        format = 'coco_instances'
        with TestDir() as test_dir:
            dataset_dir = osp.join(test_dir, 'dataset')
            expected_dataset.export(dataset_dir, format, save_images=True)
            os.rename(osp.join(dataset_dir, 'annotations', 'instances_default.json'),
                osp.join(dataset_dir, 'annotations', 'aa_bbbb_cccc.json'))

            imported_dataset = Dataset.import_from(dataset_dir, format)
            compare_datasets(self, expected_dataset, imported_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_instances_with_original_cat_ids(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='train', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Bbox(2, 2, 3, 1, label=2,
                        group=1, id=1, attributes={'is_crowd': False})
                ]
            ),
        ], categories=['class-0', 'a', 'b', 'class-3', 'c'])

        actual_dataset = Dataset.import_from(
            osp.join(DUMMY_DATASET_DIR, 'coco_instances',
                'annotations', 'instances_train.json'),
            'coco_instances',
            keep_original_category_ids=True)
        compare_datasets(self, expected_dataset, actual_dataset,
            require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_captions(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='train', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Caption('hello', id=1, group=1),
                ]),

            DatasetItem(id='b', subset='val', image=np.ones((10, 5, 3)),
                attributes={'id': 40},
                annotations=[
                    Caption('world', id=1, group=1),
                    Caption('text', id=2, group=2),
                ]),
        ])

        formats = ['coco', 'coco_captions']
        paths = [
            ('', osp.join(DUMMY_DATASET_DIR, 'coco_captions')),
            ('train', osp.join(DUMMY_DATASET_DIR, 'coco_captions',
                'annotations', 'captions_train.json')),
            ('val', osp.join(DUMMY_DATASET_DIR, 'coco_captions',
                'annotations', 'captions_val.json')),
        ]
        for format, (subset, path) in product(formats, paths):
            if subset:
                expected = expected_dataset.get_subset(subset)
            else:
                expected = expected_dataset

            with self.subTest(path=path, format=format, subset=subset):
                dataset = Dataset.import_from(path, format)
                compare_datasets(self, expected, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_captions_with_any_annotation_filename(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='default', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Caption('hello', id=1, group=1),
                ]),
        ])

        format = 'coco_captions'
        with TestDir() as test_dir:
            dataset_dir = osp.join(test_dir, 'dataset')
            expected_dataset.export(dataset_dir, format, save_images=True)
            os.rename(osp.join(dataset_dir, 'annotations', 'captions_default.json'),
                osp.join(dataset_dir, 'annotations', 'aa_bbbb_cccc.json'))

            imported_dataset = Dataset.import_from(dataset_dir, format)
            compare_datasets(self, expected_dataset, imported_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_labels(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='train', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Label(1, id=1, group=1),
                ]),

            DatasetItem(id='b', subset='val', image=np.ones((10, 5, 3)),
                attributes={'id': 40},
                annotations=[
                    Label(0, id=1, group=1),
                    Label(1, id=2, group=2),
                ]),
        ], categories=['a', 'b'])

        formats = ['coco', 'coco_labels']
        paths = [
            ('', osp.join(DUMMY_DATASET_DIR, 'coco_labels')),
            ('train', osp.join(DUMMY_DATASET_DIR, 'coco_labels',
                'annotations', 'labels_train.json')),
            ('val', osp.join(DUMMY_DATASET_DIR, 'coco_labels',
                'annotations', 'labels_val.json')),
        ]
        for format, (subset, path) in product(formats, paths):
            if subset:
                expected = expected_dataset.get_subset(subset)
            else:
                expected = expected_dataset

            with self.subTest(path=path, format=format, subset=subset):
                dataset = Dataset.import_from(path, format)
                compare_datasets(self, expected, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_labels_with_any_annotation_filename(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='default', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Label(1, id=1, group=1),
                ]),
        ], categories=['a', 'b'])

        format = 'coco_labels'
        with TestDir() as test_dir:
            dataset_dir = osp.join(test_dir, 'dataset')
            expected_dataset.export(dataset_dir, format, save_images=True)
            os.rename(osp.join(dataset_dir, 'annotations', 'labels_default.json'),
                osp.join(dataset_dir, 'annotations', 'aa_bbbb_cccc.json'))

            imported_dataset = Dataset.import_from(dataset_dir, format)
            compare_datasets(self, expected_dataset, imported_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_keypoints(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='train', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Points([0, 0, 0, 2, 4, 1], [0, 1, 2], label=1,
                        id=1, group=1, attributes={'is_crowd': False}),
                    Bbox(2, 2, 3, 1, label=1,
                        id=1, group=1, attributes={'is_crowd': False}),
                ]),

            DatasetItem(id='b', subset='val', image=np.ones((10, 5, 3)),
                attributes={'id': 40},
                annotations=[
                    Points([1, 2, 3, 4, 2, 3], label=0,
                        id=1, group=1, attributes={'is_crowd': False,
                            'x': 1, 'y': 'hello'}),
                    Polygon([0, 0, 1, 0, 1, 2, 0, 2], label=0,
                        id=1, group=1, attributes={'is_crowd': False,
                            'x': 1, 'y': 'hello'}),

                    Points([2, 4, 4, 4, 4, 2], label=1,
                        id=2, group=2, attributes={'is_crowd': True}),
                    Mask(np.array( [[1, 1, 0, 0, 0]] * 10 ), label=1,
                        id=2, group=2, attributes={'is_crowd': True}),
                ]),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(['a', 'b']),
            AnnotationType.points: PointsCategories.from_iterable(
                (i, None, [[0, 1], [1, 2]]) for i in range(2)
            ),
        })

        formats = ['coco', 'coco_person_keypoints']
        paths = [
            ('', osp.join(DUMMY_DATASET_DIR, 'coco_person_keypoints')),
            ('train', osp.join(DUMMY_DATASET_DIR, 'coco_person_keypoints',
                'annotations', 'person_keypoints_train.json')),
            ('val', osp.join(DUMMY_DATASET_DIR, 'coco_person_keypoints',
                'annotations', 'person_keypoints_val.json')),
        ]
        for format, (subset, path) in product(formats, paths):
            if subset:
                expected = expected_dataset.get_subset(subset)
            else:
                expected = expected_dataset

            with self.subTest(path=path, format=format, subset=subset):
                dataset = Dataset.import_from(path, format)
                compare_datasets(self, expected, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_keypoints_with_any_annotation_filename(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='default', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Points([0, 0, 0, 2, 4, 1], [0, 1, 2], label=1,
                        id=1, group=1, attributes={'is_crowd': False}),
                    Bbox(2, 2, 3, 1, label=1,
                        id=1, group=1, attributes={'is_crowd': False}),
                ]),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(['a', 'b']),
            AnnotationType.points: PointsCategories.from_iterable(
                (i, None, [[0, 1], [1, 2]]) for i in range(2)
            ),
        })

        format = 'coco_person_keypoints'
        with TestDir() as test_dir:
            dataset_dir = osp.join(test_dir, 'dataset')
            expected_dataset.export(dataset_dir, format, save_images=True)
            os.rename(osp.join(dataset_dir, 'annotations',
                'person_keypoints_default.json'), osp.join(dataset_dir,
                'annotations', 'aa_bbbb_cccc.json')
            )

            imported_dataset = Dataset.import_from(dataset_dir, format)
            compare_datasets(self, expected_dataset, imported_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_keypoints_with_original_cat_ids(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='train', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Points([0, 0, 0, 2, 4, 1], [0, 1, 2], label=2,
                        id=1, group=1, attributes={'is_crowd': False}),
                    Bbox(2, 2, 3, 1, label=2,
                        id=1, group=1, attributes={'is_crowd': False}),
                ]),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                ['class-0', 'a', 'b']
            ),
            AnnotationType.points: PointsCategories.from_iterable(
                [(i, None, [[0, 1], [1, 2]]) for i in range(1, 3)],
            ),
        })

        actual_dataset = Dataset.import_from(
            osp.join(DUMMY_DATASET_DIR, 'coco_person_keypoints',
                'annotations', 'person_keypoints_train.json'),
            'coco_person_keypoints',
            keep_original_category_ids=True)

        compare_datasets(self, expected_dataset, actual_dataset,
            require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_image_info(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='train', image=np.ones((5, 10, 3)),
                attributes={'id': 5}),
            DatasetItem(id='b', subset='val', image=np.ones((10, 5, 3)),
                attributes={'id': 40})
        ])

        formats = ['coco', 'coco_image_info']
        paths = [
            ('', osp.join(DUMMY_DATASET_DIR, 'coco_image_info')),
            ('train', osp.join(DUMMY_DATASET_DIR, 'coco_image_info',
                'annotations', 'image_info_train.json')),
            ('val', osp.join(DUMMY_DATASET_DIR, 'coco_image_info',
                'annotations', 'image_info_val.json')),
        ]
        for format, (subset, path) in product(formats, paths):
            if subset:
                expected = expected_dataset.get_subset(subset)
            else:
                expected = expected_dataset

            with self.subTest(path=path, format=format, subset=subset):
                dataset = Dataset.import_from(path, format)
                compare_datasets(self, expected, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_image_info_with_any_annotation_filename(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='default', image=np.ones((5, 10, 3)),
                attributes={'id': 5}),
        ])

        format = 'coco_image_info'
        with TestDir() as test_dir:
            dataset_dir = osp.join(test_dir, 'dataset')
            expected_dataset.export(dataset_dir, format, save_images=True)
            os.rename(osp.join(dataset_dir, 'annotations', 'image_info_default.json'),
                osp.join(dataset_dir, 'annotations', 'aa_bbbb_cccc.json'))

            imported_dataset = Dataset.import_from(dataset_dir, format)
            compare_datasets(self, expected_dataset, imported_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_panoptic(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='train', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Mask(np.ones((5, 5)), label=0, id=460551,
                        group=460551, attributes={'is_crowd': False}),
                ]),

            DatasetItem(id='b', subset='val', image=np.ones((10, 5, 3)),
                attributes={'id': 40},
                annotations=[
                    Mask(np.array( [[1, 1, 0, 0, 0]] * 10 ), label=0,
                        id=7, group=7, attributes={'is_crowd': False}),
                    Mask(np.array( [[0, 0, 1, 1, 0]] * 10 ), label=1,
                        id=20, group=20, attributes={'is_crowd': True}),
                ]),
        ], categories=['a', 'b'])

        formats = ['coco', 'coco_panoptic']
        paths = [
            ('', osp.join(DUMMY_DATASET_DIR, 'coco_panoptic')),
            ('train', osp.join(DUMMY_DATASET_DIR, 'coco_panoptic',
                'annotations', 'panoptic_train.json')),
            ('val', osp.join(DUMMY_DATASET_DIR, 'coco_panoptic',
                'annotations', 'panoptic_val.json')),
        ]
        for format, (subset, path) in product(formats, paths):
            if subset:
                expected = expected_dataset.get_subset(subset)
            else:
                expected = expected_dataset

            with self.subTest(path=path, format=format, subset=subset):
                dataset = Dataset.import_from(path, format)
                compare_datasets(self, expected, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_panoptic_with_any_annotation_filename(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='default', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Mask(np.ones((5, 5)), label=0, id=460551,
                        group=460551, attributes={'is_crowd': False}),
                ]),
        ], categories=['a', 'b'])

        format = 'coco_panoptic'
        with TestDir() as test_dir:
            dataset_dir = osp.join(test_dir, 'dataset')
            expected_dataset.export(dataset_dir, format, save_images=True)
            os.rename(osp.join(dataset_dir, 'annotations', 'panoptic_default'),
                osp.join(dataset_dir, 'annotations', 'aa_bbbb_cccc'))
            os.rename(osp.join(dataset_dir, 'annotations', 'panoptic_default.json'),
                osp.join(dataset_dir, 'annotations', 'aa_bbbb_cccc.json'))

            imported_dataset = Dataset.import_from(dataset_dir, format)
            compare_datasets(self, expected_dataset, imported_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_panoptic_with_original_cat_ids(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='train', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Mask(np.ones((5, 5)), label=1, id=460551,
                        group=460551, attributes={'is_crowd': False}),
                ]),
        ], categories=['class-0', 'a', 'b'])

        actual_dataset = Dataset.import_from(
            osp.join(DUMMY_DATASET_DIR, 'coco_panoptic',
                'annotations', 'panoptic_train.json'),
            'coco_panoptic',
            keep_original_category_ids=True)
        compare_datasets(self, expected_dataset, actual_dataset,
            require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_stuff(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='train', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Mask(np.array(
                        [[0, 0, 1, 1, 0, 1, 1, 0, 0, 0]] * 5
                        ), label=0,
                        id=7, group=7, attributes={'is_crowd': False}),
                ]),

            DatasetItem(id='b', subset='val', image=np.ones((10, 5, 3)),
                attributes={'id': 40},
                annotations=[
                    Mask(np.array( [[1, 1, 0, 0, 0]] * 10 ), label=1,
                        id=2, group=2, attributes={'is_crowd': False}),
                ]),
        ], categories=['a', 'b'])

        formats = ['coco', 'coco_stuff']
        paths = [
            ('', osp.join(DUMMY_DATASET_DIR, 'coco_stuff')),
            ('train', osp.join(DUMMY_DATASET_DIR, 'coco_stuff',
                'annotations', 'stuff_train.json')),
            ('val', osp.join(DUMMY_DATASET_DIR, 'coco_stuff',
                'annotations', 'stuff_val.json')),
        ]
        for format, (subset, path) in product(formats, paths):
            if subset:
                expected = expected_dataset.get_subset(subset)
            else:
                expected = expected_dataset

            with self.subTest(path=path, format=format, subset=subset):
                dataset = Dataset.import_from(path, format)
                compare_datasets(self, expected, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_stuff_with_any_annotation_filename(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='a', subset='default', image=np.ones((5, 10, 3)),
                attributes={'id': 5},
                annotations=[
                    Mask(np.array(
                        [[0, 0, 1, 1, 0, 1, 1, 0, 0, 0]] * 5
                        ), label=0,
                        id=7, group=7, attributes={'is_crowd': False}),
                ]),
        ], categories=['a', 'b'])

        format = 'coco_stuff'
        with TestDir() as test_dir:
            dataset_dir = osp.join(test_dir, 'dataset')
            expected_dataset.export(dataset_dir, format, save_images=True)
            os.rename(osp.join(dataset_dir, 'annotations', 'stuff_default.json'),
                osp.join(dataset_dir, 'annotations', 'aa_bbbb_cccc.json'))

            imported_dataset = Dataset.import_from(dataset_dir, format)
            compare_datasets(self, expected_dataset, imported_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        subdirs = [
            'coco',
            'coco_captions',
            'coco_image_info',
            'coco_instances',
            'coco_labels',
            'coco_panoptic',
            'coco_person_keypoints',
            'coco_stuff',
        ]

        env = Environment()

        for subdir in subdirs:
            with self.subTest(subdir=subdir):
                dataset_dir = osp.join(DUMMY_DATASET_DIR, subdir)

                detected_formats = env.detect_dataset(dataset_dir)
                self.assertEqual([CocoImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_673)
    def test_can_pickle(self):
        subdirs = [
            'coco',
            'coco_captions',
            'coco_image_info',
            'coco_instances',
            'coco_labels',
            'coco_panoptic',
            'coco_person_keypoints',
            'coco_stuff',
        ]

        for subdir in subdirs:
            with self.subTest(fmt=subdir, subdir=subdir):
                dataset_dir = osp.join(DUMMY_DATASET_DIR, subdir)
                source = Dataset.import_from(dataset_dir, format=subdir)

                parsed = pickle.loads(pickle.dumps(source))

                compare_datasets_strict(self, source, parsed)

class CocoConverterTest(TestCase):
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None, **kwargs):
        return check_save_and_load(self, source_dataset, converter, test_dir,
            importer='coco',
            target_dataset=target_dataset, importer_args=importer_args, **kwargs)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_captions(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train',
                annotations=[
                    Caption('hello', id=1, group=1),
                    Caption('world', id=2, group=2),
                ], attributes={'id': 1}),
            DatasetItem(id=2, subset='train',
                annotations=[
                    Caption('test', id=3, group=3),
                ], attributes={'id': 2}),

            DatasetItem(id=3, subset='val',
                annotations=[
                    Caption('word', id=1, group=1),
                ], attributes={'id': 1}),
            ])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                CocoCaptionsConverter.convert, test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_instances(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', image=np.ones((4, 4, 3)),
                annotations=[
                    # Bbox + single polygon
                    Bbox(0, 1, 2, 2,
                        label=2, group=1, id=1,
                        attributes={ 'is_crowd': False }),
                    Polygon([0, 1, 2, 1, 2, 3, 0, 3],
                        attributes={ 'is_crowd': False },
                        label=2, group=1, id=1),
                ], attributes={'id': 1}),
            DatasetItem(id=2, subset='train', image=np.ones((4, 4, 3)),
                annotations=[
                    # Mask + bbox
                    Mask(np.array([
                            [0, 1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 1, 1, 1],
                            [0, 0, 0, 0]],
                        ),
                        attributes={ 'is_crowd': True },
                        label=4, group=3, id=3),
                    Bbox(1, 0, 2, 2, label=4, group=3, id=3,
                        attributes={ 'is_crowd': True }),
                ], attributes={'id': 2}),

            DatasetItem(id=3, subset='val', image=np.ones((4, 4, 3)),
                annotations=[
                    # Bbox + mask
                    Bbox(0, 1, 2, 2, label=4, group=3, id=3,
                        attributes={ 'is_crowd': True }),
                    Mask(np.array([
                            [0, 0, 0, 0],
                            [1, 1, 1, 0],
                            [1, 1, 0, 0],
                            [0, 0, 0, 0]],
                        ),
                        attributes={ 'is_crowd': True },
                        label=4, group=3, id=3),
                ], attributes={'id': 1}),
            ], categories=[str(i) for i in range(10)])

        target_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', image=np.ones((4, 4, 3)),
                annotations=[
                    Polygon([0, 1, 2, 1, 2, 3, 0, 3],
                        attributes={ 'is_crowd': False },
                        label=2, group=1, id=1),
                ], attributes={'id': 1}),
            DatasetItem(id=2, subset='train', image=np.ones((4, 4, 3)),
                annotations=[
                    Mask(np.array([
                            [0, 1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 1, 1, 1],
                            [0, 0, 0, 0]],
                        ),
                        attributes={ 'is_crowd': True },
                        label=4, group=3, id=3),
                ], attributes={'id': 2}),

            DatasetItem(id=3, subset='val', image=np.ones((4, 4, 3)),
                annotations=[
                    Mask(np.array([
                            [0, 0, 0, 0],
                            [1, 1, 1, 0],
                            [1, 1, 0, 0],
                            [0, 0, 0, 0]],
                        ),
                        attributes={ 'is_crowd': True },
                        label=4, group=3, id=3),
                ], attributes={'id': 1})
            ], categories=[str(i) for i in range(10)])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                CocoInstancesConverter.convert, test_dir,
                target_dataset=target_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_panoptic(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', image=np.ones((4, 4, 3)),
                annotations=[
                    Mask(image=np.array([
                            [0, 1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 1, 1, 1],
                            [0, 0, 0, 0]
                        ]),
                        attributes={ 'is_crowd': False },
                        label=4, group=3, id=3),
                ], attributes={'id': 1}),

            DatasetItem(id=2, subset='val', image=np.ones((5, 5, 3)),
                annotations=[
                    Mask(image=np.array([
                            [0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]
                        ]),
                        attributes={ 'is_crowd': False },
                        label=4, group=3, id=3),
                    Mask(image=np.array([
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1]
                        ]),
                        attributes={ 'is_crowd': False },
                        label=2, group=2, id=2),
                ], attributes={'id': 2}),
            ], categories=[str(i) for i in range(10)])

        with TestDir() as test_dir:
            self._test_save_and_load(dataset,
                partial(CocoPanopticConverter.convert, save_images=True),
                test_dir, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_stuff(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', image=np.ones((4, 4, 3)),
                annotations=[
                    Mask(np.array([
                            [0, 1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 1, 1, 1],
                            [0, 0, 0, 0]],
                        ),
                        attributes={ 'is_crowd': False },
                        label=4, group=3, id=3),
                ], attributes={'id': 2}),

            DatasetItem(id=2, subset='val', image=np.ones((4, 4, 3)),
                annotations=[
                    Mask(np.array([
                            [0, 0, 0, 0],
                            [1, 1, 1, 0],
                            [1, 1, 0, 0],
                            [0, 0, 0, 0]],
                        ),
                        attributes={ 'is_crowd': False },
                        label=4, group=3, id=3),
                ], attributes={'id': 1}),
            ], categories=[str(i) for i in range(10)])

        with TestDir() as test_dir:
            self._test_save_and_load(dataset,
                CocoStuffConverter.convert, test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_polygons_on_loading(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.zeros((6, 10, 3)),
                annotations=[
                    Polygon([0, 0, 4, 0, 4, 4],
                        label=3, id=4, group=4),
                    Polygon([5, 0, 9, 0, 5, 5],
                        label=3, id=4, group=4),
                ]
            ),
        ], categories=[str(i) for i in range(10)])

        target_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.zeros((6, 10, 3)),
                annotations=[
                    Mask(np.array([
                        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        # only internal fragment (without the border),
                        # but not everywhere...
                    ),
                    label=3, id=4, group=4,
                    attributes={ 'is_crowd': False }),
                ], attributes={'id': 1}
            ),
        ], categories=[str(i) for i in range(10)])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                CocoInstancesConverter.convert, test_dir,
                importer_args={'merge_instance_polygons': True},
                target_dataset=target_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_crop_covered_segments(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.zeros((5, 5, 3)),
                annotations=[
                    Mask(np.array([
                            [0, 0, 1, 1, 1],
                            [0, 0, 1, 1, 1],
                            [1, 1, 0, 1, 1],
                            [1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0]],
                        ),
                        label=2, id=1, z_order=0),
                    Polygon([1, 1, 4, 1, 4, 4, 1, 4],
                        label=1, id=2, z_order=1),
                ]
            ),
        ], categories=[str(i) for i in range(10)])

        target_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.zeros((5, 5, 3)),
                annotations=[
                    Mask(np.array([
                            [0, 0, 1, 1, 1],
                            [0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0]],
                        ),
                        attributes={ 'is_crowd': True },
                        label=2, id=1, group=1),

                    Polygon([1, 1, 4, 1, 4, 4, 1, 4],
                        label=1, id=2, group=2,
                        attributes={ 'is_crowd': False }),
                ], attributes={'id': 1}
            ),
        ], categories=[str(i) for i in range(10)])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                 partial(CocoInstancesConverter.convert, crop_covered=True),
                 test_dir, target_dataset=target_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_convert_polygons_to_mask(self):
        """
        <b>Description:</b>
        Ensure that the dataset polygon annotation can be properly converted into dataset segmentation mask.

        <b>Expected results:</b>
        Dataset segmentation mask converted from dataset polygon annotation is equal to expected mask.

        <b>Steps:</b>
        1. Prepare dataset with polygon annotation (source dataset)
        2. Prepare dataset with expected mask segmentation mode (target dataset)
        3. Convert source dataset to target, with conversion of annotation from polygon to mask. Verify that result
        segmentation mask is equal to expected mask.

        """

        # 1. Prepare dataset with polygon annotation (source dataset)
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.zeros((6, 10, 3)),
                annotations=[
                    Polygon([0, 0, 4, 0, 4, 4],
                        label=3, id=4, group=4),
                    Polygon([5, 0, 9, 0, 5, 5],
                        label=3, id=4, group=4),
                ]
            ),
        ], categories=[str(i) for i in range(10)])

        # 2. Prepare dataset with expected mask segmentation mode (target dataset)
        target_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.zeros((6, 10, 3)),
                annotations=[
                    Mask(np.array([
                            [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                            # only internal fragment (without the border),
                            # but not everywhere...
                        ),
                        attributes={ 'is_crowd': True },
                        label=3, id=4, group=4),
                ], attributes={'id': 1}
            ),
        ], categories=[str(i) for i in range(10)])

        # 3. Convert source dataset to target, with conversion of annotation from polygon to mask. Verify that result
        # segmentation mask is equal to expected mask.
        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(CocoInstancesConverter.convert, segmentation_mode='mask'),
                test_dir, target_dataset=target_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_convert_masks_to_polygons(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.zeros((5, 10, 3)),
                annotations=[
                    Mask(np.array([
                            [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ]),
                        label=3, id=4, group=4),
                ]
            ),
        ], categories=[str(i) for i in range(10)])

        target_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.zeros((5, 10, 3)),
                annotations=[
                    Polygon(
                        [1, 0, 3, 2, 3, 0, 1, 0],
                        label=3, id=4, group=4,
                        attributes={ 'is_crowd': False }),
                    Polygon(
                        [5, 0, 5, 3, 8, 0, 5, 0],
                        label=3, id=4, group=4,
                        attributes={ 'is_crowd': False }),
                ], attributes={'id': 1}
            ),
        ], categories=[str(i) for i in range(10)])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(CocoInstancesConverter.convert, segmentation_mode='polygons'),
                test_dir,
                target_dataset=target_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_images(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', attributes={'id': 1}),
            DatasetItem(id=2, subset='train', attributes={'id': 2}),

            DatasetItem(id=2, subset='val', attributes={'id': 2}),
            DatasetItem(id=3, subset='val', attributes={'id': 3}),
            DatasetItem(id=4, subset='val', attributes={'id': 4}),

            DatasetItem(id=5, subset='test', attributes={'id': 1}),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                CocoImageInfoConverter.convert, test_dir)

    @mark_requirement(Requirements.DATUM_231)
    def test_can_save_dataset_with_cjk_categories(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', image=np.ones((4, 4, 3)),
                annotations=[
                    Bbox(0, 1, 2, 2,
                        label=0, group=1, id=1,
                        attributes={ 'is_crowd': False }),
                ], attributes={'id': 1}),
            DatasetItem(id=2, subset='train', image=np.ones((4, 4, 3)),
                annotations=[
                    Bbox(1, 0, 2, 2, label=1, group=2, id=2,
                        attributes={ 'is_crowd': False }),
                ], attributes={'id': 2}),

            DatasetItem(id=3, subset='train', image=np.ones((4, 4, 3)),
                annotations=[
                    Bbox(0, 1, 2, 2, label=2, group=3, id=3,
                        attributes={ 'is_crowd': False }),
                ], attributes={'id': 3}),
            ],
            categories=[
                "고양이", "ネコ", "猫"
            ]
        )

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                CocoInstancesConverter.convert, test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='кириллица с пробелом', subset='train',
                attributes={'id': 1}),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                CocoImageInfoConverter.convert, test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_labels(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train',
                annotations=[
                    Label(4, id=1, group=1),
                    Label(9, id=2, group=2),
                ], attributes={'id': 1}),
        ], categories=[str(i) for i in range(10)])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                CocoLabelsConverter.convert, test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_keypoints(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', image=np.zeros((5, 5, 3)),
                annotations=[
                    # Full instance annotations: polygon + keypoints
                    Points([0, 0, 0, 2, 4, 1], [0, 1, 2],
                        label=3, group=1, id=1),
                    Polygon([0, 0, 4, 0, 4, 4],
                        label=3, group=1, id=1),

                    # Full instance annotations: bbox + keypoints
                    Points([1, 2, 3, 4, 2, 3], group=2, id=2),
                    Bbox(1, 2, 2, 2, group=2, id=2),

                    # Solitary keypoints
                    Points([1, 2, 0, 2, 4, 1], label=5, id=3),

                    # Some other solitary annotations (bug #1387)
                    Polygon([0, 0, 4, 0, 4, 4], label=3, id=4),

                    # Solitary keypoints with no label
                    Points([0, 0, 1, 2, 3, 4], [0, 1, 2], id=5),
                ]),
            ], categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    str(i) for i in range(10)),
                AnnotationType.points: PointsCategories.from_iterable(
                    (i, None, [[0, 1], [1, 2]]) for i in range(10)
                ),
            })

        target_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', image=np.zeros((5, 5, 3)),
                annotations=[
                    Points([0, 0, 0, 2, 4, 1], [0, 1, 2],
                        label=3, group=1, id=1,
                        attributes={'is_crowd': False}),
                    Polygon([0, 0, 4, 0, 4, 4],
                        label=3, group=1, id=1,
                        attributes={'is_crowd': False}),

                    Points([1, 2, 3, 4, 2, 3],
                        group=2, id=2,
                        attributes={'is_crowd': False}),
                    Bbox(1, 2, 2, 2,
                        group=2, id=2,
                        attributes={'is_crowd': False}),

                    Points([1, 2, 0, 2, 4, 1],
                        label=5, group=3, id=3,
                        attributes={'is_crowd': False}),
                    Bbox(0, 1, 4, 1,
                        label=5, group=3, id=3,
                        attributes={'is_crowd': False}),

                    Points([0, 0, 1, 2, 3, 4], [0, 1, 2],
                        group=5, id=5,
                        attributes={'is_crowd': False}),
                    Bbox(1, 2, 2, 2,
                        group=5, id=5,
                        attributes={'is_crowd': False}),
                ], attributes={'id': 1}),
            ], categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    str(i) for i in range(10)),
                AnnotationType.points: PointsCategories.from_iterable(
                    (i, None, [[0, 1], [1, 2]]) for i in range(10)
                ),
            })

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                CocoPersonKeypointsConverter.convert, test_dir,
                target_dataset=target_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_no_subsets(self):
        test_dataset = Dataset.from_iterable([
            DatasetItem(id=1, attributes={'id': 1}),
            DatasetItem(id=2, attributes={'id': 2}),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(test_dataset,
                CocoConverter.convert, test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_image_info(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=Image(path='1.jpg', size=(10, 15)),
                attributes={'id': 1}),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                CocoImageInfoConverter.convert, test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_relative_paths(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='1', image=np.ones((4, 2, 3)),
                attributes={'id': 1}),
            DatasetItem(id='subdir1/1', image=np.ones((2, 6, 3)),
                attributes={'id': 2}),
            DatasetItem(id='subdir2/1', image=np.ones((5, 4, 3)),
                attributes={'id': 3}),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                partial(CocoImageInfoConverter.convert, save_images=True),
                test_dir, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        expected = Dataset.from_iterable([
            DatasetItem(id='q/1', image=Image(path='q/1.JPEG',
                data=np.zeros((4, 3, 3))), attributes={'id': 1}),
            DatasetItem(id='a/b/c/2', image=Image(path='a/b/c/2.bmp',
                data=np.zeros((3, 4, 3))), attributes={'id': 2}),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(expected,
                partial(CocoImageInfoConverter.convert, save_images=True),
                test_dir, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_preserve_coco_ids(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='some/name1', image=np.ones((4, 2, 3)),
                attributes={'id': 40}),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                partial(CocoImageInfoConverter.convert, save_images=True),
                test_dir, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_annotation_attributes(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.ones((4, 2, 3)), annotations=[
                Polygon([0, 0, 4, 0, 4, 4], label=5, group=1, id=1,
                    attributes={'is_crowd': False, 'x': 5, 'y': 'abc'}),
            ], attributes={'id': 1})
        ], categories=[str(i) for i in range(10)])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                CocoConverter.convert, test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_auto_annotation_ids(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=2, image=np.ones((4, 2, 3)), annotations=[
                Polygon([0, 0, 4, 0, 4, 4], label=0),
            ])
        ], categories=[str(i) for i in range(10)])

        target_dataset = Dataset.from_iterable([
            DatasetItem(id=2, image=np.ones((4, 2, 3)), annotations=[
                Polygon([0, 0, 4, 0, 4, 4], label=0, id=1, group=1,
                    attributes={'is_crowd': False}),
            ], attributes={'id': 1})
        ], categories=[str(i) for i in range(10)])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                CocoConverter.convert, test_dir, target_dataset=target_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_subset_can_contain_underscore(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=2, image=np.ones((4, 2, 3)), subset='subset_1',
                annotations=[Polygon([0, 0, 4, 0, 4, 4], label=0, id=1, group=1,
                    attributes={'is_crowd': False}),
            ], attributes={'id': 1})
        ], categories=[str(i) for i in range(10)])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                CocoConverter.convert, test_dir)


    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=2, image=np.ones((4, 2, 3)), annotations=[
                Polygon([0, 0, 4, 0, 4, 4], label=0, id=5),
            ], attributes={'id': 22})
        ], categories=[str(i) for i in range(10)])

        target_dataset = Dataset.from_iterable([
            DatasetItem(id=2, image=np.ones((4, 2, 3)), annotations=[
                Polygon([0, 0, 4, 0, 4, 4], label=0, id=1, group=1,
                    attributes={'is_crowd': False}),
            ], attributes={'id': 1})
        ], categories=[str(i) for i in range(10)])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(CocoConverter.convert, reindex=True),
                test_dir, target_dataset=target_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_images_in_single_dir(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', image=np.ones((2, 4, 3)),
                attributes={'id': 1}),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(dataset,
                partial(CocoImageInfoConverter.convert, save_images=True,
                    merge_images=True),
                test_dir, require_images=True)
            self.assertTrue(osp.isfile(osp.join(test_dir, 'images', '1.jpg')))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_images_in_separate_dirs(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', image=np.ones((2, 4, 3)),
                attributes={'id': 1}),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(dataset,
                partial(CocoImageInfoConverter.convert, save_images=True,
                    merge_images=False),
                test_dir, require_images=True)
            self.assertTrue(osp.isfile(osp.join(
                test_dir, 'images', 'train', '1.jpg')))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self):
        expected = Dataset.from_iterable([
            DatasetItem(1, subset='a'),
            DatasetItem(2, subset='a', image=np.ones((3, 2, 3))),

            DatasetItem(2, subset='b'),
        ])

        with TestDir() as path:
            dataset = Dataset.from_iterable([
                DatasetItem(1, subset='a'),
                DatasetItem(2, subset='b'),
                DatasetItem(3, subset='c', image=np.ones((2, 2, 3))),
            ])
            dataset.export(path, 'coco', save_images=True)

            dataset.put(DatasetItem(2, subset='a', image=np.ones((3, 2, 3))))
            dataset.remove(3, 'c')
            dataset.save(save_images=True)

            self.assertEqual({'image_info_a.json', 'image_info_b.json'},
                set(os.listdir(osp.join(path, 'annotations'))))
            self.assertTrue(osp.isfile(osp.join(path, 'images', 'a', '2.jpg')))
            self.assertFalse(osp.isfile(osp.join(path, 'images', 'c', '3.jpg')))
            compare_datasets(self, expected, Dataset.import_from(path, 'coco'),
                require_images=True, ignored_attrs={'id'})

    @mark_requirement(Requirements. DATUM_BUG_425)
    def test_can_save_and_load_grouped_masks_and_polygons(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.zeros((5, 5, 3)),
                annotations=[
                    Mask(np.array([
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]],
                        ),
                        label=0, id=0, z_order=0, group=1),
                    Polygon([1, 1, 1, 3, 3, 3, 3, 1],
                        label=0, id=1, z_order=0, group=1),
                ]
            ),
        ], categories=['label_1'])

        target_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.zeros((5, 5, 3)),
                annotations=[
                    Mask(np.array([
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]],
                        ),
                        attributes={ 'is_crowd': True },
                        label=0, id=0, group=1),
                ], attributes={'id': 1}
            ),
        ], categories=['label_1'])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                 partial(CocoInstancesConverter.convert),
                 test_dir, target_dataset=target_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_panoptic_with_meta_file(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', image=np.ones((4, 4, 3)),
                annotations=[
                    Mask(image=np.array([
                            [0, 1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 1, 1, 1],
                            [0, 0, 0, 0]
                        ]),
                        attributes={ 'is_crowd': False },
                        label=4, group=3, id=3),
                ], attributes={'id': 1}),

            DatasetItem(id=2, subset='val', image=np.ones((5, 5, 3)),
                annotations=[
                    Mask(image=np.array([
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1]
                        ]),
                        attributes={ 'is_crowd': False },
                        label=2, group=2, id=2),
                ], attributes={'id': 2}),
            ], categories=[str(i) for i in range(10)])

        with TestDir() as test_dir:
            self._test_save_and_load(dataset,
                partial(CocoPanopticConverter.convert, save_images=True,
                    save_dataset_meta=True),
                test_dir, require_images=True)
            self.assertTrue(osp.isfile(osp.join(test_dir, 'dataset_meta.json')))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_stuff_with_meta_file(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', image=np.ones((4, 4, 3)),
                annotations=[
                    Mask(np.array([
                            [0, 1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 1, 1, 1],
                            [0, 0, 0, 0]],
                        ),
                        attributes={ 'is_crowd': False },
                        label=4, group=3, id=3),
                ], attributes={'id': 2}),

            DatasetItem(id=2, subset='val', image=np.ones((4, 4, 3)),
                annotations=[
                    Mask(np.array([
                            [0, 0, 0, 0],
                            [1, 1, 1, 0],
                            [1, 1, 0, 0],
                            [0, 0, 0, 0]],
                        ),
                        attributes={ 'is_crowd': False },
                        label=4, group=3, id=3),
                ], attributes={'id': 1}),
            ], categories=[str(i) for i in range(10)])

        with TestDir() as test_dir:
            self._test_save_and_load(dataset,
                partial(CocoPanopticConverter.convert, save_images=True,
                    save_dataset_meta=True),
                test_dir, require_images=True)
            self.assertTrue(osp.isfile(osp.join(test_dir, 'dataset_meta.json')))
