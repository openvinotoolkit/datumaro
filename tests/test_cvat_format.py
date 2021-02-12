from functools import partial
import numpy as np
import os
import os.path as osp

from unittest import TestCase
from datumaro.components.project import Dataset
from datumaro.components.extractor import (DatasetItem,
    AnnotationType, Points, Polygon, PolyLine, Bbox, Label,
    LabelCategories,
)
from datumaro.plugins.cvat_format.extractor import CvatImporter
from datumaro.plugins.cvat_format.converter import CvatConverter
from datumaro.util.image import Image
from datumaro.util.test_utils import (TestDir, compare_datasets,
    test_save_and_load)


DUMMY_IMAGE_DATASET_DIR = osp.join(osp.dirname(__file__),
    'assets', 'cvat_dataset', 'for_images')

DUMMY_VIDEO_DATASET_DIR = osp.join(osp.dirname(__file__),
    'assets', 'cvat_dataset', 'for_video')

class CvatImporterTest(TestCase):
    def test_can_detect_image(self):
        self.assertTrue(CvatImporter.detect(DUMMY_IMAGE_DATASET_DIR))

    def test_can_detect_video(self):
        self.assertTrue(CvatImporter.detect(DUMMY_VIDEO_DATASET_DIR))

    def test_can_load_image(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='img0', subset='train',
                image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=0, z_order=1,
                        attributes={
                            'occluded': True,
                            'a1': True, 'a2': 'v3'
                        }),
                    PolyLine([1, 2, 3, 4, 5, 6, 7, 8],
                        attributes={'occluded': False}),
                ], attributes={'frame': 0}),
            DatasetItem(id='img1', subset='train',
                image=np.ones((10, 10, 3)),
                annotations=[
                    Polygon([1, 2, 3, 4, 6, 5], z_order=1,
                        attributes={'occluded': False}),
                    Points([1, 2, 3, 4, 5, 6], label=1, z_order=2,
                        attributes={'occluded': False}),
                ], attributes={'frame': 1}),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable([
                ['label1', '', {'a1', 'a2'}],
                ['label2'],
            ])
        })

        parsed_dataset = Dataset.import_from(DUMMY_IMAGE_DATASET_DIR, 'cvat')

        compare_datasets(self, expected_dataset, parsed_dataset)

    def test_can_load_video(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000010', subset='annotations',
                image=255 * np.ones((20, 25, 3)),
                annotations=[
                    Bbox(3, 4, 7, 1, label=2,
                        id=0,
                        attributes={
                            'occluded': True,
                            'outside': False, 'keyframe': True,
                            'track_id': 0
                        }),
                    Points([21.95, 8.00, 2.55, 15.09, 2.23, 3.16],
                        label=0,
                        id=1,
                        attributes={
                            'occluded': False,
                            'outside': False, 'keyframe': True,
                            'track_id': 1, 'hgl': 'hgkf',
                        }),
                ], attributes={'frame': 10}),
            DatasetItem(id='frame_000013', subset='annotations',
                image=255 * np.ones((20, 25, 3)),
                annotations=[
                    Bbox(7, 6, 7, 2, label=2,
                        id=0,
                        attributes={
                            'occluded': False,
                            'outside': True, 'keyframe': True,
                            'track_id': 0
                        }),
                    Points([21.95, 8.00, 9.55, 15.09, 5.23, 1.16],
                        label=0,
                        id=1,
                        attributes={
                            'occluded': False,
                            'outside': True, 'keyframe': True,
                            'track_id': 1, 'hgl': 'jk',
                        }),
                    PolyLine([7.85, 13.88, 3.50, 6.67, 15.90, 2.00, 13.31, 7.21],
                        label=2,
                        id=2,
                        attributes={
                            'occluded': False,
                            'outside': False, 'keyframe': True,
                            'track_id': 2,
                        }),
                ], attributes={'frame': 13}),
            DatasetItem(id='frame_000016', subset='annotations',
                image=Image(path='frame_0000016.png', size=(20, 25)),
                annotations=[
                    Bbox(8, 7, 6, 10, label=2,
                        id=0,
                        attributes={
                            'occluded': False,
                            'outside': True, 'keyframe': True,
                            'track_id': 0
                        }),
                    PolyLine([7.85, 13.88, 3.50, 6.67, 15.90, 2.00, 13.31, 7.21],
                        label=2,
                        id=2,
                        attributes={
                            'occluded': False,
                            'outside': True, 'keyframe': True,
                            'track_id': 2,
                        }),
                ], attributes={'frame': 16}),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable([
                ['klhg', '', {'hgl'}],
                ['z U k'],
                ['II']
            ]),
        })

        parsed_dataset = Dataset.import_from(DUMMY_VIDEO_DATASET_DIR, 'cvat')

        compare_datasets(self, expected_dataset, parsed_dataset)

class CvatConverterTest(TestCase):
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None):
        return test_save_and_load(self, source_dataset, converter, test_dir,
            importer='cvat',
            target_dataset=target_dataset, importer_args=importer_args)

    def test_can_save_and_load(self):
        label_categories = LabelCategories()
        for i in range(10):
            label_categories.add(str(i))
        label_categories.items[2].attributes.update(['a1', 'a2', 'empty'])
        label_categories.attributes.update(['occluded'])

        source_dataset = Dataset.from_iterable([
            DatasetItem(id=0, subset='s1', image=np.zeros((5, 10, 3)),
                annotations=[
                    Polygon([0, 0, 4, 0, 4, 4],
                        label=1, group=4,
                        attributes={ 'occluded': True}),
                    Points([1, 1, 3, 2, 2, 3],
                        label=2,
                        attributes={ 'a1': 'x', 'a2': 42, 'empty': '',
                            'unknown': 'bar' }),
                    Label(1),
                    Label(2, attributes={ 'a1': 'y', 'a2': 44 }),
                ]
            ),
            DatasetItem(id=1, subset='s1',
                annotations=[
                    PolyLine([0, 0, 4, 0, 4, 4],
                        label=3, id=4, group=4),
                    Bbox(5, 0, 1, 9,
                        label=3, id=4, group=4),
                ]
            ),

            DatasetItem(id=2, subset='s2', image=np.ones((5, 10, 3)),
                annotations=[
                    Polygon([0, 0, 4, 0, 4, 4], z_order=1,
                        label=3, group=4,
                        attributes={ 'occluded': False }),
                    PolyLine([5, 0, 9, 0, 5, 5]), # will be skipped as no label
                ]
            ),

            DatasetItem(id=3, subset='s3', image=Image(
                path='3.jpg', size=(2, 4))),
        ], categories={
            AnnotationType.label: label_categories,
        })

        target_dataset = Dataset.from_iterable([
            DatasetItem(id=0, subset='s1', image=np.zeros((5, 10, 3)),
                annotations=[
                    Polygon([0, 0, 4, 0, 4, 4],
                        label=1, group=4,
                        attributes={ 'occluded': True }),
                    Points([1, 1, 3, 2, 2, 3],
                        label=2,
                        attributes={ 'occluded': False, 'empty': '',
                            'a1': 'x', 'a2': 42 }),
                    Label(1),
                    Label(2, attributes={ 'a1': 'y', 'a2': 44 }),
                ], attributes={'frame': 0}
            ),
            DatasetItem(id=1, subset='s1',
                annotations=[
                    PolyLine([0, 0, 4, 0, 4, 4],
                        label=3, group=4,
                        attributes={ 'occluded': False }),
                    Bbox(5, 0, 1, 9,
                        label=3, group=4,
                        attributes={ 'occluded': False }),
                ], attributes={'frame': 1}
            ),

            DatasetItem(id=2, subset='s2', image=np.ones((5, 10, 3)),
                annotations=[
                    Polygon([0, 0, 4, 0, 4, 4], z_order=1,
                        label=3, group=4,
                        attributes={ 'occluded': False }),
                ], attributes={'frame': 0}
            ),

            DatasetItem(id=3, subset='s3', image=Image(
                    path='3.jpg', size=(2, 4)),
                attributes={'frame': 0}),
        ], categories={
            AnnotationType.label: label_categories,
        })

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(CvatConverter.convert, save_images=True), test_dir,
                target_dataset=target_dataset)

    def test_relative_paths(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', image=np.ones((4, 2, 3))),
            DatasetItem(id='subdir1/1', image=np.ones((2, 6, 3))),
            DatasetItem(id='subdir2/1', image=np.ones((5, 4, 3))),
        ], categories={ AnnotationType.label: LabelCategories() })

        target_dataset = Dataset.from_iterable([
            DatasetItem(id='1', image=np.ones((4, 2, 3)),
                attributes={'frame': 0}),
            DatasetItem(id='subdir1/1', image=np.ones((2, 6, 3)),
                attributes={'frame': 1}),
            DatasetItem(id='subdir2/1', image=np.ones((5, 4, 3)),
                attributes={'frame': 2}),
        ], categories={
            AnnotationType.label: LabelCategories()
        })

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(CvatConverter.convert, save_images=True), test_dir,
                target_dataset=target_dataset)

    def test_preserve_frame_ids(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='some/name1', image=np.ones((4, 2, 3)),
                attributes={'frame': 40}),
        ], categories={
            AnnotationType.label: LabelCategories()
        })

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                CvatConverter.convert, test_dir)

    def test_reindex(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='some/name1', image=np.ones((4, 2, 3)),
                attributes={'frame': 40}),
        ], categories={ AnnotationType.label: LabelCategories() })

        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='some/name1', image=np.ones((4, 2, 3)),
                attributes={'frame': 0}),
        ], categories={ AnnotationType.label: LabelCategories() })

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(CvatConverter.convert, reindex=True), test_dir,
                target_dataset=expected_dataset)

    def test_inplace_save_writes_only_updated_data(self):
        with TestDir() as path:
            # generate initial dataset
            dataset = Dataset.from_iterable([
                DatasetItem(1, subset='a'),
                DatasetItem(2, subset='b'),
                DatasetItem(3, subset='c', image=np.ones((2, 2, 3))),
            ], categories=[])
            dataset.export(path, 'cvat', save_images=True)
            os.unlink(osp.join(path, 'a.xml'))
            os.unlink(osp.join(path, 'b.xml'))
            os.unlink(osp.join(path, 'c.xml'))
            self.assertFalse(osp.isfile(osp.join(path, 'images', '2.jpg')))
            self.assertTrue(osp.isfile(osp.join(path, 'images', '3.jpg')))

            dataset.put(DatasetItem(2, subset='a', image=np.ones((3, 2, 3))))
            dataset.remove(3, 'c')
            dataset.save(save_images=True)

            self.assertTrue(osp.isfile(osp.join(path, 'a.xml')))
            self.assertFalse(osp.isfile(osp.join(path, 'b.xml')))
            self.assertTrue(osp.isfile(osp.join(path, 'c.xml')))
            self.assertTrue(osp.isfile(osp.join(path, 'images', '2.jpg')))
            self.assertFalse(osp.isfile(osp.join(path, 'images', '3.jpg')))