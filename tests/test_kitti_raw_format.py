from functools import partial
from unittest import TestCase
import os
import os.path as osp

from datumaro.components.extractor import (
    AnnotationType, Cuboid3d, DatasetItem, LabelCategories,
)
from datumaro.components.project import Dataset
from datumaro.plugins.kitti_raw_format.converter import KittiRawConverter
from datumaro.plugins.kitti_raw_format.extractor import KittiRawImporter
from datumaro.util.test_utils import (
    Dimensions, TestDir, compare_datasets_3d, test_save_and_load,
)

from tests.requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(osp.dirname(
    __file__), 'assets', 'kitti_dataset', 'kitti_raw')


class KittiRawImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        self.assertTrue(KittiRawImporter.detect(DUMMY_DATASET_DIR))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load(self):
        pcd1 = osp.join(DUMMY_DATASET_DIR,
            'velodyne_points', 'data', '0000000000.pcd')
        pcd2 = osp.join(DUMMY_DATASET_DIR,
            'velodyne_points', 'data', '0000000001.pcd')
        pcd3 = osp.join(DUMMY_DATASET_DIR,
            'velodyne_points', 'data', '0000000002.pcd')

        image1 = osp.join(DUMMY_DATASET_DIR,
            'IMAGE_00', 'data', '0000000000.png')
        image2 = osp.join(DUMMY_DATASET_DIR,
            'IMAGE_00', 'data', '0000000001.png')
        image3 = osp.join(DUMMY_DATASET_DIR,
            'IMAGE_00', 'data', '0000000002.png')

        expected_label_cat = LabelCategories(attributes={'occluded'})
        expected_label_cat.add('bus')
        expected_label_cat.add('car')
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='0000000000',
                annotations=[
                    Cuboid3d(position=[1, 2, 3], scale=[7.95, -3.62, -1.03],
                        label=1, attributes={'occluded': False, 'track_id': 1}),

                    Cuboid3d(position=[1, 1, 0], scale=[8.34, 23.01, -0.76],
                        label=0, attributes={'occluded': False, 'track_id': 2})
                ],
                point_cloud=pcd1, related_images=[image1],
                attributes={'frame': 0}),

            DatasetItem(id='0000000001',
                annotations=[
                    Cuboid3d(position=[0, 1, 0], scale=[8.34, 23.01, -0.76],
                        rotation=[1, 1, 3],
                        label=0, attributes={'occluded': True, 'track_id': 2})
                ],
                point_cloud=pcd2, related_images=[image2],
                attributes={'frame': 1}),

            DatasetItem(id='0000000002',
                annotations=[
                    Cuboid3d(position=[1, 2, 3], scale=[-9.41, 13.54, 0.24],
                        label=1, attributes={'occluded': False, 'track_id': 3})
                ],
                point_cloud=pcd3, related_images=[image3],
                attributes={'frame': 2})

        ], categories={AnnotationType.label: expected_label_cat})

        parsed_dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'kitti_raw')

        compare_datasets_3d(self, expected_dataset, parsed_dataset,
            require_point_cloud=True)


class KittiRawConverterTest(TestCase):
    pcd1 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
        'velodyne_points', 'data', '0000000000.pcd'))
    pcd2 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
        'velodyne_points', 'data', '0000000001.pcd'))
    pcd3 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
        'velodyne_points', 'data', '0000000002.pcd'))

    image1 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
        'IMAGE_00', 'data', '0000000000.png'))
    image2 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
        'IMAGE_00', 'data', '0000000001.png'))
    image3 = osp.abspath(osp.join(DUMMY_DATASET_DIR,
        'IMAGE_00', 'data', '0000000002.png'))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None, **kwargs):
        kwargs.setdefault('dimension', Dimensions.dim_3d)
        return test_save_and_load(self, source_dataset, converter, test_dir,
            importer='kitti_raw', target_dataset=target_dataset,
            importer_args=importer_args, **kwargs)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='0000000000',
                annotations=[
                    Cuboid3d(position=[13.54, -9.41, 0.24], label=0,
                        attributes={'occluded': False, 'track_id': 1}),

                    Cuboid3d(position=[3.4, -2.11, 4.4], label=1,
                        attributes={'occluded': True, 'track_id': 2})
                ],
                point_cloud=self.pcd1, related_images=[self.image1],
                attributes={'frame': 0}
            ),

            DatasetItem(id='0000000001',
                annotations=[
                    Cuboid3d(position=[1.4, 2.1, 1.4], label=1,
                        attributes={'track_id': 2}),

                    Cuboid3d(position=[11.4, -0.1, 4.2], scale=[2, 1, 2],
                        label=0, attributes={'track_id': 3})
                ],
            ),

            DatasetItem(id='0000000002',
                annotations=[
                    Cuboid3d(position=[0.4, -1, 2.24], scale=[2, 1, 2],
                        label=0, attributes={'track_id': 3}),
                ],
                point_cloud=self.pcd3,
                attributes={'frame': 2}
            ),
        ], categories=['cat', 'dog'])

        with TestDir() as test_dir:
            target_label_cat = LabelCategories(attributes={'occluded'})
            target_label_cat.add('cat')
            target_label_cat.add('dog')

            target_dataset = Dataset.from_iterable([
                DatasetItem(id='0000000000',
                    annotations=[
                        Cuboid3d(position=[13.54, -9.41, 0.24], label=0,
                            attributes={
                                'occluded': False, 'track_id': 1}),

                        Cuboid3d(position=[3.4, -2.11, 4.4], label=1,
                            attributes={
                                'occluded': True, 'track_id': 2})
                    ],
                    point_cloud=osp.join(test_dir,
                        'velodyne_points', 'data', '0000000000.pcd'),
                    related_images=[osp.join(test_dir,
                        'image_00', 'data', '0000000000.png')
                    ],
                    attributes={'frame': 0}
                ),

                DatasetItem(id='0000000001',
                    annotations=[
                        Cuboid3d(position=[1.4, 2.1, 1.4], label=1,
                            attributes={'occluded': False, 'track_id': 2}),

                        Cuboid3d(position=[11.4, -0.1, 4.2], scale=[2, 1, 2],
                            label=0, attributes={
                                'occluded': False, 'track_id': 3})
                    ],
                    attributes={'frame': 1}
                ),

                DatasetItem(id='0000000002',
                    annotations=[
                        Cuboid3d(position=[0.4, -1, 2.24], scale=[2, 1, 2],
                            label=0, attributes={
                                'occluded': False, 'track_id': 3}),
                    ],
                    point_cloud=osp.join(test_dir,
                        'velodyne_points', 'data', '0000000002.pcd'),
                    attributes={'frame': 2}
                ),
            ], categories={AnnotationType.label: target_label_cat})

            self._test_save_and_load(source_dataset,
                partial(KittiRawConverter.convert, save_images=True),
                test_dir, target_dataset=target_dataset,
                require_point_cloud=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_preserve_frame_ids(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='abc', attributes={'frame': 40})
        ], categories=[])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                KittiRawConverter.convert, test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex_frames(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='abc')
        ], categories=[])

        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='abc', attributes={'frame': 0})
        ], categories=[])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(KittiRawConverter.convert, reindex=True),
                test_dir, target_dataset=expected_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_requires_track_id(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='abc',
                annotations=[
                    Cuboid3d(position=[0.4, -1, 2.24], label=0),
                ]
            )
        ], categories=['dog'])

        with TestDir() as test_dir:
            with self.assertRaisesRegex(Exception, 'track_id'):
                KittiRawConverter.convert(source_dataset, test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex_allows_single_annotations(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='abc',
                annotations=[
                    Cuboid3d(position=[0.4, -1, 2.24], label=0),
                ]
            )
        ], categories=['dog'])

        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='abc',
                annotations=[
                    Cuboid3d(position=[0.4, -1, 2.24], label=0,
                        attributes={'track_id': 1, 'occluded': False}),
                ],
                attributes={'frame': 0})
        ], categories=['dog'])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(KittiRawConverter.convert, reindex=True),
                test_dir, target_dataset=expected_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_attributes(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='0000000000',
                annotations=[
                    Cuboid3d(position=[13.54, -9.41, 0.24], label=0,
                        attributes={'track_id': 1,
                            'occluded': True, 'a': 'w', 'b': 5})
                ],
                attributes={'frame': 0}
            )
        ], categories=['cat'])

        target_label_cat = LabelCategories(attributes={'occluded'})
        target_label_cat.add('cat', attributes=['a', 'b'])
        target_dataset = Dataset.from_iterable([
            DatasetItem(id='0000000000',
                annotations=[
                    Cuboid3d(position=[13.54, -9.41, 0.24], label=0,
                        attributes={'track_id': 1,
                            'occluded': True, 'a': 'w', 'b': 5})
                ],
                attributes={'frame': 0}
            )
        ], categories={AnnotationType.label: target_label_cat})

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(KittiRawConverter.convert, allow_attrs=True),
                test_dir, target_dataset=target_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_discard_attributes(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='0000000000',
                annotations=[
                    Cuboid3d(position=[13.54, -9.41, 0.24], label=0,
                        attributes={'track_id': 1, 'a': 'w', 'b': 5})
                ],
                attributes={'frame': 0}
            )
        ], categories=['cat'])

        target_label_cat = LabelCategories(attributes={'occluded'})
        target_label_cat.add('cat')
        target_dataset = Dataset.from_iterable([
            DatasetItem(id='0000000000',
                annotations=[
                    Cuboid3d(position=[13.54, -9.41, 0.24], label=0,
                        attributes={'track_id': 1, 'occluded': False})
                ],
                attributes={'frame': 0}
            )
        ], categories={AnnotationType.label: target_label_cat})

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                KittiRawConverter.convert,
                test_dir, target_dataset=target_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_without_annotations(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='0000000000', attributes={'frame': 0})
        ], categories=[])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                KittiRawConverter.convert, test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_arbitrary_paths(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a/d',
                annotations=[
                    Cuboid3d(position=[1, 2, 3], label=0,
                        attributes={'track_id': 1})
                ],
                point_cloud=self.pcd1, related_images=[self.image1],
                attributes={'frame': 3}
            ),
        ], categories=['cat'])

        with TestDir() as test_dir:
            target_label_cat = LabelCategories(attributes={'occluded'})
            target_label_cat.add('cat')
            target_dataset = Dataset.from_iterable([
                DatasetItem(id='a/d',
                    annotations=[
                        Cuboid3d(position=[1, 2, 3], label=0,
                            attributes={'track_id': 1, 'occluded': False})
                    ],
                    point_cloud=osp.join(test_dir,
                        'velodyne_points', 'data', 'a', 'd.pcd'),
                    related_images=[
                        osp.join(test_dir, 'image_00', 'data', 'a', 'd.png'),
                    ],
                    attributes={'frame': 3}
                ),
            ], categories={AnnotationType.label: target_label_cat})

            self._test_save_and_load(source_dataset,
                partial(KittiRawConverter.convert, save_images=True),
                test_dir, target_dataset=target_dataset,
                require_point_cloud=True)
            self.assertTrue(osp.isfile(osp.join(
                test_dir, 'image_00', 'data', 'a', 'd.png')))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_multiple_related_images(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a/d',
                annotations=[
                    Cuboid3d(position=[1, 2, 3], label=0,
                        attributes={'track_id': 1})
                ],
                point_cloud=self.pcd1,
                related_images=[self.image1, self.image2, self.image3],
                attributes={'frame': 3}
            ),
        ], categories=['cat'])

        with TestDir() as test_dir:
            target_label_cat = LabelCategories(attributes={'occluded'})
            target_label_cat.add('cat')
            target_dataset = Dataset.from_iterable([
                DatasetItem(id='a/d',
                    annotations=[
                        Cuboid3d(position=[1, 2, 3], label=0,
                            attributes={'track_id': 1, 'occluded': False})
                    ],
                    point_cloud=osp.join(test_dir,
                        'velodyne_points', 'data', 'a', 'd.pcd'),
                    related_images=[
                        osp.join(test_dir, 'image_00', 'data', 'a', 'd.png'),
                        osp.join(test_dir, 'image_01', 'data', 'a', 'd.png'),
                        osp.join(test_dir, 'image_02', 'data', 'a', 'd.png'),
                    ],
                    attributes={'frame': 3}
                ),
            ], categories={AnnotationType.label: target_label_cat})

            self._test_save_and_load(source_dataset,
                partial(KittiRawConverter.convert, save_images=True),
                test_dir, target_dataset=target_dataset,
                require_point_cloud=True)
            self.assertTrue(osp.isfile(osp.join(
                test_dir, 'image_00', 'data', 'a', 'd.png')))
            self.assertTrue(osp.isfile(osp.join(
                test_dir, 'image_01', 'data', 'a', 'd.png')))
            self.assertTrue(osp.isfile(osp.join(
                test_dir, 'image_02', 'data', 'a', 'd.png')))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self):
        with TestDir() as path:
            dataset = Dataset.from_iterable([
                DatasetItem(id='frame1',
                    annotations=[
                        Cuboid3d(position=[3.5, 9.8, 0.3], label=0,
                            attributes={'track_id': 1})
                    ],
                    point_cloud=self.pcd1, related_images=[self.image1],
                    attributes={'frame': 0}
                )
            ], categories=['car', 'bus'])
            dataset.export(path, 'kitti_raw', save_images=True)

            dataset.put(DatasetItem('frame2',
                annotations=[
                    Cuboid3d(position=[1, 2, 0], label=1,
                        attributes={'track_id': 1})
                ],
                point_cloud=self.pcd2, related_images=[self.image2],
                attributes={'frame': 1}
            ))
            dataset.remove('frame1')
            dataset.save(save_images=True)

            self.assertEqual({'frame2.png'}, set(os.listdir(
                osp.join(path, 'image_00', 'data'))))
            self.assertEqual({'frame2.pcd'}, set(os.listdir(
                osp.join(path, 'velodyne_points', 'data'))))
