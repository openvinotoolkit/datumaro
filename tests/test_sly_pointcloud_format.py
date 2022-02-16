from functools import partial
from unittest import TestCase
import os
import os.path as osp

from datumaro.components.annotation import (
    AnnotationType, Cuboid3d, LabelCategories,
)
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image, PointCloud
from datumaro.components.project import Dataset
from datumaro.plugins.sly_pointcloud_format.converter import (
    SuperviselyPointCloudConverter,
)
from datumaro.plugins.sly_pointcloud_format.extractor import (
    SuperviselyPointCloudImporter,
)
from datumaro.util.test_utils import (
    Dimensions, TestDir, check_save_and_load, compare_datasets_3d,
)

from .requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(osp.dirname(
    __file__), 'assets', 'sly_pointcloud_dataset')


class SuperviselyPointcloudImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertIn(SuperviselyPointCloudImporter.NAME, detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load(self):
        pcd1 = osp.join(DUMMY_DATASET_DIR, 'ds0', 'pointcloud', 'frame1.pcd')
        pcd2 = osp.join(DUMMY_DATASET_DIR, 'ds0', 'pointcloud', 'frame2.pcd')

        image1 = Image(path=osp.join(DUMMY_DATASET_DIR,
            'ds0', 'related_images', 'frame1_pcd', 'img2.png'))
        image2 = Image(path=osp.join(DUMMY_DATASET_DIR,
            'ds0', 'related_images', 'frame2_pcd', 'img1.png'))

        label_cat = LabelCategories(attributes={'tag1', 'tag3'})
        label_cat.add('car')
        label_cat.add('bus')

        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='frame1',
                annotations=[
                    Cuboid3d(id=755220128, label=0,
                        position=[0.47, 0.23, 0.79], scale=[0.01, 0.01, 0.01],
                        attributes={'track_id': 231825,
                            'tag1': 'fd', 'tag3': '4s'}),

                    Cuboid3d(id=755337225, label=0,
                        position=[0.36, 0.64, 0.93], scale=[0.01, 0.01, 0.01],
                        attributes={'track_id': 231831,
                            'tag1': 'v12', 'tag3': ''}),
                ],
                media=PointCloud(pcd1, extra_images=[image1]),
                attributes={'frame': 0, 'description': '',
                    'tag1': '25dsd', 'tag2': 65}
            ),

            DatasetItem(id='frame2',
                annotations=[
                    Cuboid3d(id=216, label=1,
                        position=[0.59, 14.41, -0.61],
                        attributes={'track_id': 36, 'tag1': '', 'tag3': ''})
                ],
                media=PointCloud(pcd2, extra_images=[image2]),
                attributes={'frame': 1, 'description': ''}
            ),
        ], categories={AnnotationType.label: label_cat})

        parsed_dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'sly_pointcloud')

        compare_datasets_3d(self, expected_dataset, parsed_dataset,
            require_point_cloud=True)


class PointCloudConverterTest(TestCase):
    pcd1 = osp.join(DUMMY_DATASET_DIR, 'ds0', 'pointcloud', 'frame1.pcd')
    pcd2 = osp.join(DUMMY_DATASET_DIR, 'ds0', 'pointcloud', 'frame2.pcd')

    image1 = Image(path=osp.join(DUMMY_DATASET_DIR,
        'ds0', 'related_images', 'frame1_pcd', 'img2.png'))
    image2 = Image(path=osp.join(DUMMY_DATASET_DIR,
        'ds0', 'related_images', 'frame2_pcd', 'img1.png'))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None, **kwargs):
        kwargs.setdefault('dimension', Dimensions.dim_3d)
        return check_save_and_load(self, source_dataset, converter, test_dir,
            importer='sly_pointcloud', target_dataset=target_dataset,
            importer_args=importer_args, **kwargs)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        src_label_cat = LabelCategories(attributes={'occluded'})
        src_label_cat.add('car', attributes=['x'])
        src_label_cat.add('bus')

        source_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_1',
                annotations=[
                    Cuboid3d(id=206, label=0,
                        position=[320.86, 979.18, 1.04],
                        attributes={'occluded': False, 'track_id': 1, 'x': 1}),

                    Cuboid3d(id=207, label=1,
                        position=[318.19, 974.65, 1.29],
                        attributes={'occluded': True, 'track_id': 2}),
                ],
                media=PointCloud(self.pcd1),
                attributes={'frame': 0, 'description': 'zzz'}
            ),

            DatasetItem(id='frm2',
                annotations=[
                    Cuboid3d(id=208, label=1,
                        position=[23.04, 8.75, -0.78],
                        attributes={'occluded': False, 'track_id': 2})
                ],
                media=PointCloud(self.pcd2, extra_images=[self.image2]),
                attributes={'frame': 1}
            ),
        ], categories={ AnnotationType.label: src_label_cat })

        with TestDir() as test_dir:
            target_label_cat = LabelCategories(attributes={'occluded'})
            target_label_cat.add('car', attributes=['x'])
            target_label_cat.add('bus')

            target_dataset = Dataset.from_iterable([
                DatasetItem(id='frame_1',
                    annotations=[
                        Cuboid3d(id=206, label=0,
                            position=[320.86, 979.18, 1.04],
                            attributes={'occluded': False,
                                'track_id': 1, 'x': 1}),

                        Cuboid3d(id=207, label=1,
                            position=[318.19, 974.65, 1.29],
                            attributes={'occluded': True, 'track_id': 2}),
                    ],
                    media=PointCloud(osp.join(test_dir,
                        'ds0', 'pointcloud', 'frame_1.pcd')),
                    attributes={'frame': 0, 'description': 'zzz'}),

                DatasetItem(id='frm2',
                    annotations=[
                        Cuboid3d(id=208, label=1,
                            position=[23.04, 8.75, -0.78],
                            attributes={'occluded': False, 'track_id': 2}),
                    ],
                    media=PointCloud(osp.join(test_dir,
                            'ds0', 'pointcloud', 'frm2.pcd'),
                        extra_images=[Image(path=osp.join(test_dir,
                            'ds0', 'related_images', 'frm2_pcd', 'img1.png'))
                    ]),
                    attributes={'frame': 1, 'description': ''})
            ], categories={ AnnotationType.label: target_label_cat })

            self._test_save_and_load(source_dataset,
                partial(SuperviselyPointCloudConverter.convert, save_media=True),
                test_dir, target_dataset=target_dataset,
                require_point_cloud=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_preserve_frame_ids(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='abc', attributes={'frame': 20}),
        ], categories=[])

        with TestDir() as test_dir:
            self._test_save_and_load(dataset,
                SuperviselyPointCloudConverter.convert, test_dir,
                ignored_attrs={'description'})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='somename', attributes={'frame': 1234})
        ])

        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='somename', attributes={'frame': 1})
        ], categories=[])

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(SuperviselyPointCloudConverter.convert, reindex=True),
                test_dir, target_dataset=expected_dataset,
                ignored_attrs={'description'})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_keep_undeclared_attributes(self):
        src_label_cat = LabelCategories(attributes={'occluded'})
        src_label_cat.add('label1', attributes={'a'})

        source_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                annotations=[
                    Cuboid3d(id=206, label=0, position=[320.86, 979.18, 1.04],
                        attributes={'track_id': 1, 'occluded': False,
                            'a': 5, 'undeclared': 'y'}),
                ],
                attributes={'frame': 0}),
        ], categories={AnnotationType.label: src_label_cat})

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(SuperviselyPointCloudConverter.convert, save_media=True,
                    allow_undeclared_attrs=True),
                test_dir, ignored_attrs=['description'])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_drop_undeclared_attributes(self):
        src_label_cat = LabelCategories(attributes={'occluded'})
        src_label_cat.add('label1', attributes={'a'})

        source_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                annotations=[
                    Cuboid3d(id=206, label=0, position=[320.86, 979.18, 1.04],
                        attributes={'occluded': False,
                            'a': 5, 'undeclared': 'y'}),
                ],
                attributes={'frame': 0}),
        ], categories={AnnotationType.label: src_label_cat})

        target_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                annotations=[
                    Cuboid3d(id=206, label=0, position=[320.86, 979.18, 1.04],
                        attributes={'track_id': 206, 'occluded': False, 'a': 5}),
                ],
                attributes={'frame': 0}),
        ], categories={AnnotationType.label: src_label_cat})

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(SuperviselyPointCloudConverter.convert, save_media=True),
                test_dir, target_dataset=target_dataset,
                ignored_attrs=['description'])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_have_arbitrary_item_ids(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a/b/c235',
                media=PointCloud(self.pcd1, extra_images=[self.image1]),
                attributes={'frame': 20}),
        ])

        with TestDir() as test_dir:
            pcd_path = osp.join(test_dir, 'ds0', 'pointcloud',
                'a', 'b', 'c235.pcd')
            img_path = osp.join(test_dir, 'ds0', 'related_images',
                'a', 'b', 'c235_pcd', 'img2.png')
            target_dataset = Dataset.from_iterable([
                DatasetItem(id='a/b/c235',
                    media=PointCloud(pcd_path, extra_images=[Image(path=img_path)]),
                    attributes={'frame': 20}),
            ], categories=[])

            self._test_save_and_load(source_dataset,
                partial(SuperviselyPointCloudConverter.convert, save_media=True),
                test_dir, target_dataset=target_dataset,
                ignored_attrs={'description'}, require_point_cloud=True)

            self.assertTrue(osp.isfile(
                osp.join(test_dir, 'ds0', 'ann', 'a', 'b', 'c235.pcd.json')))
            self.assertTrue(osp.isfile(pcd_path))
            self.assertTrue({'img2.png', 'img2.png.json'},
                set(os.listdir(osp.join(test_dir, 'ds0', 'related_images',
                    'a', 'b', 'c235_pcd'))))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self):
        with TestDir() as path:
            dataset = Dataset.from_iterable([
                DatasetItem(id='frame1',
                    annotations=[
                        Cuboid3d(id=215,
                            position=[320.59, 979.48, 1.03], label=0)
                    ],
                    media=PointCloud(self.pcd1, extra_images=[self.image1]),
                    attributes={'frame': 0})
            ], categories=['car', 'bus'])
            dataset.export(path, 'sly_pointcloud', save_media=True)

            dataset.put(DatasetItem(id='frame2',
                annotations=[
                    Cuboid3d(id=216, position=[0.59, 14.41, -0.61], label=1)
                ],
                media=PointCloud(self.pcd2, extra_images=[self.image2]),
                attributes={'frame': 1})
            )

            dataset.remove('frame1')
            dataset.save(save_media=True)

            self.assertEqual({'frame2.pcd.json'},
                set(os.listdir(osp.join(path, 'ds0', 'ann'))))
            self.assertEqual({'frame2.pcd'},
                set(os.listdir(osp.join(path, 'ds0', 'pointcloud'))))
            self.assertTrue(osp.isfile(osp.join(path,
                'ds0', 'related_images', 'frame2_pcd', 'img1.png')))
            self.assertFalse(osp.isfile(osp.join(path,
                'ds0', 'related_images', 'frame1_pcd', 'img2.png')))
