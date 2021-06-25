from functools import partial
import os
import os.path as osp

from unittest import TestCase
from datumaro.components.project import Dataset
from datumaro.components.extractor import (DatasetItem,
                                           AnnotationType, Cuboid3D,
                                           LabelCategories,
                                           )
from datumaro.plugins.velodynepoints_format.extractor import VelodynePointsImporter
from datumaro.plugins.velodynepoints_format.converter import VelodynePointsConverter
from datumaro.util.test_utils import (TestDir, compare_datasets,
                                      test_save_and_load)
from tests.requirements import mark_requirement, Requirements

DUMMY_PCD_DATASET_DIR = osp.join(osp.dirname(
    __file__), 'assets', 'velodynepoints_dataset')


class VelodynePointsImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_image(self):
        self.assertTrue(VelodynePointsImporter.detect(DUMMY_PCD_DATASET_DIR))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_pcd(self):
        pcd1 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR,
                           r"velodyne_points/data/0000000000.pcd"))
        pcd2 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR,
                           r"velodyne_points/data/0000000001.pcd"))
        pcd3 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR,
                           r"velodyne_points/data/0000000002.pcd"))

        image1 = osp.abspath(
            osp.join(DUMMY_PCD_DATASET_DIR, r"IMAGE_00/data/0000000000.png"))
        image2 = osp.abspath(
            osp.join(DUMMY_PCD_DATASET_DIR, r"IMAGE_00/data/0000000001.png"))
        image3 = osp.abspath(
            osp.join(DUMMY_PCD_DATASET_DIR, r"IMAGE_00/data/0000000003.png"))

        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000', annotations=[Cuboid3D(id=0, attributes={'occluded': 0}, group=0, points=[-3.62, 7.95, -1.03, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], label=0, z_order=0),
                                                        Cuboid3D(id=0, attributes={'occluded': 0}, group=0, points=[23.01, 8.34, -0.76, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], label=1, z_order=0)],
                        subset='tracklets', path=[], image=None, pcd=pcd1, related_images=[{"name": "0000000000.png", "save_path": "IMAGE_00",
                                                                                            "path": image1},
                                                                                           ], attributes={'frame': 0}),
            DatasetItem(id='frame_000001', annotations=[Cuboid3D(id=0, attributes={'occluded': 0}, group=0, points=[0.39, 7.28, -0.89, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], label=1, z_order=0)],
                        subset='tracklets', path=[], image=None, pcd=pcd2, related_images=[{"name": "0000000001.png", "save_path": "IMAGE_00",
                                                                                            "path": image2},
                                                                                           ], attributes={'frame': 1}),
            DatasetItem(id='frame_000002', annotations=[Cuboid3D(id=0, attributes={'occluded': 0}, group=0,
                                                                 points=[13.54, -9.41,
                                                                         0.24, 0.0, 0.0, 0.0, 1.0, 1.0,
                                                                         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], label=0,
                                                                 z_order=0)], subset='tracklets', path=[], image=None,
                        pcd=pcd3, related_images=[{"name": "0000000002.png", "save_path": "IMAGE_00",
                                                   "path": image3},
                                                  ], attributes={'frame': 2})

        ], categories={
            AnnotationType.label: LabelCategories.from_iterable([
                ['car'],
                ['bus'],
            ])
        })

        parsed_dataset = Dataset.import_from(
            DUMMY_PCD_DATASET_DIR, 'velodyne_points')

        compare_datasets(self, expected_dataset, parsed_dataset, ignored_attrs=["occluded"])


class VelodynePointsConverterTest(TestCase):
    pcd1 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR,
                       r"velodyne_points/data/0000000000.pcd"))
    pcd2 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR,
                       r"velodyne_points/data/0000000001.pcd"))
    pcd3 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR,
                       r"velodyne_points/data/0000000002.pcd"))

    image1 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR,
                         r"IMAGE_00/data/0000000000.png"))
    image2 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR,
                         r"IMAGE_00/data/0000000001.png"))
    image3 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR,
                         r"IMAGE_00/data/0000000003.png"))

    dimension = {"dimension": "3d"}

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def _test_save_and_load(self, source_dataset, converter, test_dir,
                            target_dataset=None, importer_args=None, **kwargs):
        return test_save_and_load(self, source_dataset, converter, test_dir,
                                  importer='velodyne_points',
                                  target_dataset=target_dataset, importer_args=importer_args, **kwargs)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        src_label_cat = LabelCategories(attributes={'occluded'})
        src_label_cat.add('car')

        source_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                        annotations=[Cuboid3D(id=0,
                                              attributes={'occluded': 0},
                                              group=0,
                                              points=[13.54, -9.41, 0.24, 0.0,
                                                      0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0, z_order=0)],
                        subset='tracklets',
                        path=[],
                        image=None,
                        pcd=self.pcd1,
                        related_images=[{"name": "0000000000.png", "save_path": "IMAGE_00",
                                         "path": self.image1},
                                        ],
                        attributes={'frame': 0}
                        ),
        ], categories={AnnotationType.label: src_label_cat})

        target_label_cat = LabelCategories(attributes={'occluded'})
        target_label_cat.add("car")

        target_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                        annotations=[Cuboid3D(id=0,
                                              attributes={'occluded': 0},
                                              group=0,
                                              points=[13.54, -9.41, 0.24, 0.0,
                                                      0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0, z_order=0)],
                        subset='tracklets',
                        path=[],
                        image=None,
                        pcd=self.pcd1,
                        related_images=[{"name": "0000000000.png", "save_path": "IMAGE_00",
                                         "path": self.image1},
                                        ],
                        attributes={'frame': 0}
                        ),
        ], categories={AnnotationType.label: target_label_cat})

        with TestDir() as test_dir:

            self._test_save_and_load(source_dataset,
                                     partial(VelodynePointsConverter.convert,
                                             save_images=True), test_dir,
                                     target_dataset=target_dataset, ignored_attrs=["occluded"], **self.dimension)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_preserve_frame_ids(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000040',
                        annotations=[Cuboid3D(id=0,
                                              attributes={'occluded': 0},
                                              group=0,
                                              points=[13.54, -9.41, 0.24, 0.0,
                                                      0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0, z_order=0)],
                        subset='tracklets',
                        path=[],
                        image=None,
                        pcd=self.pcd1,
                        related_images=[{"name": "0000000000.png", "save_path": "IMAGE_00",
                                         "path": self.image1},
                                        ],
                        attributes={'frame': 40}
                        )
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(['car'])
        })

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                                     VelodynePointsConverter.convert, test_dir, ignored_attrs=["occluded"], **self.dimension)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_reindex(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                        annotations=[Cuboid3D(id=0,
                                              attributes={'occluded': 0},
                                              group=0,
                                              points=[13.54, -9.41, 0.24, 0.0,
                                                      0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0, z_order=0)],
                        subset='tracklets',
                        path=[],
                        image=None,
                        pcd=self.pcd1,
                        related_images=[{"name": "0000000000.png", "save_path": "IMAGE_00",
                                         "path": self.image1},
                                        ],
                        attributes={'frame': 20}
                        )
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(['car'])
        })

        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                        annotations=[Cuboid3D(id=0,
                                              attributes={'occluded': 0},
                                              group=0,
                                              points=[13.54, -9.41, 0.24, 0.0,
                                                      0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0, z_order=0)],
                        subset='tracklets',
                        path=[],
                        image=None,
                        pcd=self.pcd1,
                        related_images=[{"name": "0000000002.png", "save_path": "IMAGE_00",
                                         "path": self.image1},
                                        ],
                        attributes={'frame': 0}
                        )
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(['car'])
        })

        with TestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                                     partial(VelodynePointsConverter.convert,
                                             reindex=True), test_dir,
                                     target_dataset=expected_dataset, ignored_attrs=["occluded"], **self.dimension)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self):

        with TestDir() as path:
            # generate initial dataset

            dataset = Dataset.from_iterable([
                DatasetItem('0000000000.pcd', subset="tracklets"),
                DatasetItem('0000000001.pcd', subset="tracklets"),
                DatasetItem('0000000002.pcd',
                            subset='tracklets',
                            pcd=self.pcd3,
                            related_images=[{"name": "0000000002.png", "save_path": "IMAGE_00",
                                             "path": self.image2}],
                            )
            ])

            dataset.export(path, 'velodyne_points', save_images=True)

            os.unlink(osp.join(path, 'tracklets.xml'))
            self.assertTrue(osp.isfile(osp.abspath(
                osp.join(path, 'IMAGE_00/data', '0000000002.png'))))
            self.assertFalse(osp.isfile(osp.abspath(
                osp.join(path, 'IMAGE_00/data', '0000000001.png'))))

            dataset.put(DatasetItem(2,
                                    subset='tracklets',
                                    pcd=self.pcd2,
                                    related_images=[{"name": "0000000001.png", "save_path": "IMAGE_00",
                                                     "path": self.image2}],
                                    ))

            dataset.remove("0000000002.pcd", "tracklets")
            related_image_path = {'related_paths': [
                "IMAGE_00/data"], "image_names": ["0000000002.png"]}
            dataset.save(save_images=True, **related_image_path)

            self.assertFalse(osp.isfile(osp.abspath(
                osp.join(path, 'velodyne_points/data', '0000000000.pcd'))))
            self.assertFalse(osp.isfile(osp.abspath(
                osp.join(path, 'velodyne_points/data', '0000000002.pcd'))))
            self.assertTrue(osp.isfile(osp.abspath(
                osp.join(path, 'velodyne_points/data', '0000000001.pcd'))))

            self.assertFalse(osp.isfile(osp.abspath(
                osp.join(path, 'IMAGE_00/data', '0000000000.png'))))
            self.assertFalse(osp.isfile(osp.abspath(
                osp.join(path, 'IMAGE_00/data', '0000000002.png'))))
            self.assertTrue(osp.isfile(osp.abspath(
                osp.join(path, 'IMAGE_00/data', '0000000001.png'))))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_without_related_images(self):
        src_label_cat = LabelCategories(attributes={'occluded'})
        src_label_cat.add('car')
        src_label_cat.items[0].attributes.update(['a1', 'a2'])

        source_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                        annotations=[Cuboid3D(id=0,
                                              attributes={"a1": "true", "a2": 0, 'occluded': 0},
                                              group=0,
                                              points=[13.54, -9.41, 0.24, 0.0,
                                                      0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0, z_order=0)],
                        subset='tracklets',
                        path=[],
                        image=None,
                        pcd=self.pcd1,
                        related_images=[],
                        attributes={'frame': 0}
                        ),
        ], categories={AnnotationType.label: src_label_cat})

        target_label_cat = LabelCategories(attributes={'occluded'})
        target_label_cat.add("car")
        target_label_cat.items[0].attributes.update(['a1', 'a2'])

        target_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                        annotations=[Cuboid3D(id=0,
                                              attributes={"a1": "true", "a2": 0, 'occluded': 0},
                                              group=0,
                                              points=[13.54, -9.41, 0.24, 0.0,
                                                      0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                              label=0, z_order=0)],
                        subset='tracklets',
                        path=[],
                        image=None,
                        pcd=self.pcd1,
                        related_images=[],
                        attributes={'frame': 0}
                        ),
        ], categories={AnnotationType.label: target_label_cat})

        with TestDir() as test_dir:

            self._test_save_and_load(source_dataset,
                                     partial(VelodynePointsConverter.convert,
                                             save_images=True), test_dir,
                                     target_dataset=target_dataset, ignored_attrs=["occluded"], **self.dimension)
