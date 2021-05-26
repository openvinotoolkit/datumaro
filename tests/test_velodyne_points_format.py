from functools import partial
import os
import os.path as osp

from unittest import TestCase
from datumaro.components.project import Dataset
from datumaro.components.extractor import (DatasetItem,
    AnnotationType, Cuboid,
    LabelCategories,
)
from datumaro.plugins.velodynepoints_format.extractor import VelodynePointsImporter
from datumaro.plugins.velodynepoints_format.converter import VelodynePointsConverter
from datumaro.util.test_utils import (TestDir, compare_datasets,
    test_save_and_load)


DUMMY_PCD_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'velodynepoints_dataset')

class VelodynePointsImporterTest(TestCase):
    def test_can_detect_image(self):
        self.assertTrue(VelodynePointsImporter.detect(DUMMY_PCD_DATASET_DIR))

    def test_can_load_pcd(self):
        pcd1 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR, r"velodyne_points/data/0000000000.pcd"))
        pcd2 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR, r"velodyne_points/data/0000000001.pcd"))
        pcd3 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR, r"velodyne_points/data/0000000002.pcd"))

        image1 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR, r"IMAGE_00/data/0000000000.png"))
        image2 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR, r"IMAGE_00/data/0000000001.png"))
        image3 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR, r"IMAGE_00/data/0000000003.png"))

        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000', annotations=[Cuboid(id=0, attributes={'occluded': 0}, group=0, points=[-3.6271575019618414, 7.954996769991751, -1.0343550199580118, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], label=0, z_order=0),
                                                        Cuboid(id=0, attributes={'occluded': 0}, group=0, points=[23.0169506240644, 8.343682404650442, -0.7699940133040206, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], label=1, z_order=0)],
                        subset='tracklets', path=[], image=None, pcd=pcd1, related_images=[{"name": "0000000000.png", "save_path": "IMAGE_00",
                                         "path": image1},
           ], attributes={'frame': 0}),
            DatasetItem(id='frame_000001', annotations=[Cuboid(id=0, attributes={'occluded': 0}, group=0, points=[0.39720775117329943, 7.286424562074529, -0.8997166481217596, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], label=1, z_order=0)],
                        subset='tracklets', path=[], image=None, pcd=pcd2, related_images=[{"name": "0000000001.png", "save_path": "IMAGE_00",
                                         "path": image2},
           ], attributes={'frame': 1}),
            DatasetItem(id='frame_000002', annotations=[Cuboid(id=0, attributes={'occluded': 0}, group=0,
                                                               points=[13.54034048060633, -9.410120134481405,
                                                                       0.24972983624747647, 0.0, 0.0, 0.0, 1.0, 1.0,
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

        parsed_dataset = Dataset.import_from(DUMMY_PCD_DATASET_DIR, 'velodyne_points')

        compare_datasets(self, expected_dataset, parsed_dataset)


class VelodynePointsConverterTest(TestCase):
    pcd1 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR, r"velodyne_points/data/0000000000.pcd"))
    pcd2 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR, r"velodyne_points/data/0000000001.pcd"))
    pcd3 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR, r"velodyne_points/data/0000000002.pcd"))

    image1 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR, r"IMAGE_00/data/0000000000.png"))
    image2 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR, r"IMAGE_00/data/0000000001.png"))
    image3 = osp.abspath(osp.join(DUMMY_PCD_DATASET_DIR, r"IMAGE_00/data/0000000003.png"))

    dimension = {"dimension": "3d"}

    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None, **kwargs):
        return test_save_and_load(self, source_dataset, converter, test_dir,
            importer='velodyne_points',
            target_dataset=target_dataset, importer_args=importer_args, **kwargs)

    def test_can_save_and_load(self):
        src_label_cat = LabelCategories(attributes={'occluded'})
        src_label_cat.add('car')

        source_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                        annotations=[Cuboid(id=0,
                                            attributes={'occluded': 0},
                                            group=0,
                                            points=[13.54034048060633, -9.410120134481405, 0.24972983624747647, 0.0,
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
        ], categories={ AnnotationType.label: src_label_cat })

        target_label_cat = LabelCategories(attributes={'occluded'})
        target_label_cat.add("car")

        target_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                        annotations=[Cuboid(id=0,
                                            attributes={'occluded': 0},
                                            group=0,
                                            points=[13.54034048060633, -9.410120134481405, 0.24972983624747647, 0.0,
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
                partial(VelodynePointsConverter.convert, save_images=True), test_dir,
                target_dataset=target_dataset, **self.dimension)

    def test_preserve_frame_ids(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000040',
                        annotations=[Cuboid(id=0,
                                            attributes={'occluded': 0},
                                            group=0,
                                            points=[13.54034048060633, -9.410120134481405, 0.24972983624747647, 0.0,
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
                VelodynePointsConverter.convert, test_dir, **self.dimension)

    def test_reindex(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='frame_000000',
                        annotations=[Cuboid(id=0,
                                            attributes={'occluded': 0},
                                            group=0,
                                            points=[13.54034048060633, -9.410120134481405, 0.24972983624747647, 0.0,
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
                        annotations=[Cuboid(id=0,
                                            attributes={'occluded': 0},
                                            group=0,
                                            points=[13.54034048060633, -9.410120134481405, 0.24972983624747647, 0.0,
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
                partial(VelodynePointsConverter.convert, reindex=True), test_dir,
                target_dataset=expected_dataset, **self.dimension)

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

            os.unlink(osp.join(path,'tracklets.xml'))
            self.assertTrue(osp.isfile(osp.abspath(osp.join(path,'IMAGE_00/data', '0000000002.png'))))
            self.assertFalse(osp.isfile(osp.abspath(osp.join(path,'IMAGE_00/data', '0000000001.png'))))

            dataset.put(DatasetItem(2,
                            subset='tracklets',
                            pcd=self.pcd2,
                            related_images=[{"name": "0000000001.png", "save_path": "IMAGE_00",
                                         "path": self.image2}],
                            ))

            dataset.remove("0000000002.pcd", "tracklets")
            related_image_path = {'related_paths': ["IMAGE_00/data"], "image_names": ["0000000002.png"]}
            dataset.save(save_images=True, **related_image_path)

            self.assertFalse(osp.isfile(osp.abspath(osp.join(path,'velodyne_points/data', '0000000000.pcd'))))
            self.assertFalse(osp.isfile(osp.abspath(osp.join(path, 'velodyne_points/data', '0000000002.pcd'))))
            self.assertTrue(osp.isfile(osp.abspath(osp.join(path, 'velodyne_points/data', '0000000001.pcd'))))

            self.assertFalse(osp.isfile(osp.abspath(osp.join(path, 'IMAGE_00/data', '0000000000.png'))))
            self.assertFalse(osp.isfile(osp.abspath(osp.join(path, 'IMAGE_00/data', '0000000002.png'))))
            self.assertTrue(osp.isfile(osp.abspath(osp.join(path, 'IMAGE_00/data', '0000000001.png'))))

