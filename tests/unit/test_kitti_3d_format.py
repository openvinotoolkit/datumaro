import os.path as osp
from unittest import TestCase

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image, PointCloud
from datumaro.components.project import Dataset
from datumaro.plugins.data_formats.kitti_3d.importer import Kitti3dImporter

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets_3d

DUMMY_DATASET_DIR = get_test_asset_path("kitti_dataset", "kitti_3d", "training")


class Kitti3DImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([Kitti3dImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load(self):
        """
        <b>Description:</b>
        Ensure that the dataset can be loaded correctly from the KITTI3D format.

        <b>Expected results:</b>
        The loaded dataset should have the same number of data items as the expected dataset.
        The data items in the loaded dataset should have the same attributes and values as the expected data items.
        The point clouds and images associated with the data items should be loaded correctly.

        <b>Steps:</b>
        1. Prepare an expected dataset with known data items, point clouds, images, and attributes.
        2. Load the dataset from the KITTI3D format.
        3. Compare the loaded dataset with the expected dataset.
        """
        pcd1 = osp.join(DUMMY_DATASET_DIR, "velodyne", "000001.bin")

        image1 = Image.from_file(path=osp.join(DUMMY_DATASET_DIR, "image_2", "000001.png"))

        expected_label_cat = LabelCategories(
            attributes={"occluded", "truncated", "alpha", "dimensions", "location", "rotation_y"}
        )
        expected_label_cat.add("Truck")
        expected_label_cat.add("Car")
        expected_label_cat.add("DontCare")
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="000001",
                    annotations=[
                        Bbox(
                            600,  # x1
                            150,  # y1
                            30,  # x2-x1
                            40,  # y2-y1
                            label=0,
                            id=0,
                            attributes={
                                "truncated": 0.0,
                                "occluded": 0,
                                "alpha": -1.57,
                                "dimensions": [2.85, 2.63, 12.34],
                                "location": [0.47, 1.49, 69.44],
                                "rotation_y": -1.56,
                            },
                            z_order=0,
                        ),
                        Bbox(
                            650,  # x1
                            160,  # y1
                            50,  # x2-x1
                            40,  # y2-y1
                            label=1,
                            id=1,
                            attributes={
                                "truncated": 0.0,
                                "occluded": 3,
                                "alpha": -1.65,
                                "dimensions": [1.86, 0.6, 2.02],
                                "location": [4.59, 1.32, 45.84],
                                "rotation_y": -1.55,
                            },
                            z_order=0,
                        ),
                        Bbox(
                            500,  # x1
                            170,  # y1
                            90,  # x2-x1
                            20,  # y2-y1
                            label=2,
                            id=2,
                            attributes={
                                "truncated": -1.0,
                                "occluded": -1,
                                "alpha": -10.0,
                                "dimensions": [-1.0, -1.0, -1.0],
                                "location": [-1000.0, -1000.0, -1000.0],
                                "rotation_y": -10.0,
                            },
                        ),
                    ],
                    media=PointCloud.from_file(path=pcd1, extra_images=[image1]),
                    attributes={"calib_path": osp.join(DUMMY_DATASET_DIR, "calib", "000001.txt")},
                ),
            ],
            categories={AnnotationType.label: expected_label_cat},
            media_type=PointCloud,
        )

        parsed_dataset = Dataset.import_from(DUMMY_DATASET_DIR, "kitti3d")

        compare_datasets_3d(self, expected_dataset, parsed_dataset, require_point_cloud=True)
