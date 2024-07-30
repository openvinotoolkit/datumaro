# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import copy
import os
import os.path as osp
import pickle  # nosec - disable B403:import_pickle check
import random
import shutil

import numpy as np
import pytest
import yaml
from PIL import ExifTags
from PIL import Image as PILImage

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    LabelCategories,
    Points,
    PointsCategories,
    Polygon,
    Skeleton,
)
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.errors import (
    AnnotationImportError,
    DatasetExportError,
    DatasetImportError,
    DatasetNotFoundError,
    InvalidAnnotationError,
    ItemImportError,
    UndeclaredLabelError,
)
from datumaro.components.extractor import DatasetItem
from datumaro.components.format_detection import FormatDetectionContext, FormatRequirementsUnmet
from datumaro.components.media import Image
from datumaro.plugins.yolo_format.converter import (
    YoloConverter,
    YOLOv8Converter,
    YOLOv8OrientedBoxesConverter,
    YOLOv8PoseConverter,
    YOLOv8SegmentationConverter,
)
from datumaro.plugins.yolo_format.extractor import (
    YoloExtractor,
    YOLOv8Extractor,
    YOLOv8OrientedBoxesExtractor,
    YOLOv8PoseExtractor,
    YOLOv8SegmentationExtractor,
)
from datumaro.plugins.yolo_format.importer import (
    YoloImporter,
    YOLOv8Importer,
    YOLOv8OrientedBoxesImporter,
    YOLOv8PoseImporter,
    YOLOv8SegmentationImporter,
)
from datumaro.util.image import save_image
from datumaro.util.test_utils import compare_datasets, compare_datasets_strict

from ...requirements import Requirements, mark_requirement
from ...utils.assets import get_test_asset_path


@pytest.fixture(autouse=True)
def seed_random():
    random.seed(1234)


def randint(a, b):
    return random.randint(a, b)  # nosec B311 NOSONAR


class CompareDatasetMixin:
    @pytest.fixture(autouse=True)
    def setup(self, helper_tc):
        self.helper_tc = helper_tc

    def compare_datasets(self, expected, actual, **kwargs):
        compare_datasets(self.helper_tc, expected, actual, **kwargs)


class CompareDatasetsRotationMixin(CompareDatasetMixin):
    def compare_datasets(self, expected, actual, **kwargs):
        actual_copy = copy.deepcopy(actual)
        compare_datasets(self.helper_tc, expected, actual, ignored_attrs=["rotation"], **kwargs)
        for item_a, item_b in zip(expected, actual_copy):
            for ann_a, ann_b in zip(item_a.annotations, item_b.annotations):
                assert ("rotation" in ann_a.attributes) == ("rotation" in ann_b.attributes)
                assert (
                    abs(ann_a.attributes.get("rotation", 0) - ann_b.attributes.get("rotation", 0))
                    < 0.01
                )


class YoloConverterTest(CompareDatasetMixin):
    CONVERTER = YoloConverter
    IMPORTER = YoloImporter

    def _generate_random_bbox(self, n_of_labels=10, **kwargs):
        return Bbox(
            x=randint(0, 4),
            y=randint(0, 4),
            w=randint(1, 4),
            h=randint(1, 4),
            label=randint(0, n_of_labels - 1),
            attributes=kwargs,
        )

    def _generate_random_annotation(self, n_of_labels=10):
        return self._generate_random_bbox(n_of_labels=n_of_labels)

    @staticmethod
    def _make_image_path(test_dir: str, subset_name: str, image_id: str):
        return osp.join(test_dir, f"obj_{subset_name}_data", image_id)

    def _generate_random_dataset(self, recipes, n_of_labels=10):
        items = [
            DatasetItem(
                id=recipe.get("id", index + 1),
                subset=recipe.get("subset", "train"),
                media=recipe.get(
                    "media",
                    Image(data=np.ones((randint(8, 10), randint(8, 10), 3))),
                ),
                annotations=[
                    self._generate_random_annotation(n_of_labels=n_of_labels)
                    for _ in range(recipe.get("annotations", 1))
                ],
            )
            for index, recipe in enumerate(recipes)
        ]
        return Dataset.from_iterable(
            items,
            categories=["label_" + str(i) for i in range(n_of_labels)],
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {"annotations": 2},
                {"annotations": 3},
                {"annotations": 4},
            ]
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)

        self.compare_datasets(source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_image_info(self, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {
                    "annotations": 2,
                    "media": Image(path="1.jpg", size=(10, 15)),
                },
            ]
        )

        self.CONVERTER.convert(source_dataset, test_dir)

        save_image(
            self._make_image_path(test_dir, "train", "1.jpg"), np.ones((10, 15, 3))
        )  # put the image for dataset
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)

        self.compare_datasets(source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_dataset_with_exact_image_info(self, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {
                    "annotations": 2,
                    "media": Image(path="1.jpg", size=(10, 15)),
                },
            ]
        )

        self.CONVERTER.convert(source_dataset, test_dir)
        parsed_dataset = Dataset.import_from(
            test_dir, self.IMPORTER.NAME, image_info={"1": (10, 15)}
        )
        self.compare_datasets(source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {
                    "id": "кириллица с пробелом",
                    "annotations": 2,
                },
            ]
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        self.compare_datasets(source_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("save_media", [True, False])
    def test_relative_paths(self, save_media, test_dir):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="train", media=Image(data=np.ones((4, 2, 3)))),
                DatasetItem(id="subdir1/1", subset="train", media=Image(data=np.ones((2, 6, 3)))),
                DatasetItem(id="subdir2/1", subset="train", media=Image(data=np.ones((5, 4, 3)))),
            ],
            categories=[],
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=save_media)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        self.compare_datasets(source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self, test_dir):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "q/1", subset="train", media=Image(path="q/1.JPEG", data=np.zeros((4, 3, 3)))
                ),
                DatasetItem(
                    "a/b/c/2",
                    subset="valid",
                    media=Image(path="a/b/c/2.bmp", data=np.zeros((3, 4, 3))),
                ),
            ],
            categories=[],
        )

        self.CONVERTER.convert(dataset, test_dir, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        self.compare_datasets(dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self, test_dir):
        expected = Dataset.from_iterable(
            [
                DatasetItem(1, subset="train", media=Image(data=np.ones((2, 4, 3)))),
                DatasetItem(2, subset="train", media=Image(data=np.ones((3, 2, 3)))),
            ],
            categories=[],
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, subset="train", media=Image(data=np.ones((2, 4, 3)))),
                DatasetItem(2, subset="train", media=Image(path="2.jpg", size=(3, 2))),
                DatasetItem(3, subset="valid", media=Image(data=np.ones((2, 2, 3)))),
            ],
            categories=[],
        )
        dataset.export(test_dir, self.CONVERTER.NAME, save_media=True)

        dataset.put(DatasetItem(2, subset="train", media=Image(data=np.ones((3, 2, 3)))))
        dataset.remove(3, "valid")
        dataset.save(save_media=True)

        self._check_inplace_save_writes_only_updated_data(test_dir, expected)

    def _check_inplace_save_writes_only_updated_data(self, test_dir, expected):
        assert set(os.listdir(osp.join(test_dir, "obj_train_data"))) == {
            "1.txt",
            "2.txt",
            "1.jpg",
            "2.jpg",
        }
        assert set(os.listdir(osp.join(test_dir, "obj_valid_data"))) == set()
        self.compare_datasets(
            expected,
            Dataset.import_from(test_dir, "yolo"),
            require_media=True,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_meta_file(self, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {"annotations": 2},
                {"annotations": 3},
                {"annotations": 4},
            ]
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True, save_dataset_meta=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)

        assert osp.isfile(osp.join(test_dir, "dataset_meta.json"))
        self.compare_datasets(source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_565)
    def test_can_save_and_load_with_custom_subset_name(self, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {"annotations": 2, "subset": "anything", "id": 3},
            ]
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)

        self.compare_datasets(source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_565)
    @pytest.mark.parametrize("subset", ["backup", "classes"])
    def test_cant_save_with_reserved_subset_name(self, test_dir, subset):
        self._check_cant_save_with_reserved_subset_name(test_dir, subset)

    def _check_cant_save_with_reserved_subset_name(self, test_dir, subset):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset=subset,
                    media=Image(data=np.ones((8, 8, 3))),
                ),
            ],
            categories=["a"],
        )

        with pytest.raises(DatasetExportError, match=f"Can't export '{subset}' subset"):
            self.CONVERTER.convert(dataset, test_dir)

    @mark_requirement(Requirements.DATUM_609)
    def test_can_save_and_load_without_path_prefix(self, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {"subset": "valid", "id": 3},
            ],
            n_of_labels=2,
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True, add_path_prefix=False)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)

        with open(osp.join(test_dir, "obj.data"), "r") as f:
            lines = f.readlines()
            assert "valid = valid.txt\n" in lines

        with open(osp.join(test_dir, "valid.txt"), "r") as f:
            lines = f.readlines()
            assert "obj_valid_data/3.jpg\n" in lines

        self.compare_datasets(source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_export_rotated_bbox(self, test_dir):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        self._generate_random_bbox(n_of_labels=2),
                        self._generate_random_bbox(n_of_labels=2),
                    ],
                ),
            ],
            categories=["a", "b"],
        )
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=list(expected_dataset)[0].annotations
                    + [
                        self._generate_random_bbox(n_of_labels=2, rotation=30.0),
                    ],
                ),
            ],
            categories=["a", "b"],
        )
        source_dataset.export(test_dir, self.CONVERTER.NAME, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        self.compare_datasets(expected_dataset, parsed_dataset)


class YOLOv8ConverterTest(YoloConverterTest):
    CONVERTER = YOLOv8Converter
    IMPORTER = YOLOv8Importer

    @staticmethod
    def _make_image_path(test_dir: str, subset_name: str, image_id: str):
        return osp.join(test_dir, "images", subset_name, image_id)

    @mark_requirement(Requirements.DATUM_565)
    @pytest.mark.parametrize("subset", ["backup", "classes", "path", "names"])
    def test_cant_save_with_reserved_subset_name(self, test_dir, subset):
        self._check_cant_save_with_reserved_subset_name(test_dir, subset)

    @mark_requirement(Requirements.DATUM_609)
    def test_can_save_and_load_with_custom_config_file(self, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {"subset": "valid", "id": 3},
            ],
            n_of_labels=2,
        )
        filename = "custom_config_name.yaml"
        self.CONVERTER.convert(
            source_dataset, test_dir, save_media=True, add_path_prefix=False, config_file=filename
        )
        assert not osp.exists(osp.join(test_dir, "data.yaml"))
        assert osp.isfile(osp.join(test_dir, filename))
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME, config_file=filename)
        self.compare_datasets(source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_609)
    def test_can_save_and_load_without_path_prefix(self, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {"subset": "valid", "id": 3},
            ],
            n_of_labels=2,
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True, add_path_prefix=False)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)

        with open(osp.join(test_dir, "data.yaml"), "r") as f:
            config = yaml.safe_load(f)
            assert config.get("valid") == "valid.txt"

        with open(osp.join(test_dir, "valid.txt"), "r") as f:
            lines = f.readlines()
            assert "images/valid/3.jpg\n" in lines

        self.compare_datasets(source_dataset, parsed_dataset)

    def _check_inplace_save_writes_only_updated_data(self, test_dir, expected):
        assert set(os.listdir(osp.join(test_dir, "images", "train"))) == {
            "1.jpg",
            "2.jpg",
        }
        assert set(os.listdir(osp.join(test_dir, "labels", "train"))) == {
            "1.txt",
            "2.txt",
        }
        assert set(os.listdir(osp.join(test_dir, "images", "valid"))) == set()
        assert set(os.listdir(osp.join(test_dir, "labels", "valid"))) == set()
        self.compare_datasets(
            expected,
            Dataset.import_from(test_dir, self.IMPORTER.NAME),
            require_media=True,
        )


class YOLOv8SegmentationConverterTest(YOLOv8ConverterTest):
    CONVERTER = YOLOv8SegmentationConverter
    IMPORTER = YOLOv8SegmentationImporter

    def _generate_random_annotation(self, n_of_labels=10):
        return Polygon(
            points=[randint(0, 6) for _ in range(randint(3, 7) * 2)],
            label=randint(0, n_of_labels - 1),
        )

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_export_rotated_bbox(self, test_dir):
        pass


class YOLOv8OrientedBoxesConverterTest(CompareDatasetsRotationMixin, YOLOv8ConverterTest):
    CONVERTER = YOLOv8OrientedBoxesConverter
    IMPORTER = YOLOv8OrientedBoxesImporter

    def _generate_random_annotation(self, n_of_labels=10):
        return self._generate_random_bbox(n_of_labels=n_of_labels, rotation=randint(10, 350))

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_export_rotated_bbox(self, test_dir):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        self._generate_random_bbox(n_of_labels=2, rotation=30.0),
                    ],
                ),
            ],
            categories=["a", "b"],
        )
        source_dataset.export(test_dir, self.CONVERTER.NAME, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        assert abs(list(parsed_dataset)[0].annotations[0].attributes["rotation"] - 30) < 0.001
        self.compare_datasets(source_dataset, parsed_dataset)


class YOLOv8PoseConverterTest(YOLOv8ConverterTest):
    CONVERTER = YOLOv8PoseConverter
    IMPORTER = YOLOv8PoseImporter

    def _generate_random_skeleton_annotation(self, skeleton_label_to_point_labels, n_of_labels=10):
        label_id = random.choice(list(skeleton_label_to_point_labels.keys()))  # nosec B311 NOSONAR
        return Skeleton(
            [
                Points(
                    [randint(1, 7), randint(1, 7)],
                    [randint(0, 2)],
                    label=label,
                )
                for label in skeleton_label_to_point_labels[label_id]
            ],
            label=label_id,
        )

    def _generate_random_dataset(self, recipes, n_of_labels=10):
        n_of_points_in_skeleton = randint(3, 8)
        labels = [f"skeleton_label_{index}" for index in range(n_of_labels)] + [
            (f"skeleton_label_{parent_index}_point_{point_index}", f"skeleton_label_{parent_index}")
            for parent_index in range(n_of_labels)
            for point_index in range(n_of_points_in_skeleton)
        ]
        skeleton_label_to_point_labels = {
            skeleton_label_id: [
                label_id
                for label_id, label in enumerate(labels)
                if isinstance(label, tuple) and label[1] == f"skeleton_label_{skeleton_label_id}"
            ]
            for skeleton_label_id, skeleton_label in enumerate(labels)
            if isinstance(skeleton_label, str)
        }
        items = [
            DatasetItem(
                id=recipe.get("id", index + 1),
                subset=recipe.get("subset", "train"),
                media=recipe.get(
                    "media",
                    Image(data=np.ones((randint(8, 10), randint(8, 10), 3))),
                ),
                annotations=[
                    self._generate_random_skeleton_annotation(
                        skeleton_label_to_point_labels,
                        n_of_labels=n_of_labels,
                    )
                    for _ in range(recipe.get("annotations", 1))
                ],
            )
            for index, recipe in enumerate(recipes)
        ]

        point_categories = PointsCategories.from_iterable(
            [
                (
                    index,
                    [
                        f"skeleton_label_{index}_point_{point_index}"
                        for point_index in range(n_of_points_in_skeleton)
                    ],
                    set(),
                )
                for index in range(n_of_labels)
            ]
        )

        return Dataset.from_iterable(
            items,
            categories={
                AnnotationType.label: LabelCategories.from_iterable(labels),
                AnnotationType.points: point_categories,
            },
        )

    def test_export_rotated_bbox(self, test_dir):
        pass

    @staticmethod
    def _make_dataset_with_edges_and_point_labels():
        items = [
            DatasetItem(
                id="1",
                subset="train",
                media=Image(data=np.ones((5, 10, 3))),
                annotations=[
                    Skeleton(
                        [
                            Points([1.5, 2.0], [2], label=4),
                            Points([4.5, 4.0], [2], label=5),
                        ],
                        label=3,
                    ),
                ],
            ),
        ]
        return Dataset.from_iterable(
            items,
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        "skeleton_label_1",
                        ("point_label_1", "skeleton_label_1"),
                        ("point_label_2", "skeleton_label_1"),
                        "skeleton_label_2",
                        ("point_label_3", "skeleton_label_2"),
                        ("point_label_4", "skeleton_label_2"),
                    ]
                ),
                AnnotationType.points: PointsCategories.from_iterable(
                    [
                        (0, ["point_label_1", "point_label_2"], {(0, 1)}),
                        (3, ["point_label_3", "point_label_4"], {}),
                    ],
                ),
            },
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_loses_some_info_on_save_load_without_meta_file(self, test_dir):
        # loses point labels
        # loses edges
        # loses label ids - groups skeleton labels to the start
        source_dataset = self._make_dataset_with_edges_and_point_labels()
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    media=Image(data=np.ones((5, 10, 3))),
                    annotations=[
                        Skeleton(
                            [
                                Points([1.5, 2.0], [2], label=4),
                                Points([4.5, 4.0], [2], label=5),
                            ],
                            label=1,
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        "skeleton_label_1",
                        "skeleton_label_2",
                        ("skeleton_label_1_point_0", "skeleton_label_1"),
                        ("skeleton_label_1_point_1", "skeleton_label_1"),
                        ("skeleton_label_2_point_0", "skeleton_label_2"),
                        ("skeleton_label_2_point_1", "skeleton_label_2"),
                    ]
                ),
                AnnotationType.points: PointsCategories.from_iterable(
                    [
                        (0, ["skeleton_label_1_point_0", "skeleton_label_1_point_1"], set()),
                        (1, ["skeleton_label_2_point_0", "skeleton_label_2_point_1"], set()),
                    ],
                ),
            },
        )
        self.CONVERTER.convert(source_dataset, test_dir, save_media=True)

        # check that annotation with label 3 was saved as 1
        with open(osp.join(test_dir, "labels", "train", "1.txt"), "r") as f:
            assert f.readlines()[0].startswith("1 ")

        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        self.compare_datasets(expected_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_meta_file(self, test_dir):
        source_dataset = self._make_dataset_with_edges_and_point_labels()
        self.CONVERTER.convert(source_dataset, test_dir, save_media=True, save_dataset_meta=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        assert osp.isfile(osp.join(test_dir, "dataset_meta.json"))
        self.compare_datasets(source_dataset, parsed_dataset)


class YoloImporterTest(CompareDatasetMixin):
    IMPORTER = YoloImporter
    ASSETS = ["yolo"]

    def test_can_detect(self):
        dataset_dir = get_test_asset_path("yolo_dataset", "yolo")
        detected_formats = Environment().detect_dataset(dataset_dir)
        assert detected_formats == [self.IMPORTER.NAME]

    @staticmethod
    def _asset_dataset():
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(3, 3, 2, 3, label=4),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = self._asset_dataset()
        for asset in self.ASSETS:
            dataset_dir = get_test_asset_path("yolo_dataset", asset)
            dataset = Dataset.import_from(dataset_dir, self.IMPORTER.NAME)
            self.compare_datasets(expected_dataset, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_with_exif_rotated_images(self, test_dir):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((15, 10, 3))),
                    annotations=[
                        Bbox(0, 3, 2.67, 3.0, label=2),
                        Bbox(2, 4.5, 1.33, 4.5, label=4),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        dataset_path = osp.join(test_dir, "dataset")
        shutil.copytree(get_test_asset_path("yolo_dataset", "yolo"), dataset_path)

        # Add exif rotation for image
        image_path = osp.join(dataset_path, "obj_train_data", "1.jpg")
        img = PILImage.open(image_path)
        exif = img.getexif()
        exif.update(
            [
                (ExifTags.Base.ResolutionUnit, 3),
                (ExifTags.Base.XResolution, 28.0),
                (ExifTags.Base.YCbCrPositioning, 1),
                (ExifTags.Base.Orientation, 6),
                (ExifTags.Base.YResolution, 28.0),
            ]
        )
        img.save(image_path, exif=exif)

        dataset = Dataset.import_from(dataset_path, "yolo")

        self.compare_datasets(expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_673)
    def test_can_pickle(self, helper_tc):
        for asset in self.ASSETS:
            dataset_dir = get_test_asset_path("yolo_dataset", asset)
            source = Dataset.import_from(dataset_dir, format=self.IMPORTER.NAME)
            parsed = pickle.loads(pickle.dumps(source))  # nosec
            compare_datasets_strict(helper_tc, source, parsed)


class YOLOv8ImporterTest(YoloImporterTest):
    IMPORTER = YOLOv8Importer
    ASSETS = [
        "yolov8",
        "yolov8_with_list_of_imgs",
        "yolov8_with_subset_txt",
        "yolov8_with_list_of_names",
    ]

    def test_can_detect(self):
        for asset in self.ASSETS:
            dataset_dir = get_test_asset_path("yolo_dataset", asset)
            detected_formats = Environment().detect_dataset(dataset_dir)
            assert set(detected_formats) == {
                YOLOv8Importer.NAME,
                YOLOv8SegmentationImporter.NAME,
                YOLOv8OrientedBoxesImporter.NAME,
            }

    def test_can_detect_and_import_with_any_yaml_as_config(self, test_dir):
        expected_dataset = self._asset_dataset()
        dataset_path = osp.join(test_dir, "dataset")
        shutil.copytree(get_test_asset_path("yolo_dataset", self.ASSETS[0]), dataset_path)
        os.rename(
            osp.join(dataset_path, "data.yaml"), osp.join(dataset_path, "custom_file_name.yaml")
        )

        self.IMPORTER.detect(FormatDetectionContext(dataset_path))
        dataset = Dataset.import_from(dataset_path, self.IMPORTER.NAME)
        self.compare_datasets(expected_dataset, dataset)

    def test_can_detect_and_import_if_multiple_yamls_with_default_among_them(self, test_dir):
        expected_dataset = self._asset_dataset()
        dataset_path = osp.join(test_dir, "dataset")
        shutil.copytree(get_test_asset_path("yolo_dataset", self.ASSETS[0]), dataset_path)
        shutil.copyfile(
            osp.join(dataset_path, "data.yaml"), osp.join(dataset_path, "custom_file_name.yaml")
        )

        self.IMPORTER.detect(FormatDetectionContext(dataset_path))
        dataset = Dataset.import_from(dataset_path, self.IMPORTER.NAME)
        self.compare_datasets(expected_dataset, dataset)

    def test_can_not_detect_or_import_if_multiple_yamls_but_no_default_among_them(self, test_dir):
        dataset_path = osp.join(test_dir, "dataset")
        shutil.copytree(get_test_asset_path("yolo_dataset", self.ASSETS[0]), dataset_path)
        shutil.copyfile(
            osp.join(dataset_path, "data.yaml"),
            osp.join(dataset_path, "custom_file_name1.yaml"),
        )
        os.rename(
            osp.join(dataset_path, "data.yaml"),
            osp.join(dataset_path, "custom_file_name2.yaml"),
        )

        with pytest.raises(FormatRequirementsUnmet):
            self.IMPORTER.detect(FormatDetectionContext(dataset_path))
        with pytest.raises(DatasetNotFoundError):
            Dataset.import_from(dataset_path, self.IMPORTER.NAME)

    def test_can_import_despite_multiple_yamls_if_config_file_provided_as_argument(self, test_dir):
        expected_dataset = self._asset_dataset()
        dataset_path = osp.join(test_dir, "dataset")
        shutil.copytree(get_test_asset_path("yolo_dataset", self.ASSETS[0]), dataset_path)
        shutil.copyfile(
            osp.join(dataset_path, "data.yaml"),
            osp.join(dataset_path, "custom_file_name1.yaml"),
        )
        os.rename(
            osp.join(dataset_path, "data.yaml"),
            osp.join(dataset_path, "custom_file_name2.yaml"),
        )

        dataset = Dataset.import_from(
            dataset_path, self.IMPORTER.NAME, config_file="custom_file_name1.yaml"
        )
        self.compare_datasets(expected_dataset, dataset)


class YOLOv8SegmentationImporterTest(YOLOv8ImporterTest):
    IMPORTER = YOLOv8SegmentationImporter
    ASSETS = [
        "yolov8_segmentation",
    ]

    @staticmethod
    def _asset_dataset():
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((10, 15, 3))),
                    annotations=[
                        Polygon([1.5, 1.0, 6.0, 1.0, 6.0, 5.0], label=2),
                        Polygon([3.0, 1.5, 6.0, 1.5, 6.0, 7.5, 4.5, 7.5, 3.75, 3.0], label=4),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )


class YOLOv8OrientedBoxesImporterTest(CompareDatasetsRotationMixin, YOLOv8ImporterTest):
    IMPORTER = YOLOv8OrientedBoxesImporter
    ASSETS = ["yolov8_oriented_boxes"]

    @staticmethod
    def _asset_dataset():
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=2, attributes=dict(rotation=30)),
                        Bbox(3, 2, 6, 2, label=4, attributes=dict(rotation=120)),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )


class YOLOv8PoseImporterTest(YOLOv8ImporterTest):
    IMPORTER = YOLOv8PoseImporter
    ASSETS = [
        "yolov8_pose",
        "yolov8_pose_two_values_per_point",
    ]

    def test_can_detect(self):
        for asset in self.ASSETS:
            dataset_dir = get_test_asset_path("yolo_dataset", asset)
            detected_formats = Environment().detect_dataset(dataset_dir)
            assert detected_formats == [self.IMPORTER.NAME]

    @staticmethod
    def _asset_dataset():
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    media=Image(data=np.ones((5, 10, 3))),
                    annotations=[
                        Skeleton(
                            [
                                Points([1.5, 2.0], [2], label=1),
                                Points([4.5, 4.0], [2], label=2),
                                Points([7.5, 6.0], [2], label=3),
                            ],
                            label=0,
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        "skeleton_label",
                        ("skeleton_label_point_0", "skeleton_label"),
                        ("skeleton_label_point_1", "skeleton_label"),
                        ("skeleton_label_point_2", "skeleton_label"),
                    ]
                ),
                AnnotationType.points: PointsCategories.from_iterable(
                    [
                        (
                            0,
                            [
                                "skeleton_label_point_0",
                                "skeleton_label_point_1",
                                "skeleton_label_point_2",
                            ],
                            set(),
                        )
                    ],
                ),
            },
        )


class YoloExtractorTest:
    IMPORTER = YoloImporter
    EXTRACTOR = YoloExtractor

    def _prepare_dataset(self, path: str, anno=None) -> Dataset:
        if anno is None:
            anno = Bbox(1, 1, 2, 4, label=0)
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "a",
                    subset="train",
                    media=Image(np.ones((5, 10, 3))),
                    annotations=[anno],
                )
            ],
            categories=["test"],
        )
        dataset.export(path, self.EXTRACTOR.NAME, save_media=True)
        return dataset

    @staticmethod
    def _get_annotation_dir(subset="train"):
        return f"obj_{subset}_data"

    @staticmethod
    def _get_image_dir(subset="train"):
        return f"obj_{subset}_data"

    @staticmethod
    def _make_some_annotation_values():
        return [0.5, 0.5, 0.5, 0.5]

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_parse(self, helper_tc, test_dir):
        expected = self._prepare_dataset(test_dir)
        actual = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        compare_datasets(helper_tc, expected, actual)

    def test_can_report_invalid_data_file(self, test_dir):
        with pytest.raises(DatasetImportError, match="Can't read dataset descriptor file"):
            self.EXTRACTOR(test_dir)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_ann_line_format(self, test_dir):
        self._prepare_dataset(test_dir)
        with open(osp.join(test_dir, self._get_annotation_dir(), "a.txt"), "w") as f:
            f.write("1 2 3\n")

        with pytest.raises(AnnotationImportError) as capture:
            Dataset.import_from(test_dir, self.IMPORTER.NAME).init_cache()
        assert isinstance(capture.value.__cause__, InvalidAnnotationError)
        assert "Unexpected field count" in str(capture.value.__cause__)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_label(self, test_dir):
        self._prepare_dataset(test_dir)
        with open(osp.join(test_dir, self._get_annotation_dir(), "a.txt"), "w") as f:
            f.write(" ".join(str(v) for v in [10] + self._make_some_annotation_values()))

        with pytest.raises(AnnotationImportError) as capture:
            Dataset.import_from(test_dir, self.IMPORTER.NAME).init_cache()
        assert isinstance(capture.value.__cause__, UndeclaredLabelError)
        assert capture.value.__cause__.id == "10"

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    @pytest.mark.parametrize(
        "field, field_name",
        [
            (1, "bbox center x"),
            (2, "bbox center y"),
            (3, "bbox width"),
            (4, "bbox height"),
        ],
    )
    def test_can_report_invalid_field_type(self, field, field_name, test_dir):
        self._check_can_report_invalid_field_type(field, field_name, test_dir)

    def _check_can_report_invalid_field_type(self, field, field_name, test_dir):
        self._prepare_dataset(test_dir)
        with open(osp.join(test_dir, self._get_annotation_dir(), "a.txt"), "w") as f:
            values = [0] + self._make_some_annotation_values()
            values[field] = "a"
            f.write(" ".join(str(v) for v in values))

        with pytest.raises(AnnotationImportError) as capture:
            Dataset.import_from(test_dir, self.IMPORTER.NAME).init_cache()
        assert isinstance(capture.value.__cause__, InvalidAnnotationError)
        assert field_name in str(capture.value.__cause__)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_ann_file(self, test_dir):
        self._prepare_dataset(test_dir)
        os.remove(osp.join(test_dir, self._get_annotation_dir(), "a.txt"))

        with pytest.raises(ItemImportError) as capture:
            Dataset.import_from(test_dir, self.IMPORTER.NAME).init_cache()
        assert isinstance(capture.value.__cause__, FileNotFoundError)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_image_info(self, test_dir):
        self._prepare_dataset(test_dir)
        os.remove(osp.join(test_dir, self._get_image_dir(), "a.jpg"))

        with pytest.raises(ItemImportError) as capture:
            Dataset.import_from(test_dir, self.IMPORTER.NAME).init_cache()
        assert isinstance(capture.value.__cause__, DatasetImportError)
        assert "Can't find image info" in str(capture.value.__cause__)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_subset_info(self, test_dir):
        self._prepare_dataset(test_dir)
        os.remove(osp.join(test_dir, "train.txt"))

        with pytest.raises(InvalidAnnotationError, match="subset list file"):
            Dataset.import_from(test_dir, self.IMPORTER.NAME).init_cache()


class YOLOv8ExtractorTest(YoloExtractorTest):
    IMPORTER = YOLOv8Importer
    EXTRACTOR = YOLOv8Extractor

    @staticmethod
    def _get_annotation_dir(subset="train"):
        return osp.join("labels", subset)

    @staticmethod
    def _get_image_dir(subset="train"):
        return osp.join("images", subset)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_subset_folder(self, test_dir):
        dataset_path = osp.join(test_dir, "dataset")
        shutil.copytree(get_test_asset_path("yolo_dataset", self.IMPORTER.NAME), dataset_path)
        shutil.rmtree(osp.join(dataset_path, "images", "train"))

        with pytest.raises(InvalidAnnotationError, match="subset image folder"):
            Dataset.import_from(dataset_path, self.IMPORTER.NAME).init_cache()


class YOLOv8SegmentationExtractorTest(YOLOv8ExtractorTest):
    IMPORTER = YOLOv8SegmentationImporter
    EXTRACTOR = YOLOv8SegmentationExtractor

    def _prepare_dataset(self, path: str, anno=None) -> Dataset:
        return super()._prepare_dataset(
            path, anno=Polygon(points=[1, 1, 2, 4, 4, 2, 8, 8], label=0)
        )

    @staticmethod
    def _make_some_annotation_values():
        return [0.5, 0.5] * 3

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    @pytest.mark.parametrize(
        "field, field_name",
        [
            (1, "polygon point 0 x"),
            (2, "polygon point 0 y"),
            (3, "polygon point 1 x"),
            (4, "polygon point 1 y"),
            (5, "polygon point 2 x"),
            (6, "polygon point 2 y"),
        ],
    )
    def test_can_report_invalid_field_type(self, field, field_name, test_dir):
        self._check_can_report_invalid_field_type(field, field_name, test_dir)


class YOLOv8OrientedBoxesExtractorTest(YOLOv8ExtractorTest):
    IMPORTER = YOLOv8OrientedBoxesImporter
    EXTRACTOR = YOLOv8OrientedBoxesExtractor

    def _prepare_dataset(self, path: str, anno=None) -> Dataset:
        return super()._prepare_dataset(
            path, anno=Bbox(1, 1, 2, 4, label=0, attributes=dict(rotation=30))
        )

    @staticmethod
    def _make_some_annotation_values():
        return [0.5, 0.5] * 4

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_parse(self, helper_tc, test_dir):
        expected = self._prepare_dataset(test_dir)
        actual = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        assert abs(list(actual)[0].annotations[0].attributes["rotation"] - 30) < 0.001
        compare_datasets(helper_tc, expected, actual, ignored_attrs=["rotation"])

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    @pytest.mark.parametrize(
        "field, field_name",
        [
            (1, "bbox point 0 x"),
            (2, "bbox point 0 y"),
            (3, "bbox point 1 x"),
            (4, "bbox point 1 y"),
            (5, "bbox point 2 x"),
            (6, "bbox point 2 y"),
        ],
    )
    def test_can_report_invalid_field_type(self, field, field_name, test_dir):
        self._check_can_report_invalid_field_type(field, field_name, test_dir)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_shape(self, test_dir):
        self._prepare_dataset(test_dir)
        with open(osp.join(test_dir, self._get_annotation_dir(), "a.txt"), "w") as f:
            f.write("0 0.1 0.1 0.5 0.1 0.5 0.5 0.5 0.2")

        with pytest.raises(AnnotationImportError) as capture:
            Dataset.import_from(test_dir, self.IMPORTER.NAME).init_cache()
        assert isinstance(capture.value.__cause__, InvalidAnnotationError)
        assert "Given points do not form a rectangle" in str(capture.value.__cause__)


class YOLOv8PoseExtractorTest(YOLOv8ExtractorTest):
    IMPORTER = YOLOv8PoseImporter
    EXTRACTOR = YOLOv8PoseExtractor

    def _prepare_dataset(self, path: str, anno=None) -> Dataset:
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "a",
                    subset="train",
                    media=Image(np.ones((5, 10, 3))),
                    annotations=[
                        Skeleton(
                            [
                                Points([1, 2], [Points.Visibility.visible.value], label=1),
                                Points([3, 6], [Points.Visibility.visible.value], label=2),
                                Points([4, 5], [Points.Visibility.visible.value], label=3),
                                Points([8, 7], [Points.Visibility.visible.value], label=4),
                            ],
                            label=0,
                        )
                    ],
                )
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        "test",
                        ("test_point_0", "test"),
                        ("test_point_1", "test"),
                        ("test_point_2", "test"),
                        ("test_point_3", "test"),
                    ]
                ),
                AnnotationType.points: PointsCategories.from_iterable(
                    [(0, ["test_point_0", "test_point_1", "test_point_2", "test_point_3"], set())]
                ),
            },
        )
        dataset.export(path, self.EXTRACTOR.NAME, save_media=True)
        return dataset

    @staticmethod
    def _make_some_annotation_values():
        return [0.5, 0.5, 0.5, 0.5] + [0.5, 0.5, 2] * 4

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    @pytest.mark.parametrize(
        "field, field_name",
        [
            (5, "skeleton point 0 x"),
            (6, "skeleton point 0 y"),
            (7, "skeleton point 0 visibility"),
            (8, "skeleton point 1 x"),
            (9, "skeleton point 1 y"),
            (10, "skeleton point 1 visibility"),
        ],
    )
    def test_can_report_invalid_field_type(self, field, field_name, test_dir):
        self._check_can_report_invalid_field_type(field, field_name, test_dir)
