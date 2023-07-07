# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import json
import os
import os.path as osp

import numpy as np
import pycocotools.mask as mask_tools
import pytest

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Caption,
    Cuboid3d,
    Ellipse,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
    RleMask,
)
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image, PointCloud
from datumaro.components.project import Dataset
from datumaro.plugins.data_formats.datumaro.format import DatumaroPath
from datumaro.util.mask_tools import generate_colormap


@pytest.fixture
def fxt_test_datumaro_format_dataset():
    label_categories = LabelCategories(attributes={"a", "b", "score"})
    for i in range(5):
        label_categories.add("cat" + str(i), attributes={"x", "y"})

    mask_categories = MaskCategories(generate_colormap(len(label_categories.items)))

    points_categories = PointsCategories()
    for index, _ in enumerate(label_categories.items):
        points_categories.add(index, ["cat1", "cat2"], joints=[[0, 1]])

    return Dataset.from_iterable(
        [
            DatasetItem(
                id=100,
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 6, 3))),
                annotations=[
                    Caption("hello", id=1),
                    Caption("world", id=2, group=5),
                    Label(
                        2,
                        id=3,
                        attributes={
                            "x": 1,
                            "y": "2",
                        },
                    ),
                    Bbox(
                        1,
                        2,
                        3,
                        4,
                        label=4,
                        id=4,
                        z_order=1,
                        attributes={
                            "score": 1.0,
                        },
                    ),
                    Bbox(
                        5,
                        6,
                        7,
                        8,
                        id=5,
                        group=5,
                        attributes={
                            "a": 1.5,
                            "b": "text",
                        },
                    ),
                    Points(
                        [1, 2, 2, 0, 1, 1],
                        label=0,
                        id=5,
                        z_order=4,
                        attributes={
                            "x": 1,
                            "y": "2",
                        },
                    ),
                    Mask(
                        label=3,
                        id=5,
                        z_order=2,
                        image=np.ones((2, 3)),
                        attributes={
                            "x": 1,
                            "y": "2",
                        },
                    ),
                    Ellipse(
                        5,
                        6,
                        7,
                        8,
                        label=3,
                        id=5,
                        z_order=2,
                        attributes={
                            "x": 1,
                            "y": "2",
                        },
                    ),
                ],
            ),
            DatasetItem(
                id=21,
                media=Image.from_numpy(data=np.ones((10, 6, 3))),
                subset="train",
                annotations=[
                    Caption("test"),
                    Label(2),
                    Bbox(1, 2, 3, 4, label=5, id=42, group=42),
                ],
            ),
            DatasetItem(
                id=2,
                media=Image.from_numpy(data=np.ones((10, 6, 3))),
                subset="val",
                annotations=[
                    PolyLine([1, 2, 3, 4, 5, 6, 7, 8], id=11, z_order=1),
                    Polygon([1, 2, 3, 4, 5, 6, 7, 8], id=12, z_order=4),
                ],
            ),
            DatasetItem(
                id=1,
                media=Image.from_numpy(data=np.ones((10, 6, 3))),
                subset="test",
                annotations=[
                    Cuboid3d(
                        [1.0, 2.0, 3.0],
                        [2.0, 2.0, 4.0],
                        [1.0, 3.0, 4.0],
                        id=6,
                        label=0,
                        attributes={"occluded": True},
                        group=6,
                    )
                ],
            ),
            DatasetItem(
                id=42,
                media=Image.from_numpy(data=np.ones((10, 6, 3))),
                subset="test",
                attributes={"a1": 5, "a2": "42"},
            ),
            DatasetItem(
                id=42,
                media=Image.from_numpy(data=np.ones((10, 6, 3))),
                # id and group integer value can be higher than 32bits limits (COCO instances).
                annotations=[
                    Mask(
                        id=900100087038, group=900100087038, image=np.ones((2, 3), dtype=np.uint8)
                    ),
                    RleMask(
                        rle=mask_tools.encode(np.ones((2, 3), dtype=np.uint8, order="F")),
                        id=900100087038,
                        group=900100087038,
                    ),
                ],
            ),
            DatasetItem(
                id=43,
                media=Image.from_file(path="1/b/c.qq", size=(2, 4)),
            ),
        ],
        categories={
            AnnotationType.label: label_categories,
            AnnotationType.mask: mask_categories,
            AnnotationType.points: points_categories,
        },
        infos={
            "string": "test",
            "int": 0,
            "float": 0.0,
            "string_list": ["test0", "test1", "test2"],
            "int_list": [0, 1, 2],
            "float_list": [0.0, 0.1, 0.2],
        },
    )


@pytest.fixture
def fxt_wrong_version_dir(fxt_test_datumaro_format_dataset, test_dir):
    dest_dir = osp.join(test_dir, "wrong_version")
    fxt_test_datumaro_format_dataset.export(dest_dir, "datumaro")

    # exchange the dm_format version string to wrong string
    for path_annt in glob.glob(os.path.join(dest_dir, DatumaroPath.ANNOTATIONS_DIR, "**")):
        if not path_annt.endswith(DatumaroPath.ANNOTATION_EXT):
            continue

        with open(path_annt, "r") as f_annt:
            annt_json = json.load(f_annt)

        annt_json["dm_format_version"] = "wrong_version_string"

        with open(path_annt, "w") as f_annt:
            json.dump(annt_json, f_annt)

    yield dest_dir


@pytest.fixture
def fxt_relative_paths():
    return Dataset.from_iterable(
        [
            DatasetItem(id="1", media=Image.from_numpy(data=np.ones((4, 2, 3)))),
            DatasetItem(id="subdir1/1", media=Image.from_numpy(data=np.ones((2, 6, 3)))),
            DatasetItem(id="subdir2/1", media=Image.from_numpy(data=np.ones((5, 4, 3)))),
        ]
    )


@pytest.fixture
def fxt_can_save_dataset_with_cjk_categories():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id=1,
                subset="train",
                media=Image.from_numpy(data=np.ones((4, 4, 3))),
                annotations=[
                    Bbox(0, 1, 2, 2, label=0, group=1, id=1, attributes={"is_crowd": False}),
                ],
                attributes={"id": 1},
            ),
            DatasetItem(
                id=2,
                subset="train",
                media=Image.from_numpy(data=np.ones((4, 4, 3))),
                annotations=[
                    Bbox(1, 0, 2, 2, label=1, group=2, id=2, attributes={"is_crowd": False}),
                ],
                attributes={"id": 2},
            ),
            DatasetItem(
                id=3,
                subset="train",
                media=Image.from_numpy(data=np.ones((4, 4, 3))),
                annotations=[
                    Bbox(0, 1, 2, 2, label=2, group=3, id=3, attributes={"is_crowd": False}),
                ],
                attributes={"id": 3},
            ),
        ],
        categories=["고양이", "ネコ", "猫"],
    )


@pytest.fixture
def fxt_can_save_dataset_with_cyrillic_and_spaces_in_filename():
    return Dataset.from_iterable(
        [DatasetItem(id="кириллица с пробелом", media=Image.from_numpy(data=np.ones((4, 2, 3))))]
    )


@pytest.fixture
def fxt_can_save_and_load_image_with_arbitrary_extension():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="q/1",
                media=Image.from_numpy(data=np.zeros((4, 3, 3)), ext=".jpeg"),
                attributes={"frame": 1},
            ),
            DatasetItem(
                id="a/b/c/2",
                media=Image.from_numpy(data=np.zeros((3, 4, 3)), ext=".bmp"),
                attributes={"frame": 2},
            ),
        ]
    )


@pytest.fixture
def fxt_can_save_and_load_infos():
    infos = {"info 1": 1, "info 2": "test info"}

    return Dataset.from_iterable(
        [DatasetItem(3, subset="train", media=Image.from_numpy(data=np.ones((2, 2, 3))))],
        infos=infos,
    )


@pytest.fixture
def fxt_point_cloud_dataset_pair(test_dir):
    Image.from_numpy(np.ones((5, 5, 3))).save(os.path.join(test_dir, "test.jpg"))

    source_dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id=1,
                subset="test",
                media=PointCloud.from_bytes(
                    data=b"11111111",
                    extra_images=[
                        Image.from_numpy(data=np.ones((5, 5, 3))),
                        Image.from_numpy(data=np.ones((5, 4, 3))),
                        Image.from_file(path=os.path.join(test_dir, "test.jpg")),
                    ],
                ),
                annotations=[
                    Cuboid3d(
                        [2, 2, 2],
                        [1, 1, 1],
                        [3, 3, 1],
                        id=1,
                        group=1,
                        label=0,
                        attributes={"x": True},
                    )
                ],
            ),
        ],
        categories=["label"],
        media_type=PointCloud,
    )

    target_dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id=1,
                subset="test",
                media=PointCloud.from_file(
                    path=osp.join(test_dir, "point_clouds", "test", "1.pcd"),
                    extra_images=[
                        Image.from_file(
                            path=osp.join(test_dir, "images", "test", "1", "extra_image_0.jpg"),
                        ),
                        Image.from_file(
                            path=osp.join(test_dir, "images", "test", "1", "extra_image_1.jpg"),
                        ),
                        Image.from_file(
                            path=osp.join(test_dir, "images", "test", "1", "extra_image_2.jpg"),
                        ),
                    ],
                ),
                annotations=[
                    Cuboid3d(
                        [2, 2, 2],
                        [1, 1, 1],
                        [3, 3, 1],
                        id=1,
                        group=1,
                        label=0,
                        attributes={"x": True},
                    )
                ],
            ),
        ],
        categories=["label"],
        media_type=PointCloud,
    )

    yield source_dataset, target_dataset


@pytest.fixture
def fxt_legacy_dataset_pair(test_dir):
    source_dataset = Dataset.import_from(
        "./tests/assets/datumaro_dataset/legacy", format="datumaro"
    )

    target_dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="a",
                subset="train",
                media=Image.from_numpy(data=np.zeros((8, 6, 3))),
                annotations=[Label(id=0, attributes={"score": 1.0}, label=0)],
            ),
            DatasetItem(
                id="b",
                subset="train",
                media=Image.from_numpy(data=np.zeros((2, 8, 3))),
                annotations=[
                    Label(id=0, label=0),
                    Label(id=1, label=1),
                    Label(id=2, label=2),
                    Label(id=3, label=5),
                ],
            ),
            DatasetItem(
                id="c",
                subset="test",
                media=Image.from_numpy(data=np.zeros((8, 6, 3))),
                annotations=[
                    Label(id=0, attributes={"score": 1.0}, label=1),
                    Label(id=0, attributes={"score": 1.0}, label=3),
                ],
            ),
            DatasetItem(
                id="d",
                subset="validation",
                media=Image.from_numpy(data=np.zeros((2, 8, 3))),
                annotations=[],
            ),
        ],
        infos={"author": "anonymous", "task": "classification"},
        categories=["car", "bicycle", "tom", "mary"],
        media_type=Image,
    )

    yield source_dataset, target_dataset
