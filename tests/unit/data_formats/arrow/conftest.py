# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os

import numpy as np
import pytest

from datumaro.components.annotation import Cuboid3d, Label
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image, PointCloud
from datumaro.components.project import Dataset
from datumaro.util.image import encode_image

from ..datumaro.conftest import (
    fxt_can_save_and_load_image_with_arbitrary_extension,
    fxt_can_save_and_load_infos,
    fxt_can_save_dataset_with_cjk_categories,
    fxt_can_save_dataset_with_cyrillic_and_spaces_in_filename,
    fxt_relative_paths,
    fxt_test_datumaro_format_dataset,
)


@pytest.fixture
def fxt_image(test_dir, n=1000):
    items = []
    for i in range(n):
        media = None
        if i % 3 == 0:
            media = Image.from_numpy(data=np.random.randint(0, 255, (5, 5, 3)))
        elif i % 3 == 1:
            media = Image.from_bytes(
                data=encode_image(np.random.randint(0, 255, (5, 5, 3)), ".png")
            )
        elif i % 3 == 2:
            Image.from_numpy(data=np.random.randint(0, 255, (5, 5, 3))).save(
                os.path.join(test_dir, f"test{i}.jpg")
            )
            media = Image.from_file(path=os.path.join(test_dir, f"test{i}.jpg"))

        items.append(
            DatasetItem(
                id=i,
                subset="test",
                media=media,
                annotations=[Label(np.random.randint(0, 3))],
            )
        )

    source_dataset = Dataset.from_iterable(
        items,
        categories=["label"],
        media_type=Image,
    )

    yield source_dataset


@pytest.fixture
def fxt_point_cloud(test_dir, n=1000):
    items = []
    for i in range(n):
        media = None
        Image.from_numpy(data=np.random.randint(0, 255, (5, 5, 3))).save(
            os.path.join(test_dir, f"test{i}.jpg")
        )
        extra_images = [
            Image.from_numpy(data=np.random.randint(0, 255, (5, 5, 3))),
            Image.from_bytes(data=encode_image(np.random.randint(0, 255, (5, 5, 3)), ".png")),
            Image.from_file(path=os.path.join(test_dir, f"test{i}.jpg")),
        ]
        data = (f"{i}" * 10).encode()
        if i % 2 == 0:
            media = PointCloud.from_bytes(data=data, extra_images=extra_images)
        elif i % 2 == 1:
            with open(os.path.join(test_dir, f"test{i}.pcd"), "wb") as f:
                f.write(data)
            media = PointCloud.from_file(
                path=os.path.join(test_dir, f"test{i}.pcd"), extra_images=extra_images
            )

        items.append(
            DatasetItem(
                id=i,
                subset="test",
                media=media,
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
            )
        )

    source_dataset = Dataset.from_iterable(
        items,
        categories=["label"],
        media_type=PointCloud,
    )

    yield source_dataset
