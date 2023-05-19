from unittest import TestCase

import numpy as np
import pytest

import datumaro.plugins.data_formats.camvid as Camvid
import datumaro.plugins.data_formats.cityscapes as Cityscapes
import datumaro.plugins.data_formats.voc.format as VOC
from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Caption,
    HashKey,
    Label,
    LabelCategories,
    Mask,
    Points,
    Polygon,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.exporter import Exporter
from datumaro.components.media import Image
from datumaro.plugins.data_formats.camvid import CamvidExporter
from datumaro.plugins.data_formats.cifar import CifarExporter
from datumaro.plugins.data_formats.cityscapes import CityscapesExporter
from datumaro.plugins.data_formats.coco.exporter import CocoExporter
from datumaro.plugins.data_formats.icdar.exporter import (
    IcdarTextLocalizationExporter,
    IcdarTextSegmentationExporter,
    IcdarWordRecognitionExporter,
)
from datumaro.plugins.data_formats.image_dir import ImageDirExporter
from datumaro.plugins.data_formats.image_zip import ImageZipExporter
from datumaro.plugins.data_formats.kitti.exporter import KittiExporter
from datumaro.plugins.data_formats.labelme import LabelMeExporter
from datumaro.plugins.data_formats.market1501 import Market1501Exporter
from datumaro.plugins.data_formats.mnist import MnistExporter
from datumaro.plugins.data_formats.mnist_csv import MnistCsvExporter
from datumaro.plugins.data_formats.mot import MotSeqGtExporter
from datumaro.plugins.data_formats.mots import MotsPngExporter
from datumaro.plugins.data_formats.open_images import OpenImagesExporter
from datumaro.plugins.data_formats.vgg_face2 import VggFace2Exporter
from datumaro.plugins.data_formats.voc.exporter import (
    VocActionExporter,
    VocClassificationExporter,
    VocDetectionExporter,
    VocLayoutExporter,
    VocSegmentationExporter,
)
from datumaro.plugins.data_formats.voc.format import VocTask
from datumaro.plugins.data_formats.widerface import WiderFaceExporter
from datumaro.plugins.data_formats.yolo.exporter import YoloExporter
from datumaro.util.meta_file_util import has_hashkey_file, parse_hashkey_file

from tests.utils.test_utils import TestDir


@pytest.fixture
def test_cifar_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="image_2",
                subset="test",
                media=Image.from_numpy(data=np.ones((32, 32, 3))),
                annotations=[Label(0), HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8))],
            ),
            DatasetItem(
                id="image_3",
                subset="test",
                media=Image.from_numpy(data=np.ones((32, 32, 3))),
                annotations=[Label(1), HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8))],
            ),
        ],
        categories=["label_0", "label_1"],
    )
    return dataset


@pytest.fixture
def test_camvid_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="0001TP_008580",
                subset="test",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(image=np.array([[1, 1, 0, 0, 0]]), label=2),
                    Mask(image=np.array([[0, 0, 1, 0, 0]]), label=4),
                    Mask(image=np.array([[0, 0, 0, 1, 1]]), label=27),
                    HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="0001TP_006690",
                subset="train",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(image=np.array([[1, 1, 0, 1, 1]]), label=3),
                    Mask(image=np.array([[0, 0, 1, 0, 0]]), label=18),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="0016E5_07959",
                subset="val",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(image=np.array([[1, 1, 1, 0, 0]]), label=1),
                    Mask(image=np.array([[0, 0, 0, 1, 1]]), label=8),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
        ],
        categories=Camvid.make_camvid_categories(),
    )
    return dataset


@pytest.fixture
def test_cityscapes_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="defaultcity/defaultcity_000001_000031",
                subset="test",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(np.array([[1, 1, 0, 0, 0]]), label=3, attributes={"is_crowd": True}),
                    Mask(
                        np.array([[0, 0, 1, 0, 0]]),
                        id=1,
                        label=27,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        np.array([[0, 0, 0, 1, 1]]),
                        id=2,
                        label=27,
                        attributes={"is_crowd": False},
                    ),
                    HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="defaultcity/defaultcity_000002_000045",
                subset="train",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(np.array([[1, 1, 0, 1, 1]]), label=3, attributes={"is_crowd": True}),
                    Mask(
                        np.array([[0, 0, 1, 0, 0]]),
                        id=1,
                        label=24,
                        attributes={"is_crowd": False},
                    ),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="defaultcity/defaultcity_000001_000019",
                subset="val",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(np.array([[1, 0, 0, 1, 1]]), label=3, attributes={"is_crowd": True}),
                    Mask(
                        np.array([[0, 1, 1, 0, 0]]),
                        id=24,
                        label=1,
                        attributes={"is_crowd": False},
                    ),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
        ],
        categories=Cityscapes.make_cityscapes_categories(),
    )
    return dataset


@pytest.fixture()
def test_coco_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="a",
                subset="train",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                attributes={"id": 5},
                annotations=[
                    Bbox(2, 2, 3, 1, label=1, group=1, id=1, attributes={"is_crowd": False}),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="b",
                subset="val",
                media=Image.from_numpy(data=np.ones((10, 5, 3))),
                attributes={"id": 40},
                annotations=[
                    Polygon(
                        [0, 0, 1, 0, 1, 2, 0, 2],
                        label=0,
                        id=1,
                        group=1,
                        attributes={"is_crowd": False, "x": 1, "y": "hello"},
                    ),
                    Bbox(
                        0.0,
                        0.0,
                        1.0,
                        2.0,
                        id=1,
                        attributes={"x": 1, "y": "hello", "is_crowd": False},
                        group=1,
                        label=0,
                        z_order=0,
                    ),
                    Mask(
                        np.array([[1, 1, 0, 0, 0]] * 10),
                        label=1,
                        id=2,
                        group=2,
                        attributes={"is_crowd": True},
                    ),
                    Bbox(
                        0.0,
                        0.0,
                        1.0,
                        9.0,
                        id=2,
                        attributes={"is_crowd": True},
                        group=2,
                        label=1,
                        z_order=0,
                    ),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
        ],
        categories=["a", "b", "c"],
    )
    return dataset


@pytest.fixture
def test_icdar_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="a/b/1",
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 10, 3))),
                annotations=[
                    Caption("caption 0"),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            )
        ]
    )
    return dataset


@pytest.fixture
def test_image_dir_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id=1,
                media=Image.from_numpy(data=np.ones((10, 6, 3))),
                annotations=[HashKey(hash_key=np.ones((1, 64), dtype=np.uint8))],
            ),
            DatasetItem(
                id=2,
                media=Image.from_numpy(data=np.ones((5, 4, 3))),
                annotations=[HashKey(hash_key=np.ones((1, 64), dtype=np.uint8))],
            ),
        ]
    )
    return dataset


@pytest.fixture
def test_image_zip_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="subset/1",
                media=Image.from_numpy(data=np.ones((10, 10, 3))),
                annotations=[HashKey(hash_key=np.ones((1, 64), dtype=np.uint8))],
            ),
            DatasetItem(
                id="2",
                media=Image.from_numpy(data=np.ones((4, 5, 3))),
                annotations=[HashKey(hash_key=np.ones((1, 64), dtype=np.uint8))],
            ),
        ]
    )
    return dataset


@pytest.fixture
def test_kitti_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="1_2",
                subset="test",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(
                        image=np.array([[0, 0, 0, 1, 0]]),
                        label=3,
                        id=0,
                        attributes={"is_crowd": True},
                    ),
                    Mask(
                        image=np.array([[0, 1, 1, 0, 0]]),
                        label=24,
                        id=1,
                        attributes={"is_crowd": False},
                    ),
                    Mask(
                        image=np.array([[1, 0, 0, 0, 1]]),
                        label=15,
                        id=0,
                        attributes={"is_crowd": True},
                    ),
                    HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="3",
                subset="val",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(
                        image=np.array([[1, 1, 0, 1, 1]]),
                        label=3,
                        id=0,
                        attributes={"is_crowd": True},
                    ),
                    Mask(
                        image=np.array([[0, 0, 1, 0, 0]]),
                        label=5,
                        id=0,
                        attributes={"is_crowd": True},
                    ),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
        ]
    )
    return dataset


@pytest.fixture
def test_labelme_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="dir1/1",
                subset="train",
                media=Image.from_numpy(data=np.ones((16, 16, 3))),
                annotations=[
                    Bbox(0, 4, 4, 8, label=2, group=2),
                    Polygon(
                        [0, 4, 4, 4, 5, 6],
                        label=3,
                        attributes={
                            "occluded": True,
                            "a1": "qwe",
                            "a2": True,
                            "a3": 123,
                            "a4": "42",  # must be escaped and recognized as string
                            "escaped": 'a,b. = \\= \\\\ " \\" \\, \\',
                        },
                    ),
                    Mask(
                        np.array([[0, 1], [1, 0], [1, 1]]),
                        group=2,
                        attributes={"username": "test"},
                    ),
                    Bbox(1, 2, 3, 4, group=3),
                    Mask(
                        np.array([[0, 0], [0, 0], [1, 1]]),
                        group=3,
                        attributes={"occluded": True},
                    ),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
        ],
        categories=["label_" + str(label) for label in range(10)],
    )
    return dataset


@pytest.fixture
def test_market1501_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="0001_c2s3_000111_00",
                subset="query",
                media=Image.from_numpy(data=np.ones((2, 5, 3))),
                attributes={
                    "camera_id": 1,
                    "person_id": "0001",
                    "track_id": 3,
                    "frame_id": 111,
                    "bbox_id": 0,
                    "query": True,
                },
                annotations=[HashKey(hash_key=np.ones((1, 64), dtype=np.uint8))],
            ),
            DatasetItem(
                id="0001_c1s1_001051_00",
                subset="test",
                media=Image.from_numpy(data=np.ones((2, 5, 3))),
                attributes={
                    "camera_id": 0,
                    "person_id": "0001",
                    "track_id": 1,
                    "frame_id": 1051,
                    "bbox_id": 0,
                    "query": False,
                },
                annotations=[HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8))],
            ),
            DatasetItem(
                id="0002_c1s3_000151_00",
                subset="train",
                media=Image.from_numpy(data=np.ones((2, 5, 3))),
                attributes={
                    "camera_id": 0,
                    "person_id": "0002",
                    "track_id": 3,
                    "frame_id": 151,
                    "bbox_id": 0,
                    "query": False,
                },
                annotations=[HashKey(hash_key=np.ones((1, 64), dtype=np.uint8))],
            ),
        ]
    )
    return dataset


@pytest.fixture
def test_mnist_csv_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id=0,
                subset="test",
                media=Image.from_numpy(data=np.ones((28, 28))),
                annotations=[Label(0), HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8))],
            ),
            DatasetItem(
                id=1,
                subset="train",
                media=Image.from_numpy(data=np.ones((28, 28))),
                annotations=[Label(7), HashKey(hash_key=np.ones((1, 64), dtype=np.uint8))],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(str(label) for label in range(10)),
        },
    )
    return dataset


@pytest.fixture
def test_mnist_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id=0,
                subset="test",
                media=Image.from_numpy(data=np.ones((28, 28))),
                annotations=[Label(0), HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8))],
            ),
            DatasetItem(
                id=0,
                subset="train",
                media=Image.from_numpy(data=np.ones((28, 28))),
                annotations=[Label(5), HashKey(hash_key=np.ones((1, 64), dtype=np.uint8))],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(str(label) for label in range(10)),
        },
    )
    return dataset


@pytest.fixture
def test_mot_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id=1,
                media=Image.from_numpy(data=np.ones((16, 16, 3))),
                annotations=[
                    Bbox(
                        0,
                        4,
                        4,
                        8,
                        label=2,
                        attributes={
                            "occluded": False,
                            "visibility": 1.0,
                            "ignored": False,
                        },
                    ),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
        ],
        categories=["label_" + str(label) for label in range(10)],
    )
    return dataset


@pytest.fixture
def test_mots_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id=1,
                subset="train",
                media=Image.from_numpy(data=np.ones((5, 1))),
                annotations=[
                    Mask(np.array([[0, 0, 0, 1, 0]]), label=3, attributes={"track_id": 1}),
                    Mask(np.array([[0, 0, 1, 0, 0]]), label=2, attributes={"track_id": 2}),
                    Mask(np.array([[1, 1, 0, 0, 0]]), label=3, attributes={"track_id": 3}),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id=2,
                subset="train",
                media=Image.from_numpy(data=np.ones((5, 1))),
                annotations=[
                    Mask(np.array([[1, 0, 0, 0, 0]]), label=3, attributes={"track_id": 2}),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id=3,
                subset="val",
                media=Image.from_numpy(data=np.ones((5, 1))),
                annotations=[
                    Mask(np.array([[0, 1, 0, 0, 0]]), label=0, attributes={"track_id": 1}),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
        ],
        categories=["a", "b", "c", "d"],
    )
    return dataset


@pytest.fixture
def test_open_images_v5_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="aa",
                subset="train",
                media=Image.from_numpy(data=np.zeros((8, 6, 3))),
                annotations=[HashKey(hash_key=np.ones((1, 64), dtype=np.uint8))],
            ),
            DatasetItem(
                id="cc",
                subset="test",
                media=Image.from_numpy(
                    data=np.ones((10, 5, 3)),
                ),
                annotations=[HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8))],
            ),
        ],
        categories=[
            "/m/0",
            "/m/1",
        ],
    )
    return dataset


@pytest.fixture
def test_open_images_v6_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="a",
                subset="train",
                media=Image.from_numpy(data=np.zeros((8, 6, 3))),
                annotations=[
                    Label(label=0, attributes={"score": 1}),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="b",
                subset="train",
                media=Image.from_numpy(data=np.zeros((2, 8, 3))),
                annotations=[
                    Label(label=0, attributes={"score": 0}),
                    Bbox(label=0, x=1.6, y=0.6, w=6.4, h=0.4, group=1, attributes={"score": 1}),
                    Mask(
                        label=0,
                        image=np.hstack((np.ones((2, 2)), np.zeros((2, 6)))),
                        group=1,
                        attributes={
                            "box_id": "01234567",
                            "predicted_iou": 0.5,
                        },
                    ),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="c",
                subset="test",
                media=Image.from_numpy(data=np.ones((10, 5, 3))),
                annotations=[
                    Label(label=1, attributes={"score": 1}),
                    Label(label=3, attributes={"score": 1}),
                    Bbox(
                        label=3,
                        x=3.5,
                        y=0,
                        w=0.5,
                        h=5,
                        group=1,
                        attributes={
                            "score": 0.7,
                            "occluded": True,
                            "truncated": False,
                            "is_group_of": True,
                            "is_depiction": False,
                            "is_inside": False,
                        },
                    ),
                    HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="d",
                subset="validation",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8))],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                [
                    # The hierarchy file in the test dataset also includes a fake label
                    # /m/x that is set to be /m/0's parent. This is to mimic the real
                    # Open Images dataset, that assigns a nonexistent label as a parent
                    # to all labels that don't have one.
                    "/m/0",
                    ("/m/1", "/m/0"),
                    "/m/2",
                    "/m/3",
                ]
            ),
        },
    )
    return dataset


@pytest.fixture
def test_vgg_face2_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="n000001/0001_01",
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 15, 3))),
                annotations=[
                    Bbox(2, 2, 1, 2, label=0),
                    Points([2.787, 2.898, 2.965, 2.79, 2.8, 2.456, 2.81, 2.32, 2.89, 2.3], label=0),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="n000002/0001_01",
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 15, 3))),
                annotations=[
                    Bbox(2, 4, 2, 2, label=1),
                    Points([2.3, 4.9, 2.9, 4.93, 2.62, 4.745, 2.54, 4.45, 2.76, 4.43], label=1),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="n000002/0002_01",
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 15, 3))),
                annotations=[
                    Bbox(1, 3, 1, 1, label=1),
                    Points([1.2, 3.8, 1.8, 3.82, 1.51, 3.634, 1.43, 3.34, 1.65, 3.32], label=1),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="n000003/0003_01",
                subset="test",
                media=Image.from_numpy(data=np.ones((10, 15, 3))),
                annotations=[
                    Bbox(1, 1, 1, 1, label=2),
                    Points([0.2, 2.8, 0.8, 2.9, 0.5, 2.6, 0.4, 2.3, 0.6, 2.3], label=2),
                    HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8)),
                ],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                [("n000001", "Karl"), ("n000002", "Jay"), ("n000003", "Pol")]
            ),
        },
    )
    return dataset


@pytest.fixture
def test_voc_action_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="2007_000001",
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
                annotations=[
                    Bbox(
                        4.0,
                        5.0,
                        2.0,
                        2.0,
                        label=1,
                        id=0,
                        group=0,
                        attributes={
                            "difficult": False,
                            "truncated": False,
                            "occluded": False,
                            **{a.name: a.value % 2 == 1 for a in VOC.VocAction},
                        },
                    ),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="2007_000002",
                subset="test",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
                annotations=[HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8))],
            ),
        ],
        categories=VOC.make_voc_categories(task=VocTask.voc_action),
    )


@pytest.fixture
def test_voc_classification_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="a/0",
                subset="a",
                annotations=[
                    Label(1),
                    Label(2),
                    Label(3),
                    HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id=1,
                subset="b",
                annotations=[
                    Label(4),
                    HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8)),
                ],
            ),
        ],
        categories=VOC.make_voc_categories(task=VOC.VocTask.voc_classification),
    )
    return dataset


@pytest.fixture
def test_voc_layout_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="2007_000001",
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
                annotations=[
                    Bbox(
                        4.0,
                        5.0,
                        2.0,
                        2.0,
                        label=1,
                        id=0,
                        group=0,
                        attributes={
                            "difficult": False,
                            "truncated": False,
                            "occluded": False,
                        },
                    ),
                    Bbox(5.5, 6.0, 2.0, 2.0, label=2, group=0),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="2007_000002",
                subset="test",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
                annotations=[HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8))],
            ),
        ],
        categories=VOC.make_voc_categories(task=VOC.VocTask.voc_layout),
    )
    return dataset


@pytest.fixture
def test_voc_detection_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="2007_000001",
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
                annotations=[
                    Bbox(
                        1.0,
                        2.0,
                        2.0,
                        2.0,
                        label=8,
                        id=0,
                        group=0,
                        attributes={
                            "difficult": False,
                            "truncated": True,
                            "occluded": False,
                            "pose": "Unspecified",
                        },
                    ),
                    Bbox(
                        4.0,
                        5.0,
                        2.0,
                        2.0,
                        label=15,
                        id=1,
                        group=1,
                        attributes={
                            "difficult": False,
                            "truncated": False,
                            "occluded": False,
                        },
                    ),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="2007_000002",
                subset="test",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
                annotations=[HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8))],
            ),
        ],
        categories=VOC.make_voc_categories(task=VOC.VocTask.voc_detection),
    )
    return dataset


@pytest.fixture
def test_voc_segmentation_dataset():
    dataset = Dataset.from_iterable(
        [
            DatasetItem(
                id="2007_000001",
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
                annotations=[
                    Mask(image=np.ones([10, 20]), label=2, group=1),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="2007_000002",
                subset="test",
                media=Image.from_numpy(data=np.ones((10, 20, 3))),
                annotations=[HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8))],
            ),
        ],
        categories=VOC.make_voc_categories(task=VOC.VocTask.voc_segmentation),
    )
    return dataset


@pytest.fixture
def test_widerface_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="1",
                subset="train",
                media=Image.from_numpy(data=np.ones((8, 8, 3))),
                annotations=[
                    Bbox(0, 2, 4, 2, label=0),
                    Bbox(
                        0,
                        1,
                        2,
                        3,
                        label=0,
                        attributes={
                            "blur": "2",
                            "expression": "0",
                            "illumination": "0",
                            "occluded": "0",
                            "pose": "2",
                            "invalid": "0",
                        },
                    ),
                    Label(1),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="2",
                subset="val",
                media=Image.from_numpy(data=np.ones((8, 8, 3))),
                annotations=[
                    Bbox(
                        0,
                        1.1,
                        5.3,
                        2.1,
                        label=0,
                        attributes={
                            "blur": "2",
                            "expression": "1",
                            "illumination": "0",
                            "occluded": "0",
                            "pose": "1",
                            "invalid": "0",
                        },
                    ),
                    Bbox(0, 2, 3, 2, label=0, attributes={"occluded": False}),
                    Bbox(0, 3, 4, 2, label=0, attributes={"occluded": True}),
                    Bbox(0, 2, 4, 2, label=0),
                    Bbox(
                        0,
                        7,
                        3,
                        2,
                        label=0,
                        attributes={
                            "blur": "2",
                            "expression": "1",
                            "illumination": "0",
                            "occluded": "0",
                            "pose": "1",
                            "invalid": "0",
                        },
                    ),
                    HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id="3",
                subset="val",
                media=Image.from_numpy(data=np.ones((8, 8, 3))),
                annotations=[HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8))],
            ),
        ],
        categories=["face", "label_0", "label_1"],
    )


@pytest.fixture
def test_yolo_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id=1,
                subset="train",
                media=Image.from_numpy(data=np.ones((8, 8, 3))),
                annotations=[
                    Bbox(0, 2, 4, 2, label=2),
                    Bbox(0, 1, 2, 3, label=4),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id=2,
                subset="train",
                media=Image.from_numpy(data=np.ones((10, 10, 3))),
                annotations=[
                    Bbox(0, 2, 4, 2, label=2),
                    Bbox(3, 3, 2, 3, label=4),
                    Bbox(2, 1, 2, 3, label=4),
                    HashKey(hash_key=np.ones((1, 64), dtype=np.uint8)),
                ],
            ),
            DatasetItem(
                id=3,
                subset="valid",
                media=Image.from_numpy(data=np.ones((8, 8, 3))),
                annotations=[
                    Bbox(0, 1, 5, 2, label=2),
                    Bbox(0, 2, 3, 2, label=5),
                    Bbox(0, 2, 4, 2, label=6),
                    Bbox(0, 7, 3, 2, label=7),
                    HashKey(hash_key=np.zeros((1, 64), dtype=np.uint8)),
                ],
            ),
        ],
        categories=["label_" + str(i) for i in range(10)],
    )


class HashKeyTest:
    def compare_hashkey_meta(self, hashkey_meta, dataset):
        test = TestCase()
        for item in dataset:
            for annot in item.annotations:
                if isinstance(annot, HashKey):
                    test.assertEqual(
                        hashkey_meta[item.subset + "/" + item.id], annot.hash_key.tolist()
                    )

    def fxt_dataset_dir_with_hash_key(self, tmp_dir, exporter, dataset, fxt_data_format):
        fxt_export_kwargs = {}
        if fxt_data_format in ["voc_action", "voc_layout"]:
            fxt_export_kwargs = {"label_map": fxt_data_format}
        exporter.convert(dataset, tmp_dir, save_media=True, **fxt_export_kwargs)
        hash_key = parse_hashkey_file(tmp_dir)
        return tmp_dir, hash_key

    @pytest.mark.parametrize(
        ["fxt_data_format", "exporter", "fxt_dataset"],
        [
            ("cifar", CifarExporter, "test_cifar_dataset"),
            ("camvid", CamvidExporter, "test_camvid_dataset"),
            ("cityscapes", CityscapesExporter, "test_cityscapes_dataset"),
            ("coco", CocoExporter, "test_coco_dataset"),
            ("icdar_text_localization", IcdarTextLocalizationExporter, "test_icdar_dataset"),
            ("icdar_word_recognition", IcdarWordRecognitionExporter, "test_icdar_dataset"),
            ("icdar_text_segmentation", IcdarTextSegmentationExporter, "test_icdar_dataset"),
            ("image_dir", ImageDirExporter, "test_image_dir_dataset"),
            ("image_zip", ImageZipExporter, "test_image_zip_dataset"),
            ("kitti", KittiExporter, "test_kitti_dataset"),
            ("label_me", LabelMeExporter, "test_labelme_dataset"),
            ("market1501", Market1501Exporter, "test_market1501_dataset"),
            ("mnist_csv", MnistCsvExporter, "test_mnist_csv_dataset"),
            ("mnist", MnistExporter, "test_mnist_dataset"),
            ("mot_seq", MotSeqGtExporter, "test_mot_dataset"),
            ("mots", MotsPngExporter, "test_mots_dataset"),
            ("open_images", OpenImagesExporter, "test_open_images_v5_dataset"),
            ("open_images", OpenImagesExporter, "test_open_images_v6_dataset"),
            ("vgg_face2", VggFace2Exporter, "test_vgg_face2_dataset"),
            ("voc_action", VocActionExporter, "test_voc_action_dataset"),
            ("voc_classification", VocClassificationExporter, "test_voc_classification_dataset"),
            ("voc_layout", VocLayoutExporter, "test_voc_layout_dataset"),
            ("voc_detection", VocDetectionExporter, "test_voc_detection_dataset"),
            ("voc_segmentation", VocSegmentationExporter, "test_voc_segmentation_dataset"),
            ("wider_face", WiderFaceExporter, "test_widerface_dataset"),
            ("yolo", YoloExporter, "test_yolo_dataset"),
        ],
    )
    def test_save_and_load(
        self,
        fxt_data_format: str,
        exporter: Exporter,
        fxt_dataset: Dataset,
        request: pytest.FixtureRequest,
    ):
        with TestDir() as tmp_dir:
            fxt_dataset = request.getfixturevalue(fxt_dataset)
            dataset_dir, hash_key = self.fxt_dataset_dir_with_hash_key(
                tmp_dir, exporter, fxt_dataset, fxt_data_format
            )
            TestCase().assertTrue(has_hashkey_file(tmp_dir))
            dataset = Dataset.import_from(dataset_dir, fxt_data_format)
            self.compare_hashkey_meta(hash_key, dataset)
