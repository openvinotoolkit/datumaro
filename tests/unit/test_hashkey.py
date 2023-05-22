from unittest import TestCase

import numpy as np
import pytest

from datumaro.components.annotation import Bbox, HashKey
from datumaro.components.dataset import Dataset
from datumaro.util.meta_file_util import has_hashkey_file, parse_hashkey_file

from tests.utils.assets import get_test_asset_path

test_asset_dir_map = {
    "cifar": get_test_asset_path("cifar10_dataset"),
    "camvid": get_test_asset_path("camvid_dataset"),
    "cityscapes": get_test_asset_path("cityscapes_dataset/dataset"),
    "coco": get_test_asset_path("coco_dataset/coco"),
    "datumaro": get_test_asset_path("datumaro_dataset"),
    "icdar_text_localization": get_test_asset_path("icdar_dataset/text_localization"),
    "icdar_word_recognition": get_test_asset_path("icdar_dataset/word_recognition"),
    "icdar_text_segmentation": get_test_asset_path("icdar_dataset/text_segmentation"),
    "image_dir": get_test_asset_path("imagenet_dataset"),
    "image_zip": get_test_asset_path("image_zip_dataset"),
    "kitti": get_test_asset_path("kitti_dataset"),
    "label_me": get_test_asset_path("labelme_dataset"),
    "market1501": get_test_asset_path("market1501_dataset"),
    "mnist_csv": get_test_asset_path("mnist_csv_dataset"),
    "mnist": get_test_asset_path("mnist_dataset"),
    "mot_seq": get_test_asset_path("mot_dataset", "mot_seq"),
    "mots": get_test_asset_path("mots_dataset"),
    "open_images": get_test_asset_path("open_images_dataset/v5"),
    "vgg_face2": get_test_asset_path("vgg_face2_dataset"),
    "voc": get_test_asset_path("voc_dataset/voc_dataset1"),
    "wider_face": get_test_asset_path("widerface_dataset"),
    "yolo": get_test_asset_path("yolo_dataset"),
}


def check_wider_face(dataset):
    # Add label for Bbox annotation
    for item in dataset:
        for annot in item.annotations:
            if isinstance(annot, Bbox) and not annot.label:
                annot.label = 0


@pytest.fixture
def fxt_dataset_dir_with_hash_key(test_dir, fxt_data_format):
    test_asset_dir = test_asset_dir_map[fxt_data_format]
    dataset = Dataset.import_from(test_asset_dir, format=fxt_data_format)
    for item in dataset:
        hash_key = HashKey(hash_key=np.random.randint(0, 256, size=(1, 64), dtype=np.uint8))
        item.annotations += [hash_key]

    if fxt_data_format == "wider_face":
        check_wider_face(dataset)
    if fxt_data_format == "mot_seq":
        fxt_data_format = "mot_seq_gt"
    if fxt_data_format == "mots":
        fxt_data_format = "mots_png"

    dataset.export(test_dir, fxt_data_format, save_media=True, save_hashkey_meta=True)
    hashkey_meta = parse_hashkey_file(test_dir)
    return test_dir, hashkey_meta


class HashKeyTest:
    def compare_hashkey_meta(self, hashkey_meta, dataset):
        test = TestCase()
        for item in dataset:
            for annot in item.annotations:
                if isinstance(annot, HashKey):
                    test.assertEqual(
                        hashkey_meta[item.subset + "/" + item.id], annot.hash_key.tolist()
                    )

    @pytest.mark.parametrize(
        "fxt_data_format",
        [
            "cifar",
            "camvid",
            "cityscapes",
            "coco",
            "datumaro",
            "icdar_text_localization",
            "icdar_word_recognition",
            "icdar_text_segmentation",
            "image_dir",
            "image_zip",
            "kitti",
            "label_me",
            "market1501",
            "mnist_csv",
            "mnist",
            "mot_seq",
            "mots",
            "open_images",
            "vgg_face2",
            "voc",
            "wider_face",
            "yolo",
        ],
    )
    def test_save_and_load(
        self,
        fxt_dataset_dir_with_hash_key,
        fxt_data_format,
        helper_tc,
    ):
        dataset_dir, hashkey_meta = fxt_dataset_dir_with_hash_key
        helper_tc.assertTrue(has_hashkey_file(dataset_dir))
        dataset = Dataset.import_from(dataset_dir, fxt_data_format)
        self.compare_hashkey_meta(hashkey_meta, dataset)
