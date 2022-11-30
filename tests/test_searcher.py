import os.path as osp
from functools import partial
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Caption, Label, LabelCategories
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.media import Image
from datumaro.components.model_inference import hash_inference
from datumaro.components.searcher import Searcher
from datumaro.plugins.datumaro_format.converter import DatumaroConverter
from datumaro.util.image import load_image
from datumaro.util.test_utils import TestDir

from .requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), "assets", "datumaro_dataset")


class SearcherTest(TestCase):
    @property
    def test_dataset(self):
        train_img = np.full((5, 5, 3), 255, dtype=np.uint8)
        train_img[2, :] = 0
        test_img = np.full((5, 5, 3), 0, dtype=np.uint8)
        test_img[2, :] = 255

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=train_img),
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image(data=train_img),
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=3,
                    subset="test",
                    media=Image(data=test_img),
                    annotations=[Label(2, id=2), Caption("dog")],
                ),
            ]
        )
        with TestDir() as test_dir:
            converter = partial(DatumaroConverter.convert, save_media=True)
            converter(dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro", save_hash=True)
        return imported_dataset

    @property
    def test_coco_dataset(self):
        SEARCHER_DATASET_DIR = osp.join(osp.dirname(__file__), "assets", "searcher")
        train_img1 = load_image(osp.join(SEARCHER_DATASET_DIR, "000000094852.jpg"))
        train_img2 = load_image(osp.join(SEARCHER_DATASET_DIR, "000000475779.jpg"))
        test_img = load_image(osp.join(SEARCHER_DATASET_DIR, "000000572517.jpg"))

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=train_img1),
                    annotations=[Label(1, id=1), Caption("elephant")],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image(data=train_img2),
                    annotations=[Label(1, id=1), Caption("elephant")],
                ),
                DatasetItem(
                    id=3,
                    subset="test",
                    media=Image(data=test_img),
                    annotations=[Label(2, id=2), Caption("bear")],
                ),
            ]
        )
        with TestDir() as test_dir:
            converter = partial(DatumaroConverter.convert, save_media=True)
            converter(dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro", save_hash=True)
        return imported_dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inference(self):
        null_image = DatasetItem(
            id="null_img",
            subset="train",
            media=Image(data=np.zeros((3, 4, 3))),
            annotations=[Label(0)],
        )

        hash_key = hash_inference(null_image.media)
        null_hash_key = null_image.set_hash_key

        self.assertEqual(hash_key, null_hash_key)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inference_img_query(self):
        for i, item in enumerate(self.test_dataset):
            if i == 1:
                query = item
        seacher = Searcher(self.test_dataset)
        result = seacher.search_topk(query, topk=2)
        self.assertEqual(query.subset, result[1].subset)
        self.assertEqual(query.hash_key, result[1].hash_key)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inference_txt_query(self):
        seacher = Searcher(self.test_coco_dataset)
        result = seacher.search_topk("elephant", topk=2)
        self.assertEqual(result[0].subset, result[0].subset)
        self.assertEqual(result[0].hash_key, result[0].hash_key)
