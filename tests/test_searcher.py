import os.path as osp
from functools import partial
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Caption, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import MediaTypeError
from datumaro.components.media import Image
from datumaro.components.model_inference import hash_inference
from datumaro.components.searcher import Searcher
from datumaro.plugins.data_formats.datumaro.exporter import DatumaroExporter
from datumaro.util.image import load_image
from datumaro.util.test_utils import TestDir

from .requirements import Requirements, mark_requirement


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
        return dataset

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
        return dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_hash_inference(self):
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
    def test_search_img_query(self):
        """
        <b>Description:</b>
        Check that search topk data for image query.

        <b>Input data:</b>
        Image query and dataset to retrieval, in this case use white and black image.

        <b>Expected results:</b>
        Datasetitem with same hash_key as query.

        <b>Steps</b>
        1. Import dataset and set one of datasetitems as query.
        2. Set Searcher and try search_topk to find similar media of image query.
        3. Check whether each result have same subset as query.
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            for i, item in enumerate(imported_dataset):
                if i == 1:
                    query = item
            searcher = Searcher(imported_dataset)
            result = searcher.search_topk(query, topk=2)
            self.assertEqual(query.subset, result[0].subset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_search_txt_query(self):
        """
        <b>Description:</b>
        Check that search topk data for text query.

        <b>Input data:</b>
        Text query and dataset to retrieval, in this case use coco2017 val data.

        <b>Expected results:</b>
        Datasetitem with same subset as query.

        <b>Steps</b>
        1. Import dataset.
        2. Set Searcher and try search_topk to find similar media of text query.
        3. Check whether each result have same subset as query.
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_coco_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            searcher = Searcher(imported_dataset)
            result = searcher.search_topk("elephant", topk=2)
            self.assertEqual(result[0].subset, result[1].subset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_query_data_none(self):
        """
        <b>Description:</b>
        Check that query does not have any media.

        <b>Input data:</b>
        Query whose data is None.

        <b>Expected results:</b>
        Raise ValueError as Query should have hash_key.

        <b>Steps</b>
        1. Import datumaro dataset which contain None media data.
        2. Set Searcher and try search_topk to find similar media of query.
        3. Check whether ValueError raised properly or not.
        """
        imported_dataset = Dataset.import_from("./tests/assets/datumaro_dataset", "datumaro")
        for i, item in enumerate(imported_dataset):
            if i == 0:
                query = item
        searcher = Searcher(imported_dataset)
        with self.assertRaises(ValueError) as capture:
            searcher.search_topk(query, topk=2)
        self.assertEqual("Query should have hash_key", str(capture.exception))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_database_data_none(self):
        """
        <b>Description:</b>
        Check that each data of database does not have any media.

        <b>Input data:</b>
        Database whose data is None.

        <b>Expected results:</b>
        Raise ValueError as Database should have hash_key.

        <b>Steps</b>
        1. Import datumaro dataset which contain None media data.
        2. Set Searcher and try search_topk to find similar media of query.
        3. Check whether ValueError raised properly or not.
        """
        imported_dataset = Dataset.import_from("./tests/assets/voc_dataset/voc_dataset2", "voc")
        for i, item in enumerate(imported_dataset):
            if i == 0:
                query = item
        searcher = Searcher(imported_dataset)
        with self.assertRaises(ValueError) as capture:
            searcher.search_topk(query, topk=2)
        self.assertEqual("Database should have hash_key", str(capture.exception))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_multiframeimage_assert(self):
        """
        <b>Description:</b>
        Check that hash inference does not supported for MultiframeImage mediatype.

        <b>Input data:</b>
        Imported dataset which contains MultiframeImage media.

        <b>Expected results:</b>
        Raise MediaTypeError as Media type should be Image, Current type=<class 'datumaro.components.media.MultiframeImage'>.

        <b>Steps</b>
        1. Import Brats dataset which contain MultiframeImage media. (Brats Numpy Dataset also contain MultiframeImage.)
        2. Set Searcher to inference hash for imported dataset.
        3. Check whether MediaTypeError raised properly or not.
        """
        imported_dataset = Dataset.import_from("./tests/assets/brats_dataset", "brats")
        with self.assertRaises(MediaTypeError) as capture:
            Searcher(imported_dataset)
        self.assertIn("MultiframeImage", str(capture.exception))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_video_assert(self):
        """
        <b>Description:</b>
        Check that hash inference does not supported for Video mediatype.

        <b>Input data:</b>
        Imported dataset which contains Video media.

        <b>Expected results:</b>
        Raise MediaTypeError as Media type should be Image, Current type=<class 'datumaro.components.media.Video'>.

        <b>Steps</b>
        1. Import Kinetics dataset which contain Video media.
        2. Set Searcher to inference hash for imported dataset.
        3. Check whether MediaTypeError raised properly or not.
        """
        imported_dataset = Dataset.import_from("./tests/assets/kinetics_dataset", "kinetics")
        with self.assertRaises(MediaTypeError) as capture:
            Searcher(imported_dataset)
        self.assertIn("Video", str(capture.exception))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_pointcloud_assert(self):
        """
        <b>Description:</b>
        Check that hash inference does not supported for PointCloud mediatype.

        <b>Input data:</b>
        Imported dataset which contains PointCloud media.

        <b>Expected results:</b>
        Raise MediaTypeError as Media type should be Image, Current type=<class 'datumaro.components.media.PointCloud'>.

        <b>Steps</b>
        1. Import Kitti Raw dataset which contain PointCloud media.
        2. Set Searcher to inference hash for imported dataset.
        3. Check whether MediaTypeError raised properly or not.
        """
        imported_dataset = Dataset.import_from(
            "./tests/assets/kitti_dataset/kitti_raw", "kitti_raw"
        )
        with self.assertRaises(MediaTypeError) as capture:
            Searcher(imported_dataset)
        self.assertIn("PointCloud", str(capture.exception))
