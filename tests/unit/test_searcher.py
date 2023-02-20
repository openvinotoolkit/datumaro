import platform
from functools import partial
from unittest import TestCase, skipIf

import numpy as np

from datumaro.components.annotation import Caption, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import MediaTypeError
from datumaro.components.media import Image
from datumaro.components.searcher import Searcher
from datumaro.plugins.data_formats.datumaro.exporter import DatumaroExporter
from datumaro.util.test_utils import TestDir

from ..requirements import Requirements, mark_requirement


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
    def test_dataset_black_white(self):
        train_img = np.full((5, 5, 3), 255, dtype=np.uint8)
        test_img = np.full((5, 5, 3), 0, dtype=np.uint8)

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

    @skipIf(
        platform.system() == "Darwin",
        "Segmentation fault only occurs on MacOS: "
        "https://github.com/openvinotoolkit/datumaro/actions/runs/4202399957/jobs/7324077250",
    )
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

    @skipIf(
        platform.system() == "Darwin",
        "Segmentation fault only occurs on MacOS: "
        "https://github.com/openvinotoolkit/datumaro/actions/runs/4202399957/jobs/7324077250",
    )
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
            converter(self.test_dataset_black_white, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            searcher = Searcher(imported_dataset)
            result = searcher.search_topk("a photo of white background", topk=2)
            self.assertEqual(result[0].subset, result[1].subset)

    @skipIf(
        platform.system() == "Darwin",
        "Segmentation fault only occurs on MacOS: "
        "https://github.com/openvinotoolkit/datumaro/actions/runs/4202399957/jobs/7324077250",
    )
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_data_none(self):
        """
        <b>Description:</b>
        Check that data does not have any media.

        <b>Input data:</b>
        Dataset whose data is None.

        <b>Expected results:</b>
        Raise ValueError as data should have hash_key.

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
        self.assertEqual("Database should have hash_key", str(capture.exception))

    @skipIf(
        platform.system() == "Darwin",
        "Segmentation fault only occurs on MacOS: "
        "https://github.com/openvinotoolkit/datumaro/actions/runs/4202399957/jobs/7324077250",
    )
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

    @skipIf(
        platform.system() == "Darwin",
        "Segmentation fault only occurs on MacOS: "
        "https://github.com/openvinotoolkit/datumaro/actions/runs/4202399957/jobs/7324077250",
    )
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
