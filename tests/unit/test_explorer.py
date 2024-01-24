from copy import deepcopy
from functools import partial
from unittest import TestCase

import numpy as np

from datumaro.components.algorithms.hash_key_inference.explorer import Explorer
from datumaro.components.annotation import Caption, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import MediaTypeError
from datumaro.components.media import Image
from datumaro.plugins.data_formats.datumaro.exporter import DatumaroExporter

from ..requirements import Requirements, mark_requirement

from tests.utils.test_utils import TestDir


class ExplorerTest(TestCase):
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
                    media=Image.from_numpy(data=train_img),
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image.from_numpy(data=train_img),
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=3,
                    subset="test",
                    media=Image.from_numpy(data=test_img),
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
                    media=Image.from_numpy(data=train_img),
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image.from_numpy(data=train_img),
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=3,
                    subset="test",
                    media=Image.from_numpy(data=test_img),
                    annotations=[Label(2, id=2), Caption("dog")],
                ),
            ]
        )
        return dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_explore_img_query(self):
        """
        <b>Description:</b>
        Check that explore topk data for image query.

        <b>Input data:</b>
        Image query and dataset to retrieval, in this case use white and black image.

        <b>Expected results:</b>
        Datasetitem with same hash_key as query.

        <b>Steps</b>
        1. Import dataset and set one of datasetitems as query.
        2. Set Explorer and try explore_topk to find similar media of image query.
        3. Check whether each result have same subset as query.
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            for i, item in enumerate(imported_dataset):
                if i == 1:
                    query = item
            explorer = Explorer(imported_dataset)
            results = explorer.explore_topk(query, topk=2)

            for item in results:
                # There were two "train_img"s in "train" subset, and we queried "train_img"
                self.assertEqual(query.subset, item.subset)

            query_without_hash_key = deepcopy(item)
            query_without_hash_key.annotations = []

            results = explorer.explore_topk(query_without_hash_key, topk=2)

            for item in results:
                # There were two "train_img"s in "train" subset, and we queried "train_img"
                self.assertEqual(query_without_hash_key.subset, item.subset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_explore_img_list_query(self):
        """
        <b>Description:</b>
        Check that explore topk data for image list query.

        <b>Input data:</b>
        Datasetitem list for query and dataset to retrieval, in this case use white and black image.

        <b>Expected results:</b>
        Datasetitem with same hash_key as query.

        <b>Steps</b>
        1. Import dataset and set list of train datasetitems as query.
        2. Set Explorer and try explore_topk to find similar media of image query.
        3. Check whether each result have same subset as query.
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            query_list = []
            for i, item in enumerate(imported_dataset):
                if i in [1, 2]:
                    query_list.append(item)
            explorer = Explorer(imported_dataset)
            results = explorer.explore_topk(query_list, topk=2)

            self.assertEqual(results[0].subset, results[1].subset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_explore_txt_query(self):
        """
        <b>Description:</b>
        Check that explore topk data for text query.

        <b>Input data:</b>
        Text query and dataset to retrieval, in this case use white and black image.

        <b>Expected results:</b>
        Datasetitem with same subset as query.

        <b>Steps</b>
        1. Import dataset.
        2. Set Explorer and try explore_topk to find similar media of text query.
        3. Check whether each result have same subset as query.
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            explorer = Explorer(imported_dataset)
            results = explorer.explore_topk(
                "a photo of a upper white and bottom black background", topk=2
            )
            self.assertEqual(results[0].subset, results[1].subset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_explore_txt_list_query(self):
        """
        <b>Description:</b>
        Check that explore topk data for text list query.

        <b>Input data:</b>
        List of text query and dataset to retrieval, in this case use white and black image.

        <b>Expected results:</b>
        Datasetitem with same subset as query.

        <b>Steps</b>
        1. Import dataset.
        2. Set Explorer and try explore_topk to find similar media of list of text query.
        3. Check whether each result have same subset as query.
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            explorer = Explorer(imported_dataset)
            results = explorer.explore_topk(
                ["a photo of a upper white and bottom black background"],
                topk=2,
            )
            self.assertEqual(results[0].subset, results[1].subset)

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
        2. Set Explorer to inference hash for imported dataset.
        3. Check whether MediaTypeError raised properly or not.
        """
        imported_dataset = Dataset.import_from("./tests/assets/brats_dataset", "brats")
        with self.assertRaises(MediaTypeError) as capture:
            Explorer(imported_dataset)
        self.assertIn("MultiframeImage", str(capture.exception))

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
        2. Set Explorer to inference hash for imported dataset.
        3. Check whether MediaTypeError raised properly or not.
        """
        imported_dataset = Dataset.import_from(
            "./tests/assets/kitti_dataset/kitti_raw", "kitti_raw"
        )
        with self.assertRaises(MediaTypeError) as capture:
            Explorer(imported_dataset)
        self.assertIn("PointCloud", str(capture.exception))
