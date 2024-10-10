from unittest import TestCase
from unittest.mock import patch

import numpy as np

from datumaro.components.algorithms.hash_key_inference.explorer import Explorer
from datumaro.components.annotation import AnnotationType, HashKey
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import MediaTypeError
from datumaro.util.meta_file_util import load_hash_key

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path


class ExplorerTest(TestCase):
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
        dataset = Dataset.import_from(get_test_asset_path("explore_dataset"), "imagenet")
        query_item = dataset[0]

        explorer = Explorer(dataset)
        results = explorer.explore_topk(query_item, topk=3)

        for item in results:
            self.assertEqual(query_item.annotations[0].label, item.annotations[0].label)

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
        dataset = Dataset.import_from(get_test_asset_path("explore_dataset"), "imagenet")
        query_list = [dataset[i] for i in [0, 1]]

        explorer = Explorer(dataset)
        results = explorer.explore_topk(query_list, topk=3)

        for item in results:
            self.assertEqual(query_list[0].annotations[0].label, item.annotations[0].label)
            self.assertEqual(query_list[1].annotations[0].label, item.annotations[0].label)

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
        dataset = Dataset.import_from(get_test_asset_path("explore_dataset"), "imagenet")
        query_txt = "dog"
        query_label, _ = dataset.categories()[AnnotationType.label].find(query_txt)

        explorer = Explorer(dataset)
        results = explorer.explore_topk(query_txt, topk=3)

        for item in results:
            self.assertEqual(query_label, item.annotations[0].label)

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
        dataset = Dataset.import_from(get_test_asset_path("explore_dataset"), "imagenet")
        query_list = ["dog", "fluffy"]
        query_label, _ = dataset.categories()[AnnotationType.label].find(query_list[0])

        explorer = Explorer(dataset)
        results = explorer.explore_topk(query_list, topk=3)

        for item in results:
            self.assertEqual(query_label, item.annotations[0].label)

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


class MetaFileTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_no_hashkey_dir(self):
        """
        Test that the function returns the original dataset if the hashkey directory doesn't exist.
        """
        dataset = [DatasetItem(id="000001", subset="test")]
        with patch("os.path.isdir") as mock_isdir:
            mock_isdir.return_value = False
            result = load_hash_key("invalid_path", dataset)
            self.assertEqual(result, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_no_hashkey_file(self):
        """
        Test that the function returns the original dataset if the hashkey file doesn't exist.
        """
        dataset = [DatasetItem(id="000001", subset="test")]
        with patch("os.path.isdir") as mock_isdir, patch(
            "datumaro.util.meta_file_util.has_hashkey_file"
        ) as mock_has_hashkey_file:
            mock_isdir.return_value = True
            mock_has_hashkey_file.return_value = False
            result = load_hash_key("hashkey_dir", dataset)
            self.assertEqual(result, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_load_hash_key(self):
        """
        Test that the function successfully parses the hashkey file and adds HashKey annotations to the dataset items.
        """
        dataset = [
            DatasetItem(id="000001", subset="train", annotations=[]),
            DatasetItem(id="000002", subset="val", annotations=[]),
        ]
        expected_hashkey1 = np.ones((96,), dtype=np.uint8)
        expected_hashkey2 = np.zeros((96,), dtype=np.uint8)
        hashkey_dict = {
            "train/000001": expected_hashkey1.tolist(),
            "val/000002": expected_hashkey2.tolist(),
        }

        with patch("os.path.isdir") as mock_isdir, patch(
            "datumaro.util.meta_file_util.has_hashkey_file"
        ) as mock_has_hashkey_file, patch(
            "datumaro.util.meta_file_util.parse_hashkey_file"
        ) as mock_parse_hashkey_file:
            mock_isdir.return_value = True
            mock_has_hashkey_file.return_value = True
            mock_parse_hashkey_file.return_value = hashkey_dict

            result = load_hash_key("hashkey_dir", dataset)

            self.assertEqual(len(result), len(dataset))
            self.assertEqual(result[0].id, dataset[0].id)
            self.assertEqual(result[0].subset, dataset[0].subset)

            # Check if HashKey annotations are added
            self.assertEqual(len(result[0].annotations), 1)
            self.assertIsInstance(result[0].annotations[0], HashKey)
            self.assertTrue(np.array_equal(result[0].annotations[0].hash_key, expected_hashkey1))

            # Check if HashKey annotations are added for the second item as well
            self.assertEqual(len(result[1].annotations), 1)
            self.assertIsInstance(result[1].annotations[0], HashKey)
            self.assertTrue(np.array_equal(result[1].annotations[0].hash_key, expected_hashkey2))
