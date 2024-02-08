import os.path as osp
from copy import deepcopy
from functools import partial
from unittest import TestCase

import numpy as np

from datumaro.components.algorithms.hash_key_inference.explorer import Explorer
from datumaro.components.annotation import AnnotationType, Caption, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import MediaTypeError
from datumaro.components.media import Image
from datumaro.plugins.data_formats.datumaro.exporter import DatumaroExporter

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir


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
