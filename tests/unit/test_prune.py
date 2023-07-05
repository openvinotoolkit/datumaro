from collections import Counter
from functools import partial

import numpy as np
import pytest

from datumaro.components.algorithms.hash_key_inference.prune import (
    Prune,
    match_num_item_for_cluster,
)
from datumaro.components.annotation import Caption, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.data_formats.datumaro.exporter import DatumaroExporter

from ..requirements import Requirements, mark_requirement

from tests.utils.test_utils import TestDir


class PruneTest:
    @pytest.fixture
    def fxt_dataset(self):
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
                DatasetItem(
                    id=4,
                    subset="test",
                    media=Image.from_numpy(data=test_img),
                    annotations=[Label(2, id=2), Caption("dog")],
                ),
            ],
            categories=["1", "2"],
        )
        return dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_match_num_item_for_cluster(self):
        """ """
        ratio = 0.5
        total_num_items = 100
        cluster_num_items = [20, 30, 15, 10, 25]

        result = match_num_item_for_cluster(ratio, total_num_items, cluster_num_items)

        # Assert the expected result based on the given inputs
        expected_result = [10, 15, 7, 5, 12]
        assert result == expected_result

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_random(self, fxt_dataset):
        """
        <b>Description:</b>
        Check that pruned subset with random.

        <b>Input data:</b>
        Dataset with train and test subset that each datasetitem consists of same images.

        <b>Expected results:</b>
        Pruned dataset that each subset contains one datasetitem.

        <b>Steps</b>
        1. Prepare dataset with each subset contains same images.
        2. Set Prune and try get_pruned set method as random to extract representative subset.
        3. Check whether each subset contains one datasetitem.
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(fxt_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, cluster_method="random")

            result = prune.get_pruned(0.5)
            result_subsets = [item.subset for item in result]
            assert Counter(result_subsets) == {"test": 1, "train": 1}

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_clustered_random(self, fxt_dataset):
        """
        <b>Description:</b>
        Check that pruned subset with clustered random.

        <b>Input data:</b>
        Dataset with train and test subset that each datasetitem consists of same images.

        <b>Expected results:</b>
        Pruned dataset that each subset contains one datasetitem.

        <b>Steps</b>
        1. Prepare dataset with each subset contains same images.
        2. Set Prune and try get_pruned set method as clustered_random to extract representative subset.
        3. Check whether each subset contains one datasetitem.
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(fxt_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, cluster_method="cluster_random")

            result = prune.get_pruned(0.5)
            result_subsets = [item.subset for item in result]
            assert Counter(result_subsets) == {"test": 1, "train": 1}

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_centroid(self, fxt_dataset):
        """
        <b>Description:</b>
        Check that pruned subset with centroid.

        <b>Input data:</b>
        Dataset with train and test subset that each datasetitem consists of same images.

        <b>Expected results:</b>
        Pruned dataset that each subset contains one datasetitem.

        <b>Steps</b>
        1. Prepare dataset with each subset contains same images.
        2. Set Prune and try get_pruned set method as centroid to extract representative subset.
        3. Check whether each subset contains one datasetitem.
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(fxt_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, cluster_method="centroid")

            result = prune.get_pruned(0.5)
            result_subsets = [item.subset for item in result]
            assert Counter(result_subsets) == {"test": 1, "train": 1}

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_query_clust_img_hash(self, fxt_dataset):
        """
        <b>Description:</b>
        Check that pruned subset with clustering with query through image hash.

        <b>Input data:</b>
        Dataset with train and test subset that each datasetitem consists of same images.

        <b>Expected results:</b>
        Pruned dataset that each subset contains one datasetitem.

        <b>Steps</b>
        1. Prepare dataset with each subset contains same images.
        2. Set Prune and try get_pruned set method as query_clust to extract representative subset.
        3. Check whether each subset contains one datasetitem.
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(fxt_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, cluster_method="query_clust")

            result = prune.get_pruned(0.5)
            result_subsets = [item.subset for item in result]
            assert Counter(result_subsets) == {"test": 1, "train": 1}

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_query_clust_txt_hash(self, fxt_dataset):
        """
        <b>Description:</b>
        Check that pruned subset with clustering with query through text hash.

        <b>Input data:</b>
        Dataset with train and test subset that each datasetitem consists of same images.

        <b>Expected results:</b>
        Pruned dataset that each subset contains one datasetitem.

        <b>Steps</b>
        1. Prepare dataset with each subset contains same images.
        2. Set Prune and try get_pruned set method as query_clust to extract representative subset.
        3. Check whether each subset contains one datasetitem.
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(fxt_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, cluster_method="query_clust", hash_type="txt")

            result = prune.get_pruned(0.5)
            result_subsets = [item.subset for item in result]
            assert Counter(result_subsets) == {"test": 1, "train": 1}

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_entropy(self, fxt_dataset):
        """
        <b>Description:</b>
        Check that pruned subset with entropy.

        <b>Input data:</b>
        Dataset with train and test subset that each datasetitem consists of same images.

        <b>Expected results:</b>
        Pruned dataset that each subset contains one datasetitem.

        <b>Steps</b>
        1. Prepare dataset with each subset contains same images.
        2. Set Prune and try get_pruned set method as entropy to extract representative subset.
        3. Check whether each subset contains one datasetitem.
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(fxt_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, cluster_method="entropy")

            result = prune.get_pruned(0.5)
            result_subsets = [item.subset for item in result]
            assert Counter(result_subsets) == {"test": 1, "train": 1}

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_ndr(self, fxt_dataset):
        """
        <b>Description:</b>
        Check that pruned subset with ndr.

        <b>Input data:</b>
        Dataset with train and test subset that each datasetitem consists of same images.

        <b>Expected results:</b>
        Pruned dataset that each subset contains one datasetitem.

        <b>Steps</b>
        1. Prepare dataset with each subset contains same images.
        2. Set Prune and try get_pruned set method as ndr to extract representative subset.
        3. Check whether each subset contains one datasetitem.
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(fxt_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, cluster_method="ndr")

            result = prune.get_pruned(0.5)
            result_subsets = [item.subset for item in result]
            assert Counter(result_subsets) == {"test": 1, "train": 1}
