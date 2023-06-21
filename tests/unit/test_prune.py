from collections import Counter
from functools import partial
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Caption, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.components.prune import Prune
from datumaro.plugins.data_formats.datumaro.exporter import DatumaroExporter

from ..requirements import Requirements, mark_requirement

from tests.utils.test_utils import TestDir


class PruneTest(TestCase):
    @property
    def test_dataset(self):
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
    def test_prune_random(self):
        """ """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, cluster_method="random")

            result = prune.get_pruned(0.5)
            result_subsets = [item.subset for item in result[0]]
            self.assertEqual(Counter(result_subsets), {"test": 1, "train": 1})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_clustered_random(self):
        """ """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, cluster_method="cluster_random")

            result = prune.get_pruned(0.5)
            result_subsets = [item.subset for item in result[0]]
            self.assertEqual(Counter(result_subsets), {"test": 1, "train": 1})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_centroid(self):
        """ """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, cluster_method="centroid")

            result = prune.get_pruned(0.5)
            result_subsets = [item.subset for item in result[0]]
            self.assertEqual(Counter(result_subsets), {"test": 1, "train": 1})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_query_clust(self):
        """ """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, cluster_method="query_clust")

            result = prune.get_pruned(0.5)
            result_subsets = [item.subset for item in result[0]]
            self.assertEqual(Counter(result_subsets), {"test": 1, "train": 1})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_entropy(self):
        """ """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, cluster_method="entropy")

            result = prune.get_pruned(0.5)
            result_subsets = [item.subset for item in result[0]]
            self.assertEqual(Counter(result_subsets), {"test": 1, "train": 1})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_ndr(self):
        """
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, cluster_method="ndr")

            result = prune.get_pruned(0.5)
            
