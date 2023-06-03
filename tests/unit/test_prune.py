from functools import partial
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Caption, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.prune import Prune
from datumaro.components.media import Image
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
            categories=['1', '2'],
        )
        return dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_random(self):
        """
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset,'random')

            result = prune.get_pruned(0.5)
            for item in result:
                self.assertIn(item.id, ['2', '4'])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_clustered_random(self):
        """
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset,'cluster_random')

            result = prune.get_pruned(0.5)
            for item in result:
                self.assertIn(item.id, ['1', '3'])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_query_clust(self):
        """
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, "query_clust")

            result = prune.get_pruned(0.5)
            for item in result:
                self.assertIn(item.id, ['1', '3'])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_prune_entropy(self):
        """
        """
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro")
            prune = Prune(imported_dataset, "entropy")

            result = prune.get_pruned(0.5)
            for item in result:
                self.assertIn(item.id, ['1', '3'])
