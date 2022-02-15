from unittest import TestCase

import numpy as np

from datumaro.components.annotation import Label, Bbox
from datumaro.components.extractor import DatasetItem
from datumaro.components.dataset import Dataset
import datumaro.plugins.corrector as corrector

from .requirements import Requirements, mark_requirement


class DeleteImage(TestCase):

    @mark_requirement(Requirements.DATUM_API)
    def test_delete_dataset_items(self):
        id = '1'
        dataset = Dataset.from_iterable([
            DatasetItem(id=id, subset='test',
                        image=np.ones((10, 20, 3))),
        ], categories=[('cat', '', ['truncated', 'difficult']),
                       ('dog', '', ['truncated', 'difficult']),
                       ('person', '', ['truncated', 'difficult']),
                       ('car', '', ['truncated', 'difficult']), ])

        dataset_fixed = corrector.delete_dataset_items(dataset, id)
        self.assertEqual(len(dataset_fixed), 0)
