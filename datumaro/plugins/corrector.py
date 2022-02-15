# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Dict, List
import argparse

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.extractor import DatasetItem, IExtractor, Transform
from datumaro.components.dataset import Dataset
from datumaro.components.annotation import Label


class Corrector(Transform, CliPlugin):
    """
    Corrector is a post-process component of Datumaro's Validator that,|n
    fixes annotation problems in datasets.
    """

    def __init__(self, extractor: IExtractor, ids: str):
        super().__init__(extractor)

        self._ids = ids

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('-i', '--ids', type=str, required=True,
                            help="Datasetitem ids to run trasform")
        return parser

    @staticmethod
    def _get_item_subset(dataset: Dataset) -> Dict:
        id_subset = {}
        for item in dataset:
            id_subset[item.id] = item.subset
        return id_subset


class DeleteImage(Corrector):
    """
    DeleteImage that supports deleting images with annotation errors in items.
    """

    @staticmethod
    def delete_dataset_items(
        dataset: Dataset, item_ids: List[str]
    ) -> Dataset:
        """
        Returns the dataset from which datasetitems of the received item_ids have been removed.

        :param dataset: Dataset, which consists of datasetitems
        :param item_ids: a list with datasetitem ids to be deleted
        :return Dataset from which items to be deleted have been removed
        """
        if len(item_ids) > 0:
            id_subset = _get_item_subset(dataset)
            for id in item_ids:
                dataset.remove(id, id_subset[id])
        return dataset


class DeleteAnnotation(Transform, CliPlugin):
    """
    DeleteAnnotation that supports deleting annotations with errors in items.
    """

    @staticmethod
    def delete_dataset_annotations(
        dataset: Dataset, item_ids: List[str]
    ) -> Dataset:
        """
        Returns the dataset with the annotations removed in the datasetitems of the received item_ids.

        :param dataset: Dataset, which consists of datasetitems
        :param item_ids: a list with datasetitem ids to delete annotations
        :return Dataset with the annotations removed in the datasetitems
        """
        if len(item_ids) > 0:
            id_subset = _get_item_subset(dataset)
            for id in item_ids:
                item = dataset.get(id, id_subset[id])
                item.annotations = []
        return dataset


class DeleteAttribute(Transform, CliPlugin):
    """
    DeleteAttribute that supports deleting attributes with errors in items.
    """

    @staticmethod
    def delete_dataset_attributes(
        dataset: Dataset, item_ids: List[str]
    ) -> Dataset:
        """
        Returns the dataset with the attributes removed in the datasetitems of the received item_ids.

        :param dataset: Dataset, which consists of datasetitems
        :param item_ids: a list with datasetitem ids to delete attributes
        :return Dataset with the attributes removed in the datasetitems
        """
        if len(item_ids) > 0:
            id_subset = _get_item_subset(dataset)
            for id in item_ids:
                item = dataset.get(id, id_subset[id])
                item.attributes = {}

                for anno in item.annotations:
                    if isinstance(anno, Label):
                        anno.attributes = {}
        return dataset
