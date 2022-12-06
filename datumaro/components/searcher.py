# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Optional, Union

import numpy as np

from datumaro.components.annotation import HashKey
from datumaro.components.dataset import IDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.model_inference import hash_inference


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1]  # max inner product value
    distH = 0.5 * (q - B1 @ B2.transpose())
    return distH


class Searcher:
    def __init__(
        self,
        dataset: IDataset,
        topk: int = 10,
    ) -> None:
        """
        Searcher for Datumaro dataitems

        Parameters
        ----------
        dataset:
            Datumaro dataset to search similar dataitem.
        topk:
            Number of images
        """
        self._dataset = dataset
        self._topk = topk

        retrieval_keys = []
        item_list = []
        for datasetitem in self._dataset:
            hash_key = None
            # if hash_key=None, it means not inferenced
            for annotation in datasetitem.annotations:
                if isinstance(annotation, HashKey):
                    hash_key = annotation.hash_key
                    break

            if not hash_key:
                hash_key = datasetitem.set_hash_key

            # if hash_key is empty, it means data is None or not proper data type
            if hash_key:
                hash_key = hash_key[0]
                hash_key = self.unpack_hash_key(hash_key)
                retrieval_keys.append(hash_key)
                item_list.append(datasetitem)

        self._retrieval_keys = retrieval_keys
        self._item_list = item_list

    def unpack_hash_key(self, hash_key: List):
        hash_key_list = [hash_key[i : i + 2] for i in range(0, len(hash_key), 2)]
        hash_key = np.array([int(s, 16) for s in hash_key_list], dtype="uint8")
        hash_key = np.unpackbits(hash_key, axis=-1)
        return hash_key

    def search_topk(self, query: Union[DatasetItem, str], topk: Optional[int] = None):
        """
        Search topk similar results based on hamming distance for query DatasetItem
        """
        if not topk:
            topk = self._topk

        if isinstance(query, DatasetItem):
            query_key = None
            for annotation in query.annotations:
                if isinstance(annotation, HashKey):
                    query_key = annotation.hash_key
                    break

            if not query_key:
                query_key = query.set_hash_key
        elif isinstance(query, str):
            query_key = hash_inference(query)
        else:
            raise ValueError("Query should be DatasetItem or string")

        if not query_key:
            # media.data is None case
            raise ValueError("Query should have hash_key")

        query_key = self.unpack_hash_key(query_key[0])

        retrieval_keys = np.stack(self._retrieval_keys, axis=0)

        logits = calculate_hamming(query_key, retrieval_keys)
        ind = np.argsort(logits)

        item_list = np.array(self._item_list)[ind]
        result = item_list[:topk].tolist()

        return result
