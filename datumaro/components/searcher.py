# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Optional, Union, List

import numpy as np

from datumaro.components.dataset import IDataset
from datumaro.components.extractor import DatasetItem
from datumaro.components.model_inference import inference


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1] # max inner product value
    distH = 0.5 * (q - B1@B2.transpose())
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

    def unpack_hash_key(self, hash_key: List):
        hash_key_list = [hash_key[i:i+2] for i in range(0, len(hash_key), 2)]
        hash_key = np.array([int(s, 16) for s in hash_key_list], dtype='uint8')
        hash_key = np.unpackbits(hash_key, axis=-1)
        return hash_key

    def search_topk(self, query: Union[DatasetItem, str], topk: Optional[int]=None):
        """
        Search topk similar results based on hamming distance for query DatasetItem
        """
        if not topk:
            topk = self._topk

        if isinstance(query, DatasetItem):
            query_key = query.hash_key[0]
        elif isinstance(query, str):
            query_key = inference(query)[0]
        else:
            raise ValueError("Query should be DatasetItem or string")

        query_key = self.unpack_hash_key(query_key)

        retrieval_keys = []
        item_list = []
        for datasetitem in self._dataset:
            hash_key = datasetitem.hash_key[0]
            hash_key = self.unpack_hash_key(hash_key)
            retrieval_keys.append(hash_key)
            item_list.append(datasetitem)
        
        retrieval_keys = np.stack(retrieval_keys, axis=0)

        logits = calculate_hamming(query_key, retrieval_keys)
        ind = np.argsort(logits)

        item_list =np.array(item_list)[ind]
        result = item_list[:topk].tolist()

        return result
