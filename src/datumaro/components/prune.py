# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import copy
import logging as log
import math
import random
from typing import List

import numpy as np
from sklearn.cluster import KMeans

import datumaro.plugins.ndr as ndr
from datumaro.components.annotation import HashKey, Label, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.media import MediaElement
from datumaro.plugins.explorer import ExplorerLauncher
from datumaro.util.hashkey_util import (
    calculate_hamming,
    format_templates,
    select_uninferenced_dataset,
    templates,
)


def match_num_item_for_cluster(ratio, dataset_len, cluster_num_item_list):
    total_num_selected_item = math.ceil(dataset_len * ratio)

    cluster_num_item_list = [
        float(i) / sum(cluster_num_item_list) * total_num_selected_item
        for i in cluster_num_item_list
    ]
    norm_cluster_num_item_list = [int(np.round(i)) for i in cluster_num_item_list]
    zero_cluster_indexes = list(np.where(np.array(norm_cluster_num_item_list) == 0)[0])
    add_clust_dist = np.sort(np.array(cluster_num_item_list)[zero_cluster_indexes])[::-1][
        : total_num_selected_item - sum(norm_cluster_num_item_list),
    ]
    for dist in set(add_clust_dist):
        indices = [i for i, x in enumerate(cluster_num_item_list) if x == dist]
        for index in indices:
            norm_cluster_num_item_list[index] += 1
    if total_num_selected_item > sum(norm_cluster_num_item_list):
        diff_num_item_list = np.argsort(
            np.array(
                [x - norm_cluster_num_item_list[i] for i, x in enumerate(cluster_num_item_list)]
            )
        )[::-1]
        for diff_idx in diff_num_item_list[
            : total_num_selected_item - sum(norm_cluster_num_item_list)
        ]:
            norm_cluster_num_item_list[diff_idx] += 1
    elif total_num_selected_item < sum(norm_cluster_num_item_list):
        diff_num_item_list = np.argsort(
            np.array(
                [x - norm_cluster_num_item_list[i] for i, x in enumerate(cluster_num_item_list)]
            )
        )
        for diff_idx in diff_num_item_list[
            : sum(norm_cluster_num_item_list) - total_num_selected_item
        ]:
            norm_cluster_num_item_list[diff_idx] -= 1
    return norm_cluster_num_item_list


def random_select(ratio, num_centers, database_keys, labels, item_list, source):
    """
    Select item through naive random method among dataset.
    """
    random.seed(0)
    dataset_len = len(item_list)

    num_selected_item = math.ceil(dataset_len * ratio)
    random_list = random.sample(range(dataset_len), num_selected_item)
    removed_items = list(range(dataset_len))
    for idx in random_list:
        removed_items.remove(idx)
    selected_items = (np.array(item_list)[random_list]).tolist()
    return selected_items, None


def centroid(ratio, num_centers, database_keys, labels, item_list, source):
    """
    Select item through clustering with centers that targeted result number.
    """
    num_centers = math.ceil(len(item_list) * ratio)
    kmeans = KMeans(n_clusters=num_centers, random_state=0)
    clusters = kmeans.fit_predict(database_keys)
    cluster_centers = kmeans.cluster_centers_
    cluster_ids = np.unique(clusters)

    selected_items = []
    dist_tuples = []
    for cluster_id in cluster_ids:
        cluster_center = cluster_centers[cluster_id]
        cluster_items_idx = np.where(clusters == cluster_id)[0]
        num_selected_item = 1
        cluster_items = database_keys[cluster_items_idx,]
        dist = calculate_hamming(cluster_center, cluster_items)
        ind = np.argsort(dist)
        item_idx_list = cluster_items_idx[ind]
        for i, idx in enumerate(item_idx_list[:num_selected_item]):
            selected_items.append(item_list[idx])
            dist_tuples.append((cluster_id, item_list[idx].id, item_list[idx].subset, dist[ind][i]))
    return selected_items, dist_tuples


def clustered_random(ratio, num_centers, database_keys, labels, item_list, source):
    """
    Select item through clustering and choose randomly in each cluster.
    """
    kmeans = KMeans(n_clusters=num_centers, random_state=0)
    clusters = kmeans.fit_predict(database_keys)
    cluster_ids, cluster_num_item_list = np.unique(clusters, return_counts=True)

    norm_cluster_num_item_list = match_num_item_for_cluster(
        ratio, len(database_keys), cluster_num_item_list
    )

    selected_items = []
    for cluster_id in cluster_ids:
        random.seed(0)
        cluster_items_idx = np.where(clusters == cluster_id)[0]

        num_selected_item = norm_cluster_num_item_list[cluster_id]
        random.shuffle(cluster_items_idx)
        for idx in cluster_items_idx[:num_selected_item]:
            selected_items.append(item_list[idx])
    return selected_items, None


def query_clust(ratio, num_centers, database_keys, labels, item_list, source):
    """
    Select item through clustering with inits that implys each label.
    """
    center_dict = {i: [] for i in range(1, num_centers)}
    for item in item_list:
        for anno in item.annotations:
            if isinstance(anno, Label):
                label_ = anno.label
                if not center_dict.get(label_):
                    center_dict[label_] = item
            if all(center_dict.values()):
                break

    for item in item_list:
        for anno in item.annotations:
            if isinstance(anno, Label):
                label_ = anno.label
                if not center_dict.get(label_):
                    center_dict[label_] = item
        if all(center_dict.values()):
            break
    item_id_list = [item.id.split("/")[-1] for item in item_list]
    centroids = [
        database_keys[item_id_list.index(i.id.split(":")[-1])] for i in list(center_dict.values())
    ]
    kmeans = KMeans(n_clusters=num_centers, n_init=1, init=centroids, random_state=0)

    clusters = kmeans.fit_predict(database_keys)
    cluster_centers = kmeans.cluster_centers_
    cluster_ids, cluster_num_item_list = np.unique(clusters, return_counts=True)

    norm_cluster_num_item_list = match_num_item_for_cluster(
        ratio, len(database_keys), cluster_num_item_list
    )

    selected_items = []
    dist_tuples = []
    for cluster_id in cluster_ids:
        cluster_center = cluster_centers[cluster_id]
        cluster_items_idx = np.where(clusters == cluster_id)[0]
        num_selected_item = norm_cluster_num_item_list[cluster_id]

        cluster_items = database_keys[cluster_items_idx,]
        dist = calculate_hamming(cluster_center, cluster_items)
        ind = np.argsort(dist)
        item_idx_list = cluster_items_idx[ind]
        for i, idx in enumerate(item_idx_list[:num_selected_item]):
            selected_items.append(item_list[idx])
            dist_tuples.append((cluster_id, item_list[idx].id, item_list[idx].subset, dist[ind][i]))
    return selected_items, dist_tuples


def entropy(ratio, num_centers, database_keys, labels, item_list, source):
    """
    Select item through clustering and choose it based on label entropy in each cluster.
    """
    kmeans = KMeans(n_clusters=num_centers, random_state=0)
    clusters = kmeans.fit_predict(database_keys)
    cluster_ids, cluster_num_item_list = np.unique(clusters, return_counts=True)

    norm_cluster_num_item_list = match_num_item_for_cluster(
        ratio, len(database_keys), cluster_num_item_list
    )

    selected_item_indexes = []
    for cluster_id in cluster_ids:
        cluster_items_idx = np.where(clusters == cluster_id)[0]
        num_selected_item = norm_cluster_num_item_list[cluster_id]

        cluster_classes = np.array(labels)[cluster_items_idx]
        _, inv, cnts = np.unique(cluster_classes, return_inverse=True, return_counts=True)
        weights = 1 / cnts
        probs = weights[inv]
        probs = probs / probs.sum()

        choices = np.random.choice(range(len(inv)), size=num_selected_item, p=probs, replace=False)
        selected_item_indexes += cluster_items_idx[choices].tolist()
    selected_items = (np.array(item_list)[selected_item_indexes]).tolist()
    return selected_items, None


def ndr_select(ratio, num_centers, database_keys, labels, item_list, source):
    """
    Select item based on NDR among each subset.
    """
    subset_lists = list(source.subsets().keys())

    selected_items = []
    for subset_ in subset_lists:
        subset_len = len(source.get_subset(subset_))
        num_selected_subset_item = math.ceil(subset_len * ratio)
        ndr_result = ndr.NDR(source, working_subset=subset_, num_cut=num_selected_subset_item)
        for item in ndr_result.get_subset(subset_):
            selected_items.append(item)
    return selected_items, None


class Prune:
    def __init__(
        self,
        *dataset: Dataset,
        cluster_method: str = "random",
        hash_type: str = "img",
    ) -> None:
        """
        Prune make a representative and manageable subset.
        """
        if isinstance(dataset, tuple):
            try:
                self._dataset = copy.deepcopy(dataset[0]._dataset)
            except AttributeError:
                self._dataset = dataset[0]

        self._cluster_method = cluster_method
        self._hash_type = hash_type

        self._model = None
        self._text_model = None
        self._num_centers = None

        database_keys = None
        item_list = []
        labels = []

        if self._hash_type == "txt":
            category_dict = self.prompting(dataset)

        if self._cluster_method == "random":
            for item in self._dataset:
                item_list.append(item)
        else:
            datasets_to_infer = select_uninferenced_dataset(self._dataset)
            dataset = self.compute_hash_key([self._dataset], [datasets_to_infer])[0]

            # check number of category
            for category in dataset.categories().values():
                if isinstance(category, LabelCategories):
                    num_centers = len(list(category._indices.keys()))
            self._num_centers = num_centers

            for item in dataset:
                for annotation in item.annotations:
                    if isinstance(annotation, Label):
                        labels.append(annotation.label)
                    if isinstance(annotation, HashKey):
                        hash_key = annotation.hash_key[0]
                        if self._hash_type == "txt":
                            inputs = category_dict.get(item.annotations[0].label)
                            if isinstance(inputs, List):
                                inputs = (" ").join(inputs)
                            hash_key_txt = self.text_model.infer_text(inputs)
                            hash_key = np.concatenate([hash_key, hash_key_txt])
                        hash_key = np.unpackbits(hash_key, axis=-1)
                        if database_keys is None:
                            database_keys = hash_key.reshape(1, -1)
                        else:
                            database_keys = np.concatenate(
                                (database_keys, hash_key.reshape(1, -1)), axis=0
                            )
                item_list.append(item)

        self._database_keys = database_keys
        self._item_list = item_list
        self._labels = labels

    @property
    def model(self):
        if self._model is None:
            self._model = ExplorerLauncher(model_name="clip_visual_ViT-B_32")
        return self._model

    @property
    def text_model(self):
        if self._text_model is None:
            self._text_model = ExplorerLauncher(model_name="clip_text_ViT-B_32")
        return self._text_model

    def prompting(self):
        category_dict = {}
        detected_format = self._dataset.format
        template = format_templates.get(detected_format, templates)
        for label, indice in list(self._dataset.categories().values())[0]._indices.items():
            category_dict[indice] = [temp.format(label) for temp in template]
        return category_dict

    def compute_hash_key(self, datasets, datasets_to_infer):
        for dataset_to_infer in datasets_to_infer:
            if len(dataset_to_infer) > 0:
                dataset_to_infer.run_model(self.model, append_annotation=True)
        for dataset, dataset_to_infer in zip(datasets, datasets_to_infer):
            updated_items = []
            for item in dataset_to_infer:
                item_ = dataset.get(item.id, item.subset)
                updated_items.append(item_.wrap(annotations=item.annotations))
            dataset.update(updated_items)
        return datasets

    def get_pruned(self, ratio: float) -> None:
        method = {
            "random": random_select,
            "cluster_random": clustered_random,
            "centroid": centroid,
            "query_clust": query_clust,
            "entropy": entropy,
            "ndr": ndr_select,
        }

        removed_items, dist_tuples = method[self._cluster_method](
            ratio=ratio,
            num_centers=self._num_centers,
            labels=self._labels,
            database_keys=self._database_keys,
            item_list=self._item_list,
            source=self._dataset,
        )

        result_dataset = Dataset(media_type=MediaElement)
        for item in removed_items:
            result_dataset.put(item)

        if dist_tuples:
            for center, id_, subset_, d in dist_tuples:
                log.info(f"item {id_} of subset {subset_} has distance {d} for cluster {center}")

        log.info(f"Pruned dataset with {ratio} from {len(self._dataset)} to {len(result_dataset)}")
        return result_dataset
