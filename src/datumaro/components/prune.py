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
from datumaro.plugins.explorer import ExplorerLauncher
from datumaro.util.hashkey_util import (
    calculate_hamming,
    format_templates,
    select_uninferenced_dataset,
    templates,
)


def match_num_item_for_cluster(ratio, dataset_len, cluster_num_item_list):
    total_num_selected_item = math.ceil(dataset_len * ratio)

    cluster_weights = np.array(cluster_num_item_list) / sum(cluster_num_item_list)
    norm_cluster_num_item_list = (cluster_weights * total_num_selected_item).astype(int)
    remaining_items = total_num_selected_item - sum(norm_cluster_num_item_list)

    if remaining_items > 0:
        zero_cluster_indexes = np.where(norm_cluster_num_item_list == 0)[0]
        add_clust_dist = np.sort(cluster_weights[zero_cluster_indexes])[::-1][:remaining_items]

        for dist in set(add_clust_dist):
            indices = np.where(cluster_weights == dist)[0]
            for index in indices:
                norm_cluster_num_item_list[index] += 1

    elif remaining_items < 0:
        diff_num_item_list = np.argsort(cluster_weights - norm_cluster_num_item_list)
        for diff_idx in diff_num_item_list[: abs(remaining_items)]:
            norm_cluster_num_item_list[diff_idx] -= 1

    return norm_cluster_num_item_list.tolist()


def random_select(ratio, num_centers, database_keys, labels, item_list, source):
    """
    Select items randomly from the dataset.
    """
    random.seed(0)
    dataset_len = len(item_list)
    num_selected_item = math.ceil(dataset_len * ratio)
    random_indices = random.sample(range(dataset_len), num_selected_item)
    selected_items = [item_list[idx] for idx in random_indices]
    return selected_items, None


def centroid(ratio, num_centers, database_keys, labels, item_list, source):
    """
    Select items through clustering with centers targeting the desired number.
    """
    num_selected_centers = math.ceil(len(item_list) * ratio)
    kmeans = KMeans(n_clusters=num_selected_centers, random_state=0)
    clusters = kmeans.fit_predict(database_keys)
    cluster_centers = kmeans.cluster_centers_
    cluster_ids = np.unique(clusters)

    selected_items = []
    dist_tuples = []
    for cluster_id in cluster_ids:
        cluster_center = cluster_centers[cluster_id]
        cluster_items_idx = np.where(clusters == cluster_id)[0]
        num_selected_items = 1
        cluster_items = database_keys[cluster_items_idx,]
        dist = calculate_hamming(cluster_center, cluster_items)
        ind = np.argsort(dist)
        item_idx_list = cluster_items_idx[ind]
        for i, idx in enumerate(item_idx_list[:num_selected_items]):
            selected_items.append(item_list[idx])
            dist_tuples.append((cluster_id, item_list[idx].id, item_list[idx].subset, dist[ind][i]))
    return selected_items, dist_tuples


def clustered_random(ratio, num_centers, database_keys, labels, item_list, source):
    """
    Select items through clustering and choose randomly within each cluster.
    """
    kmeans = KMeans(n_clusters=num_centers, random_state=0)
    clusters = kmeans.fit_predict(database_keys)
    cluster_ids, cluster_num_item_list = np.unique(clusters, return_counts=True)

    norm_cluster_num_item_list = match_num_item_for_cluster(
        ratio, len(database_keys), cluster_num_item_list
    )

    selected_items = []
    random.seed(0)
    for i, cluster_id in enumerate(cluster_ids):
        cluster_items_idx = np.where(clusters == cluster_id)[0]
        num_selected_items = norm_cluster_num_item_list[i]
        random.shuffle(cluster_items_idx)
        selected_items.extend(item_list[idx] for idx in cluster_items_idx[:num_selected_items])
    return selected_items, None


def query_clust(ratio, num_centers, database_keys, labels, item_list, source):
    """
    Select items through clustering with inits that imply each label.
    """
    center_dict = {i: None for i in range(1, num_centers)}
    for item in item_list:
        for anno in item.annotations:
            if isinstance(anno, Label):
                label_ = anno.label
                if center_dict.get(label_) is None:
                    center_dict[label_] = item
        if all(center_dict.values()):
            break

    item_id_list = [item.id.split("/")[-1] for item in item_list]
    centroids = [
        database_keys[item_id_list.index(i.id.split(":")[-1])]
        for i in list(center_dict.values())
        if i
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
    for i, cluster_id in enumerate(cluster_ids):
        cluster_center = cluster_centers[cluster_id]
        cluster_items_idx = np.where(clusters == cluster_id)[0]
        num_selected_item = norm_cluster_num_item_list[i]

        cluster_items = database_keys[cluster_items_idx]
        dist = calculate_hamming(cluster_center, cluster_items)
        ind = np.argsort(dist)
        item_idx_list = cluster_items_idx[ind]
        for i, idx in enumerate(item_idx_list[:num_selected_item]):
            selected_items.append(item_list[idx])
            dist_tuples.append((cluster_id, item_list[idx].id, item_list[idx].subset, dist[ind][i]))
    return selected_items, dist_tuples


def entropy(ratio, num_centers, database_keys, labels, item_list, source):
    """
    Select items through clustering and choose them based on label entropy in each cluster.
    """
    kmeans = KMeans(n_clusters=num_centers, random_state=0)
    clusters = kmeans.fit_predict(database_keys)

    cluster_ids, cluster_num_item_list = np.unique(clusters, return_counts=True)
    norm_cluster_num_item_list = match_num_item_for_cluster(
        ratio, len(database_keys), cluster_num_item_list
    )

    selected_item_indexes = []
    for cluster_id, num_selected_item in zip(cluster_ids, norm_cluster_num_item_list):
        cluster_items_idx = np.where(clusters == cluster_id)[0]

        cluster_classes = np.array(labels)[cluster_items_idx]
        _, inv, cnts = np.unique(cluster_classes, return_inverse=True, return_counts=True)
        weights = 1 / cnts
        probs = weights[inv]
        probs /= probs.sum()

        choices = np.random.choice(len(inv), size=num_selected_item, p=probs, replace=False)
        selected_item_indexes.extend(cluster_items_idx[choices])

    selected_items = np.array(item_list)[selected_item_indexes].tolist()
    return selected_items, None


def ndr_select(ratio, num_centers, database_keys, labels, item_list, source):
    """
    Select items based on NDR among each subset.
    """
    subset_lists = list(source.subsets().keys())

    selected_items = []
    for subset_ in subset_lists:
        subset_len = len(source.get_subset(subset_))
        num_selected_subset_item = math.ceil(subset_len * (1 - ratio))
        ndr_result = ndr.NDR(source, working_subset=subset_, num_cut=num_selected_subset_item)
        selected_items.extend(ndr_result.get_subset(subset_))

    return selected_items, None


class Prune:
    def __init__(
        self,
        *datasets: Dataset,
        cluster_method: str = "random",
        hash_type: str = "img",
    ) -> None:
        """
        Prune make a representative and manageable subset.
        """
        self._dataset = datasets[0] if isinstance(datasets, tuple) else datasets[0]
        self._cluster_method = cluster_method
        self._hash_type = hash_type

        self._model = None
        self._text_model = None
        self._num_centers = None

        self._database_keys = None
        self._item_list = []
        self._labels = []

        self._prepare_data()

    def _prepare_data(self):
        if self._hash_type == "txt":
            category_dict = self._prompting()

        if self._cluster_method == "random":
            self._item_list = list(self._dataset)
        else:
            datasets_to_infer = select_uninferenced_dataset(self._dataset)
            datasets = self._compute_hash_key([self._dataset], [datasets_to_infer])[0]

            for category in datasets.categories().values():
                if isinstance(category, LabelCategories):
                    self._num_centers = len(category._indices.keys())

            for item in datasets:
                for annotation in item.annotations:
                    if isinstance(annotation, Label):
                        self._labels.append(annotation.label)
                    if isinstance(annotation, HashKey):
                        hash_key = annotation.hash_key[0]
                        if self._hash_type == "txt":
                            inputs = category_dict.get(item.annotations[0].label)
                            if isinstance(inputs, List):
                                inputs = " ".join(inputs)
                            hash_key_txt = self._text_model.infer_text(inputs)
                            hash_key = np.concatenate([hash_key, hash_key_txt])
                        hash_key = np.unpackbits(hash_key, axis=-1)
                        if self._database_keys is None:
                            self._database_keys = hash_key.reshape(1, -1)
                        else:
                            self._database_keys = np.concatenate(
                                (self._database_keys, hash_key.reshape(1, -1)), axis=0
                            )
                self._item_list.append(item)

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

    def _compute_hash_key(self, datasets, datasets_to_infer):
        for dataset_to_infer in datasets_to_infer:
            if dataset_to_infer:
                dataset_to_infer.run_model(self.model, append_annotation=True)
        for dataset, dataset_to_infer in zip(datasets, datasets_to_infer):
            updated_items = [
                dataset.get(item.id, item.subset).wrap(annotations=item.annotations)
                for item in dataset_to_infer
            ]
            dataset.update(updated_items)
        return datasets

    def get_pruned(self, ratio: float = 0.5) -> Dataset:
        method = {
            "random": random_select,
            "cluster_random": clustered_random,
            "centroid": centroid,
            "query_clust": query_clust,
            "entropy": entropy,
            "ndr": ndr_select,
        }

        selected_items, dist_tuples = method[self._cluster_method](
            ratio=ratio,
            num_centers=self._num_centers,
            labels=self._labels,
            database_keys=self._database_keys,
            item_list=self._item_list,
            source=self._dataset,
        )

        result_dataset = Dataset(media_type=self._dataset.media_type())
        result_dataset._source_path = self._dataset._source_path
        result_dataset.define_categories(self._dataset.categories())
        for item in selected_items:
            result_dataset.put(item)

        if dist_tuples:
            for center, id_, subset_, d in dist_tuples:
                log.info(f"item {id_} of subset {subset_} has distance {d} for cluster {center}")

        log.info(f"Pruned dataset with {ratio} from {len(self._dataset)} to {len(result_dataset)}")
        return result_dataset
