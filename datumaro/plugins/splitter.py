# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import numpy as np

from datumaro.components.extractor import (
    Transform,
    AnnotationType,
    DEFAULT_SUBSET_NAME,
)

NEAR_ZERO = 1e-7


class _TaskSpecificSplit(Transform):
    def __init__(self, dataset, splits, seed):
        super().__init__(dataset)

        snames, sratio = self._validate_splits(splits)

        self._snames = snames
        self._sratio = sratio

        self._seed = seed

        self._subsets = {"train", "val", "test"}  # output subset names
        self._parts = []
        self._length = "parent"

        self._initialized = False

    def _set_parts(self, by_splits):
        self._parts = []
        for subset in self._subsets:
            self._parts.append((set(by_splits[subset]), subset))

    @staticmethod
    def _get_uniq_annotations(dataset):
        annotations = []
        for _, item in enumerate(dataset):
            labels = []
            for ann in item.annotations:
                if ann.type == AnnotationType.label:
                    labels.append(ann)
            assert (
                len(labels) == 1
            ), "Expected exact one label for a DatasetItem"
            ann = labels[0]
            annotations.append(ann)
        return annotations

    @staticmethod
    def _validate_splits(splits, valid=None):
        snames = []
        ratios = []
        if valid is None:
            valid = ["train", "val", "test"]
        for subset, ratio in splits:
            assert subset in valid, (
                "Subset name \
                must be one of %s, but got %s"
                % (valid, subset)
            )
            assert ratio >= 0.0 and ratio <= 1.0, (
                "Ratio is expected to be \
                in the range [0, 1], but got %s for %s"
                % (ratio, subset)
            )
            snames.append(subset)
            ratios.append(float(ratio))
        ratios = np.array(ratios)

        total_ratio = np.sum(ratios)
        if not abs(total_ratio - 1.0) <= NEAR_ZERO:
            raise Exception(
                "Sum of ratios is expected to be 1, got %s, which is %s"
                % (splits, total_ratio)
            )
        return snames, ratios

    @staticmethod
    def _get_required(ratio):
        min_value = np.max(ratio)
        for i in ratio:
            if i < min_value and i > NEAR_ZERO:
                min_value = i
        required = int(np.around(1.0) / min_value)
        return required

    @staticmethod
    def _get_sections(dataset_size, ratio):
        n_splits = [int(np.around(dataset_size * r)) for r in ratio[:-1]]
        n_splits.append(dataset_size - np.sum(n_splits))

        # if there are splits with zero samples even if ratio is not 0,
        # borrow one from the split who has one or more.
        for ii, num_split in enumerate(n_splits):
            if num_split == 0 and ratio[ii] > NEAR_ZERO:
                midx = np.argmax(n_splits)
                if n_splits[midx] > 0:
                    n_splits[ii] += 1
                    n_splits[midx] -= 1
        sections = np.add.accumulate(n_splits[:-1])
        return sections

    @staticmethod
    def _group_by_attr(items):
        """
        Args:
            items: list of (idx, ann). ann is the annotation from Label object.
        Returns:
            by_attributes: dict of { combination-of-attrs : list of index }
        """
        # group by attributes
        by_attributes = dict()
        for idx, ann in items:
            attributes = tuple(sorted(ann.attributes.items()))
            if attributes not in by_attributes:
                by_attributes[attributes] = []
            by_attributes[attributes].append(idx)
        return by_attributes

    def _split_by_attr(
        self, datasets, snames, ratio, out_splits, dataset_key="label"
    ):
        required = self._get_required(ratio)
        for key, items in datasets.items():
            np.random.shuffle(items)
            by_attributes = self._group_by_attr(items)
            for attributes, indice in by_attributes.items():
                gname = "%s: %s, attrs: %s" % (dataset_key, key, attributes)
                splits = self._split_indice(indice, gname, ratio, required)
                for subset, split in zip(snames, splits):
                    if len(split) > 0:
                        out_splits[subset].extend(split)

    def _split_indice(self, indice, group_name, ratio, required):
        filtered_size = len(indice)
        if required > filtered_size:
            log.warning("Not enough samples for group, '%s'" % group_name)
        sections = self._get_sections(filtered_size, ratio)
        splits = np.array_split(indice, sections)
        return splits

    def _find_split(self, index):
        for subset_indices, subset in self._parts:
            if index in subset_indices:
                return subset
        return DEFAULT_SUBSET_NAME  # all the possible remainder --> default

    def _split_dataset(self):
        raise NotImplementedError()

    def __iter__(self):
        # lazy splitting
        if self._initialized is False:
            self._split_dataset()
            self._initialized = True
        for i, item in enumerate(self._extractor):
            yield self.wrap_item(item, subset=self._find_split(i))


class ClassificationSplit(_TaskSpecificSplit):
    """
    Splits dataset into train/val/test set in class-wise manner. |n
    |n
    Notes:|n
    - Single label is expected for each DatasetItem.|n
    - If there are not enough images in some class or attributes group,
      the split ratio can't be guaranteed.|n
    """

    def __init__(self, dataset, splits, seed=None):
        """
        Parameters
        ----------
        dataset : Dataset
        splits : list
            A list of (subset(str), ratio(float))
            Subset is expected to be one of ["train", "val", "test"].
            The sum of ratios is expected to be 1.
        seed : int, optional
        """
        super().__init__(dataset, splits, seed)

    def _split_dataset(self):
        np.random.seed(self._seed)

        # support only single label for a DatasetItem
        # 1. group by label
        by_labels = dict()
        annotations = self._get_uniq_annotations(self._extractor)
        for idx, ann in enumerate(annotations):
            label = ann.label if hasattr(ann, "label") else None
            if label not in by_labels:
                by_labels[label] = []
            by_labels[label].append((idx, ann))

        by_splits = dict()
        for subset in self._subsets:
            by_splits[subset] = []

        # 2. group by attributes
        self._split_by_attr(by_labels, self._snames, self._sratio, by_splits)
        self._set_parts(by_splits)


class MatchingReIDSplit(_TaskSpecificSplit):
    """
    Splits dataset for matching, especially re-id task.|n
    First, splits dataset into 'train+val' and 'test' sets by person id.|n
    Note that this splitting is not by DatasetItem. |n
    Then, tags 'test' into 'gallery'/'query' in class-wise random manner.|n
    Then, splits 'train+val' into 'train'/'val' sets in the same way.|n
    Therefore, the final subsets would be 'train', 'val', 'test'. |n
    And 'gallery', 'query' are tagged using anntoation group.|n
    You can get the 'gallery' and 'query' sets using 'get_subset_by_group'.|n
    Notes:|n
    - Single label is expected for each DatasetItem.|n
    - Each label is expected to have attribute representing the person id. |n
    """

    _group_map = dict()

    def __init__(
        self, dataset, splits, test_splits, pid_name="PID", seed=None
    ):
        """
        Parameters
        ----------
        dataset : Dataset
        splits : list
            A list of (subset(str), ratio(float))
            Subset is expected to be one of ["train", "val", "test"].
            The sum of ratios is expected to be 1.
        test_splits : list
            A list of (subset(str), ratio(float))
            Subset is expected to be one of ["gallery", "query"].
            The sum of ratios is expected to be 1.
        pid_name: str
            attribute name representing the person id. (default: PID)
        seed : int, optional
        """
        super().__init__(dataset, splits, seed)

        self._test_splits = test_splits
        self._pid_name = pid_name

    def _split_dataset(self):
        np.random.seed(self._seed)

        id_snames, id_ratio = self._snames, self._sratio

        pid_name = self._pid_name
        dataset = self._extractor

        groups = set()

        # group by PID(pid_name)
        by_pid = dict()
        annotations = self._get_uniq_annotations(dataset)
        for idx, ann in enumerate(annotations):
            attributes = dict(ann.attributes.items())
            assert pid_name in attributes, (
                "'%s' is expected as attribute name" % pid_name
            )
            person_id = attributes[pid_name]
            if person_id not in by_pid:
                by_pid[person_id] = []
            by_pid[person_id].append((idx, ann))
            groups.add(ann.group)

        max_group_id = max(groups)
        self._group_map["gallery"] = max_group_id + 1
        self._group_map["query"] = max_group_id + 2

        required = self._get_required(id_ratio)
        if len(by_pid) < required:
            log.warning(
                "There's not enough IDs, which is {}, \
                so train/val/test ratio can't be guaranteed."
                % len(by_pid)
            )

        # 1. split dataset into trval and test
        #    IDs in test set should not exist in train/val set.
        test = id_ratio[id_snames.index("test")] if "test" in id_snames else 0
        if test > NEAR_ZERO:  # has testset
            split_ratio = np.array([test, 1.0 - test])
            person_ids = list(by_pid.keys())
            np.random.shuffle(person_ids)
            sections = self._get_sections(len(person_ids), split_ratio)
            splits = np.array_split(person_ids, sections)
            testset = {pid: by_pid[pid] for pid in splits[0]}
            trval = {pid: by_pid[pid] for pid in splits[1]}

            # follow the ratio of datasetitems as possible.
            # naive heuristic: exchange the best item one by one.
            expected_count = int(len(self._extractor) * split_ratio[0])
            testset_total = int(np.sum([len(v) for v in testset.values()]))
            self._rebalancing(testset, trval, expected_count, testset_total)
        else:
            testset = dict()
            trval = by_pid

        by_splits = dict()
        for subset in self._subsets:
            by_splits[subset] = []

        # 2. split 'test' into 'gallery' and 'query'
        if len(testset) > 0:
            for person_id, items in testset.items():
                indice = [idx for idx, _ in items]
                by_splits["test"].extend(indice)

            valid = ["gallery", "query"]
            test_splits = self._test_splits
            test_snames, test_ratio = self._validate_splits(test_splits, valid)
            by_groups = {s: [] for s in test_snames}
            self._split_by_attr(
                testset,
                test_snames,
                test_ratio,
                by_groups,
                dataset_key=pid_name,
            )

            # tag using group
            for idx, item in enumerate(self._extractor):
                for subset, split in by_groups.items():
                    if idx in split:
                        group_id = self._group_map[subset]
                        item.annotations[0].group = group_id
                        break

        # 3. split 'trval' into  'train' and 'val'
        trval_snames = ["train", "val"]
        trval_ratio = []
        for subset in trval_snames:
            if subset in id_snames:
                val = id_ratio[id_snames.index(subset)]
            else:
                val = 0.0
            trval_ratio.append(val)
        trval_ratio = np.array(trval_ratio)
        total_ratio = np.sum(trval_ratio)
        if total_ratio < NEAR_ZERO:
            trval_splits = list(zip(["train", "val"], trval_ratio))
            log.warning(
                "Sum of ratios is expected to be positive,\
                got %s, which is %s"
                % (trval_splits, total_ratio)
            )
        else:
            trval_ratio /= total_ratio  # normalize
            self._split_by_attr(
                trval,
                trval_snames,
                trval_ratio,
                by_splits,
                dataset_key=pid_name,
            )

        self._set_parts(by_splits)

    @staticmethod
    def _rebalancing(test, trval, expected_count, testset_total):
        diffs = dict()
        for id_test, items_test in test.items():
            count_test = len(items_test)
            for id_trval, items_trval in trval.items():
                count_trval = len(items_trval)
                diff = count_trval - count_test
                if diff == 0:
                    continue  # exchange has no effect
                if diff not in diffs:
                    diffs[diff] = [(id_test, id_trval)]
                else:
                    diffs[diff].append((id_test, id_trval))
        exchanges = []
        while True:
            target_diff = expected_count - testset_total
            # find nearest diff.
            keys = np.array(list(diffs.keys()))
            idx = (np.abs(keys - target_diff)).argmin()
            nearest = keys[idx]
            if abs(target_diff - nearest) >= abs(target_diff):
                break
            choice = np.random.choice(range(len(diffs[nearest])))
            pid_test, pid_trval = diffs[nearest][choice]
            testset_total += nearest
            new_diffs = dict()
            for diff, person_ids in diffs.items():
                new_list = []
                for id1, id2 in person_ids:
                    if id1 == pid_test or id2 == pid_trval:
                        continue
                    new_list.append((id1, id2))
                if len(new_list) > 0:
                    new_diffs[diff] = new_list
            diffs = new_diffs
            exchanges.append((pid_test, pid_trval))
        # exchange
        for pid_test, pid_trval in exchanges:
            test[pid_trval] = trval.pop(pid_trval)
            trval[pid_test] = test.pop(pid_test)

    def get_subset_by_group(self, group: str):
        available = list(self._group_map.keys())
        assert (
            group in self._group_map
        ), "Unknown group '%s', available groups: %s" % (
            group,
            available,
        )
        group_id = self._group_map[group]
        return self.select(lambda item: item.annotations[0].group == group_id)


class DetectionSplit(_TaskSpecificSplit):
    """
    Splits dataset into train/val/test set for detection task.|n
    For detection dataset, each image can have multiple bbox annotations.|n
    Since one DataItem can't be included in multiple subsets at the same time,
    the dataset can't be divided according to the bbox annotations.|n
    Thus, we split dataset based on DatasetItem
    while preserving label distribution as possible.|n
    |n
    Notes:|n
    - Each DatsetItem is expected to have one or more Bbox annotations.|n
    - Label annotations are ignored. We only focus on the Bbox annotations.|n
    """

    def __init__(self, dataset, splits, seed=None):
        """
        Parameters
        ----------
        dataset : Dataset
        splits : list
            A list of (subset(str), ratio(float))
            Subset is expected to be one of ["train", "val", "test"].
            The sum of ratios is expected to be 1.
        seed : int, optional
        """
        super().__init__(dataset, splits, seed)

    @staticmethod
    def _group_by_bbox_labels(dataset):
        by_labels = dict()
        for idx, item in enumerate(dataset):
            bbox_anns = []
            for ann in item.annotations:
                if ann.type == AnnotationType.bbox:
                    bbox_anns.append(ann)
            assert (
                len(bbox_anns) > 0
            ), "Expected more than one bbox annotation."
            for ann in bbox_anns:
                label = ann.label if hasattr(ann, "label") else None
                if label not in by_labels:
                    by_labels[label] = [(idx, ann)]
                else:
                    by_labels[label].append((idx, ann))
        return by_labels

    def _split_dataset(self):
        np.random.seed(self._seed)

        subsets, sratio = self._snames, self._sratio

        # 1. group by bbox label
        by_labels = self._group_by_bbox_labels(self._extractor)

        # 2. group by attributes
        by_combinations = dict()
        for label, items in by_labels.items():
            by_attributes = self._group_by_attr(items)
            for attributes, indice in by_attributes.items():
                gname = "label: %s, attributes: %s" % (label, attributes)
                by_combinations[gname] = indice

        # total number of GT samples per label-attr combinations
        n_combs = {k: len(v) for k, v in by_combinations.items()}

        # 3-1. initially count per-image GT samples
        scores_all = {}
        init_scores = {}
        for idx, _ in enumerate(self._extractor):
            counts = {k: v.count(idx) for k, v in by_combinations.items()}
            scores_all[idx] = counts
            init_scores[idx] = np.sum(
                [v / n_combs[k] for k, v in counts.items()]
            )

        by_splits = dict()
        for sname in subsets:
            by_splits[sname] = []

        total = len(self._extractor)
        target_size = dict()
        expected = []  # expected numbers of per split GT samples
        for sname, ratio in zip(subsets, sratio):
            target_size[sname] = total * ratio
            expected.append(
                (sname, {k: v * ratio for k, v in n_combs.items()})
            )

        ##
        # functions for keep the # of annotations not exceed the expected num
        def compute_penalty(counts, n_combs):
            p = 0
            for k, v in counts.items():
                p += max(0, (v / n_combs[k]) - 1.0)
            return p

        def update_nc(counts, n_combs):
            for k, v in counts.items():
                n_combs[k] = max(0, n_combs[k] - v)
                if n_combs[k] == 0:
                    n_combs[k] = -1
            return n_combs

        ##

        # 3-2. assign each DatasetItem to a split, one by one
        for idx, _ in sorted(
            init_scores.items(), key=lambda item: item[1], reverse=True
        ):
            counts = scores_all[idx]

            # shuffling split order to add randomness
            # when two or more splits have the same penalty value
            np.random.shuffle(expected)

            pp = []
            for sname, nc in expected:
                if len(by_splits[sname]) >= target_size[sname]:
                    # the split has enough images,
                    # stop adding more images to this split
                    pp.append(1e08)
                else:
                    # compute penalty based on the number of GT samples
                    # added in the split
                    pp.append(compute_penalty(counts, nc))

            # we push an image to a split with the minimum penalty
            midx = np.argmin(pp)

            sname, nc = expected[midx]
            by_splits[sname].append(idx)
            update_nc(counts, nc)

        self._set_parts(by_splits)
