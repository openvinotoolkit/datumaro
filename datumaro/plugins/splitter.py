# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import numpy as np

from datumaro.components.dataset import Dataset
from datumaro.components.extractor import (Transform, AnnotationType, \
    DEFAULT_SUBSET_NAME)
from datumaro.components.cli_plugin import CliPlugin

class TaskSpecificSplitter(Transform):
    _parts = []

    def __init__(self, dataset:Dataset, seed):
        super().__init__(dataset)

        np.random.seed(seed)

    def _get_required(self, ratio):
        min_value = np.max(ratio)
        for i in ratio:
            if i < min_value and i > 1e-7:
                min_value = i
        required = int (np.around(1.0) / min_value)
        return required

    def _normalize_ratio(ratio):
        assert ratio is not None, "Ratio shouldn't be None"
        assert 0 < len(ratio), "Expected at least one split"
        assert all(0.0 < r and r < 1.0 for r in ratio), \
            "Ratios are expected to be in the range (0; 1), but got %s" % ratio

        ratio = np.array(ratio)
        ratio /= np.sum(ratio)
        return ratio

    def _get_sections(self, dataset_size, ratio):
        n_splits = [int(np.around(dataset_size * r)) for r in ratio[:-1]]
        n_splits.append(dataset_size - np.sum(n_splits))

        # if there are splits with zero samples even if ratio is not 0,
        # borrow one from the split who has one or more.
        for ii in range(len(n_splits)):
            if n_splits[ii] == 0 and ratio[ii] > 1e-7:
                midx = np.argmax(n_splits)
                if n_splits[midx] > 0:
                    n_splits[ii] += 1
                    n_splits[midx] -= 1
        sections = np.add.accumulate(n_splits[:-1])
        return sections

    def _group_by_attributes(self, items):
        '''
        Args:
            items: list of (idx, ann). ann is the annotation from Label object.
        Returns:
            by_attributes: dict of { combination-of-attributes : list of index }
        '''
        ## group by attributes
        by_attributes = dict()
        for idx, ann in items:
            attributes = tuple(sorted(ann.attributes.items()))
            if attributes not in by_attributes:
                by_attributes[attributes] = []
            by_attributes[attributes].append(idx)
        return by_attributes

    def _split_indice(self, indice, group_name, ratio, required):
        filtered_size = len(indice)
        if required > filtered_size:
            log.warning("There's not enough samples for filtered group, \
                '{}'}'".format(group_name))
        sections = self._get_sections(filtered_size, ratio)
        splits = np.array_split(indice, sections)
        assert len(ratio)==len(splits)
        return splits

    def _find_split(self, index):
        for subset_indices, subset in self._parts:
            if index in subset_indices:
                return subset
        return DEFAULT_SUBSET_NAME # all the possible remainder --> default

    def __iter__(self):
        for i, item in enumerate(self._extractor):
            yield self.wrap_item(item, subset=self._find_split(i))

class SplitforClassification(TaskSpecificSplitter, CliPlugin):
    """
    Splits dataset into train/val/test set in class-wise manner. |n
    |n
    Notes:|n
    - Single label is expected for each DatasetItem.|n
    - If there are not enough images in some class or attributes group,
      the split ratio can't be guaranteed.|n
    |n
    Example:|n
    |s|s%(prog)s --train .5 --val .2 --test .3
    """
    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('-t', '--train', type=float,
            help="Ratio for train set")
        parser.add_argument('-v', '--val', type=float,
            help="Ratio for validation set")
        parser.add_argument('-e', '--test',type=float,
            help="Ratio for test set")
        parser.add_argument('--seed', type=int, help="Random seed")
        return parser

    def __init__(self, dataset:Dataset,
        train=0.0, val=0.0, test=0.0, seed=None):
        super().__init__(dataset, seed)

        subsets = ['train', 'val', 'test']
        sratio = np.array([train, val, test])
        splits = list(zip(subsets, sratio))

        assert all(0.0 <= r and r <= 1.0 for _, r in splits), \
            "Ratios are expected to be in the range [0, 1], but got %s" % splits

        total_ratio = np.sum(sratio)
        if not abs(total_ratio - 1.0) <= 1e-7:
            raise Exception(
                "Sum of ratios is expected to be 1, got %s, which is %s" %
                (splits, total_ratio))

        ## support only single label for a DatasetItem
        ## 1. group by label
        by_labels = dict()
        for idx, item in enumerate(self._extractor):
            labels = []
            for ann in item.annotations:
                if ann.type == AnnotationType.label:
                    labels.append(ann)
            assert len(labels) == 1, \
                "Expected exact one label for a DatasetItem"
            ann = labels[0]
            if not hasattr(ann, 'label') or ann.label is None:
                label = None
            else:
                label = str(ann.label)
            if label not in by_labels:
                by_labels[label] = []
            by_labels[label].append((idx, ann))

        self._subsets = set(subsets) # output subset names
        by_splits = dict()
        for subset in subsets:
            by_splits[subset] = []

        ## 2. group by attributes
        required = self._get_required(sratio)
        for label, items in by_labels.items():
            np.random.shuffle(items)
            by_attributes = self._group_by_attributes(items)
            for attributes, indice in by_attributes.items():
                gname = 'label: {}, attributes: {}'.format(label, attributes)
                splits = self._split_indice(indice, gname, sratio, required)
                for subset, split in zip(subsets, splits):
                    if len(split) > 0:
                        by_splits[subset].extend(split)

        parts = []
        for subset in self._subsets:
            parts.append((set(by_splits[subset]), subset))
        self._parts = parts

        self._length = 'parent'


class SplitforMatchingReID(TaskSpecificSplitter, CliPlugin):
    """
    Splits dataset for matching, especially re-id task.|n
    First, splits dataset into 'train+val' and 'test' sets by "PID".|n
    Note that this splitting is not by DatasetItem. |n
    Then, tags 'test' into 'gallery'/'query' in class-wise random manner.|n
    Then, splits 'train+val' into 'train'/'val' sets in the same way.|n
    Therefore, the final subsets would be 'train', 'val', 'test'. |n
    And 'gallery', 'query' are tagged using anntoation group.|n
    You can get the 'gallery' and 'query' subsets using 'get_subset_by_group'.|n
    Notes:|n
    - Single label is expected for each DatasetItem.|n
    - Each label is expected to have "PID" attribute. |n
    |n
    Example:|n
    |s|s%(prog)s --train .5 --val .2 --test .3 --gallery .5 --query .5
    """
    _group_map = dict()

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('-t', '--train', type=float,
            help="Ratio for train set")
        parser.add_argument('-v', '--val', type=float,
            help="Ratio for validation set")
        parser.add_argument('-e', '--test',type=float,
            help="Ratio for test set")
        parser.add_argument('-g', '--gallery', type=float,
            help="Ratio for gallery in test set")
        parser.add_argument('-q', '--query',type=float,
            help="Ratio for query in test set")
        parser.add_argument('--seed', type=int, help="Random seed")
        return parser

    def __init__(self, dataset:Dataset, train=0.0, val=0.0, test=0.0,\
        gallery=0.0, query=0.0, attr_pid="PID", seed=None):
        super().__init__(dataset, seed)

        id_subsets = ['train', 'val', 'test']
        id_ratio = np.array([train, val, test])
        id_splits = list(zip(id_subsets, id_ratio))
        assert all(0.0 <= r and r <= 1.0 for _, r in id_splits), "Ratios \
            are expected to be in the range [0, 1], but got %s" % id_splits
        total_ratio = np.sum(id_ratio)
        if not abs(total_ratio - 1.0) <= 1e-7:
            raise Exception(
                "Sum of ratios is expected to be 1, got %s, which is %s" %
                (id_splits, total_ratio))

        test_subsets = ['gallery', 'query']
        test_ratio = np.array([gallery, query])
        test_splits = list(zip(test_subsets, test_ratio))
        assert all(0.0 <= r and r <= 1.0 for _, r in test_splits), \
            "Ratios are expected to be in the range [0, 1], but got %s"\
            % test_splits

        groups = set()
        ## group by PID(attr_pid)
        by_pid = dict()
        for idx, item in enumerate(self._extractor):
            labels = []
            for ann in item.annotations:
                if ann.type == AnnotationType.label:
                    labels.append(ann)
            assert len(labels) == 1, \
                "Expected exact one label for a DatasetItem"
            ann = labels[0]
            attributes = dict(ann.attributes.items())
            assert attr_pid in attributes, \
                "'{}' is expected as attribute name".format(attr_pid)
            person_id = attributes[attr_pid]
            if person_id not in by_pid:
                by_pid[person_id] = []
            by_pid[person_id].append((idx, ann))
            groups.add(ann.group)

        max_group_id = max(groups)
        self._group_map["gallery"] = max_group_id + 1
        self._group_map["query"] = max_group_id + 2

        required = self._get_required(id_ratio)
        if len(by_pid) < required:
            log.warning("There's not enough IDs, which is {}, \
                so train/val/test ratio can't be guaranteed." % len(by_pid))

        self._subsets = set(id_subsets) # output subset names
        by_splits = dict()
        for subset in self._subsets:
            by_splits[subset] = []

        ## 1. split dataset into trainval and test
        ##    IDs in test set should not exist in train/val set.
        if test > 1e-7: # has testset
            split_ratio = np.array([test, (train+val)])
            split_ratio /= np.sum(split_ratio) # normalize
            person_ids = list(by_pid.keys())
            np.random.shuffle(person_ids)
            sections = self._get_sections(len(person_ids), split_ratio)
            splits = np.array_split(person_ids, sections)
            testset = { pid: by_pid[pid] for pid in splits[0] }
            trainval = { pid: by_pid[pid] for pid in splits[1] }

            ## follow the ratio of datasetitems as possible.
            ## naive heuristic: exchange the best item one by one.
            expected_count = int(len(self._extractor) * split_ratio[0])
            testset_total = int(np.sum([len(v) for v in testset.values()]))
            if testset_total != expected_count:
                diffs = dict()
                for id_test, items_test in testset.items():
                    count_test = len(items_test)
                    for id_trval, items_trval in trainval.items():
                        count_trval = len(items_trval)
                        diff = count_trval - count_test
                        if diff==0:
                            continue # exchange has no effect
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
                    if abs(target_diff-nearest) >= abs(target_diff):
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
                            new_list.append((id1,id2))
                        if len(new_list)>0:
                            new_diffs[diff] = new_list
                    diffs = new_diffs
                    exchanges.append((pid_test, pid_trval))
                # exchange
                for pid_test, pid_trval in exchanges:
                    testset[pid_trval] = trainval.pop(pid_trval)
                    trainval[pid_test] = testset.pop(pid_test)
        else:
            testset = dict()
            trainval = by_pid

        ## 2. split 'test' into 'gallery' and 'query'
        if len(testset)>0:
            for person_id, items in testset.items():
                indice = [idx for idx, _ in items]
                by_splits['test'].extend(indice)

            total_ratio = np.sum(test_ratio)
            if not abs(total_ratio - 1.0) <= 1e-7:
                raise Exception(
                    "Sum of ratios is expected to be 1, got %s, which is %s" %
                    (test_splits, total_ratio))

            required = self._get_required(test_ratio)
            for person_id, items in testset.items():
                np.random.shuffle(items)
                by_attributes = self._group_by_attributes(items)
                for attributes, indice in by_attributes.items():
                    gname = 'person_id: {}, attributes: {}'.format(
                        person_id, attributes)
                    splits = self._split_indice(indice, gname,
                        test_ratio, required)

                    # tag using group
                    for idx, item in enumerate(self._extractor):
                        for subset, split in zip(test_subsets, splits):
                            if idx in split:
                                group_id = self._group_map[subset]
                                item.annotations[0].group = group_id
                                break

        ## 3. split 'trainval' into  'train' and 'val'
        trainval_subsets = ["train", "val"]
        trainval_ratio = np.array([train, val])
        total_ratio = np.sum(trainval_ratio)
        if total_ratio < 1e-7:
            trainval_splits = list(zip(["train", "val"], trainval_ratio))
            log.warning(
                "Sum of ratios is expected to be positive,\
                got %s, which is %s" % (trainval_splits, total_ratio))
        else:
            trainval_ratio /= total_ratio # normalize
            required = self._get_required(trainval_ratio)
            for person_id, items in trainval.items():
                np.random.shuffle(items)
                by_attributes = self._group_by_attributes(items)
                for attributes, indice in by_attributes.items():
                    gname = 'person_id: {}, attributes: {}'.format(
                        person_id, attributes)
                    splits = self._split_indice(indice, gname,
                        trainval_ratio, required)
                    for subset, split in zip(trainval_subsets, splits):
                        if len(split) > 0:
                            by_splits[subset].extend(split)

        parts = []
        for subset in self._subsets:
            parts.append((set(by_splits[subset]), subset))
        self._parts = parts

        self._length = 'parent'

    def get_subset_by_group(self, group:str):
        assert group in self._group_map, \
            "Unknown group '{}', available groups: {}".format(
                group, self._group_map.keys()
            )
        group_id = self._group_map[group]
        subset = self.select(lambda item: item.annotations[0].group == group_id)
        return subset

class SplitforDetection(TaskSpecificSplitter, CliPlugin):
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
    |n
    Example:|n
    |s|s%(prog)s --train .5 --val .2 --test .3
    """
    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('-t', '--train', type=float,
            help="Ratio for train set")
        parser.add_argument('-v', '--val', type=float,
            help="Ratio for validation set")
        parser.add_argument('-e', '--test',type=float,
            help="Ratio for test set")
        parser.add_argument('--seed', type=int, help="Random seed")
        return parser

    def __init__(self, dataset:Dataset,
        train=0.0, val=0.0, test=0.0, seed=None):
        super().__init__(dataset, seed)

        subsets = ['train', 'val', 'test']
        sratio = np.array([train, val, test])
        splits = list(zip(subsets, sratio))

        assert all(0.0 <= r and r <= 1.0 for _, r in splits), \
            "Ratios are expected to be in the range [0, 1], but got %s" % splits

        total_ratio = np.sum(sratio)
        if not abs(total_ratio - 1.0) <= 1e-7:
            raise Exception(
                "Sum of ratios is expected to be 1, got %s, which is %s" %
                (splits, total_ratio))

        ## 1. group by bbox label
        by_labels = dict()
        for idx, item in enumerate(self._extractor):
            for ann in item.annotations:
                if ann.type == AnnotationType.bbox:
                    if not hasattr(ann, 'label') or ann.label is None:
                        label = None
                    else:
                        label = str(ann.label)
                    if label not in by_labels:
                        by_labels[label] = [(idx, ann)]
                    else:
                        by_labels[label].append((idx, ann))

        ## 2. group by attributes
        by_combinations = dict()
        for label, items in by_labels.items():
            by_attributes = self._group_by_attributes(items)
            for attributes, indice in by_attributes.items():
                gname = 'label: {}, attributes: {}'.format(label, attributes)
                by_combinations[gname] = indice

        ## total number of GT samples per label-attr combinations
        NC = {k: len(v) for k, v in by_combinations.items()}

        ## 3-1. initially count per-image GT samples
        scores_all = {}
        init_scores = {}
        for idx, item in enumerate(self._extractor):
            counts = { k: v.count(idx) for k, v in by_combinations.items() }
            scores_all[idx] = counts
            init_scores[idx] = np.sum( [v / NC[k] for k, v in counts.items()] )

        self._subsets = set(subsets) # output subset names
        by_splits = dict()
        for sname in subsets:
            by_splits[sname] = []

        total = len(self._extractor)
        target_size = dict()
        NC_all = [] # expected numbers of per split GT samples
        for sname, ratio in zip (subsets, sratio):
            target_size[sname] = total * ratio
            NC_all.append((sname, {k: v * ratio for k, v in NC.items()}))

        ###
        # functions for keep the # of annotations not exceed the expected number
        def compute_penalty(counts, NC):
            p = 0
            for k, v in counts.items():
                p += max(0, (v / NC[k]) - 1.0)
            return p
        def update_nc(counts, NC):
            for k, v in counts.items():
                NC[k] = max(0, NC[k] - v)
                if NC[k] == 0:
                    NC[k] = -1
            return NC
        ###

        # 3-2. assign each DatasetItem to a split, one by one
        for idx, _ in sorted(init_scores.items(), \
            key=lambda item: item[1], reverse=True):
            counts = scores_all[idx]

            # shuffling split order to add randomness
            # when two or more splits have the same penalty value
            np.random.shuffle(NC_all)

            pp = []
            for sname, nc in NC_all:
                if len(by_splits[sname]) >= target_size[sname]:
                    # the split has enough images, stop adding more images to this split
                    pp.append(1e+08)
                else:
                    # compute penalty based on number of GT samples added in the split
                    pp.append(compute_penalty(counts, nc))

            # we push an image to a split with the minimum penalty
            midx = np.argmin(pp)

            sname, nc = NC_all[midx]
            by_splits[sname].append(idx)
            update_nc(counts, nc)

        parts = []
        for subset in self._subsets:
            parts.append((set(by_splits[subset]), subset))
        self._parts = parts

        self._length = 'parent'