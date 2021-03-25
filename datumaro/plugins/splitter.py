# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import numpy as np

from datumaro.components.extractor import (Transform, AnnotationType,
    DEFAULT_SUBSET_NAME)
from datumaro.components.cli_plugin import CliPlugin

NEAR_ZERO = 1e-7


class _TaskSpecificSplit(Transform, CliPlugin):
    _default_split = [('train', 0.5), ('val', 0.2), ('test', 0.3)]

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('-s', '--subset', action='append',
            type=cls._split_arg, dest='splits',
            help="Subsets in the form: '<subset>:<ratio>' "
                "(repeatable, default: %s)" % dict(cls._default_split))
        parser.add_argument('--seed', type=int, help="Random seed")
        return parser

    @staticmethod
    def _split_arg(s):
        parts = s.split(':')
        if len(parts) != 2:
            import argparse
            raise argparse.ArgumentTypeError()
        return (parts[0], float(parts[1]))

    def __init__(self, dataset, splits, seed):
        super().__init__(dataset)

        if splits is None:
            splits = self._default_split

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
        for item in dataset:
            labels = [a for a in item.annotations
                if a.type == AnnotationType.label]
            if len(labels) != 1:
                raise Exception("Item '%s' contains %s labels, "
                    "but exactly one is expected" % (item.id, len(labels)))
            annotations.append(labels[0])
        return annotations

    @staticmethod
    def _validate_splits(splits, valid=None):
        snames = []
        ratios = []
        if valid is None:
            valid = ["train", "val", "test"]
        for subset, ratio in splits:
            assert subset in valid, \
                "Subset name must be one of %s, but got %s" % (valid, subset)
            assert 0.0 <= ratio and ratio <= 1.0, \
                "Ratio is expected to be in the range " \
                "[0, 1], but got %s for %s" % (ratio, subset)
            # ignore near_zero ratio because it may produce partition error.
            if ratio > NEAR_ZERO:
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
            if NEAR_ZERO < i and i < min_value:
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
            if num_split == 0 and NEAR_ZERO < ratio[ii]:
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

    def _split_by_attr(self, datasets, snames, ratio, out_splits,
            dataset_key=None):
        required = self._get_required(ratio)
        if dataset_key is None:
            dataset_key = "label"
        for key, items in datasets.items():
            np.random.shuffle(items)
            by_attributes = self._group_by_attr(items)
            for attributes, indice in by_attributes.items():
                gname = "%s: %s, attrs: %s" % (dataset_key, key, attributes)
                splits = self._split_indice(indice, gname, ratio, required)
                for subset, split in zip(snames, splits):
                    if 0 < len(split):
                        out_splits[subset].extend(split)

    def _split_indice(self, indice, group_name, ratio, required):
        filtered_size = len(indice)
        if filtered_size < required:
            log.warning("Not enough samples for a group, '%s'" % group_name)
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
    Splits dataset images in the specified ratio, keeping the initial class
    distribution.|n
    |n
    Notes:|n
    - Each image is expected to have only one Label|n
    - If Labels also have attributes, also splits by attribute values.|n
    - If there is not enough images in some class or attributes group,
      the split ratio can't be guaranteed.|n
    |n
    Example:|n
    |s|s%(prog)s --subset train:.5 --subset val:.2 --subset test:.3
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
            label = getattr(ann, 'label', None)
            if label not in by_labels:
                by_labels[label] = []
            by_labels[label].append((idx, ann))

        by_splits = dict()
        for subset in self._subsets:
            by_splits[subset] = []

        # 2. group by attributes
        self._split_by_attr(by_labels, self._snames, self._sratio, by_splits)
        self._set_parts(by_splits)


class ReidentificationSplit(_TaskSpecificSplit):
    """
    Splits a dataset for re-identification task.|n
    Produces a split with a specified ratio of images, avoiding having same
    labels in different subsets.|n
    |n
    In this task, the test set should consist of images of unseen
    people or objects during the training phase. |n
    This function splits a dataset in the following way:|n
    1. Splits the dataset into 'train + val' and 'test' sets|n
    |s|sbased on person or object ID.|n
    2. Splits 'test' set into 'test-gallery' and 'test-query' sets|n
    |s|sin class-wise manner.|n
    3. Splits the 'train + val' set into 'train' and 'val' sets|n
    |s|sin the same way.|n
    The final subsets would be
    'train', 'val', 'test-gallery' and 'test-query'. |n
    |n
    Notes:|n
    - Each image is expected to have a single Label|n
    - Object ID can be described by Label, or by attribute (--attr parameter)|n
    - The splits of the test set are controlled by '--query' parameter. |n
    |s|sGallery ratio would be 1.0 - query.|n
    |n
    Example: split a dataset in the specified ratio, split the test set|n
    |s|s|s|sinto gallery and query in 1:1 ratio|n
    |s|s%(prog)s --subset train:.5 --subset val:.2 --subset test:.3 --query .5|n
    Example: use 'person_id' attribute for splitting|n
    |s|s%(prog)s --attr person_id
    """

    _default_query_ratio = 0.5

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('--query', type=float,
            help="Query ratio in the test set (default: %.3f)"
            % cls._default_query_ratio)
        parser.add_argument('--attr', type=str, dest='attr_for_id',
            help="Attribute name representing the ID (default: use label)")
        return parser

    def __init__(self, dataset, splits, query=None,
            attr_for_id=None, seed=None):
        """
        Parameters
        ----------
        dataset : Dataset
        splits : list
            A list of (subset(str), ratio(float))
            Subset is expected to be one of ["train", "val", "test"].
            The sum of ratios is expected to be 1.
        query : float
            The ratio of 'test-query' set.
            The ratio of 'test-gallery' set would be 1.0 - query.
        attr_for_id: str
            attribute name representing the person/object id.
            if this is not specified, label would be used.
        seed : int, optional
        """
        super().__init__(dataset, splits, seed)

        if query is None:
            query = self._default_query_ratio

        assert 0.0 <= query and query <= 1.0, \
            "Query ratio is expected to be in the range " \
            "[0, 1], but got %f" % query
        test_splits = [('test-query', query), ('test-gallery', 1.0 - query)]

        # reset output subset names
        self._subsets = {"train", "val", "test-gallery", "test-query"}
        self._test_splits = test_splits
        self._attr_for_id = attr_for_id

    def _split_dataset(self):
        np.random.seed(self._seed)

        id_snames, id_ratio = self._snames, self._sratio

        attr_for_id = self._attr_for_id
        dataset = self._extractor

        # group by ID(attr_for_id)
        by_id = dict()
        annotations = self._get_uniq_annotations(dataset)
        if attr_for_id is None:  # use label
            for idx, ann in enumerate(annotations):
                ID = getattr(ann, 'label', None)
                if ID not in by_id:
                    by_id[ID] = []
                by_id[ID].append((idx, ann))
        else:  # use attr_for_id
            for idx, ann in enumerate(annotations):
                attributes = dict(ann.attributes.items())
                assert attr_for_id in attributes, \
                    "'%s' is expected as an attribute name" % attr_for_id
                ID = attributes[attr_for_id]
                if ID not in by_id:
                    by_id[ID] = []
                by_id[ID].append((idx, ann))

        required = self._get_required(id_ratio)
        if len(by_id) < required:
            log.warning("There's not enough IDs, which is %s, "
                "so train/val/test ratio can't be guaranteed."
                % len(by_id)
            )

        # 1. split dataset into trval and test
        #    IDs in test set should not exist in train/val set.
        test = id_ratio[id_snames.index("test")] if "test" in id_snames else 0
        if NEAR_ZERO < test:  # has testset
            split_ratio = np.array([test, 1.0 - test])
            IDs = list(by_id.keys())
            np.random.shuffle(IDs)
            sections = self._get_sections(len(IDs), split_ratio)
            splits = np.array_split(IDs, sections)
            testset = {pid: by_id[pid] for pid in splits[0]}
            trval = {pid: by_id[pid] for pid in splits[1]}

            # follow the ratio of datasetitems as possible.
            # naive heuristic: exchange the best item one by one.
            expected_count = int(len(self._extractor) * split_ratio[0])
            testset_total = int(np.sum([len(v) for v in testset.values()]))
            self._rebalancing(testset, trval, expected_count, testset_total)
        else:
            testset = dict()
            trval = by_id

        by_splits = dict()
        for subset in self._subsets:
            by_splits[subset] = []

        # 2. split 'test' into 'test-gallery' and 'test-query'
        if 0 < len(testset):
            test_snames = []
            test_ratio = []
            for sname, ratio in self._test_splits:
                test_snames.append(sname)
                test_ratio.append(float(ratio))

            self._split_by_attr(testset, test_snames, test_ratio, by_splits,
                dataset_key=attr_for_id)

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
            log.warning("Sum of ratios is expected to be positive, "
                "got %s, which is %s"
                % (trval_splits, total_ratio)
            )
        else:
            trval_ratio /= total_ratio  # normalize
            self._split_by_attr(trval, trval_snames, trval_ratio, by_splits,
                dataset_key=attr_for_id)

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
        if len(diffs) == 0:  # nothing would be changed by exchange
            return

        exchanges = []
        while True:
            target_diff = expected_count - testset_total
            # find nearest diff.
            keys = np.array(list(diffs.keys()))
            idx = (np.abs(keys - target_diff)).argmin()
            nearest = keys[idx]
            if abs(target_diff) <= abs(target_diff - nearest):
                break
            choice = np.random.choice(range(len(diffs[nearest])))
            id_test, id_trval = diffs[nearest][choice]
            testset_total += nearest
            new_diffs = dict()
            for diff, IDs in diffs.items():
                new_list = []
                for id1, id2 in IDs:
                    if id1 == id_test or id2 == id_trval:
                        continue
                    new_list.append((id1, id2))
                if 0 < len(new_list):
                    new_diffs[diff] = new_list
            diffs = new_diffs
            exchanges.append((id_test, id_trval))

        # exchange
        for id_test, id_trval in exchanges:
            test[id_trval] = trval.pop(id_trval)
            trval[id_test] = test.pop(id_test)


class DetectionSplit(_TaskSpecificSplit):
    """
    Splits a dataset into train/val/test subsets for detection task,
    using object annotations as a basis for splitting.|n
    Tries to produce an image split with the specified ratio, keeping the
    initial distribution of class objects.|n
    |n
    In a detection dataset, each image can have multiple object annotations -
    instance bounding boxes. Since an image shouldn't be included
    in multiple subsets at the same time, and image annotations
    shoudln't be split, in general, dataset annotations are unlikely to be split
    exactly in the specified ratio. |n
    This split tries to split dataset images as close as possible
    to the specified ratio, keeping the initial class distribution.|n
    |n
    Notes:|n
    - Each image is expected to have one or more Bbox annotations.|n
    - Only Bbox annotations are considered.|n
    |n
    Example: split dataset so that each object class annotations were split|n
    |s|s|s|sin the specified ratio between subsets|n
    |s|s%(prog)s --subset train:.5 --subset val:.2 --subset test:.3
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
            bbox_anns = [a for a in item.annotations
                if a.type == AnnotationType.bbox]
            assert 0 < len(bbox_anns), \
                "Expected more than one bbox annotation in the dataset"
            for ann in bbox_anns:
                label = getattr(ann, 'label', None)
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
        for sname in self._subsets:
            by_splits[sname] = []

        total = len(self._extractor)
        target_size = dict()
        expected = []  # expected numbers of per split GT samples
        for sname, ratio in zip(subsets, sratio):
            target_size[sname] = total * ratio
            expected.append(
                (sname, {k: v * ratio for k, v in n_combs.items()})
            )

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
                if target_size[sname] <= len(by_splits[sname]):
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
