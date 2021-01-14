# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import random
import numpy as np

from datumaro.components.extractor import (Transform, AnnotationType)
from datumaro.components.cli_plugin import CliPlugin

class TaskSpecificSplitter(Transform):
    # avoid https://bugs.python.org/issue16399
    _default_split = [('train', 0.67), ('test', 0.33)]
    _parts = []

    def __init__(self, extractor, splits, seed):
        super().__init__(extractor)
        
        random.seed(seed)   

        if splits is None:
            splits = self._default_split

        assert 0 < len(splits), "Expected at least one split"
        assert all(0.0 <= r and r <= 1.0 for _, r in splits), \
            "Ratios are expected to be in the range [0; 1], but got %s" % splits

        total_ratio = sum(s[1] for s in splits)
        if not abs(total_ratio - 1.0) <= 1e-7:
            raise Exception(
                "Sum of ratios is expected to be 1, got %s, which is %s" %
                (splits, total_ratio))
    
    def _get_split_size(self, dataset_size, ratio):
        n_splits = [int(np.around(dataset_size * r)) for r in ratio[:-1]]
        n_splits.append(dataset_size - np.sum(n_splits))

        for ii in range(len(n_splits)):
            # if there are splits with zero samples, 
            # borrow one from the split who has one or more.
            if n_splits[ii] == 0:
                midx = np.argmax(n_splits)
                if n_splits[midx] > 0:
                    n_splits[ii] += 1
                    n_splits[midx] -= 1   
        return n_splits

    def _find_split(self, index):
        for subset_indices, subset in self._parts:
            if index in subset_indices:
                return subset
        return subset # all the possible remainder goes to the last split

    def __iter__(self):
        for i, item in enumerate(self._extractor):
            yield self.wrap_item(item, subset=self._find_split(i))

class SplitforClassification(TaskSpecificSplitter, CliPlugin):
    """
    Joins all subsets into one and splits the result into few parts
    in class-wise manner.
    Single label for a DatasetItem is expected. 

    It is expected that item ids are unique and subset ratios sum up to 1.|n
    |n
    Example:|n
    |s|s%(prog)s --subset train:.67 --subset test:.33
    If there are not enough images in some class or attributes group,
    split ratio among subsets wouldn't be guaranteed.        
    """
    @staticmethod
    def _split_arg(s):
        parts = s.split(':')
        if len(parts) != 2:
            import argparse
            raise argparse.ArgumentTypeError()
        return (parts[0], float(parts[1]))

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('-s', '--subset', action='append',
            type=cls._split_arg, dest='splits',
            help="Subsets in the form: '<subset>:<ratio>' "
                "(repeatable, default: %s)" % dict(cls._default_split))
        parser.add_argument('--seed', type=int, help="Random seed")
        return parser

    def __init__(self, extractor, splits, seed=None):
        super().__init__(extractor, splits, seed)

        ## support only single label for a DatasetItem    
        ## 1. group by label
        by_labels = dict()
        for idx, item in enumerate(extractor):
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
        
        subsets = [s[0] for s in splits]
        sratio = np.array([float(s[1]) for s in splits])
        n_required_samples = int (np.around(1.0) / np.min(sratio)) 

        by_splits = dict()
        for subset in subsets:
            by_splits[subset] = []
     
        ## 2. group by attributes    
        for label, items in by_labels.items():
            random.shuffle(items)

            by_attributes = dict()
            for idx, ann in items:
                attributes = tuple(sorted(ann.attributes.items()))
                if attributes not in by_attributes:
                    by_attributes[attributes] = []
                by_attributes[attributes].append(idx)
            
            for attributes, indice in by_attributes.items():                    
                filtered_size = len(indice)  
                if n_required_samples > filtered_size:
                    log.warning("There's not enough samples for filtered group\
                        (label : {}, attributes: {})".format(label, attributes))
                n_splits = super()._get_split_size(filtered_size, sratio)
                
                k = 0
                for subset, ns in zip(subsets, n_splits):
                    if ns > 0:
                        by_splits[subset].extend(indice[k:k+ns])
                        k += ns
        
        parts = []
        for subset in subsets:
            parts.append((set(by_splits[subset]), subset))

        self._parts = parts
        self._subsets = set(subsets)
        self._length = 'parent'
    