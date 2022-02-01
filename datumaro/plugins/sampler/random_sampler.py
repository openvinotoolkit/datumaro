# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from random import Random
from typing import Optional

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.extractor import IExtractor, Transform


class RandomSampler(Transform, CliPlugin):
    """
    Sampler that keeps no more than required number of items in the dataset.|n
    |n
    Notes:|n
    - Items are selected uniformly|n
    - Requesting a sample larger than the number of all images will
    return all images|n
    |n
    Example: select subset of 20 images randomly|n
    |s|s%(prog)s -k 20 |n
    Example: select subset of 20 images, modify only 'train' subset|n
    |s|s%(prog)s -k 20 -s train
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('-k', '--count', type=int, required=True,
            help="Maximum number of items to sample")
        parser.add_argument('-s', '--subset', default=None,
            help="Limit changes to this subset (default: affect all dataset)")
        parser.add_argument('--seed', type=int,
            help="Initial value for random number generator")
        return parser

    def __init__(self, extractor: IExtractor, count: int, *,
            subset: Optional[str] = None, seed: Optional[int] = None):
        super().__init__(extractor)

        self._seed = seed
        self._count = count
        self._indices = None
        self._subset = subset

    def __iter__(self):
        if self._indices is None:
            rng = Random(self._seed)

            if self._subset:
                n = len(self._extractor.get_subset(self._subset))
            else:
                n = len(self._extractor)
            self._indices = rng.sample(range(n), min(self._count, n))
            self._indices.sort()

        idx_iter = iter(self._indices)

        try:
            next_pick = next(idx_iter)
        except StopIteration:
            if not self._subset:
                return
            else:
                next_pick = -1

        i = 0
        for item in self._extractor:
            if self._subset and self._subset != item.subset:
                yield item
            else:
                if i == next_pick:
                    yield item

                    try:
                        next_pick = next(idx_iter)
                    except StopIteration:
                        if self._subset:
                            next_pick = -1
                            continue
                        else:
                            return

                i += 1
