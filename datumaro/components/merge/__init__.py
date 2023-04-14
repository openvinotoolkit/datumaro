# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from .base import Merger
from .exact_merge import ExactMerge
from .intersect_merge import IntersectMerge
from .union_merge import UnionMerge

DEFAULT_MERGE_POLICY = "exact"


def get_merger(merge_policy: str = DEFAULT_MERGE_POLICY, *args, **kwargs) -> Merger:
    if merge_policy == "union":
        return UnionMerge(*args, **kwargs)
    elif merge_policy == "intersect":
        return IntersectMerge(*args, **kwargs)
    elif merge_policy == "exact":
        return ExactMerge(*args, **kwargs)

    raise ValueError(f"{merge_policy} is invalid Merger name.")
