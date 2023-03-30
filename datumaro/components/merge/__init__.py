# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from .base import Merger

DEFAULT_MERGE_POLICY = "exact"


def get_merger(merge_policy: str = DEFAULT_MERGE_POLICY, *args, **kwargs) -> Merger:
    if merge_policy == "union":
        from .union_merge import UnionMerge

        return UnionMerge(*args, **kwargs)
    elif merge_policy == "intersect":
        from .intersect_merge import IntersectMerge

        return IntersectMerge(*args, **kwargs)
    elif merge_policy == "exact":
        from .exact_merge import ExactMerge

        return ExactMerge(*args, **kwargs)

    raise ValueError(f"{merge_policy} is invalid Merger name.")
