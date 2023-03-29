# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from .base import Merger

DEFAULT_MERGE_POLICY = "exact"


def get_merger(merge_policy: str = DEFAULT_MERGE_POLICY, *args, **kwargs) -> Merger:
    if merge_policy == "union":
        from datumaro.components.operations import UnionMerge

        return UnionMerge(*args, **kwargs)
    elif merge_policy == "intersect":
        from datumaro.components.operations import IntersectMerge

        return IntersectMerge(*args, **kwargs)
    elif merge_policy == "exact":
        from datumaro.components.operations import ExactMerge

        return ExactMerge(*args, **kwargs)

    raise ValueError(f"{merge_policy} is invalid Merger name.")
