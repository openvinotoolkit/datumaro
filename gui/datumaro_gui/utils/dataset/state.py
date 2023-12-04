# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


def reset_subset(state):
    subset_list = ["subset", "subset_1", "subset_2"]
    for subset in subset_list:
        if subset in state.keys() and state[subset] is None:
            state[subset] = 0


def reset_state(keys, state):
    for k in keys:
        if k not in state:
            state[k] = None
    reset_subset(state)
