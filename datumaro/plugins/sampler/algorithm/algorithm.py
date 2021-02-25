# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT


class InferenceResultAnalyzer:
    """
    Basic interface for IRA (Inference Result Analyzer)
    """

    def __init__(self, dataset, inference):
        self.data = dataset
        self.inference = inference

    def get_sample(self, method: str, k: int):
        raise NotImplementedError()
