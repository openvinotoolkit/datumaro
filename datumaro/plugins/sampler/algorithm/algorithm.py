# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum

SamplingMethod = Enum("SamplingMethod", ["topk", "lowk", "randk", "mixk", "randtopk"])
Algorithm = Enum("Algorithm", ["entropy"])


class InferenceResultAnalyzer:
    """
    Basic interface for IRA (Inference Result Analyzer)
    """

    def __init__(self, dataset, inference):
        self.data = dataset
        self.inference = inference
        self.sampling_method = SamplingMethod

    def get_sample(self, method: str, k: int):
        raise NotImplementedError()
