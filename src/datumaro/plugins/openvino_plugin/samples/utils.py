# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np

from datumaro.components.annotation import HashKey


def gen_hash_key(features: np.ndarray) -> HashKey:
    features = np.sign(features)
    hash_key = np.clip(features, 0, None)
    hash_key = hash_key.astype(np.uint8)
    hash_key = np.packbits(hash_key, axis=-1)
    return HashKey(hash_key)
