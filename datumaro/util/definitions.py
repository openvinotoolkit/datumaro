# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from typing import Tuple

DEFAULT_SUBSET_NAME = "default"
BboxIntCoords = Tuple[int, int, int, int]  # (x, y, w, h)
SUBSET_NAME_BLACKLIST = {"labels", "images", "annotations", "instances"}

DATUMARO_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "datumaro")

if not osp.exists(DATUMARO_CACHE_DIR):
    os.makedirs(DATUMARO_CACHE_DIR)
