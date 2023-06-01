# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from typing import Tuple

DEFAULT_SUBSET_NAME = "default"
BboxIntCoords = Tuple[int, int, int, int]  # (x, y, w, h)
SUBSET_NAME_BLACKLIST = {"labels", "images", "annotations", "instances"}

_CACHE_DIR = osp.expanduser(os.getenv("XDG_CACHE_HOME", osp.join("~", ".cache")))
DATUMARO_CACHE_DIR = osp.join(_CACHE_DIR, "datumaro")

if not osp.exists(DATUMARO_CACHE_DIR):
    os.makedirs(DATUMARO_CACHE_DIR)
