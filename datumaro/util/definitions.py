# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Tuple

DEFAULT_SUBSET_NAME = "default"
BboxIntCoords = Tuple[int, int, int, int]  # (x, y, w, h)
SUBSET_NAME_BLACKLIST = {"labels", "images", "annotations", "instances"}
