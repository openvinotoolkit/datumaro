# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
"""Transforms using Segment Anything Model"""
from .bbox_to_inst_mask import SAMBboxToInstanceMask

__all__ = ["SAMBboxToInstanceMask"]
