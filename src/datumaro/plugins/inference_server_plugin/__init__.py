# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.plugins.inference_server_plugin.ovms import OVMSLauncher
from datumaro.plugins.inference_server_plugin.triton import TritonLauncher

__all__ = ["OVMSLauncher", "TritonLauncher"]
