# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp


def get_samples_path():
    cur_dir = osp.dirname(__file__)
    return osp.abspath(osp.join(cur_dir, "..", "plugins", "openvino_plugin", "samples"))
