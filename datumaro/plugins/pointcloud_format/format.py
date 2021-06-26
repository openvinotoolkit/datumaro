# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

class PointCloudPath:
    BASE_DIR = 'ds0'
    ANNNOTATION_DIR = 'ann'

    DEFAULT_IMAGE_EXT = '.jpg'

    POINT_CLOUD_DIR = 'pointcloud'
    RELATED_IMAGES_DIR = 'related_images'

    KEY_ID_FILE = 'key_id_map.json'
    META_FILE = 'meta.json'

    BUILTIN_ATTRS = {'frame', 'object'}
