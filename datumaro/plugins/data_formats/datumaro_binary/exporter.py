# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=no-self-use

import os
import os.path as osp
import shutil

import numpy as np
import pycocotools.mask as mask_utils

from datumaro.components.annotation import (
    Annotation,
    Bbox,
    Caption,
    Cuboid3d,
    Ellipse,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
    RleMask,
    _Shape,
)
from datumaro.components.dataset import ItemStatus
from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetItem, IDataset
from datumaro.components.exporter import Exporter
from datumaro.components.media import Image, MediaElement, PointCloud
from datumaro.plugins.data_formats.datumaro.exporter import DatumaroExporter
from datumaro.plugins.data_formats.datumaro.exporter import _SubsetWriter as __SubsetWriter
from datumaro.util import cast, dump_json_file

from .format import DatumaroBinaryPath


class _SubsetWriter(__SubsetWriter):
    """"""


class DatumaroBinaryExporter(DatumaroExporter):
    DEFAULT_IMAGE_EXT = DatumaroBinaryPath.IMAGE_EXT
    WRITER_CLS = _SubsetWriter
    PATH_CLS = DatumaroBinaryPath
