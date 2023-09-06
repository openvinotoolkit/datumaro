# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import warnings

warnings.warn(
    "We are planning to clean up the entities in datumaro/__init__.py until datumaro==1.5.0. "
    'This will affect the following import pattern: "import datumaro as dm; dm.<entity>". '
    "If you are using this pattern in your code, please revisit it after upgrading datumaro>=1.5.0.",
    DeprecationWarning,
)

from . import errors as errors
from . import ops as ops
from . import project as project
from .components.algorithms import RISE, LossDynamicsAnalyzer
from .components.annotation import (
    NO_GROUP,
    Annotation,
    AnnotationType,
    Bbox,
    BinaryMaskImage,
    Caption,
    Categories,
    Colormap,
    CompiledMask,
    CompiledMaskImage,
    Cuboid3d,
    Ellipse,
    IndexMaskImage,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
    RgbColor,
    RleMask,
)
from .components.cli_plugin import CliPlugin
from .components.dataset import Dataset, DatasetPatch, DatasetSubset, eager_mode
from .components.dataset_base import (
    DEFAULT_SUBSET_NAME,
    CategoriesInfo,
    DatasetBase,
    DatasetItem,
    IDataset,
    SubsetBase,
)
from .components.dataset_item_storage import ItemStatus
from .components.environment import Environment, PluginRegistry
from .components.exporter import Exporter, ExportErrorPolicy, FailingExportErrorPolicy
from .components.hl_ops import HLOps
from .components.importer import FailingImportErrorPolicy, Importer, ImportErrorPolicy
from .components.launcher import Launcher
from .components.media import Image, MediaElement, Video, VideoFrame
from .components.media_manager import MediaManager
from .components.progress_reporting import (
    NullProgressReporter,
    ProgressReporter,
    SimpleProgressReporter,
    TQDMProgressReporter,
)
from .components.transformer import ItemTransform, ModelTransform, Transform
from .components.validator import Validator
from .components.visualizer import Visualizer
from .version import __version__
