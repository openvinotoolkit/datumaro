# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from abc import ABC, abstractclassmethod
from importlib import import_module
from typing import Type, Union, get_args

from datumaro.components.dataset_base import DatasetBase
from datumaro.components.exporter import Exporter
from datumaro.components.generator import DatasetGenerator
from datumaro.components.importer import Importer
from datumaro.components.launcher import Launcher
from datumaro.components.transformer import Transform
from datumaro.components.validator import Validator

# Transform should be in front of DatasetBase,
# since Transform inherits DatasetBase.
_PLUGIN_TYPES = Union[
    Transform,
    Exporter,
    DatasetGenerator,
    Importer,
    Launcher,
    Validator,
    DatasetBase,
]
PLUGIN_TYPES = Type[_PLUGIN_TYPES]
STR_TO_PLUGIN_TYPES = {t.__name__: t for t in get_args(_PLUGIN_TYPES)}


class LazyPlugin(ABC):
    NAME: str

    @abstractclassmethod
    def get_plugin_cls(cls) -> PLUGIN_TYPES:
        pass


def get_lazy_plugin(
    import_path: str,
    plugin_name: str,
    plugin_type: str,
) -> LazyPlugin:
    plugin_type_cls = STR_TO_PLUGIN_TYPES[plugin_type]

    class LazyPluginImpl(LazyPlugin, plugin_type_cls):
        NAME = plugin_name

        @classmethod
        def get_plugin_cls(cls) -> PLUGIN_TYPES:
            splits = import_path.split(".")
            module_name = ".".join(splits[:-1])
            class_name = splits[-1]
            module = import_module(module_name)
            plugin_cls = getattr(module, class_name)
            return plugin_cls

    return LazyPluginImpl
