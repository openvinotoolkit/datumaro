# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import importlib
import logging as log
import os.path as osp
from functools import partial
from inspect import isclass
from typing import (
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
)

from datumaro.components.cli_plugin import CliPlugin, plugin_types
from datumaro.components.format_detection import (
    DetectedFormat,
    FormatDetectionConfidence,
    RejectionReason,
    detect_dataset_format,
)
from datumaro.components.lazy_plugin import PLUGIN_TYPES, LazyPlugin
from datumaro.util.os_util import import_foreign_module, split_path

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self):
        self._items: Dict[str, T] = {}

    def register(self, name: str, value: T) -> T:
        self._items[name] = value
        return value

    def unregister(self, name: str) -> Optional[T]:
        return self._items.pop(name, None)

    def get(self, key: str):
        """Returns a class or a factory function"""
        return self._items[key]

    def __getitem__(self, key: str) -> T:
        return self.get(key)

    def __contains__(self, key) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def items(self) -> Generator[Tuple[str, T], None, None]:
        for key in self:
            yield key, self.get(key)


class PluginRegistry(Registry[Type[CliPlugin]]):
    def __init__(
        self, filter: Callable[[Type[CliPlugin]], bool] = None
    ):  # pylint: disable=redefined-builtin
        super().__init__()
        self._filter = filter

    def get(self, key: str) -> PLUGIN_TYPES:
        """Returns a class or a factory function"""
        item = self._items[key]
        if issubclass(item, LazyPlugin):
            return item.get_plugin_cls()
        return item

    def batch_register(self, values: Iterable[CliPlugin]):
        for v in values:
            if self._filter and not self._filter(v):
                continue

            self.register(v.NAME, v)


class Environment:
    _builtin_plugins = None

    @classmethod
    def _make_filter(cls, accept, decline=None, skip=None):
        accept = (accept,) if isclass(accept) else tuple(accept)
        skip = {skip} if isclass(skip) else set(skip or [])
        skip = tuple(skip | set(accept))
        return partial(cls._check_type, accept=accept, decline=decline, skip=skip)

    @staticmethod
    def _check_type(t, *, accept, decline, skip):
        if not issubclass(t, accept) or t in skip or (decline and issubclass(t, decline)):
            return False
        if getattr(t, "__not_plugin__", None):
            return False
        return True

    def __init__(self, use_lazy_import: bool = True):
        from datumaro.components.dataset_base import DatasetBase, SubsetBase
        from datumaro.components.exporter import Exporter
        from datumaro.components.generator import DatasetGenerator
        from datumaro.components.importer import Importer
        from datumaro.components.launcher import Launcher
        from datumaro.components.transformer import ItemTransform, Transform
        from datumaro.components.validator import Validator

        _filter = self._make_filter
        self._extractors = PluginRegistry(
            _filter(
                DatasetBase,
                decline=Transform,
                skip=(SubsetBase, Transform, ItemTransform),
            )
        )
        self._importers = PluginRegistry(_filter(Importer))
        self._launchers = PluginRegistry(_filter(Launcher))
        self._exporters = PluginRegistry(_filter(Exporter))
        self._generators = PluginRegistry(_filter(DatasetGenerator))
        self._transforms = PluginRegistry(_filter(Transform, skip=ItemTransform))
        self._validators = PluginRegistry(_filter(Validator))
        self._builtins_initialized = False
        self._use_lazy_import = use_lazy_import

    def _get_plugin_registry(self, name):
        if not self._builtins_initialized:
            self._builtins_initialized = True
            self._register_builtin_plugins()
        return getattr(self, name)

    @property
    def extractors(self) -> PluginRegistry:
        return self._get_plugin_registry("_extractors")

    @property
    def importers(self) -> PluginRegistry:
        return self._get_plugin_registry("_importers")

    @property
    def launchers(self) -> PluginRegistry:
        return self._get_plugin_registry("_launchers")

    @property
    def exporters(self) -> PluginRegistry:
        return self._get_plugin_registry("_exporters")

    @property
    def generators(self) -> PluginRegistry:
        return self._get_plugin_registry("_generators")

    @property
    def transforms(self) -> PluginRegistry:
        return self._get_plugin_registry("_transforms")

    @property
    def validators(self) -> PluginRegistry:
        return self._get_plugin_registry("_validators")

    @staticmethod
    def _find_plugins(plugins_dir):
        plugins = []

        for pattern in ("*.py", "*/*.py", "*/*/*.py"):
            for path in glob.glob(osp.join(glob.escape(plugins_dir), pattern)):
                if not osp.isfile(path):
                    continue

                path_rel = osp.relpath(path, plugins_dir)
                name_parts = split_path(osp.splitext(path_rel)[0])

                # a module with a dot in the name won't load correctly
                if any("." in part for part in name_parts):
                    log.warning(
                        "Python file '%s' in directory '%s' can't be imported "
                        "due to a dot in the name; skipping.",
                        path_rel,
                        plugins_dir,
                    )
                    continue
                plugins.append(".".join(name_parts))

        return plugins

    @classmethod
    def _get_plugin_exports(cls, module, types):
        exports = []
        if hasattr(module, "exports"):
            exports = module.exports
        else:
            for symbol in dir(module):
                if symbol.startswith("_"):
                    continue
                exports.append(getattr(module, symbol))

        exports = [s for s in exports if isclass(s) and issubclass(s, types) and s not in types]

        return exports

    @classmethod
    def _load_plugins(cls, module_names, *, importer, types=None):
        types = tuple(types or plugin_types())

        all_exports = []
        for module_name in module_names:
            try:
                module = importer(module_name)
                exports = cls._get_plugin_exports(module, types)
            except Exception as e:
                module_search_error = ModuleNotFoundError

                message = ["Failed to import module '%s': %s", module_name, e]
                if isinstance(e, module_search_error):
                    log.debug(*message)
                else:
                    log.warning(*message)
                continue

            log.debug(
                "Imported the following symbols from %s: %s"
                % (module_name, ", ".join(s.__name__ for s in exports))
            )
            all_exports.extend(exports)

        return all_exports

    @classmethod
    def _load_builtin_plugins(cls):
        """Load builtin plugins which will be imported lazily using plugin spec files"""
        if cls._builtin_plugins is None:
            from datumaro.plugins.specs import get_lazy_plugins

            cls._builtin_plugins = get_lazy_plugins()
        return cls._builtin_plugins

    @classmethod
    def _load_builtin_plugins_from_importlib(cls):
        """Load builtin plugins from importlib, not lazy import from plugin spec files"""
        if cls._builtin_plugins is None:
            import datumaro.plugins

            plugins_dir = osp.dirname(datumaro.plugins.__file__)
            module_names = [
                datumaro.plugins.__name__ + "." + name for name in cls._find_plugins(plugins_dir)
            ]
            cls._builtin_plugins = cls._load_plugins(module_names, importer=importlib.import_module)
        return cls._builtin_plugins

    def load_plugins(self, plugins_dir):
        module_names = self._find_plugins(plugins_dir)
        plugins = self._load_plugins(
            module_names, importer=partial(import_foreign_module, path=plugins_dir)
        )

        self.register_plugins(plugins)

    def _register_builtin_plugins(self):
        self.register_plugins(
            self._load_builtin_plugins()
            if self._use_lazy_import
            else self._load_builtin_plugins_from_importlib()
        )

    def register_plugins(self, plugins):
        self.extractors.batch_register(plugins)
        self.importers.batch_register(plugins)
        self.launchers.batch_register(plugins)
        self.exporters.batch_register(plugins)
        self.generators.batch_register(plugins)
        self.transforms.batch_register(plugins)
        self.validators.batch_register(plugins)

    def make_extractor(self, name, *args, **kwargs):
        return self.extractors.get(name)(*args, **kwargs)

    def make_importer(self, name, *args, **kwargs):
        return self.importers.get(name)(*args, **kwargs)

    def make_launcher(self, name, *args, **kwargs):
        return self.launchers.get(name)(*args, **kwargs)

    def make_exporter(self, name, *args, **kwargs):
        result = self.exporters.get(name)
        if isclass(result):
            result = result.convert
        return partial(result, *args, **kwargs)

    def make_transform(self, name, *args, **kwargs):
        return partial(self.transforms.get(name), *args, **kwargs)

    def is_format_known(self, name):
        return name in self.importers or name in self.extractors

    def detect_dataset(
        self,
        path: str,
        depth: int = 1,
        rejection_callback: Optional[Callable[[str, RejectionReason, str], None]] = None,
    ) -> List[str]:
        ignore_dirs = {"__MSOSX", "__MACOSX"}
        all_matched_formats: Set[DetectedFormat] = set()

        for _ in range(depth + 1):
            detected_formats = detect_dataset_format(
                (
                    (format_name, importer.detect)
                    for format_name, importer in self.importers.items()
                ),
                path,
                rejection_callback=rejection_callback,
            )

            if detected_formats:
                all_matched_formats |= set(detected_formats)

            paths = glob.glob(osp.join(path, "*"))
            path = "" if len(paths) != 1 else paths[0]
            if not osp.isdir(path) or osp.basename(path) in ignore_dirs:
                break

        max_conf = (
            max(all_matched_formats).confidence
            if len(all_matched_formats) > 0
            else FormatDetectionConfidence.NONE
        )

        return sorted(
            [str(format) for format in all_matched_formats if format.confidence == max_conf]
        )

    def __reduce__(self):
        return (self.__class__, ())

    @classmethod
    def merge(cls, envs: Sequence["Environment"]) -> "Environment":
        if all([env == DEFAULT_ENVIRONMENT for env in envs]):
            return DEFAULT_ENVIRONMENT

        merged = Environment()

        def _register(registry: PluginRegistry):
            merged.register_plugins(plugin for plugin in registry)

        for env in envs:
            _register(env.extractors)
            _register(env.importers)
            _register(env.launchers)
            _register(env.exporters)
            _register(env.generators)
            _register(env.transforms)
            _register(env.validators)

        return merged

    @classmethod
    def release_builtin_plugins(cls):
        cls._builtin_plugins = None


DEFAULT_ENVIRONMENT = Environment()
