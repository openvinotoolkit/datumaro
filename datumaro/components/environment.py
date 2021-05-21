# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from functools import partial
from glob import glob
import inspect
import logging as log
import os
import os.path as osp

from datumaro.components.config_model import TreeConfig, TreeLayout
from datumaro.util.os_util import import_foreign_module


class Registry:
    def __init__(self):
        self.items = {}

    def register(self, name, value):
        self.items[name] = value
        return value

    def unregister(self, name):
        return self.items.pop(name, None)

    def get(self, key):
        """Returns a class or a factory function"""
        return self.items[key]

    def __getitem__(self, key):
        return self.get(key)

    def __contains__(self, key):
        return key in self.items

    def __iter__(self):
        return iter(self.items)

class PluginRegistry(Registry):
    def __init__(self, builtin=None, local=None):
        super().__init__()

        from datumaro.components.cli_plugin import CliPlugin

        if builtin is not None:
            for v in builtin:
                k = CliPlugin._get_name(v)
                self.register(k, v)
        if local is not None:
            for v in local:
                k = CliPlugin._get_name(v)
                self.register(k, v)


class Environment:
    _builtin_plugins = None
    PROJECT_EXTRACTOR_NAME = 'datumaro_project'

    def __init__(self, config: TreeConfig = None):
        if not isinstance(config, TreeConfig):
            config = TreeConfig(config)

        if config.base_dir and osp.isdir(config.base_dir):
            custom = self._load_plugins(
                osp.join(config.base_dir, TreeLayout.plugins_dir))
        else:
            custom = []
        builtin = self._load_builtin_plugins()
        select = lambda seq, t: [e for e in seq if issubclass(e, t)]
        from datumaro.components.project import load_project_as_dataset
        from datumaro.components.converter import Converter
        from datumaro.components.extractor import (Importer, Extractor,
            SourceExtractor, Transform)
        from datumaro.components.launcher import Launcher
        self.extractors = PluginRegistry(
            builtin=[e for e in select(builtin, Extractor)
                if e != SourceExtractor],
            local=[e for e in select(custom, Extractor)
                if e != SourceExtractor]
        )
        self.extractors.register(self.PROJECT_EXTRACTOR_NAME,
            load_project_as_dataset)

        self.importers = PluginRegistry(
            builtin=select(builtin, Importer),
            local=select(custom, Importer)
        )
        self.launchers = PluginRegistry(
            builtin=select(builtin, Launcher),
            local=select(custom, Launcher)
        )
        self.converters = PluginRegistry(
            builtin=select(builtin, Converter),
            local=select(custom, Converter)
        )
        self.transforms = PluginRegistry(
            builtin=select(builtin, Transform),
            local=select(custom, Transform)
        )

    @staticmethod
    def _find_plugins(plugins_dir):
        plugins = []
        if not osp.exists(plugins_dir):
            return plugins

        for plugin_name in os.listdir(plugins_dir):
            p = osp.join(plugins_dir, plugin_name)
            if osp.isfile(p) and p.endswith('.py'):
                plugins.append((plugins_dir, plugin_name, None))
            elif osp.isdir(p):
                plugins += [(plugins_dir,
                        osp.splitext(plugin_name)[0] + '.' + osp.basename(p),
                        osp.splitext(plugin_name)[0]
                    )
                    for p in glob(osp.join(p, '*.py'))]
        return plugins

    @classmethod
    def _import_module(cls, module_dir, module_name, types, package=None):
        module = import_foreign_module(osp.splitext(module_name)[0], module_dir,
            package=package)

        exports = []
        if hasattr(module, 'exports'):
            exports = module.exports
        else:
            for symbol in dir(module):
                if symbol.startswith('_'):
                    continue
                exports.append(getattr(module, symbol))

        exports = [s for s in exports
            if inspect.isclass(s) and issubclass(s, types) and not s in types]

        return exports

    @classmethod
    def _load_plugins(cls, plugins_dir, types=None):
        if not types:
            from datumaro.components.converter import Converter
            from datumaro.components.extractor import (Extractor, Importer,
                Transform)
            from datumaro.components.launcher import Launcher
            types = [Extractor, Converter, Importer, Launcher, Transform]

        types = tuple(types)

        plugins = cls._find_plugins(plugins_dir)

        all_exports = []
        for module_dir, module_name, package in plugins:
            try:
                exports = cls._import_module(module_dir, module_name, types,
                    package)
            except Exception as e:
                module_search_error = ImportError
                try:
                    module_search_error = ModuleNotFoundError # python 3.6+
                except NameError:
                    pass

                message = ["Failed to import module '%s': %s", module_name, e]
                if isinstance(e, module_search_error):
                    log.debug(*message)
                else:
                    log.warning(*message)
                continue

            log.debug("Imported the following symbols from %s: %s" % \
                (
                    module_name,
                    ', '.join(s.__name__ for s in exports)
                )
            )
            all_exports.extend(exports)

        return all_exports

    @classmethod
    def _load_builtin_plugins(cls):
        if not cls._builtin_plugins:
            plugins_dir = osp.join(
                __file__[: __file__.rfind(osp.join('datumaro', 'components'))],
                osp.join('datumaro', 'plugins')
            )
            assert osp.isdir(plugins_dir), plugins_dir
            cls._builtin_plugins = cls._load_plugins(plugins_dir)
        return cls._builtin_plugins

    def make_extractor(self, name, *args, **kwargs):
        return self.extractors.get(name)(*args, **kwargs)

    def make_importer(self, name, *args, **kwargs):
        return self.importers.get(name)(*args, **kwargs)

    def make_launcher(self, name, *args, **kwargs):
        return self.launchers.get(name)(*args, **kwargs)

    def make_converter(self, name, *args, **kwargs):
        result = self.converters.get(name)
        if inspect.isclass(result):
            result = result.convert
        return partial(result, *args, **kwargs)

    def make_transform(self, name, *args, **kwargs):
        return partial(self.transforms.get(name), *args, **kwargs)

    def is_format_known(self, name):
        return name in self.importers or name in self.extractors

    def detect_dataset(self, path):
        matches = []

        for format_name, importer in self.importers.items.items():
            log.debug("Checking '%s' format...", format_name)
            try:
                match = importer.detect(path)
                if match:
                    log.debug("format matched")
                    matches.append(format_name)
            except NotImplementedError:
                log.debug("Format '%s' does not support auto detection.",
                    format_name)

        return matches
