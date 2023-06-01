# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pytest

import datumaro.components.lazy_plugin
from datumaro.components.environment import Environment

real_import_module = datumaro.components.lazy_plugin.import_module


class EnvironmentTest:
    @pytest.fixture
    def fxt_lazy_import(self):
        Environment.release_builtin_plugins()
        env = Environment(use_lazy_import=True)
        _ = env.importers
        yield env
        Environment.release_builtin_plugins()

    @pytest.fixture
    def fxt_no_lazy_import(self):
        Environment.release_builtin_plugins()
        env = Environment(use_lazy_import=False)
        _ = env.importers
        yield env
        Environment.release_builtin_plugins()

    def test_equivalance(self, fxt_lazy_import: Environment, fxt_no_lazy_import: Environment):
        assert sorted(fxt_lazy_import.extractors) == sorted(fxt_no_lazy_import.extractors)
        assert sorted(fxt_lazy_import.importers) == sorted(fxt_no_lazy_import.importers)
        assert sorted(fxt_lazy_import.launchers) == sorted(fxt_no_lazy_import.launchers)
        assert sorted(fxt_lazy_import.exporters) == sorted(fxt_no_lazy_import.exporters)
        assert sorted(fxt_lazy_import.generators) == sorted(fxt_no_lazy_import.generators)
        assert sorted(fxt_lazy_import.transforms) == sorted(fxt_no_lazy_import.transforms)
        assert sorted(fxt_lazy_import.validators) == sorted(fxt_no_lazy_import.validators)

    @pytest.fixture
    def fxt_tf_failure_env(self, monkeypatch):
        def _patch(name, package=None):
            if name == "tensorflow":
                raise ImportError()
            return real_import_module(name, package)

        monkeypatch.setattr(datumaro.components.lazy_plugin, "import_module", _patch)

        Environment.release_builtin_plugins()
        env = Environment(use_lazy_import=True)
        _ = env.importers
        yield env
        Environment.release_builtin_plugins()

    def test_extra_deps_req(self, fxt_tf_failure_env):
        """Plugins affected by the import failure: `ac` and `tf_detection_api`."""

        env = fxt_tf_failure_env

        loaded_plugin_names = set(
            sorted(env.extractors)
            + sorted(env.importers)
            + sorted(env.launchers)
            + sorted(env.exporters)
            + sorted(env.generators)
            + sorted(env.transforms)
            + sorted(env.validators)
        )

        assert "ac" not in loaded_plugin_names
        assert "tf_detection_api" not in loaded_plugin_names
