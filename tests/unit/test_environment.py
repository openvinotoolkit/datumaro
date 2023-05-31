# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import builtins
import sys

import pytest

from datumaro.components.environment import Environment

real_import = builtins.__import__


class EnvironmentTest:
    @pytest.fixture
    def fxt_lazy_import(self):
        with Environment.release_builtin_plugins():
            env = Environment(use_lazy_import=True)
            _ = env.importers
            yield env

    @pytest.fixture
    def fxt_no_lazy_import(self):
        with Environment.release_builtin_plugins():
            env = Environment(use_lazy_import=False)
            _ = env.importers
            yield env

    def test_equivalance(self, fxt_lazy_import: Environment, fxt_no_lazy_import: Environment):
        assert sorted(fxt_lazy_import.extractors) == sorted(fxt_no_lazy_import.extractors)
        assert sorted(fxt_lazy_import.importers) == sorted(fxt_no_lazy_import.importers)
        assert sorted(fxt_lazy_import.launchers) == sorted(fxt_no_lazy_import.launchers)
        assert sorted(fxt_lazy_import.exporters) == sorted(fxt_no_lazy_import.exporters)
        assert sorted(fxt_lazy_import.generators) == sorted(fxt_no_lazy_import.generators)
        assert sorted(fxt_lazy_import.transforms) == sorted(fxt_no_lazy_import.transforms)
        assert sorted(fxt_lazy_import.validators) == sorted(fxt_no_lazy_import.validators)

    @pytest.fixture
    def fxt_tf_import_error(self, monkeypatch):
        # Simulate `tensorflow` import failure

        def monkey_import_notfound(name, globals=None, locals=None, fromlist=(), level=0):
            if name in ("tensorflow",):
                raise ModuleNotFoundError(f"Mocked module not found {name}")
            return real_import(name, globals=globals, locals=locals, fromlist=fromlist, level=level)

        monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
        monkeypatch.setattr(builtins, "__import__", monkey_import_notfound)

    @pytest.fixture
    def fxt_tf_failure_env(self, fxt_tf_import_error):
        with Environment.release_builtin_plugins():
            env = Environment(use_lazy_import=True)
            _ = env.importers
            yield env

    def test_extra_deps_req(self, fxt_tf_failure_env: Environment):
        """Plugins affected by the import failure: `ac` and `tf_detection_api`."""
        loaded_plugin_names = set(
            sorted(fxt_tf_failure_env.extractors)
            + sorted(fxt_tf_failure_env.importers)
            + sorted(fxt_tf_failure_env.launchers)
            + sorted(fxt_tf_failure_env.exporters)
            + sorted(fxt_tf_failure_env.generators)
            + sorted(fxt_tf_failure_env.transforms)
            + sorted(fxt_tf_failure_env.validators)
        )

        assert "ac" not in loaded_plugin_names
        assert "tf_detection_api" not in loaded_plugin_names
