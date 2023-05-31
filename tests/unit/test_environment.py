# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import sys

import pytest

from datumaro.components.environment import Environment


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
    def fxt_tf_failure(self, monkeypatch):
        # Simulate `openvino.tools` and `tensorflow` import failure
        monkeypatch.setitem(sys.modules, "openvino.tools", None)
        monkeypatch.setitem(sys.modules, "tensorflow", None)

        with Environment.release_builtin_plugins():
            env = Environment(use_lazy_import=True)
            _ = env.importers
            yield env

    def test_extra_deps_req(self, fxt_tf_failure: Environment):
        """Plugins affected by the import failure: `ac` and `tf_detection_api`."""
        loaded_plugin_names = set(
            sorted(fxt_tf_failure.extractors)
            + sorted(fxt_tf_failure.importers)
            + sorted(fxt_tf_failure.launchers)
            + sorted(fxt_tf_failure.exporters)
            + sorted(fxt_tf_failure.generators)
            + sorted(fxt_tf_failure.transforms)
            + sorted(fxt_tf_failure.validators)
        )

        assert "ac" not in loaded_plugin_names
        assert "tf_detection_api" not in loaded_plugin_names
