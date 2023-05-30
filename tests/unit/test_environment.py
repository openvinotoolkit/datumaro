# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

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
