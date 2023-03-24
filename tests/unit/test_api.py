# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

# ruff: noqa: F401

from ..requirements import Requirements, mark_requirement


class ApiTest:
    @mark_requirement(Requirements.DATUM_API)
    def test_can_import_core(self):
        import datumaro as dm

        assert hasattr(dm, "Dataset")

    @mark_requirement(Requirements.DATUM_API)
    def test_can_reach_module_alias_symbols_from_base(self):
        import datumaro as dm

        assert hasattr(dm.ops, "ExactMerge")
        assert hasattr(dm.project, "Project")
        assert hasattr(dm.errors, "DatumaroError")

    @mark_requirement(Requirements.DATUM_API)
    def test_can_import_from_module_aliases(self):
        from datumaro.errors import DatumaroError
        from datumaro.ops import ExactMerge
        from datumaro.project import Project
