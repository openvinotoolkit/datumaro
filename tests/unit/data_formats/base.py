# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
from typing import Any, Dict, Optional

import pytest

from datumaro.components.dataset import Dataset
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.exporter import Exporter
from datumaro.components.importer import Importer

from ...requirements import Requirements, mark_requirement

from tests.utils.test_utils import compare_datasets


class TestDataFormatBase:
    IMPORTER: Importer
    EXPORTER: Exporter
    USE_TEST_CAN_EXPORT_AND_IMPORT = True

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self, fxt_dataset_dir: str):
        detected_formats = DEFAULT_ENVIRONMENT.detect_dataset(fxt_dataset_dir)
        assert [self.IMPORTER.NAME] == detected_formats

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        fxt_import_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
    ):
        helper_tc = request.getfixturevalue("helper_tc")
        dataset = Dataset.import_from(fxt_dataset_dir, self.IMPORTER.NAME, **fxt_import_kwargs)
        compare_datasets(helper_tc, fxt_expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_export_and_import(
        self,
        fxt_expected_dataset: Dataset,
        test_dir: str,
        fxt_import_kwargs: Dict[str, Any],
        fxt_export_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
        exporter: Optional[Exporter] = None,
        importer: Optional[Importer] = None,
    ):
        if exporter is None:
            exporter = getattr(self, "EXPORTER", None)
        if importer is None:
            importer = getattr(self, "IMPORTER", None)

        if exporter is None or importer is None:
            pytest.skip(reason="exporter or importer is None.")

        helper_tc = request.getfixturevalue("helper_tc")

        exporter.convert(
            fxt_expected_dataset, save_dir=test_dir, save_media=True, **fxt_export_kwargs
        )
        dataset = Dataset.import_from(test_dir, importer.NAME, **fxt_import_kwargs)
        compare_datasets(helper_tc, fxt_expected_dataset, dataset, require_media=True)
