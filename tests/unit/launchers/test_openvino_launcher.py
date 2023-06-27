# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


from unittest.mock import patch

import numpy as np
import pytest

from datumaro.plugins.openvino_plugin.launcher import OpenvinoLauncher

from ...requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path


class OpenvinoLauncherTest:
    @pytest.fixture
    def fxt_input(self):
        return np.zeros([1, 10, 10, 3], dtype=np.uint8)

    @pytest.fixture
    def fxt_normal(self):
        model_dir = get_test_asset_path("rise")
        return OpenvinoLauncher(
            interpreter="model_interp.py",
            description="model.xml",
            weights="model.bin",
            model_dir=model_dir,
        )

    @pytest.fixture
    def fxt_override_interpreter_by_builtin_model_name(self):
        model_dir = get_test_asset_path("rise")
        return OpenvinoLauncher(
            interpreter="model_interp.py",
            description="model.xml",
            weights="model.bin",
            model_dir=model_dir,
            model_name="otx_custom_object_detection_gen3_atss",
        )

    @pytest.fixture(params=["fxt_normal", "fxt_override_interpreter_by_builtin_model_name"])
    def fxt_launcher(self, request):
        return request.getfixturevalue(request.param)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_launchers(self, fxt_launcher, fxt_input):
        with patch.object(fxt_launcher._request, "infer") as mock_request:
            fxt_launcher.infer(fxt_input)
            mock_request.assert_called()
