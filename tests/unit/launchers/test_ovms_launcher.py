# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import datumaro.plugins.ovms_plugin.samples.face_detection as face_det_model_interp
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.ovms_plugin.launcher import OVMSLauncher

from ...requirements import Requirements, mark_requirement


class OVMSLauncherTest:
    @pytest.fixture
    def fxt_input(self) -> List[DatasetItem]:
        return [
            DatasetItem(
                id="test",
                media=Image.from_numpy(np.zeros(shape=[10, 10, 3], dtype=np.uint8)),
                annotations=[],
            )
        ]

    @pytest.fixture
    def fxt_output(self) -> np.ndarray:
        # Output of face-detection model
        np.random.seed(3003)
        return np.random.rand(1, 1, 200, 7)

    @pytest.fixture
    def fxt_metadata(self) -> Dict:
        # Metadata of face-detection model
        return {
            "model_version": 1,
            "inputs": {"data": {"shape": [1, 3, 400, 600], "dtype": "DT_FLOAT"}},
            "outputs": {"detection_out": {"shape": [1, 1, 200, 7], "dtype": "DT_FLOAT"}},
        }

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_launchers(self, fxt_input, fxt_output, fxt_metadata):
        mock_client = MagicMock()
        mock_client.get_model_metadata.return_value = fxt_metadata
        mock_client.predict.return_value = fxt_output

        with patch(
            "datumaro.plugins.ovms_plugin.launcher.make_grpc_client",
            return_value=mock_client,
        ):
            launcher = OVMSLauncher(
                model_name="face-detection",
                model_interpreter_path=os.path.abspath(face_det_model_interp.__file__),
            )

            mock_client.get_model_metadata.assert_called_once()
            mock_client.get_model_status.assert_called_once()

            outputs = launcher.launch(fxt_input)
            mock_client.predict.assert_called_once()

            assert len(outputs) > 0
            for anns in outputs:
                assert len(anns) > 0
