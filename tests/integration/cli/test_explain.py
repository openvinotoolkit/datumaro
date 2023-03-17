# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import platform
from unittest import TestCase, skipIf

import numpy as np

from datumaro.util.image import save_image

from ...requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir
from tests.utils.test_utils import run_datum as run


class ExplainIntegrationScenarios(TestCase):
    @skipIf(
        platform.system() == "Darwin",
        "Segmentation fault only occurs on MacOS: "
        "https://github.com/openvinotoolkit/datumaro/actions/runs/4084466205/jobs/7041196279",
    )
    @mark_requirement(Requirements.DATUM_BUG_721)
    def test_rise(self):
        rise_dir = get_test_asset_path("rise")

        with TestDir() as project_dir:
            # Create project
            run(self, "create", "-o", project_dir)

            model_name = "my-model"

            # Add model
            run(
                self,
                "model",
                "add",
                "-n",
                model_name,
                "-l",
                "openvino",
                "-p",
                project_dir,
                "--",
                "-d",
                osp.join(rise_dir, "model.xml"),
                "-w",
                osp.join(rise_dir, "model.bin"),
                "-i",
                osp.join(rise_dir, "model_interp.py"),
            )

            input_fpath = osp.join(project_dir, "input", "input.jpg")
            save_image(input_fpath, np.zeros([224, 224, 3], dtype=np.uint8), create_dir=True)

            output_dir = osp.join(project_dir, "output")

            # Run explain
            run(
                self,
                "explain",
                "-m",
                model_name,
                "-o",
                output_dir,
                "-p",
                project_dir,
                input_fpath,
                "rise",
            )

            num_heatmap = len(os.listdir(output_dir))
            self.assertTrue(num_heatmap > 0)
