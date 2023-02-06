# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from contextlib import redirect_stdout
from io import StringIO
from unittest import TestCase

from datumaro.util.test_utils import TestDir
from datumaro.util.test_utils import run_datum as run

from ...requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path


class ModelIntegrationScenarios(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_model_add_and_remove(self):
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

            # Check whether the model is added
            with StringIO() as buf:
                with redirect_stdout(buf):
                    run(self, "model", "info", "-p", project_dir)
                self.assertIn(model_name, buf.getvalue())

            # Remove model
            run(
                self,
                "model",
                "remove",
                model_name,
                "-p",
                project_dir,
            )

            # Check whether the model is removed
            with StringIO() as buf:
                with redirect_stdout(buf):
                    run(self, "model", "info", "-p", project_dir)
                self.assertNotIn(model_name, buf.getvalue())
