# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=exec-used

import inspect
import logging as log
import os.path as osp
import shutil
import urllib
from importlib.util import module_from_spec, spec_from_file_location
from typing import Dict, Optional

import cv2
import numpy as np
from openvino.runtime import Core
from tqdm import tqdm

from datumaro.components.abstracts import IModelInterpreter
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.launcher import Launcher
from datumaro.errors import DatumaroError
from datumaro.util.definitions import DATUMARO_CACHE_DIR
from datumaro.util.samples import get_samples_path


class _OpenvinoImporter(CliPlugin):
    @staticmethod
    def _parse_output_layers(s):
        return [s.strip() for s in s.split(",")]

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-d", "--description", required=True, help="Path to the model description file (.xml)"
        )
        parser.add_argument(
            "-w", "--weights", required=True, help="Path to the model weights file (.bin)"
        )
        parser.add_argument(
            "-i",
            "--interpreter",
            required=True,
            help="Path to the network output interprter script (.py)",
        )
        parser.add_argument("--device", default="CPU", help="Target device (default: %(default)s)")
        parser.add_argument(
            "--output-layers",
            type=cls._parse_output_layers,
            help="A comma-separated list of extra output layers",
        )
        return parser

    @staticmethod
    def copy_model(model_dir, model):
        shutil.copy(model["description"], osp.join(model_dir, osp.basename(model["description"])))
        model["description"] = osp.basename(model["description"])

        shutil.copy(model["weights"], osp.join(model_dir, osp.basename(model["weights"])))
        model["weights"] = osp.basename(model["weights"])

        shutil.copy(model["interpreter"], osp.join(model_dir, osp.basename(model["interpreter"])))
        model["interpreter"] = osp.basename(model["interpreter"])


class OpenvinoLauncher(Launcher):
    cli_plugin = _OpenvinoImporter
    ALLOWED_CHANNEL_FORMATS = {"NCHW", "NHWC"}

    def __init__(
        self,
        description: Optional[str] = None,
        weights: Optional[str] = None,
        interpreter: Optional[str] = None,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        output_layers=None,
        device: Optional[str] = None,
        compile_model_config: Optional[Dict] = None,
        channel_format: str = "NCHW",
        to_rgb: bool = True,
    ):
        if model_name:
            model_dir = DATUMARO_CACHE_DIR

            # Please visit open-model-zoo repository for OpenVINO public models if you are interested in
            # https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md
            url_folder = "https://storage.openvinotoolkit.org/repositories/datumaro/models/"

            description = osp.join(model_dir, model_name + ".xml")
            if not osp.exists(description):
                cached_description_url = osp.join(url_folder, model_name + ".xml")
                log.info('Downloading: "{}" to {}\n'.format(cached_description_url, description))
                self._download_file(cached_description_url, description)

            weights = osp.join(model_dir, model_name + ".bin")
            if not osp.exists(weights):
                cached_weights_url = osp.join(url_folder, model_name + ".bin")
                log.info('Downloading: "{}" to {}\n'.format(cached_weights_url, weights))
                self._download_file(cached_weights_url, weights)

            if not interpreter:
                openvino_plugin_samples_dir = get_samples_path()
                interpreter = osp.join(openvino_plugin_samples_dir, model_name + "_interp.py")

        if not model_dir:
            model_dir = ""

        if not osp.isfile(description):
            description = osp.join(model_dir, description)
        if not osp.isfile(description):
            raise DatumaroError('Failed to open model description file "%s"' % (description))

        if not osp.isfile(weights):
            weights = osp.join(model_dir, weights)
        if not osp.isfile(weights):
            raise DatumaroError('Failed to open model weights file "%s"' % (weights))

        if not osp.isfile(interpreter):
            interpreter = osp.join(model_dir, interpreter)
        if not osp.isfile(interpreter):
            raise DatumaroError('Failed to open model interpreter script file "%s"' % (interpreter))

        self._interpreter = self._load_interpreter(file_path=interpreter)

        self._device = device or "CPU"
        self._output_blobs = output_layers
        self._compile_model_config = compile_model_config

        self._core = Core()
        self._network = self._core.read_model(description, weights)
        self._check_model_support(self._network, self._device)
        self._load_executable_net()

        if channel_format not in self.ALLOWED_CHANNEL_FORMATS:
            raise DatumaroError(
                f"channel_format={channel_format} is not in "
                f"ALLOWED_CHANNEL_FORMATS={self.ALLOWED_CHANNEL_FORMATS}."
            )
        self._channel_format = channel_format
        self._to_rgb = to_rgb

    def _load_interpreter(self, file_path: str) -> IModelInterpreter:
        fname, _ = osp.splitext(osp.basename(file_path))
        spec = spec_from_file_location(fname, file_path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, IModelInterpreter)
                and obj is not IModelInterpreter
            ):
                log.info(f"Load {name} for model interpreter.")
                return obj()

        raise DatumaroError(f"{file_path} has no class derived from IModelInterpreter.")

    def _download_file(self, url: str, file_root: str):
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as source, open(file_root, "wb") as output:  # nosec B310
            with tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))
        return 0

    def _check_model_support(self, net, device):
        not_supported_layers = set(
            name for name, dev in self._core.query_model(net, device).items() if not dev
        )
        if len(not_supported_layers) != 0:
            log.error(
                "The following layers are not supported "
                "by the plugin for device '%s': %s." % (device, ", ".join(not_supported_layers))
            )
            raise NotImplementedError("Some layers are not supported on the device")

    def _load_executable_net(self, batch_size: int = 1):
        network = self._network

        if self._output_blobs:
            network.add_outputs([self._output_blobs])

        iter_inputs = iter(network.inputs)
        self._input_blob = next(iter_inputs)

        is_dynamic_layout = False
        try:
            self._input_layout = self._input_blob.shape
        except ValueError:
            # In case of that the input has dynamic shape
            self._input_layout = self._input_blob.partial_shape
            is_dynamic_layout = True

        if is_dynamic_layout:
            self._input_layout[0] = batch_size
            network.reshape({self._input_blob: self._input_layout})
        else:
            model_batch_size = self._input_layout[0]
            if batch_size != model_batch_size:
                log.warning(
                    "Input layout of the model is static, so that we cannot change "
                    f"the model batch size ({model_batch_size}) to batch size ({batch_size})! "
                    "Set the batch size to {model_batch_size}."
                )
                batch_size = model_batch_size

        self._batch_size = batch_size

        self._net = self._core.compile_model(
            model=network,
            device_name=self._device,
            config=self._compile_model_config,
        )
        self._request = self._net.create_infer_request()

    def infer(self, inputs):
        inputs = self.process_inputs(inputs)
        results = self._request.infer(inputs)
        if len(results) == 1:
            return next(iter(results.values()))
        else:
            return results

    def launch(self, inputs):
        batch_size = len(inputs)
        if self._batch_size < batch_size:
            self._load_executable_net(batch_size)

        outputs = self.infer(inputs)
        results = self.process_outputs(inputs, outputs)
        return results

    def categories(self):
        return self._interpreter.get_categories()

    def process_outputs(self, inputs, outputs):
        return self._interpreter.process_outputs(inputs, outputs)

    def process_inputs(self, inputs):
        assert len(inputs.shape) == 4, "Expected an input image in (N, H, W, C) format, got %s" % (
            inputs.shape,
        )

        if inputs.shape[3] == 1:  # A batch of single-channel images
            inputs = np.repeat(inputs, 3, axis=3)

        assert inputs.shape[3] == 3, "Expected BGR input, got %s" % (inputs.shape,)

        # Resize
        inputs = self._interpreter.resize(inputs)

        if self._channel_format == "NCHW":
            n, c, h, w = self._input_layout
        elif self._channel_format == "NHWC":
            n, h, w, c = self._input_layout
        else:
            raise DatumaroError(f"Invliad channel_format: {self._channel_format}.")

        if inputs.shape[1:3] != (h, w):
            resized_inputs = np.empty((n, h, w, c), dtype=inputs.dtype)
            for inp, resized_input in zip(inputs, resized_inputs):
                cv2.resize(inp, (w, h), resized_input)
            inputs = resized_inputs

        if self._channel_format == "NCHW":
            inputs = inputs.transpose((0, 3, 1, 2))  # NHWC to NCHW

        if self._to_rgb:
            inputs = inputs[:, :, :, ::-1]  # Convert from BGR to RGB

        inputs = self._interpreter.normalize(inputs)

        inputs = {self._input_blob.get_any_name(): inputs}

        return inputs
