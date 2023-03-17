# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=exec-used

import logging as log
import os
import os.path as osp
import shutil
import urllib

import cv2
import numpy as np
from openvino.inference_engine import IECore
from tqdm import tqdm

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.launcher import Launcher
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


class InterpreterScript:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            script = f.read()

        context = {}
        exec(script, context, context)

        normalize = context.get("normalize")
        if not callable(normalize):
            raise Exception("Can't find 'normalize' function in the interpreter script")
        self.__dict__["normalize"] = normalize

        process_outputs = context.get("process_outputs")
        if not callable(process_outputs):
            raise Exception("Can't find 'process_outputs' function in the interpreter script")
        self.__dict__["process_outputs"] = process_outputs

        get_categories = context.get("get_categories")
        assert get_categories is None or callable(get_categories)
        if get_categories:
            self.__dict__["get_categories"] = get_categories

    @staticmethod
    def get_categories():
        return None

    @staticmethod
    def process_outputs(inputs, outputs):
        raise NotImplementedError("Function should be implemented in the interpreter script")

    @staticmethod
    def normalize(inputs):
        raise NotImplementedError("Function should be implemented in the interpreter script")


class OpenvinoLauncher(Launcher):
    cli_plugin = _OpenvinoImporter

    def __init__(
        self,
        description=None,
        weights=None,
        interpreter=None,
        model_dir=None,
        model_name=None,
        output_layers=None,
        device=None,
    ):
        if model_name:
            model_dir = os.path.join(os.path.expanduser("~"), ".cache", "datumaro")
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

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
            raise Exception('Failed to open model description file "%s"' % (description))

        if not osp.isfile(weights):
            weights = osp.join(model_dir, weights)
        if not osp.isfile(weights):
            raise Exception('Failed to open model weights file "%s"' % (weights))

        if not osp.isfile(interpreter):
            interpreter = osp.join(model_dir, interpreter)
        if not osp.isfile(interpreter):
            raise Exception('Failed to open model interpreter script file "%s"' % (interpreter))

        self._interpreter = InterpreterScript(interpreter)

        self._device = device or "CPU"
        self._output_blobs = output_layers

        self._ie = IECore()
        self._network = self._ie.read_network(description, weights)
        self._check_model_support(self._network, self._device)
        self._load_executable_net()

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
            name for name, dev in self._ie.query_network(net, device).items() if not dev
        )
        if len(not_supported_layers) != 0:
            log.error(
                "The following layers are not supported "
                "by the plugin for device '%s': %s." % (device, ", ".join(not_supported_layers))
            )
            raise NotImplementedError("Some layers are not supported on the device")

    def _load_executable_net(self, batch_size=1):
        network = self._network

        if self._output_blobs:
            network.add_outputs(self._output_blobs)

        iter_inputs = iter(network.input_info)
        self._input_blob = next(iter_inputs)

        # NOTE: handling for the inclusion of `image_info` in OpenVino2019
        self._require_image_info = "image_info" in network.input_info
        if self._input_blob == "image_info":
            self._input_blob = next(iter_inputs)

        self._input_layout = network.input_info[self._input_blob].input_data.shape
        self._input_layout[0] = batch_size
        network.reshape({self._input_blob: self._input_layout})
        self._batch_size = batch_size

        self._net = self._ie.load_network(network=network, num_requests=1, device_name=self._device)

    def infer(self, inputs):
        inputs = self.process_inputs(inputs)
        results = self._net.infer(inputs)
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

        n, c, h, w = self._input_layout
        if inputs.shape[1:3] != (h, w):
            resized_inputs = np.empty((n, h, w, c), dtype=inputs.dtype)
            for inp, resized_input in zip(inputs, resized_inputs):
                cv2.resize(inp, (w, h), resized_input)
            inputs = resized_inputs
        inputs = inputs.transpose((0, 3, 1, 2))  # NHWC to NCHW

        inputs = self._interpreter.normalize(inputs)

        inputs = {self._input_blob: inputs}
        if self._require_image_info:
            info = np.zeros([1, 3])
            info[0, 0] = h
            info[0, 1] = w
            info[0, 2] = 1.0  # scale
            inputs["image_info"] = info

        return inputs
