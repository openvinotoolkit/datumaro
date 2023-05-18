# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.components.errors import MediaTypeError
from datumaro.components.media import Image
from datumaro.plugins.openvino_plugin.launcher import OpenvinoLauncher


class ShiftAnalyzerLauncher(OpenvinoLauncher):
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
        super().__init__(
            description, weights, interpreter, model_dir, model_name, output_layers, device
        )

        self._device = device or "cpu"
        self._output_blobs = next(iter(self._net.outputs))
        self._input_blobs = next(iter(self._net.input_info))

    def infer(self, inputs):
        inputs = self.process_inputs(inputs)
        features = self._net.infer(inputs)
        return features[self._output_blobs]

    def launch(self, inputs):
        return self.infer(inputs)

    def type_check(self, item):
        if not isinstance(item.media, Image):
            raise MediaTypeError(f"Media type should be Image, Current type={type(item.media)}")
        return True
