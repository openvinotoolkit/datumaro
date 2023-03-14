# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import cv2
import numpy as np

from datumaro.components.errors import MediaTypeError
from datumaro.components.media import Image
from datumaro.plugins.openvino_plugin.launcher import OpenvinoLauncher


def img_center_crop(image, size):
    width, height = image.shape[1], image.shape[0]
    mid_w, mid_h = int(width / 2), int(height / 2)

    crop_w = size if size < image.shape[1] else image.shape[1]
    crop_h = size if size < image.shape[0] else image.shape[0]
    mid_cw, mid_ch = int(crop_w / 2), int(crop_h / 2)

    cropped_image = image[mid_h - mid_ch : mid_h + mid_ch, mid_w - mid_cw : mid_w + mid_cw]
    return cropped_image


def img_normalize(image):
    mean = np.array([127.5] * 3)
    std = np.array([127.5] * 3)

    image = image.transpose(-1, 0, 1)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image


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
        color_space_dict = {2: "COLOR_GRAY2RGB", 3: "COLOR_BGR2RGB", 4: "COLOR_RGBA2RGB"}
        if isinstance(inputs, str):
            if len(inputs.split()) > 1:
                prompt_text = inputs
            else:
                prompt_text = f"a photo of a {inputs}"
            inputs = self._tokenize(prompt_text)
        else:
            # media.data is None case
            if not inputs.any():
                return None
            inputs = inputs.squeeze().astype(np.uint8)
            inputs = cv2.cvtColor(inputs, getattr(cv2, color_space_dict.get(inputs.ndim)))
            inputs = cv2.resize(inputs, (299, 299))
            inputs = img_normalize(inputs)
            inputs = np.expand_dims(inputs, axis=0)

        features = self._net.infer(inputs={self._input_blobs: inputs})
        return features[self._output_blobs]

    def launch(self, inputs):
        return self.infer(inputs)

    def type_check(self, item):
        if not isinstance(item.media, Image):
            raise MediaTypeError(f"Media type should be Image, Current type={type(item.media)}")
        return True
