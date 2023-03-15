# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import cv2
import numpy as np
from tokenizers import Tokenizer

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
    mean = 255 * np.array([0.485, 0.456, 0.406])
    std = 255 * np.array([0.229, 0.224, 0.225])

    image = image.transpose(-1, 0, 1)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image


class SearcherLauncher(OpenvinoLauncher):
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
        self._tokenizer = None

    def _tokenize(self, texts: str, context_length: int = 77, truncate: bool = True):
        if not self._tokenizer:
            checkpoint = "openai/clip-vit-base-patch32"
            self._tokenizer = Tokenizer.from_pretrained(checkpoint)
        tokens = self._tokenizer.encode(texts).ids
        result = np.zeros((1, context_length))

        if len(tokens) > context_length:
            if truncate:
                eot_token = tokens.ids[-1]
                tokens = tokens[:context_length]
                tokens[-1] = eot_token

        for i, token in enumerate(tokens):
            result[:, i] = token
        return result

    def _compute_hash(self, features):
        features = np.sign(features)
        hash_key = np.clip(features, 0, None)
        hash_key = hash_key.astype(np.uint8)
        hash_key = np.packbits(hash_key, axis=-1)
        return hash_key

    def infer(self, inputs):
        color_space_dict = {2: "COLOR_GRAY2RGB", 3: "COLOR_BGR2RGB", 4: "COLOR_RGBA2RGB"}
        if isinstance(inputs, str):
            if len(inputs.split()) > 1:
                prompt_text = inputs
            else:
                prompt_text = f"a photo of a {inputs}"
            inputs = self._tokenize(prompt_text)
        else:
            inputs = inputs.squeeze()
            if not inputs.any():
                # media.data is None case
                return None

            inputs = cv2.cvtColor(inputs, getattr(cv2, color_space_dict.get(inputs.ndim)))
            inputs = cv2.resize(inputs, (256, 256))
            inputs = img_center_crop(inputs, 224)
            inputs = img_normalize(inputs)
            inputs = np.expand_dims(inputs, axis=0)

        results = self._net.infer(inputs={self._input_blobs: inputs})
        hash_key = self._compute_hash(results[self._output_blobs])
        return hash_key

    def launch(self, inputs):
        hash_key = self.infer(inputs)
        results = self.process_outputs(inputs, hash_key)
        return results

    def type_check(self, item):
        if not isinstance(item.media, Image):
            raise MediaTypeError(f"Media type should be Image, Current type={type(item.media)}")
        return True
