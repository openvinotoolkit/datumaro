# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
from tokenizers import Tokenizer

from datumaro.components.errors import MediaTypeError
from datumaro.components.media import Image
from datumaro.plugins.openvino_plugin.launcher import OpenvinoLauncher


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
        if isinstance(inputs, str):
            if len(inputs.split()) > 1:
                prompt_text = inputs
            else:
                prompt_text = f"a photo of a {inputs}"
            inputs = self._tokenize(prompt_text)
            inputs = {self._input_blob: inputs}
        else:
            if not inputs.any():
                # media.data is None case
                return None

            # when processing a query key, we expand HWC to NHWC
            if len(inputs.shape) == 3:
                inputs = np.expand_dims(inputs, axis=0)

            inputs = self.process_inputs(inputs)

        results = self._net.infer(inputs)
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
