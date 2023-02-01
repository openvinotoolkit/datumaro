# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os.path as osp

import cv2
import numpy as np
import torch

from datumaro.components.model_inference import (
    compute_hash,
    download_file,
    img_center_crop,
    img_normalize,
    tokenize,
)
from datumaro.plugins.openvino_plugin.launcher import OpenvinoLauncher


class SearcherLauncher(OpenvinoLauncher):
    def __init__(
        self,
        description=None,
        weights=None,
        interpreter=None,
        device=None,
        model_dir=None,
        output_layers=None,
    ):
        model_name = "clip_visual_ViT-B_32"
        if description or weights:
            model_name = osp.splitext(description)[0]

        url_folder = "http://s3.toolbox.iotg.sclab.intel.com/test/components-data/datumaro/models/"
        if not model_dir:
            model_dir = "tests/assets/searcher"

        description = osp.join(model_dir, model_name + ".xml")
        if not osp.exists(description):
            cached_description_url = osp.join(url_folder, model_name + ".xml")
            log.info('Downloading: "{}" to {}\n'.format(cached_description_url, description))
            download_file(cached_description_url, description)

        weights = osp.join(model_dir, model_name + ".bin")
        if not osp.exists(weights):
            cached_weights_url = osp.join(url_folder, model_name + ".bin")
            log.info('Downloading: "{}" to {}\n'.format(cached_weights_url, weights))
            download_file(cached_weights_url, weights)

        if not interpreter:
            interpreter = osp.join(model_dir, "clip_ViT-B_32_interp.py")

        super().__init__(description, weights, interpreter, device, model_dir, output_layers)

        self._device = device or "cpu"
        self._output_blobs = next(iter(self._net.outputs))
        self._input_blobs = next(iter(self._net.input_info))

    def infer(self, inputs):
        if isinstance(inputs, str):
            if len(inputs.split()) > 1:
                prompt_text = inputs
            else:
                prompt_text = f"a photo of a {inputs}"
            inputs = tokenize(prompt_text).to("cpu", dtype=torch.float)
        else:
            inputs = inputs.squeeze()
            if not inputs.any():
                # media.data is None case
                return None

            if np.array(inputs).ndim == 2:
                inputs = cv2.cvtColor(inputs, cv2.COLOR_GRAY2RGB)
            elif np.array(inputs).ndim == 4:
                inputs = cv2.cvtColor(inputs, cv2.COLOR_RGBA2RGB)
            inputs = cv2.resize(inputs, (256, 256))
            inputs = img_center_crop(inputs, 224)
            inputs = img_normalize(inputs)
            inputs = np.expand_dims(inputs, axis=0)

        results = self._net.infer(inputs={self._input_blobs: inputs})
        results = torch.from_numpy(results[self._output_blobs])
        hash_string = compute_hash(results)
        return hash_string

    def launch(self, inputs):
        hash_key = self.infer(inputs)
        results = self.process_outputs(inputs, hash_key)
        return results
