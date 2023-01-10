# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from datumaro.components.model_inference import compute_hash, tokenize
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
        if not model_dir:
            model_dir = "tests/assets/searcher"
        if not description:
            description = osp.join(model_dir, "clip_visual_ViT-B_32.xml")
        if not weights:
            weights = osp.join(model_dir, "clip_visual_ViT-B_32.bin")
        if not interpreter:
            interpreter = osp.join(model_dir, "clip_visual_ViT-B_32_interp.py")
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
            if None in inputs:
                # media.data is None case
                return None

            trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]),
                ]
            )

            inputs = np.uint8(inputs)
            inputs = Image.fromarray(inputs)

            if np.array(inputs).ndim == 2 or inputs.mode == "RGBA":
                inputs = inputs.convert("RGB")
            inputs = trans(inputs)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = torch.Tensor(inputs)
            inputs = inputs.to("cpu", dtype=torch.float)

        results = self._net.infer(inputs={self._input_blobs: inputs.cpu()})
        results = torch.from_numpy(results[self._output_blobs])
        hash_string = compute_hash(results)
        return hash_string

    def launch(self, inputs):
        hash_key = self.infer(inputs)
        results = self.process_outputs(inputs, hash_key)
        return results
