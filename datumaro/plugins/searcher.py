# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
import urllib

import cv2
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from datumaro.components.errors import MediaTypeError
from datumaro.components.media import Image
from datumaro.plugins.openvino_plugin.launcher import OpenvinoLauncher
from datumaro.util.samples import get_samples_path


def download_file(url: str, file_root: str):
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


def compute_hash(features):
    features = np.sign(features)
    hash_key = np.clip(features, 0, None)
    hash_key = hash_key.astype(np.uint8)
    hash_key = np.packbits(hash_key, axis=-1)
    return hash_key


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


def tokenize(texts: str, context_length: int = 77, truncate: bool = True):
    checkpoint = "openai/clip-vit-base-patch32"
    tokenizer = Tokenizer.from_pretrained(checkpoint)
    tokens = tokenizer.encode(texts).ids
    result = np.zeros((1, context_length))

    if len(tokens) > context_length:
        if truncate:
            eot_token = tokens.ids[-1]
            tokens = tokens[:context_length]
            tokens[-1] = eot_token

    for i, token in enumerate(tokens):
        result[:, i] = token
    return result


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
            model_dir = os.path.join(os.path.expanduser("~"), ".cache/datumaro")
            if not osp.exists(model_dir):
                os.mkdir(model_dir)

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
            openvino_plugin_samples_dir = get_samples_path()
            interpreter = osp.join(openvino_plugin_samples_dir, "clip_ViT-B_32_interp.py")

        super().__init__(description, weights, interpreter, device, model_dir, output_layers)

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
            inputs = tokenize(prompt_text)
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
        hash_key = compute_hash(results[self._output_blobs])
        return hash_key

    def launch(self, inputs):
        hash_key = self.infer(inputs)
        results = self.process_outputs(inputs, hash_key)
        return results

    def type_check(self, item):
        if not isinstance(item.media, Image):
            raise MediaTypeError(f"Media type should be Image, Current type={type(item.media)}")
        return True
