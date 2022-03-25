# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
import sys
from importlib.resources import open_text
from multiprocessing import Pool
from random import Random
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from datumaro.components.dataset_generator import DatasetGenerator

from .utils import IFSFunction, augment, colorize, download_colorization_model


class ImageGenerator(DatasetGenerator):
    """
    ImageGenerator generates 3-channel synthetic images with provided shape.
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-o",
            "--output-dir",
            type=osp.abspath,
            required=True,
            help="Path to the directory where dataset are saved",
        )
        parser.add_argument(
            "-k", "--count", type=int, required=True, help="Number of data to generate"
        )
        parser.add_argument(
            "--shape",
            nargs="+",
            type=int,
            required=True,
            help="Data shape. For example, for image with height = 256 and width = 224: --shape 256 224",
        )

        parser.add_argument(
            "--model-path",
            type=osp.abspath,
            help="Path where colorization model is located or path to save model",
        )

        return parser

    def __init__(self, output_dir: str, count: int, shape: Tuple[int, int], model_path: str = None):
        self._count = count
        self._cpu_count = min(os.cpu_count(), self._count)
        assert len(shape) == 2
        self._height, self._width = shape

        self._output_dir = output_dir
        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        self._weights = self._create_weights(IFSFunction.NUM_PARAMS)
        self._threshold = 0.2
        self._iterations = 200000
        self._num_of_points = 100000

        self._path = model_path if model_path is not None else os.getcwd()
        download_colorization_model(self._path)
        self._initialize_params()

    def generate_dataset(self) -> None:
        log.warning(
            "Generation of '%d' 3-channel images with height = '%d' and width = '%d'",
            self._count,
            self._height,
            self._width,
        )
        use_multiprocessing = sys.platform != "darwin" or sys.version_info > (3, 7)
        if use_multiprocessing:
            with Pool(processes=self._cpu_count) as pool:
                params = pool.map(
                    self._generate_category, [Random(i) for i in range(self._categories)]
                )
        else:
            params = []
            for i in range(self._categories):
                param = self._generate_category(Random(i))
                params.append(param)

        instances_weights = np.repeat(self._weights, self._instances, axis=0)
        weight_per_img = np.tile(instances_weights, (self._categories, 1))
        repeated_params = np.repeat(params, self._weights.shape[0] * self._instances, axis=0)
        repeated_params = repeated_params[: self._count]
        weight_per_img = weight_per_img[: self._count]
        assert weight_per_img.shape[0] == len(repeated_params) == self._count

        splits = min(self._cpu_count, self._count)
        params_per_proc = np.array_split(repeated_params, splits)
        weights_per_proc = np.array_split(weight_per_img, splits)

        generation_params = []
        offset = 0
        for i, (param, w) in enumerate(zip(params_per_proc, weights_per_proc)):
            indices = list(range(offset, offset + len(param)))
            offset += len(param)
            generation_params.append((i, Random(i), param, w, indices))

        if use_multiprocessing:
            with Pool(processes=self._cpu_count) as pool:
                pool.starmap(self._generate_image_batch, generation_params)
        else:
            for i, param in enumerate(generation_params):
                self._generate_image_batch(*param)

    def _generate_image_batch(
        self, job_id: int, rng: Random, params: np.ndarray, weights: np.ndarray, indices: List[int]
    ) -> None:
        print(f"{job_id}: started")
        proto = osp.join(self._path, "colorization_deploy_v2.prototxt")
        model = osp.join(self._path, "colorization_release_v2.caffemodel")
        npy = osp.join(self._path, "pts_in_hull.npy")
        pts_in_hull = np.load(npy).transpose().reshape(2, 313, 1, 1).astype(np.float32)

        content = open_text(__package__, "synthetic_background.txt")
        synthetic_background = np.loadtxt(content)

        print(f"Job {job_id}: loading caffe model")
        net = cv.dnn.readNetFromCaffe(proto, model)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [pts_in_hull]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, np.float32)]

        print(f"Job {job_id}: starting loop")
        for i, param, w in zip(indices, params, weights):
            print(f"Job {job_id}: processing image #{i}")
            image = self._generate_image(
                rng, param, self._iterations, self._height, self._width, draw_point=False, weight=w
            )
            color_image = colorize(image, net)
            aug_image = augment(rng, color_image, synthetic_background)
            cv.imwrite(osp.join(self._output_dir, "{:06d}.png".format(i)), aug_image)
        print(f"Job {job_id}: finished")

    def _generate_image(
        self,
        rng: Random,
        params: np.ndarray,
        iterations: int,
        height: int,
        width: int,
        draw_point: bool = True,
        weight: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ifs_function = IFSFunction(rng, prev_x=0.0, prev_y=0.0)
        for param in params:
            ifs_function.add_param(
                param[: ifs_function.NUM_PARAMS], param[ifs_function.NUM_PARAMS], weight
            )
        ifs_function.calculate(iterations)
        img = ifs_function.draw(height, width, draw_point)
        return img

    def _generate_category(self, rng: Random, base_h: int = 512, base_w: int = 512) -> np.ndarray:
        pixels = -1
        i = 0
        while pixels < self._threshold and i < self._iterations:
            param_size = rng.randint(2, 7)
            params = np.zeros((param_size, IFSFunction.NUM_PARAMS + 1), dtype=np.float32)

            sum_proba = 1e-5
            for p_idx in range(param_size):
                a, b, c, d, e, f = [rng.uniform(-1.0, 1.0) for _ in range(IFSFunction.NUM_PARAMS)]
                prob = abs(a * d - b * c)
                sum_proba += prob
                params[p_idx] = a, b, c, d, e, f, prob
            params[:, IFSFunction.NUM_PARAMS] /= sum_proba

            fractal_img = self._generate_image(rng, params, self._num_of_points, base_h, base_w)
            pixels = np.count_nonzero(fractal_img) / (base_h * base_w)
            i += 1
        return params

    def _initialize_params(self) -> None:
        if self._count < self._weights.shape[0]:
            self._weights = self._weights[: self._count, :]

        instances_categories = np.ceil(self._count / self._weights.shape[0])
        self._instances = np.ceil(np.sqrt(instances_categories)).astype(np.int)
        self._categories = np.ceil(instances_categories / self._instances).astype(np.int)

    @staticmethod
    def _create_weights(num_params):
        # weights from https://openaccess.thecvf.com/content/ACCV2020/papers/Kataoka_Pre-training_without_Natural_Images_ACCV_2020_paper.pdf
        BASE_WEIGHTS = np.ones((num_params,))
        WEIGHT_INTERVAL = 0.4
        INTERVAL_MULTIPLIERS = (-2, -1, 1, 2)
        weight_vectors = [BASE_WEIGHTS]

        for weight_index in range(num_params):
            for multiplier in INTERVAL_MULTIPLIERS:
                modified_weights = BASE_WEIGHTS.copy()
                modified_weights[weight_index] += multiplier * WEIGHT_INTERVAL
                weight_vectors.append(modified_weights)
        weights = np.array(weight_vectors)
        return weights
