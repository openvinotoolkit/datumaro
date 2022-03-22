# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from importlib.resources import read_text
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

    def __init__(self, output_dir: str, count: int, shape: Tuple[int, int]):
        super().__init__(output_dir, count, shape)
        assert len(self._shape) == 2
        self._height, self._width = self._shape

        if self._height < 13 or self._width < 13:
            raise ValueError(
                "Image generation with height or width of less than 13 is not supported"
            )

        self._weights = self._create_weights(IFSFunction.NUM_PARAMS)
        self._threshold = 0.2
        self._iterations = 200000

        self._path = os.getcwd()
        download_colorization_model(self._path)
        self._initialize_params()

    def generate_dataset(self) -> None:
        log.info(
            "Generating 3-channel images with height = '%d' and width = '%d'",
            self._height,
            self._width,
        )

        with Pool(processes=self._cpu_count) as pool:
            params = pool.map(self._generate_category, [Random(i) for i in range(self._categories)])

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
            generation_params.append((Random(i), param, w, indices))

        with Pool(processes=self._cpu_count) as pool:
            pool.starmap(self._generate_image_batch, generation_params)

    def _generate_image_batch(
        self, rng: Random, params: np.ndarray, weights: np.ndarray, indices: List[int]
    ) -> None:
        proto = osp.join(self._path, "colorization_deploy_v2.prototxt")
        model = osp.join(self._path, "colorization_release_v2.caffemodel")
        npy = osp.join(self._path, "pts_in_hull.npy")
        pts_in_hull = np.load(npy).transpose().reshape(2, 313, 1, 1).astype(np.float32)

        content = read_text(__package__, "synthetic_background.txt")
        source = content.replace(" ", "").split(",")
        synthetic_background = np.array(list(map(float, source))).reshape(-1, 3)

        net = cv.dnn.readNetFromCaffe(proto, model)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [pts_in_hull]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, np.float32)]

        for i, param, w in zip(indices, params, weights):
            image = self._generate_image(
                rng, param, self._iterations, self._height, self._width, draw_point=False, weight=w
            )
            color_image = colorize(image, net)
            aug_image = augment(rng, color_image, synthetic_background)
            cv.imwrite(osp.join(self._output_dir, "{:06d}.png".format(i)), aug_image)

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
            ifs_function.add_param(param[:6], param[6], weight)
        ifs_function.calculate(iterations)
        img = ifs_function.draw(height, width, draw_point)
        return img

    def _generate_category(self, rng: Random, base_h: int = 512, base_w: int = 512) -> np.ndarray:
        pixels = -1
        while pixels < self._threshold:
            param_size = rng.randint(2, 7)
            params = np.zeros((param_size, 7), dtype=np.float32)

            sum_proba = 1e-5
            for i in range(param_size):
                a, b, c, d, e, f = [rng.uniform(-1.0, 1.0) for _ in range(6)]
                prob = abs(a * d - b * c)
                sum_proba += prob
                params[i] = a, b, c, d, e, f, prob
            params[:, 6] /= sum_proba

            fractal_img = self._generate_image(rng, params, self._num_of_points, base_h, base_w)
            pixels = np.count_nonzero(fractal_img) / (base_h * base_w)
        return params

    def _initialize_params(self) -> None:
        default_img_size = 362 * 362
        points_coeff = max(1, int(np.round(self._height * self._width / default_img_size)))
        self._num_of_points = 100000 * points_coeff

        instances_categories = np.ceil(self._count / self._weights.shape[0])
        self._instances = np.ceil(np.sqrt(instances_categories)).astype(np.int)
        self._categories = np.floor(np.sqrt(instances_categories)).astype(np.int)

        if self._count < self._weights.shape[0]:
            self._weights = self._weights[: self._count, :]

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
        weights = np.array(weight_vectors).reshape(-1, 6)
        return weights
