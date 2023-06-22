# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=unused-variable

import cv2
import numpy as np
from openvino.runtime import Core

__all__ = ["ReciproCAM"]


class ReciproCAM:
    """
    Implements ReciproCAM: Gradient-Free Reciprocal Class Activation Map
    Glass-boxed explanation of models
    See explanations at: https://arxiv.org/pdf/2209.14074.pdf
    """

    def __init__(
        self,
        backbone,
        head,
        dsize,
        use_gaussian,
    ):
        self.backbone = backbone
        self.head = head
        self.dsize = dsize
        self.use_gaussian = use_gaussian

        ie = Core()
        self.compiled_backbone = ie.compile_model(model=backbone, device_name="CPU")
        self.backbone_output_layer = self.compiled_backbone.output(0)

        self.compiled_head = ie.compile_model(model=head, device_name="CPU")
        self.head_output_layer = self.compiled_head.output(0)

    @staticmethod
    def mosaic_feature(feature, use_gaussian=False):
        _, height, width, num_channels = feature.shape
        purturbed_feature = np.zeros((height * width, height, width, num_channels))
        if use_gaussian is False:
            for b in range(height * width):
                for i in range(height):
                    for j in range(width):
                        if b == i * width + j:
                            purturbed_feature[b, i, j, :] = feature[0, i, j, :]
            return purturbed_feature

        GAUSSIAN_FILTER = np.array(
            [
                [1 / 16.0, 1 / 8.0, 1 / 16.0],
                [1 / 8.0, 1 / 4.0, 1 / 8.0],
                [1 / 16.0, 1 / 8.0, 1 / 16.0],
            ]
        ).repeat(num_channels, axis=0)

        for b in range(height * width):
            for i in range(height):  # 0...h-1
                kx_s = max(i - 1, 0)
                kx_e = min(i + 1, height - 1)
                if i == 0:
                    sx_s = 1
                else:
                    sx_s = 0
                if i == height - 1:
                    sx_e = 1
                else:
                    sx_e = 2
                for j in range(width):  # 0...w-1
                    ky_s = max(j - 1, 0)
                    ky_e = min(j + 1, width - 1)
                    if j == 0:
                        sy_s = 1
                    else:
                        sy_s = 0
                    if j == width - 1:
                        sy_e = 1
                    else:
                        sy_e = 2
                    if b == i * width + j:
                        r_feature_map = feature[0, i, j, :].reshape(num_channels, 1, 1)
                        r_feature_map = r_feature_map.repeat(1, 3, 3)
                        score_map = r_feature_map * GAUSSIAN_FILTER
                        purturbed_feature[b, kx_s : kx_e + 1, ky_s : ky_e + 1, :] = score_map[
                            :, sx_s : sx_e + 1, sy_s : sy_e + 1
                        ]

    def generate_cam(self, feature, class_idx: int):
        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        _, height, width, _ = feature.shape
        cam = np.zeros((height, width))
        for idx, feat in enumerate(feature):
            mosaic_infer = self.compiled_head([np.expand_dims(feat, axis=0)])[
                self.head_output_layer
            ]
            mosaic_predict = softmax(mosaic_infer)
            i, j = idx // width, idx % width
            cam[i, j] = mosaic_predict[:, class_idx]

        cam_min = cam.min()
        cam = (cam - cam_min) / (cam.max() - cam_min)

        return cam

    def apply(self, image, target):
        assert len(image.shape) in [2, 3], "Expected an input image in (H, W, C) format"
        if len(image.shape) == 3:
            assert image.shape[2] in [3, 4], "Expected BGR or BGRA input"
        image = image[:, :, :3].astype(np.float32)

        # Resize to MobileNet image shape.
        input_image = cv2.resize(image, dsize=self.dsize)
        input_image = np.expand_dims(input_image, 0)

        # Run backbone network to extract features
        extracted_feature = self.compiled_backbone([input_image])[self.backbone_output_layer]

        # Purturb the feature map to get mosaic feature
        purturbed_feature = self.mosaic_feature(extracted_feature, self.use_gaussian)

        # Compute the class activation map by inferring mosaic feature into the head network
        cam = self.generate_cam(feature=purturbed_feature, class_idx=target)

        # Resize into the original image size
        resized_cam = cv2.resize(
            cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
        )

        # Normalize the saliency map values to the range [0, 255]
        normalized_cam = cv2.normalize(resized_cam, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        yield normalized_cam
