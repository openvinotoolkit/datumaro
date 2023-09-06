# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Tuple

import numpy as np

from datumaro.components.abstracts import IModelInterpreter
from datumaro.components.abstracts.model_interpreter import LauncherInputType, ModelPred, PrepInfo
from datumaro.components.annotation import Annotation, Bbox, FeatureVector, Mask
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image


class SAMDecoderForBboxInterpreter(IModelInterpreter):
    h_model = 1024
    w_model = 1024
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    mask_threshold = 0.0

    def preprocess(self, inp: DatasetItem) -> Tuple[LauncherInputType, PrepInfo]:
        img = inp.media_as(Image).data

        img_embed = inp.annotations[-1]

        assert isinstance(
            img_embed, FeatureVector
        ), "annotations should have the image embedding vector as FeatureVector."

        h_img, w_img = img.shape[:2]

        scale = min(self.h_model / h_img, self.w_model / w_img)

        bboxes = [ann for ann in inp.annotations if isinstance(ann, Bbox)]
        onnx_coord = scale * np.array(
            [(bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h) for bbox in bboxes],
            dtype=np.float32,
        ).reshape(-1, 2, 2)
        onnx_label = np.array([2, 3] * len(onnx_coord)).reshape(-1, 2).astype(np.float32)

        decoder_inputs = {
            "image_embeddings": img_embed.vector[None, :, :, :],
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": self.onnx_mask_input,
            "has_mask_input": self.onnx_has_mask_input,
            "orig_im_size": np.array(img.shape[:2], dtype=np.float32),
        }

        return decoder_inputs, bboxes

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        """Postprocesses the outputs of the SAM decoder to generate masks associated with bounding boxes.

        Parameters:
            pred: List of dictionaries containing model predictions. Each dictionary should have the 'masks' key
                corresponding to the predicted mask of which shape is (1, H, W).
            info: List of associated bounding boxes obtained from the preprocessing step.

        Returns:
            List of `Mask`s associated with the input bounding boxes. Each Mask object contains the generated mask
            along with metadata from the corresponding `Bbox`:
            `id`, `group`, `object_id`, `label`, `z_order` and `attributes`.
        """
        masks = [np.squeeze(p["masks"], axis=0) > self.mask_threshold for p in pred]

        bboxes: List[Bbox] = info  # Return from preprocess()

        return [
            Mask(
                image=mask,
                id=bbox.id,
                group=bbox.group,
                object_id=bbox.object_id,
                label=bbox.label,
                z_order=bbox.z_order,
                attributes=bbox.attributes,
            )
            for bbox, mask in zip(bboxes, masks)
        ]

    def get_categories(self):
        return None
