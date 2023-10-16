# Copyright (C) 2023 Intel Corporation
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# We implemented some of this code by referring to the codebase available at
# https://github.com/facebookresearch/segment-anything.
# This code is licensed under the Apache License 2.0, which can be found in the LICENSE file at
# https://github.com/facebookresearch/segment-anything/blob/main/LICENSE.

from typing import List, Tuple

import numpy as np
from attr import attrs, field

from datumaro.components.abstracts import IModelInterpreter
from datumaro.components.abstracts.model_interpreter import LauncherInputType, ModelPred, PrepInfo
from datumaro.components.annotation import Annotation, FeatureVector, Mask
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.util.annotation_util import nms


@attrs(slots=True, kw_only=True, order=False)
class AMGPoints(Annotation):
    """Intermediate annotation class for SAM decoder inputs.

    Attributes:
        points: Array of points (x, y) for the SAM prompt.
    """

    points: np.ndarray = field()


@attrs(slots=True, kw_only=True, order=False)
class AMGMasks(Annotation):
    """Intermediate annotation class for SAM decoder outputs.

    Attributes:
        masks: Array of masks corresponded to the points.
        iou_preds: Array of Intersection over Union (IoU)
            prediction scores corresponded to the points.
    """

    masks: np.ndarray = field()
    iou_preds: np.ndarray = field()

    @classmethod
    def cat(cls, masks: List["AMGMasks"]) -> "AMGMasks":
        """Concatenate a list of `AMGMasks` into a single `AMGMasks` object.

        Parameters:
            masks: List of `AMGMasks` to concatenate.

        Returns:
            A new AMGMasks containing the concatenated masks and IoU prediction scores.
        """
        return AMGMasks(
            masks=np.concatenate([mask.masks for mask in masks], axis=0),
            iou_preds=np.concatenate([mask.iou_preds for mask in masks], axis=0),
        )

    def postprocess(
        self,
        mask_threshold: float,
        pred_iou_thresh: float,
        stability_score_offset: float,
        stability_score_thresh: float,
        box_nms_thresh: float,
        min_mask_region_area: int,
    ) -> List[Mask]:
        """Postprocesses the masks with the given parameters.

        Parameters:
            pred_iou_thresh (float): A filtering threshold in [0,1], using the
                model's predicted mask quality.
            stability_score_thresh (float): A filtering threshold in [0,1], using
                the stability of the mask under changes to the cutoff used to binarize
                the model's mask predictions.
            stability_score_offset (float): The amount to shift the cutoff when
                calculated the stability score.
            box_nms_thresh (float): The box IoU cutoff used by non-maximal
                suppression to filter duplicate masks.
            min_mask_region_area (int): If >0, postprocessing will be applied
                to remove the binary mask which has the number of 1s less than min_mask_region_area.

        Returns:
            List of :class:`Mask`s representing the postprocessed masks.
        """
        masks, iou_preds = self.masks, self.iou_preds

        if pred_iou_thresh > 0.0:
            keep_mask = iou_preds > pred_iou_thresh
            masks = masks[keep_mask]
            iou_preds = iou_preds[keep_mask]

        if stability_score_thresh > 0.0:
            keep_mask = (
                self._calculate_stability_score(
                    masks=masks,
                    mask_threshold=mask_threshold,
                    stability_score_offset=stability_score_offset,
                )
                > stability_score_thresh
            )
            masks = masks[keep_mask]
            iou_preds = iou_preds[keep_mask]

        binary_masks = masks > mask_threshold

        if min_mask_region_area > 0:
            keep_mask = binary_masks.sum(axis=(1, 2)) > min_mask_region_area
            binary_masks = binary_masks[keep_mask]
            iou_preds = iou_preds[keep_mask]

        segments = [
            Mask(
                image=binary_mask,
                id=idx,
                group=idx,
                object_id=idx,
                label=0,
                z_order=0,
                attributes={"score": iou_pred},
            )
            for idx, (binary_mask, iou_pred) in enumerate(zip(binary_masks, iou_preds))
        ]
        segments_after_nms = nms(segments=segments, iou_thresh=box_nms_thresh)
        return segments_after_nms

    @staticmethod
    def _calculate_stability_score(
        masks: np.ndarray, mask_threshold: float, stability_score_offset: float
    ) -> np.ndarray:
        """
        Computes the stability score for a batch of masks. The stability
        score is the IoU between the binary masks obtained by thresholding
        the predicted mask logits at high and low values.
        """
        # One mask is always contained inside the other.
        # Save memory by preventing unnecessary cast to torch.int64
        intersections = (
            (masks > (mask_threshold + stability_score_offset))
            .sum(-1, dtype=np.int16)
            .sum(-1, dtype=np.int32)
        )
        unions = (
            (masks > (mask_threshold - stability_score_offset))
            .sum(-1, dtype=np.int16)
            .sum(-1, dtype=np.int32)
        )
        return intersections / unions


class SAMDecoderForAMGInterpreter(IModelInterpreter):
    """Interpreter for the automatic mask generation using SAM decoder."""

    h_model = 1024
    w_model = 1024
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    def preprocess(self, inp: DatasetItem) -> Tuple[LauncherInputType, PrepInfo]:
        img_size = inp.media_as(Image).size

        img_embed = inp.annotations[-1]

        assert isinstance(
            img_embed, FeatureVector
        ), "annotations should have the image embedding vector as FeatureVector."

        amg_points = inp.annotations[-2]

        h_img, w_img = img_size

        points = amg_points.points
        points[:, 0] *= self.w_model
        points[:, 1] *= self.h_model

        scale = min(self.h_model / h_img, self.w_model / w_img)

        if h_img <= w_img:
            points[:, 1] *= scale
        else:
            points[:, 0] *= scale

        onnx_coord = np.concatenate(
            [points.reshape(-1, 1, 2), np.zeros_like(points).reshape(-1, 1, 2)], axis=1
        ).astype(np.float32)
        onnx_label = np.array([1, -1] * len(points)).reshape(-1, 2).astype(np.float32)

        decoder_inputs = {
            "image_embeddings": img_embed.vector[None, :, :, :],
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": self.onnx_mask_input,
            "has_mask_input": self.onnx_has_mask_input,
            "orig_im_size": np.array(img_size, dtype=np.float32),
        }

        return decoder_inputs, None

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        """Postprocesses the outputs of the SAM decoder to generate masks automatically
        from the prompts which have a point uniformly distributed on a 2d grid.

        Parameters:
            pred: List of dictionaries containing model predictions. Each dictionary should have the 'masks'
            and 'iou_preds' keys. 'masks' is corresponding to the predicted mask of which shape is (1, H, W).
            'iou_preds' is corresponding to the scalar IoU prediction score.
            info: None

        Returns:
            List of :class:`AMGMasks` produced by the SAM decoder.
        """
        masks = np.concatenate([p["masks"] for p in pred], axis=0)
        iou_preds = np.concatenate([p["iou_predictions"] for p in pred], axis=0)
        return [AMGMasks(masks=masks, iou_preds=iou_preds)]

    def get_categories(self):
        return None
