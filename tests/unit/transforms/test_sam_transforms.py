# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from random import randint
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from datumaro.components.annotation import Bbox, FeatureVector, Mask, Polygon
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.inference_server_plugin.base import InferenceServerType
from datumaro.plugins.sam_transforms import SAMAutomaticMaskGeneration, SAMBboxToInstanceMask
from datumaro.plugins.sam_transforms.interpreters.sam_decoder_for_amg import (
    AMGMasks,
    AMGPoints,
    SAMDecoderForAMGInterpreter,
)
from datumaro.plugins.sam_transforms.interpreters.sam_decoder_for_bbox import (
    SAMDecoderForBboxInterpreter,
)
from datumaro.plugins.sam_transforms.interpreters.sam_encoder import SAMEncoderInterpreter


def create_bbox_item(n_bboxes: int, n_labels: int, idx: int) -> DatasetItem:
    media = Image.from_numpy(np.zeros(shape=[randint(5, 10), randint(5, 10), 3], dtype=np.uint8))
    h, w = media.size

    annotations = []
    for ann_id in range(n_bboxes):
        x1, x2 = tuple(sorted(randint(0, w - 1) for _ in range(2)))
        y1, y2 = tuple(sorted(randint(0, h - 1) for _ in range(2)))
        annotations.append(
            Bbox(
                x=x1,
                y=y1,
                w=x2 - x1,
                h=y2 - y1,
                id=ann_id,
                label=randint(0, n_labels - 1),
                group=randint(0, 10),
                object_id=randint(0, 10),
                attributes={"dummy": 0},
                z_order=randint(0, 10),
            ),
        )

    return DatasetItem(id=f"test_{idx}", media=media, annotations=annotations)


@pytest.mark.new
class SAMModelInterpreterTest:
    def test_encoder(self):
        interp = SAMEncoderInterpreter()
        item = create_bbox_item(10, 5, 0)

        inp, scale = interp.preprocess(item)

        assert isinstance(inp, np.ndarray)
        assert inp.shape[1] <= interp.h_model and inp.shape[2] <= interp.w_model
        assert inp.shape[1] == interp.h_model or inp.shape[2] == interp.w_model
        assert inp.shape[1] == int(scale * item.media_as(Image).size[0])
        assert inp.shape[2] == int(scale * item.media_as(Image).size[1])

        vector = np.random.normal(0, 1, (10,)).astype(np.float32)
        output = interp.postprocess({"image_embeddings": vector}, scale)

        assert all(isinstance(o, FeatureVector) for o in output)
        assert len(output) == 1
        assert np.allclose(output[0].vector, vector)

    def test_decoder_for_bbox(self):
        interp = SAMDecoderForBboxInterpreter()
        item = create_bbox_item(10, 5, 0)

        vector = np.random.normal(0, 1, (256, 64, 64)).astype(np.float32)
        n_bboxes = len(item.annotations)
        item.annotations += [FeatureVector(vector=vector)]

        inp, prep_info = interp.preprocess(item)

        assert isinstance(inp, dict)
        expect_keys = (
            "image_embeddings",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
            "orig_im_size",
        )
        assert set(expect_keys) == set(inp.keys())
        # Bboxes should be batched in the prompt, so that the first dimension should be n_bboxes
        assert len(inp.get("point_coords")) == n_bboxes
        assert len(inp.get("point_labels")) == n_bboxes
        # Prep info should be the given bboxes in the dataset item
        assert len(prep_info) == n_bboxes
        assert all(isinstance(info, Bbox) for info in prep_info)

        output = interp.postprocess(
            [
                {
                    "masks": np.random.normal(
                        0,
                        1,
                        size=(1, *item.media_as(Image).size),
                    ).astype(np.float32)
                }
                for _ in range(n_bboxes)
            ],
            prep_info,
        )

        assert all(isinstance(o, Mask) for o in output)
        assert len(output) == n_bboxes

    def test_decoder_for_amg(self):
        interp = SAMDecoderForAMGInterpreter()
        item = create_bbox_item(10, 5, 0)

        vector = np.random.normal(0, 1, (256, 64, 64)).astype(np.float32)
        amg_points = AMGPoints(points=np.zeros([10, 2]))
        n_points = len(amg_points.points)
        item.annotations += [amg_points, FeatureVector(vector=vector)]

        inp, prep_info = interp.preprocess(item)

        assert isinstance(inp, dict)
        expect_keys = (
            "image_embeddings",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
            "orig_im_size",
        )
        assert set(expect_keys) == set(inp.keys())
        # Points should be batched in the prompt, so that the first dimension should be n_points
        assert len(inp.get("point_coords")) == n_points
        assert len(inp.get("point_labels")) == n_points
        # Prep info should be None
        assert prep_info is None

        outputs = interp.postprocess(
            [
                {
                    "masks": np.random.normal(
                        0,
                        1,
                        size=(1, *item.media_as(Image).size),
                    ).astype(np.float32),
                    "iou_predictions": np.random.uniform(0, 1, size=(1,)),
                }
                for _ in range(n_points)
            ],
            prep_info,
        )

        assert all(isinstance(o, AMGMasks) for o in outputs)
        for o in outputs:
            assert len(o.masks) == n_points
            assert len(o.iou_preds) == n_points


class TestBase:
    @pytest.fixture
    def fxt_dataset(self, n_items: int = 10, n_bboxes: int = 5, n_labels: int = 3) -> Dataset:
        return Dataset.from_iterable(
            create_bbox_item(n_bboxes, n_labels, idx) for idx in range(n_items)
        )

    @pytest.fixture(params=list(InferenceServerType))
    def fxt_inference_server_type(self, request):
        return request.param


class SAMBboxToPolygonTest(TestBase):
    @pytest.fixture(params=[True, False])
    def fxt_to_polygon(self, request):
        return request.param

    @pytest.mark.new
    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_transform(self, fxt_dataset, fxt_inference_server_type, fxt_to_polygon, num_workers):
        if fxt_inference_server_type == InferenceServerType.ovms:
            launcher_str = "OVMSLauncher"
        elif fxt_inference_server_type == InferenceServerType.triton:
            launcher_str = "TritonLauncher"
        else:
            raise ValueError

        with patch(
            f"datumaro.plugins.sam_transforms.bbox_to_inst_mask.{launcher_str}"
        ) as mock_launcher:
            mock_sam_encoder = MagicMock()
            mock_sam_decoder = MagicMock()
            mock_launcher.side_effect = [mock_sam_encoder, mock_sam_decoder]

            transform = SAMBboxToInstanceMask(
                extractor=fxt_dataset,
                inference_server_type=fxt_inference_server_type,
                to_polygon=fxt_to_polygon,
                num_workers=num_workers,
            )

            mock_sam_encoder.launch.return_value = [[FeatureVector(vector=np.zeros([10]))]]

            def _mock_decoder_launch(items, stack):
                return [
                    [
                        Mask(
                            image=np.ones(item.media_as(Image).size, dtype=np.uint8),
                            id=ann.id,
                            group=ann.group,
                            object_id=ann.object_id,
                            label=ann.label,
                            z_order=ann.z_order,
                            attributes=ann.attributes,
                        )
                        for ann in item.annotations
                        if not isinstance(ann, FeatureVector)
                    ]
                    for item in items
                ]

            mock_sam_decoder.launch.side_effect = _mock_decoder_launch
            assert len(fxt_dataset) == len(transform)
            for e_item, a_item in zip(fxt_dataset, transform):
                assert len(e_item.annotations) == len(a_item.annotations)

                for e_ann, a_ann in zip(e_item.annotations, a_item.annotations):
                    assert isinstance(e_ann, Bbox) and isinstance(
                        a_ann, Polygon if fxt_to_polygon else Mask
                    )
                    assert e_ann.id == a_ann.id
                    assert e_ann.group == a_ann.group
                    assert e_ann.label == a_ann.label
                    assert e_ann.attributes == a_ann.attributes
                    assert e_ann.z_order == a_ann.z_order
                    assert e_ann.object_id == a_ann.object_id


class SAMAutomaticMaskGenerationTest(TestBase):
    @pytest.mark.new
    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_transform(self, fxt_dataset, fxt_inference_server_type, num_workers):
        points_per_side = 2
        points_per_batch = 4
        mask_threshold = 0.1
        pred_iou_thresh = 0.1
        stability_score_thresh = 0.1
        stability_score_offset = 0.0
        box_nms_thresh = 0.4
        min_mask_region_area = 1

        if fxt_inference_server_type == InferenceServerType.ovms:
            launcher_str = "OVMSLauncher"
        elif fxt_inference_server_type == InferenceServerType.triton:
            launcher_str = "TritonLauncher"
        else:
            raise ValueError

        with patch(
            f"datumaro.plugins.sam_transforms.automatic_mask_gen.{launcher_str}"
        ) as mock_launcher:
            mock_sam_encoder = MagicMock()
            mock_sam_decoder = MagicMock()
            mock_launcher.side_effect = [mock_sam_encoder, mock_sam_decoder]

            transform = SAMAutomaticMaskGeneration(
                extractor=fxt_dataset,
                inference_server_type=fxt_inference_server_type,
                num_workers=num_workers,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                mask_threshold=mask_threshold,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                stability_score_offset=stability_score_offset,
                box_nms_thresh=box_nms_thresh,
                min_mask_region_area=min_mask_region_area,
            )

            assert transform.points_per_side == points_per_side
            assert transform.points_per_batch == points_per_batch
            assert transform.mask_threshold == mask_threshold
            assert transform.pred_iou_thresh == pred_iou_thresh
            assert transform.stability_score_thresh == stability_score_thresh
            assert transform.stability_score_offset == stability_score_offset
            assert transform.box_nms_thresh == box_nms_thresh
            assert transform.min_mask_region_area == min_mask_region_area

            mock_sam_encoder.launch.return_value = [[FeatureVector(vector=np.zeros([10]))]]

            def _mock_decoder_launch(items, stack):
                return [
                    [
                        AMGMasks(
                            masks=np.random.uniform(
                                0,
                                1,
                                size=(points_per_batch, *item.media_as(Image).size),
                            ),
                            iou_preds=np.random.uniform(
                                0,
                                1,
                                size=(points_per_batch,),
                            ),
                        )
                    ]
                    for item in items
                ]

            mock_sam_decoder.launch.side_effect = _mock_decoder_launch
            assert len(fxt_dataset) == len(transform)
            for o_item, a_item in zip(fxt_dataset, transform):
                # Original dataset has no mask
                assert not any([isinstance(ann, Mask) for ann in o_item.annotations])
                # After the transform, it should have mask
                assert any([isinstance(ann, Mask) for ann in a_item.annotations])
