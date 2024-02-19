import os.path as osp
from collections import namedtuple
from unittest import TestCase

import cv2
import numpy as np
import pytest

from datumaro.components.algorithms.rise import RISE
from datumaro.components.annotation import Bbox, Label
from datumaro.components.launcher import LauncherWithModelInterpreter
from datumaro.plugins.openvino_plugin.launcher import OpenvinoLauncher

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path


class RiseTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_rise_can_be_applied_to_classification_model(self):
        model = OpenvinoLauncher(
            model_name="googlenet-v4-tf",
            output_layers="InceptionV4/Logits/PreLogitsFlatten/flatten_1/Reshape:0",
        )

        rise = RISE(model, num_masks=10, mask_size=7, prob=0.5)

        image = cv2.imread(osp.join(get_test_asset_path("rise"), "catdog.png"))
        saliency = next(rise.apply(image))

        logit_size = model.outputs[0].shape
        self.assertEqual(saliency.shape[0], logit_size[1])

        image_size = model.inputs[0].shape
        self.assertEqual(saliency.shape[1], image_size[1])
        self.assertEqual(saliency.shape[2], image_size[2])

        class_indices = [244, 282]  # bullmastiff and tabby of imagenet.class
        rois = [[90, 10, 190, 110], [170, 150, 250, 270]]  # location of bullmastiff and tabby

        for cls_idx in range(len(class_indices)):
            norm_saliency = saliency[class_indices[cls_idx]]
            roi = rois[cls_idx]
            saliency_dense_roi = (
                np.sum(norm_saliency[roi[1] : roi[3], roi[0] : roi[2]])
                / (roi[3] - roi[1])
                / (roi[2] - roi[0])
            )
            saliency_dense_total = (
                np.sum(norm_saliency) / norm_saliency.shape[0] / norm_saliency.shape[1]
            )

            self.assertLess(saliency_dense_total, saliency_dense_roi)

    @pytest.mark.xfail(reason="Broken unit test and need to reimplement RISE algorithm")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_rise_can_be_applied_to_detection_model(self):
        ROI = namedtuple("ROI", ["threshold", "x", "y", "w", "h", "label"])

        class TestLauncher(LauncherWithModelInterpreter):
            def __init__(self, rois, class_count, fp_count=4, pixel_jitter=20, **kwargs):
                self.rois = rois
                self.roi_base_sums = [
                    None,
                ] * len(rois)
                self.class_count = class_count
                self.fp_count = fp_count
                self.pixel_jitter = pixel_jitter

            @staticmethod
            def roi_value(roi, image):
                return np.sum(image[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w, :])

            def preprocess(self, img):
                return img, None

            def postprocess(self, pred, info):
                roi_sums = pred
                detections = []
                for i, roi in enumerate(self.rois):
                    roi_sum = roi_sums[i]
                    roi_base_sum = self.roi_base_sums[i]
                    first_run = roi_base_sum is None
                    if first_run:
                        roi_base_sum = roi_sum
                        self.roi_base_sums[i] = roi_base_sum

                    cls_conf = roi_sum / roi_base_sum

                    if roi.threshold < roi_sum / roi_base_sum:
                        cls = roi.label
                        detections.append(
                            Bbox(
                                roi.x,
                                roi.y,
                                roi.w,
                                roi.h,
                                label=cls,
                                attributes={"score": cls_conf},
                            )
                        )

                    if first_run:
                        continue
                    for j in range(self.fp_count):
                        if roi.threshold < cls_conf:
                            cls = roi.label
                        else:
                            cls = (i + j) % self.class_count
                        box = [roi.x, roi.y, roi.w, roi.h]
                        offset = (np.random.rand(4) - 0.5) * self.pixel_jitter
                        detections.append(
                            Bbox(*(box + offset), label=cls, attributes={"score": cls_conf})
                        )

                return detections

            def infer(self, inputs):
                for inp in inputs:
                    yield self._process(inp)

            def _process(self, image):
                roi_sums = []
                for _, roi in enumerate(self.rois):
                    roi_sum = self.roi_value(roi, image)
                    roi_sums += [roi_sum]

                return roi_sums

        rois = [
            ROI(0.3, 10, 40, 30, 10, 0),
            ROI(0.5, 70, 90, 7, 10, 0),
            ROI(0.7, 5, 20, 40, 60, 2),
            ROI(0.9, 30, 20, 10, 40, 1),
        ]
        model = model = TestLauncher(class_count=3, rois=rois)

        rise = RISE(model, max_samples=(7 * 7) ** 2, mask_width=7, mask_height=7)

        image = np.ones((100, 100, 3))
        heatmaps = next(rise.apply(image))
        heatmaps_class_count = len(set([roi.label for roi in rois]))
        self.assertEqual(heatmaps_class_count + len(rois), len(heatmaps))

        # import cv2
        # roi_image = image.copy()
        # for i, roi in enumerate(rois):
        #     cv2.rectangle(roi_image, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (32 * i) * 3)
        # cv2.imshow('img', roi_image)

        for c in range(heatmaps_class_count):
            class_roi = np.zeros(image.shape[:2])
            for i, roi in enumerate(rois):
                if roi.label != c:
                    continue
                class_roi[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] += roi.threshold

            heatmap = heatmaps[c]

            roi_pixels = heatmap[class_roi != 0]
            h_sum = np.sum(roi_pixels)
            h_area = np.sum(roi_pixels != 0)
            h_den = h_sum / h_area

            rest_pixels = heatmap[class_roi == 0]
            r_sum = np.sum(rest_pixels)
            r_area = np.sum(rest_pixels != 0)
            r_den = r_sum / r_area

            # print(r_den, h_den)
            # cv2.imshow('class %s' % c, heatmap)
            self.assertLess(r_den, h_den)

        for i, roi in enumerate(rois):
            heatmap = heatmaps[heatmaps_class_count + i]
            h_sum = np.sum(heatmap)
            h_area = np.prod(heatmap.shape)
            roi_sum = np.sum(heatmap[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w])
            roi_area = roi.h * roi.w
            roi_den = roi_sum / roi_area
            hrest_den = (h_sum - roi_sum) / (h_area - roi_area)
            # print(hrest_den, h_den)
            # cv2.imshow('roi %s' % i, heatmap)
            self.assertLess(hrest_den, roi_den)
        # cv2.waitKey(0)

    @staticmethod
    def DISABLED_test_roi_nms():
        ROI = namedtuple("ROI", ["conf", "x", "y", "w", "h", "label"])

        class_count = 3
        noisy_count = 3
        rois = [
            ROI(0.3, 10, 40, 30, 10, 0),
            ROI(0.5, 70, 90, 7, 10, 0),
            ROI(0.7, 5, 20, 40, 60, 2),
            ROI(0.9, 30, 20, 10, 40, 1),
        ]
        pixel_jitter = 10

        detections = []
        for i, roi in enumerate(rois):
            detections.append(
                Bbox(roi.x, roi.y, roi.w, roi.h, label=roi.label, attributes={"score": roi.conf})
            )

            for j in range(noisy_count):
                cls_conf = roi.conf * j / noisy_count
                cls = (i + j) % class_count
                box = [roi.x, roi.y, roi.w, roi.h]
                offset = (np.random.rand(4) - 0.5) * pixel_jitter
                detections.append(Bbox(*(box + offset), label=cls, attributes={"score": cls_conf}))

        import cv2

        image = np.zeros((100, 100, 3))
        for i, det in enumerate(detections):
            roi = ROI(det.attributes["score"], *det.get_bbox(), det.label)
            p1 = (int(roi.x), int(roi.y))
            p2 = (int(roi.x + roi.w), int(roi.y + roi.h))
            c = (0, 1 * (i % (1 + noisy_count) == 0), 1)
            cv2.rectangle(image, p1, p2, c)
            cv2.putText(
                image,
                "d%s-%s-%.2f" % (i, roi.label, roi.conf),
                p1,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                c,
            )
        cv2.imshow("nms_image", image)
        cv2.waitKey(0)

        nms_boxes = RISE.nms(detections, iou_thresh=0.25)
        print(len(detections), len(nms_boxes))

        for i, det in enumerate(nms_boxes):
            roi = ROI(det.attributes["score"], *det.get_bbox(), det.label)
            p1 = (int(roi.x), int(roi.y))
            p2 = (int(roi.x + roi.w), int(roi.y + roi.h))
            c = (0, 1, 0)
            cv2.rectangle(image, p1, p2, c)
            cv2.putText(
                image,
                "p%s-%s-%.2f" % (i, roi.label, roi.conf),
                p1,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                c,
            )
        cv2.imshow("nms_image", image)
        cv2.waitKey(0)
