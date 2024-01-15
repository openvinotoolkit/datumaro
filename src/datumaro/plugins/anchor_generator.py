from typing import List

import torch

from datumaro.components.dataset import Dataset


class BboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale=1.0, dtype=None):
        self.scale = scale
        self.dtype = dtype

    def _bbox_overlaps(self, bboxes1, bboxes2, eps=1e-6):
        """Calculate overlap between two set of bboxes."""
        # Either the boxes are empty or the length of boxes' last dimension is 4
        assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
        assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

        # Batch dim must be the same
        # Batch dim: (B1, B2, ... Bn)
        assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
        batch_shape = bboxes1.shape[:-2]

        rows = bboxes1.size(-2)
        cols = bboxes2.size(-2)

        if rows * cols == 0:
            return bboxes1.new(batch_shape + (rows, cols))

        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]

        union = area1[..., None] + area2[..., None, :] - overlap

        eps = union.new_tensor([eps])
        union = torch.max(union, eps)
        ious = overlap / union

        return ious

    def __call__(self, bboxes1: torch.Tensor, bboxes2: torch.Tensor):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                in <x1, y1, x2, y2> format, or shape (m, 5) in <x1, y1, x2,
                y2, score> format.
            bboxes2 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                in <x1, y1, x2, y2> format, shape (m, 5) in <x1, y1, x2, y2,
                score> format, or be empty. If ``is_aligned `` is ``True``,
                then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]

        return self._bbox_overlaps(bboxes1, bboxes2)


class DataAwareAnchorGenerator:
    def __init__(self, dataset: Dataset, img_size: tuple(int, int), strides: List[int]):
        self.img_size = img_size
        self.targets = []
        for item in dataset:
            height, width = item.media.data.shape[:2]
            scale_h, scale_w = height / img_size[0], width / img_size[1]

            for ann in item.annotations:
                x, y, w, h = ann.get_bbox()
                x /= scale_w
                y /= scale_h
                w /= scale_w
                h /= scale_h

                self.targets.append([x, y, x + w, y + h])

        self.shifts = []
        for stride in enumerate(strides):
            self.shifts.append(self.get_shifts(stride))

    def get_shifts(self, stride: int) -> torch.Tensor:
        def _meshgrid(x: torch.Tensor, y: torch.Tensor):
            # use shape instead of len to keep tracing while exporting to onnx
            xx = x.repeat(y.shape[0])
            yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
            return xx, yy

        feat_h, feat_w = self.img_size[0] / stride, self.img_size[1] / stride
        shift_x = torch.arange(0, feat_w) * stride
        shift_y = torch.arange(0, feat_h) * stride

        shift_xx, shift_yy = _meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)

        return shifts

    def get_anchors(self, base_size, shifts, scales, ratios) -> torch.Tensor:
        w, h = base_size, base_size
        x_center, y_center = 0.5 * w, 0.5 * h

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios

        ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = [
            x_center - 0.5 * ws,
            y_center - 0.5 * hs,
            x_center + 0.5 * ws,
            y_center + 0.5 * hs,
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        anchors = base_anchors[None, :, :] + shifts[:, None, :]
        anchors = anchors.view(-1, 4)

        return anchors

    def get_loss(self, targets, strides, scales, ratios, pos_thr, neg_thr):
        all_anchors = []
        for level, base_size in enumerate(strides):
            anchors = self.get_anchors(base_size, self.shifts[level], scales[level], ratios[level])
            all_anchors.append(anchors)
        all_anchors = torch.cat(all_anchors, dim=0)

        iou_calculator = BboxOverlaps2D()
        overlaps = iou_calculator(targets, all_anchors)

        max_overlaps, _ = overlaps.max(dim=0)
        print(
            f"[ANCHOR] total: {overlaps.shape[1]}, "
            f"pos: {sum(max_overlaps >= pos_thr)}, "
            f"neg: {sum(max_overlaps < neg_thr)}"
        )

        gt_max_overlaps, _ = overlaps.max(dim=1)
        print(
            f"[GT] total: {overlaps.shape[0]}, "
            f"pos: {sum(gt_max_overlaps >= pos_thr)}, "
            f"neg: {sum(gt_max_overlaps < neg_thr)}"
        )

        cost = (
            500 * gt_max_overlaps[(gt_max_overlaps >= pos_thr)].sum()
            + max_overlaps[(max_overlaps >= pos_thr) | (max_overlaps < neg_thr)].sum()
        )

        return cost

    def optimize(self, strides, scales, ratios, pos_thr, neg_thr):
        LR = 0.1
        BATCH_SIZE = 1024
        NUM_ITERS = 100

        ratios = ratios.clone().detach().requires_grad_(True)
        scales = scales.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([ratios, scales], lr=LR)

        opt_iou = 0
        opt_iter = 0
        opt_scales = scales
        opt_ratios = ratios
        for iter in range(NUM_ITERS):
            random_indices = torch.randperm(self.targets.size(0))[:BATCH_SIZE]
            iou = self.get_loss(
                targets=self.targets[random_indices],
                strides=strides,
                scales=scales,
                ratios=ratios,
                pos_thr=pos_thr,
                neg_thr=neg_thr,
            )
            print(f"iter: {iter} / iou: {iou} / scales: {scales} / ratios: {ratios}")

            if iou > opt_iou:
                opt_iter = iter
                opt_scales = scales
                opt_ratios = ratios

            optimizer.zero_grad()
            (-iou).backward()
            optimizer.step()

            ratios.data = torch.clamp(ratios.data, min=0.125, max=8)
            scales.data = torch.clamp(scales.data, min=0.0625, max=16)

        print(f"optimized scale/ratio: {opt_scales}/{opt_ratios} @ iter {opt_iter}")
        return opt_scales, opt_ratios
