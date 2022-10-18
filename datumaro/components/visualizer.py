# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import math
import warnings
from typing import Iterable, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    Bbox,
    Label,
    LabelCategories,
    Mask,
    Points,
    PolyLine,
    Polygon,
)
from datumaro.components.dataset import IDataset
from datumaro.components.extractor import DatasetItem

DEFAULT_COLOR_CYCLES: List[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _infer_grid_size(length: int, grid_size: Tuple[Optional[int], Optional[int]]):
    nrows, ncols = grid_size

    if nrows is None and ncols is None:
        nrows = ncols = int(math.sqrt(length))

        while nrows * ncols < length:
            nrows += 1
    elif nrows is None and ncols > 0:
        nrows = int(length / ncols)

        while nrows * ncols < length:
            nrows += 1
    elif nrows > 0 and ncols is None:
        ncols = int(length / nrows)

        while nrows * ncols < length:
            ncols += 1

    assert nrows > 0, "nrows should be a positive integer."
    assert ncols > 0, "ncols should be a positive integer."
    assert length <= nrows * ncols, "The number of ids should less then or equal to nrows * ncols."

    return nrows, ncols


class Visualizer:
    def __init__(
        self,
        dataset: IDataset,
        ignored_types: Optional[Iterable[AnnotationType]] = None,
        figsize: Tuple[float, float] = (8, 6),
        color_cycles: Optional[List[str]] = None,
        bbox_linewidth: float = 1.0,
        text_y_offset: float = 1.5,
        alpha: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.figsize = figsize
        self.ignored_types = set(ignored_types) if ignored_types is not None else set()
        self.color_cycles = color_cycles if color_cycles is not None else DEFAULT_COLOR_CYCLES
        self.bbox_linewidth = bbox_linewidth
        self.text_y_offset = text_y_offset
        self.alpha = alpha

        def _not_implmented(ann: Annotation, *args, **kwargs):
            raise NotImplementedError(f"{ann.type} is not implemented yet.")

        self._draw_func = {
            AnnotationType.label: self._draw_label,
            AnnotationType.mask: self._draw_mask,
            AnnotationType.points: self._draw_points,
            AnnotationType.polygon: self._draw_polygon,
            AnnotationType.polyline: self._draw_polygon,
            AnnotationType.bbox: self._draw_bbox,
            AnnotationType.caption: _not_implmented,
            AnnotationType.cuboid_3d: _not_implmented,
            AnnotationType.super_resolution_annotation: _not_implmented,
            AnnotationType.depth_annotation: _not_implmented,
        }

    def _sort_by_z_order(self, annotations: List[Annotation]) -> List[Annotation]:
        def _sort_key(ann: Annotation):
            z_order = getattr(ann, "z_order", -1)
            return z_order

        return sorted(annotations, key=_sort_key)

    def vis_gallery(
        self,
        ids: List[Union[str, DatasetItem]],
        subset: Optional[str] = None,
        grid_size: Tuple[Optional[int], Optional[int]] = (None, None),
    ) -> Figure:
        nrows, ncols = _infer_grid_size(len(ids), grid_size)
        fig, axs = plt.subplots(nrows, ncols, figsize=self.figsize)

        for dataset_id, ax in zip(ids, axs.flatten()):
            self.vis_one_sample(dataset_id, subset, ax)

        return fig

    def vis_one_sample(
        self, id: Union[str, DatasetItem], subset: Optional[str] = None, ax: Optional[Axes] = None
    ) -> Figure:
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = plt.gca()
        else:
            fig = ax.get_figure()
            plt.sca(ax)

        item: DatasetItem = self.dataset.get(id, subset)
        assert item is not None, f"Cannot find id={id}, subset={subset}"

        img = item.media.data.astype(np.uint8)
        ax.imshow(img)

        annotations = self._sort_by_z_order(item.annotations)
        categories = self.dataset.categories()
        label_categories = (
            self.dataset.categories()[AnnotationType.label]
            if AnnotationType.label in categories
            else None
        )

        for ann in annotations:
            if ann.type in self.ignored_types:
                warnings.warn(f"{ann.type} in self.ignored_types. Skip it.")
                continue

            if ann.type in self._draw_func:
                self._draw_func[ann.type](ann, label_categories, ax)
            else:
                raise

        ax.set_title(f"ID: {id}, Subset={subset}")
        ax.set_axis_off()

        return fig

    def _draw_label(
        self, ann: Label, label_categories: Optional[LabelCategories], ax: Axes
    ) -> None:
        label_text = label_categories[ann.label].name if label_categories is not None else ann.label
        color = self.color_cycles[ann.label % len(self.color_cycles)]
        ax.text(0, 0, label_text, ha="left", va="top", color=color, transform=ax.transAxes)

    def _draw_mask(self, ann: Mask, label_categories: Optional[LabelCategories], ax: Axes) -> None:
        pass

    def _draw_points(
        self, ann: Points, label_categories: Optional[LabelCategories], ax: Axes
    ) -> None:
        label_text = label_categories[ann.label].name if label_categories is not None else ann.label
        color = self.color_cycles[ann.label % len(self.color_cycles)]
        points = np.array(ann.points)
        n_points = len(points) // 2
        points = points.reshape(n_points, 2)
        visible = [viz == Points.Visibility.visible for viz in ann.visibility]
        points = points[visible]

        ax.scatter(points[:, 0], points[:, 1], color=color)

        if len(points) > 0:
            x, y, _, _ = ann.get_bbox()
            ax.text(x, y - self.text_y_offset, label_text, color=color)

    def _draw_polygon(
        self,
        ann: Union[Polygon, PolyLine],
        label_categories: Optional[LabelCategories],
        ax: Axes,
    ) -> None:
        label_text = label_categories[ann.label].name if label_categories is not None else ann.label
        color = self.color_cycles[ann.label % len(self.color_cycles)]
        points = np.array(ann.points)
        n_points = len(points) // 2
        points = points.reshape(n_points, 2)

        polyline = patches.Polygon(
            points,
            fill=False,
            linewidth=self.bbox_linewidth,
            edgecolor=color,
        )
        ax.add_patch(polyline)

        if isinstance(ann, Polygon):
            polygon = patches.Polygon(
                points,
                fill=True,
                facecolor=color if isinstance(ann, Polygon) else "none",
                alpha=self.alpha,
            )
            ax.add_patch(polygon)

        x, y, _, _ = ann.get_bbox()
        ax.text(x, y - self.text_y_offset, label_text, color=color)

    def _draw_bbox(self, ann: Bbox, label_categories: Optional[LabelCategories], ax: Axes) -> None:
        label_text = label_categories[ann.label].name if label_categories is not None else ann.label
        color = self.color_cycles[ann.label % len(self.color_cycles)]
        rect = patches.Rectangle(
            (ann.x, ann.y),
            ann.w,
            ann.h,
            linewidth=self.bbox_linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.text(ann.x, ann.y - self.text_y_offset, label_text, color=color)
        ax.add_patch(rect)
