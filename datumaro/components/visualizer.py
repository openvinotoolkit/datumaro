# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT
import math
import warnings
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import ImageColor

from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    Bbox,
    Caption,
    Cuboid3d,
    DepthAnnotation,
    Label,
    LabelCategories,
    Mask,
    Points,
    Polygon,
    PolyLine,
    SuperResolutionAnnotation,
)
from datumaro.components.dataset_base import DatasetItem, IDataset
from datumaro.components.media import Image

CAPTION_BBOX_PAD = 0.2
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
        """
        Visualizer for Datumaro annotations

        Parameters
        ----------
        dataset:
            Datumaro dataset to visualize its items.
        ignored_types:
            Categories of labels. It is used to extract label name by label id.
        figsize:
            Pyplot Figure instance used to draw annotation.
        color_cycles:
            Color cycle corresponding to each label ID.
            If the length of the color cycle is less than the label ID,
            then the label ID exceeding the color cycle length is assgined by the following rule.
            color = color_cycles[label_id % len(color_cycles)]
        bbox_linewidth:
            Line width for Bbox, Polygon and PolyLine annotation
        text_y_offset:
            Offset of y axis for texts.
            The higher value puts the text in the upper place of the annotation.
        alpha:
            Transparency value when drawing annotations. It should be in [0, 1].
            If alpha=0, we do not draw any annotations.
        """
        self.dataset = dataset
        self.figsize = figsize
        self.ignored_types = set(ignored_types) if ignored_types is not None else set()
        self.color_cycles = color_cycles if color_cycles is not None else DEFAULT_COLOR_CYCLES
        self.bbox_linewidth = bbox_linewidth
        self.text_y_offset = text_y_offset

        assert 0.0 <= alpha <= 1.0, "alpha should be in [0, 1]."
        self.alpha = alpha

    @property
    def draw_only_image(self):
        """
        If self.alpha = 0, we do not overdraw any annotation over the image.
        """
        return self.alpha == 0.0

    def _draw(
        self,
        ann: Annotation,
        label_categories: Optional[LabelCategories],
        fig: Figure,
        ax: Axes,
        context: List,
    ) -> None:
        """
        Draw annotation according to it's annotation type.

        Parameters
        ----------
        ann:
            Annotation entity to draw.
        label_categories:
            Categories of labels. It is used to extract label name by label id.
        fig:
            Pyplot Figure instance used to draw annotation.
        ax:
            Pyplot Axes instance used to draw annotation.
        context:
            It includes previously drawing history for each annotation type.
            Currently, it is necessary to avoid drawing again over an already drawn place.
            For example, multi label dataset has multiple Labels in a DatasetItem.
            If we don't keep context, it has to draw Label at the same place again and again.
        """
        if ann.type == AnnotationType.label:
            return self._draw_label(ann, label_categories, fig, ax, context)
        if ann.type == AnnotationType.mask:
            return self._draw_mask(ann, label_categories, fig, ax, context)
        if ann.type == AnnotationType.points:
            return self._draw_points(ann, label_categories, fig, ax, context)
        if ann.type == AnnotationType.polygon:
            return self._draw_polygon(ann, label_categories, fig, ax, context)
        if ann.type == AnnotationType.polyline:
            return self._draw_polygon(ann, label_categories, fig, ax, context)
        if ann.type == AnnotationType.bbox:
            return self._draw_bbox(ann, label_categories, fig, ax, context)
        if ann.type == AnnotationType.caption:
            return self._draw_caption(ann, label_categories, fig, ax, context)
        if ann.type == AnnotationType.cuboid_3d:
            return self._draw_cuboid_3d(ann, label_categories, fig, ax, context)
        if ann.type == AnnotationType.super_resolution_annotation:
            return self._draw_super_resolution_annotation(ann, label_categories, fig, ax, context)
        if ann.type == AnnotationType.depth_annotation:
            return self._draw_depth_annotation(ann, label_categories, fig, ax, context)

        raise ValueError(f"Unknown ann.type={ann.type}")

    def _get_color(self, ann: Annotation) -> str:
        color = self.color_cycles[ann.label % len(self.color_cycles)]
        return color

    def _sort_by_z_order(self, annotations: List[Annotation]) -> List[Annotation]:
        def _sort_key(ann: Annotation):
            z_order = getattr(ann, "z_order", -1)
            return z_order

        return sorted(annotations, key=_sort_key)

    def vis_gallery(
        self,
        ids: List[Union[str, DatasetItem]],
        subset: Optional[Union[str, List[str]]] = None,
        grid_size: Tuple[Optional[int], Optional[int]] = (None, None),
    ) -> Figure:
        nrows, ncols = _infer_grid_size(len(ids), grid_size)
        fig, axs = plt.subplots(nrows, ncols, figsize=self.figsize)

        if isinstance(subset, list):
            assert len(ids) == len(
                subset
            ), "If subset is a list, it should have the same length as ids."

        for i, (dataset_id, ax) in enumerate(zip(ids, axs.flatten())):
            if isinstance(subset, List):
                self.vis_one_sample(dataset_id, subset[i], ax)
            else:
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
        assert (
            item is not Image
        ), f"Media type should be Image, Current media type={type(item.media)}"

        img = item.media.data.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)

        width = ax.transAxes.transform_point((1, 0))[0] - ax.transAxes.transform_point((0, 0))[0]
        text = ax.set_title(f"ID: {id}, Subset: {subset}", loc="center", wrap=True)
        text.__get_wrapped_text = text._get_wrapped_text

        def _get_wrapped_text():
            wrapped_text = text.__get_wrapped_text()
            text._text = wrapped_text
            return wrapped_text

        text._get_wrapped_text = _get_wrapped_text
        text._get_wrap_line_width = lambda: width

        ax.set_axis_off()

        if self.draw_only_image:
            return fig

        annotations = self._sort_by_z_order(item.annotations)
        categories = self.dataset.categories()
        label_categories = (
            self.dataset.categories()[AnnotationType.label]
            if AnnotationType.label in categories
            else None
        )

        context = defaultdict(list)
        for ann in annotations:
            if ann.type in self.ignored_types:
                warnings.warn(f"{ann.type} in self.ignored_types. Skip it.")
                continue
            self._draw(ann, label_categories, fig, ax, context[ann.type])

        return fig

    def _draw_label(
        self,
        ann: Label,
        label_categories: Optional[LabelCategories],
        fig: Figure,
        ax: Axes,
        context: List,
    ) -> None:
        label_text = label_categories[ann.label].name if label_categories is not None else ann.label
        color = self._get_color(ann)

        if len(context) == 0:
            x, y = 0.01, 0.99
        else:
            # Draw below the previously drawn label.
            text: Text = context[-1]
            # We can know the position of text bbox only if drawing it actually.
            # https://stackoverflow.com/a/41271773/16880031
            fig.canvas.draw()
            bbox = text.get_window_extent()
            bbox = ax.transAxes.inverted().transform_bbox(bbox)
            x, y = 0.01, bbox.y0

        text = ax.text(x, y, label_text, ha="left", va="top", color=color, transform=ax.transAxes)
        context.append(text)

    def _draw_mask(
        self,
        ann: Mask,
        label_categories: Optional[LabelCategories],
        fig: Figure,
        ax: Axes,
        context: List,
    ) -> None:
        h, w = ann.image.shape
        mask_map = np.zeros((h, w, 4), dtype=np.uint8)
        color = self._get_color(ann)
        rgba_color = (*ImageColor.getcolor(color, "RGB"), 0.0)
        mask_map[ann.image] = rgba_color
        mask_map[ann.image, 3] = int(255 * self.alpha)

        ax.imshow(mask_map)

    def _draw_points(
        self,
        ann: Points,
        label_categories: Optional[LabelCategories],
        fig: Figure,
        ax: Axes,
        context: List,
    ) -> None:
        label_text = label_categories[ann.label].name if label_categories is not None else ann.label
        color = self._get_color(ann)
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
        fig: Figure,
        ax: Axes,
        context: List,
    ) -> None:
        label_text = label_categories[ann.label].name if label_categories is not None else ann.label
        color = self._get_color(ann)
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

    def _draw_bbox(
        self,
        ann: Bbox,
        label_categories: Optional[LabelCategories],
        fig: Figure,
        ax: Axes,
        context: List,
    ) -> None:
        label_text = label_categories[ann.label].name if label_categories is not None else ann.label
        color = self._get_color(ann)
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

    def _draw_caption(
        self,
        ann: Caption,
        label_categories: Optional[LabelCategories],
        fig: Figure,
        ax: Axes,
        context: List,
    ) -> None:
        if len(context) == 0:
            x, y = 0.5, 0.01
        else:
            # Draw on the top of the previously drawn caption.
            text: Text = context[-1]
            bbox = text.get_bbox_patch()
            # We can know the position of text bbox only if drawing it actually.
            # https://stackoverflow.com/a/41271773/16880031
            fig.canvas.draw()
            drawed_bbox = bbox.get_bbox()
            x, y = bbox.get_transform().transform(
                [0, (1.0 + 2 * CAPTION_BBOX_PAD) * drawed_bbox.height]
            )
            x, y = ax.transAxes.inverted().transform([x, y])
            x, y = 0.5, y

        width = ax.transAxes.transform_point((1, 0))[0] - ax.transAxes.transform_point((0, 0))[0]

        text = ax.text(
            x,
            y,
            ann.caption,
            ha="center",
            va="bottom",
            wrap=True,
            transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": self.alpha},
        )
        text.get_bbox_patch().set_boxstyle(f"Round,pad={CAPTION_BBOX_PAD}")
        text._get_wrap_line_width = lambda: width
        context.append(text)

    def _draw_cuboid_3d(
        self,
        ann: Cuboid3d,
        label_categories: Optional[LabelCategories],
        fig: Figure,
        ax: Axes,
        context: List,
    ) -> None:
        raise NotImplementedError(f"{ann.type} is not implemented yet.")

    def _draw_super_resolution_annotation(
        self,
        ann: SuperResolutionAnnotation,
        label_categories: Optional[LabelCategories],
        fig: Figure,
        ax: Axes,
        context: List,
    ) -> None:
        assert (
            len(context) == 0
        ), "It cannot visualize more than one SuperResolutionAnnotation per item."

        warnings.warn(
            "SuperResolutionAnnotation overdraws the high-resolution image over the original image. "
            "If you want to see the original image, set alpha=0."
        )

        hi_res_img = ann.image.data
        im = ax.imshow(hi_res_img, alpha=self.alpha)
        context.append(im)

    def _draw_depth_annotation(
        self,
        ann: DepthAnnotation,
        label_categories: Optional[LabelCategories],
        fig: Figure,
        ax: Axes,
        context: List,
    ) -> None:
        assert len(context) == 0, "It cannot visualize more than one DepthAnnotation per item."

        depth = ann.image.data

        im = ax.imshow(depth, alpha=self.alpha)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(im, cax)
        context.append(im)
