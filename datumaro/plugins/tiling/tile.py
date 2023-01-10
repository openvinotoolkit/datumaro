# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import shapely.geometry as sg
import shapely.ops as so

from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    Bbox,
    Caption,
    DepthAnnotation,
    Label,
    Mask,
    Points,
    Polygon,
    PolyLine,
)
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import DatumaroError, MediaTypeError
from datumaro.components.media import BboxIntCoords, Image, RoIImage
from datumaro.components.transformer import ItemTransform
from datumaro.plugins.tiling.util import (
    clip_x1y1x2y2,
    cxcywh_to_x1y1x2y2,
    x1y1x2y2_to_cxcywh,
    x1y1x2y2_to_xywh,
    xywh_to_x1y1x2y2,
)


def _apply_offset(geom: sg.base.BaseGeometry, roi_box: sg.Polygon) -> sg.base.BaseGeometry:
    offset_x, offset_y = roi_box.bounds[:2]
    return so.transform(lambda x, y: (x - offset_x, y - offset_y), geom)


def _tile_mask(ann: Mask, roi_int: BboxIntCoords, *args, **kwargs) -> Mask:
    x, y, w, h = roi_int
    tiled_mask = ann.image[y : y + h, x : x + w]
    return Mask(
        image=tiled_mask,
        id=ann.id,
        attributes=deepcopy(ann.attributes),
        group=ann.group,
        label=ann.label,
        z_order=ann.z_order,
    )


def _tile_points(ann: Points, roi_box: sg.Polygon, *args, **kwargs) -> Optional[Points]:
    points = sg.MultiPoint(ann.get_points())

    if not roi_box.covers(points):
        return None

    points = _apply_offset(points, roi_box)

    return Points(
        [v for point in points.geoms for v in (point.x, point.y)],
        id=ann.id,
        attributes=deepcopy(ann.attributes),
        group=ann.group,
        label=ann.label,
        z_order=ann.z_order,
        visibility=deepcopy(ann.visibility),
    )


def _tile_polygon(
    ann: Polygon, roi_box: sg.Polygon, threshold_drop_ann: float = 0.8, *args, **kwargs
) -> Optional[Polygon]:
    polygon = sg.Polygon(ann.get_points())

    if not roi_box.intersects(polygon):
        return None

    inter: sg.Polygon = polygon.intersection(roi_box)
    prop_area = inter.area / polygon.area

    if prop_area < threshold_drop_ann:
        return None

    inter = _apply_offset(inter, roi_box)

    return Polygon(
        [p for xy in inter.exterior.coords for p in xy],
        id=ann.id,
        attributes=deepcopy(ann.attributes),
        group=ann.group,
        label=ann.label,
        z_order=ann.z_order,
    )


def _tile_polyline(ann: PolyLine, roi_box: sg.Polygon, *args, **kwargs) -> Optional[PolyLine]:
    lines = sg.LineString(ann.get_points())

    if not roi_box.covers(lines):
        return None

    lines = _apply_offset(lines, roi_box)

    return PolyLine(
        [v for point in lines.coords for v in (point[0], point[1])],
        id=ann.id,
        attributes=deepcopy(ann.attributes),
        group=ann.group,
        label=ann.label,
        z_order=ann.z_order,
    )


def _tile_bbox(
    ann: Bbox, roi_box: sg.Polygon, threshold_drop_ann: float = 0.8, *args, **kwargs
) -> Optional[Bbox]:
    bbox: sg.Polygon = sg.box(*xywh_to_x1y1x2y2(*ann.get_bbox()))

    if not roi_box.intersects(bbox):
        return None

    inter: sg.Polygon = bbox.intersection(roi_box)
    prop_area = inter.area / bbox.area

    if prop_area < threshold_drop_ann:
        return None

    inter = _apply_offset(inter, roi_box)

    return Bbox(
        *x1y1x2y2_to_xywh(*inter.bounds),
        id=ann.id,
        attributes=deepcopy(ann.attributes),
        group=ann.group,
        label=ann.label,
        z_order=ann.z_order,
    )


def _tile_depth_annotation(
    ann: DepthAnnotation, roi_int: BboxIntCoords, *args, **kwargs
) -> DepthAnnotation:
    x, y, w, h = roi_int
    tiled_img = ann.image[y : y + h, x : x + w]

    return DepthAnnotation(
        tiled_img, id=ann.id, attributes=deepcopy(ann.attributes), group=ann.group
    )


def _tile_by_copy(ann: Union[Label, Caption], *args, **kwargs) -> Union[Label, Caption]:
    if ann.type == AnnotationType.label:
        return Label(ann.label, id=ann.id, attributes=deepcopy(ann.attributes), group=ann.group)
    elif ann.type == AnnotationType.caption:
        return Caption(ann.caption, id=ann.id, attributes=deepcopy(ann.attributes), group=ann.group)
    raise ValueError(f"Unknown type, type(ann)={type(ann)}.")


def _tile_not_support(ann: Annotation, *args, **kwargs) -> None:
    raise DatumaroError(f"type(ann)={type(ann)} is not support tiling.")


class TileTransform(ItemTransform, CliPlugin):
    """
    Tile dataset
    TODO: Fill in this description
    """

    _tile_ann_func_map: Dict[AnnotationType, Callable] = {
        AnnotationType.label: _tile_by_copy,
        AnnotationType.mask: _tile_mask,
        AnnotationType.points: _tile_points,
        AnnotationType.polygon: _tile_polygon,
        AnnotationType.polyline: _tile_polyline,
        AnnotationType.bbox: _tile_bbox,
        AnnotationType.caption: _tile_by_copy,
        AnnotationType.cuboid_3d: _tile_not_support,
        AnnotationType.super_resolution_annotation: _tile_not_support,
        AnnotationType.depth_annotation: _tile_depth_annotation,
    }

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--grid-size",
            type=int,
            nargs=2,
            required=True,
            metavar=("N_ROWS", "N_COLS"),
            help="Grid size, e.g. '--grid-size 2 3' will produce 2x3 tiled images.",
        )
        parser.add_argument(
            "--overlap",
            type=float,
            nargs=2,
            default=[0.1, 0.1],
            metavar=("PERC_HEIGHT", "PERC_WIDTH"),
            help="Percentage of overlaps between tiled images, e.g."
            " '--overlap 0.1 0.2' will create overlaps of 10% x height of tiled image"
            " and 20% x width of tiled image.",
        )
        parser.add_argument(
            "--threshold-drop-ann",
            type=float,
            default=0.5,
            help="Threshold for dropping Polygon and Bbox annotations when tiling."
            " Polygon and Bbox should be cropped if they exist on the edge of the tiled image."
            " If an area of the cropped annotation / an area of the original annoation  < `threshold_drop_ann`,"
            " drop the corresponding annotation.",
        )
        return parser

    def __init__(
        self,
        extractor,
        grid_size: Tuple[int, int],
        overlap: Tuple[float, float],
        threshold_drop_ann: float,
    ):
        super().__init__(extractor)

        self._grid_size = grid_size
        self._overlap = overlap
        self._threshold_drop_ann = threshold_drop_ann

    def __iter__(self):
        for item in self._extractor:
            items = self.transform_item(item)
            for item in items:
                yield item

    def transform_item(self, item: DatasetItem) -> List[DatasetItem]:
        if not isinstance(item.media, Image):
            assert MediaTypeError(
                f"item.media should be Image, but type(item.media)={type(item.media)}."
            )

        items: List[DatasetItem] = []
        rois = self.extract_rois(item.media)
        for idx, roi in enumerate(rois):
            items += [
                DatasetItem(
                    id=item.id + f"_tile_{idx}",
                    subset=item.subset,
                    media=RoIImage.create_from_image(item.media, roi),
                    attributes=self.get_tiled_attributes(item, idx, roi),
                    annotations=self.get_tiled_annotations(item, roi),
                )
            ]

        return items

    def extract_rois(self, image: Image) -> List[BboxIntCoords]:
        assert image.size is not None, "image.size is None."

        max_h, max_w = image.size
        n_row, n_col = self._grid_size
        new_h, new_w = int(max_h / n_row), int(max_w / n_col)
        h_ovl, w_ovl = self._overlap

        rois: List[BboxIntCoords] = []

        for r in range(n_row):
            for c in range(n_col):
                y1, x1 = new_h * r, new_w * c
                y2, x2 = y1 + new_h, x1 + new_w

                c_x, c_y, w, h = x1y1x2y2_to_cxcywh(x1, y1, x2, y2)
                w, h = int(w * (1 + w_ovl)), int(h * (1 + h_ovl))
                x1, y1, x2, y2 = cxcywh_to_x1y1x2y2(c_x, c_y, w, h)
                x1, y1, x2, y2 = clip_x1y1x2y2(x1, y1, x2, y2, max_w, max_h)
                rois += [x1y1x2y2_to_xywh(x1, y1, x2, y2)]
        return rois

    @staticmethod
    def get_tiled_attributes(item: DatasetItem, idx: int, roi: BboxIntCoords) -> Dict[str, Any]:
        attributes = {k: v for k, v in item.attributes.items()}
        attributes["tile_id"] = item.id
        attributes["tile_idx"] = idx
        attributes["roi"] = roi
        return attributes

    def get_tiled_annotations(self, item: DatasetItem, roi: BboxIntCoords) -> List[Annotation]:
        roi_box: sg.Polygon = sg.box(*xywh_to_x1y1x2y2(*roi))

        tiled_anns = []
        for ann in item.annotations:
            tiled_ann = self._tile_ann_func_map[ann.type](
                ann, roi_int=roi, roi_box=roi_box, threshold_drop_ann=self._threshold_drop_ann
            )
            if tiled_ann is not None:
                tiled_anns.append(tiled_ann)

        return tiled_anns
