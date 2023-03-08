# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
import struct
from typing import List, Optional, Tuple

import numpy as np
import pycocotools.mask as mask_utils

from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    Bbox,
    Caption,
    Cuboid3d,
    Ellipse,
    Label,
    Mask,
    Points,
    Polygon,
    PolyLine,
    RleMask,
    _Shape,
)

from .common import DictMapper, FloatListMapper, IntListMapper, Mapper, StringMapper

MAGIC_NUM_FOR_NONE = 2**31 - 1


class AnnotationListMapper(Mapper):
    """"""

    @staticmethod
    def forward(anns: List[Annotation]) -> bytes:
        _bytearray = bytearray()
        _bytearray.extend(struct.pack("<I", len(anns)))

        for ann in anns:
            if isinstance(ann, Label):
                _bytearray.extend(LabelMapper.forward(ann))
            elif isinstance(ann, Mask):
                _bytearray.extend(MaskMapper.forward(ann))
            elif isinstance(ann, Points):
                _bytearray.extend(PointsMapper.forward(ann))
            elif isinstance(ann, PolyLine):
                _bytearray.extend(PolyLineMapper.forward(ann))
            elif isinstance(ann, Polygon):
                _bytearray.extend(PolygonMapper.forward(ann))
            elif isinstance(ann, Bbox):
                _bytearray.extend(BboxMapper.forward(ann))
            elif isinstance(ann, Caption):
                _bytearray.extend(CaptionMapper.forward(ann))
            elif isinstance(ann, Cuboid3d):
                _bytearray.extend(Cuboid3dMapper.forward(ann))
            elif isinstance(ann, Ellipse):
                _bytearray.extend(EllipseMapper.forward(ann))
            else:
                raise NotImplementedError()

        return bytes(_bytearray)

    @staticmethod
    def backward(_bytes: bytes, offset: int = 0) -> Tuple[List[Annotation], int]:
        (n_anns,) = struct.unpack_from("<I", _bytes, offset)
        offset += 4
        anns = []
        for _ in range(n_anns):
            ann_type = AnnotationMapper.parse_ann_type(_bytes, offset)

            if ann_type == AnnotationType.label:
                ann, offset = LabelMapper.backward(_bytes, offset)
            elif ann_type == AnnotationType.mask:
                ann, offset = MaskMapper.backward(_bytes, offset)
            elif ann_type == AnnotationType.points:
                ann, offset = PointsMapper.backward(_bytes, offset)
            elif ann_type == AnnotationType.polyline:
                ann, offset = PolyLineMapper.backward(_bytes, offset)
            elif ann_type == AnnotationType.polygon:
                ann, offset = PolygonMapper.backward(_bytes, offset)
            elif ann_type == AnnotationType.bbox:
                ann, offset = BboxMapper.backward(_bytes, offset)
            elif ann_type == AnnotationType.caption:
                ann, offset = CaptionMapper.backward(_bytes, offset)
            elif ann_type == AnnotationType.cuboid_3d:
                ann, offset = Cuboid3dMapper.backward(_bytes, offset)
            elif ann_type == AnnotationType.ellipse:
                ann, offset = EllipseMapper.backward(_bytes, offset)
            else:
                raise NotImplementedError()

            anns.append(ann)

        return anns, offset


class AnnotationMapper(Mapper):
    ann_type = AnnotationType.unknown

    @classmethod
    def forward(cls, ann: Annotation) -> bytes:
        _bytearray = bytearray()
        _bytearray.extend(struct.pack("<Bqq", ann.type, ann.id, ann.group))
        _bytearray.extend(DictMapper.forward(ann.attributes))
        return bytes(_bytearray)

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[Annotation, int]:
        ann_type, id, group = struct.unpack_from("<Bqq", _bytes, offset)
        assert ann_type == cls.ann_type
        offset += 17  # struct.calcsize("<Bqq") = 17
        attributes, offset = DictMapper.backward(_bytes, offset)
        return Annotation(id=id, attributes=attributes, group=group), offset

    @staticmethod
    def forward_optional_label(label: Optional[int]) -> int:
        return label if label is not None else MAGIC_NUM_FOR_NONE

    @staticmethod
    def backward_optional_label(label: int) -> Optional[int]:
        return label if label != MAGIC_NUM_FOR_NONE else None

    @staticmethod
    def parse_ann_type(_bytes: bytes, offset: int = 0) -> AnnotationType:
        (ann_type,) = struct.unpack_from("<B", _bytes, offset)
        return AnnotationType(ann_type)


class LabelMapper(AnnotationMapper):
    ann_type = AnnotationType.label

    @classmethod
    def forward(cls, ann: Label) -> bytes:
        _bytearray = bytearray()
        _bytearray.extend(super().forward(ann))
        _bytearray.extend(struct.pack("<i", ann.label))
        return bytes(_bytearray)

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[Label, int]:
        ann, offset = super().backward(_bytes, offset)
        (label,) = struct.unpack_from("<i", _bytes, offset)
        offset += 4
        return Label(label=label, id=ann.id, attributes=ann.attributes, group=ann.group), offset


class MaskMapper(AnnotationMapper):
    ann_type = AnnotationType.mask

    @classmethod
    def forward(cls, ann: Mask) -> bytes:
        if isinstance(ann, RleMask):
            rle = ann.rle
        else:
            rle = mask_utils.encode(np.require(ann.image, dtype=np.uint8, requirements="F"))

        h, w = rle["size"]
        counts = rle["counts"]
        len_counts = len(counts)

        _bytearray = bytearray()
        _bytearray.extend(super().forward(ann))
        _bytearray.extend(struct.pack("<ii", cls.forward_optional_label(ann.label), ann.z_order))
        _bytearray.extend(struct.pack(f"<iiI{len_counts}s", h, w, len_counts, counts))
        return bytes(_bytearray)

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[Mask, int]:
        ann, offset = super().backward(_bytes, offset)
        label, z_order = struct.unpack_from("<ii", _bytes, offset)
        label = cls.backward_optional_label(label)
        offset += 8
        h, w, len_counts = struct.unpack_from("<iiI", _bytes, offset)
        offset += 12
        (counts,) = struct.unpack_from(f"<{len_counts}s", _bytes, offset)
        offset += len_counts

        return (
            RleMask(
                rle={"size": [h, w], "counts": counts},
                label=label,
                id=ann.id,
                attributes=ann.attributes,
                group=ann.group,
                z_order=z_order,
            ),
            offset,
        )


class RleMaskMapper(MaskMapper):
    """Just clone MaskMapper."""


class _ShapeMapper(AnnotationMapper):
    @classmethod
    def forward(cls, ann: _Shape) -> bytes:
        _bytearray = bytearray()
        _bytearray.extend(super().forward(ann))
        _bytearray.extend(struct.pack("<ii", cls.forward_optional_label(ann.label), ann.z_order))
        _bytearray.extend(FloatListMapper.forward(ann.points))
        return bytes(_bytearray)

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[_Shape, int]:
        ann, offset = super().backward(_bytes, offset)
        label, z_order = struct.unpack_from("<ii", _bytes, offset)
        offset += 8
        points, offset = FloatListMapper.backward(_bytes, offset)

        return (
            _Shape(
                points=points,
                id=ann.id,
                attributes=ann.attributes,
                group=ann.group,
                label=cls.backward_optional_label(label),
                z_order=z_order,
            ),
            offset,
        )


class PointsMapper(_ShapeMapper):
    ann_type = AnnotationType.points

    @classmethod
    def forward(cls, ann: Points) -> bytes:
        _bytearray = bytearray()
        _bytearray.extend(super().forward(ann))
        _bytearray.extend(IntListMapper.forward(ann.visibility))
        return bytes(_bytearray)

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[Points, int]:
        shape, offset = super().backward(_bytes, offset)
        visibility, offset = IntListMapper.backward(_bytes, offset)
        return (
            Points(
                points=shape.points,
                visibility=[Points.Visibility(v) for v in visibility],
                id=shape.id,
                attributes=shape.attributes,
                group=shape.group,
                label=shape.label,
                z_order=shape.z_order,
            ),
            offset,
        )


class PolyLineMapper(_ShapeMapper):
    ann_type = AnnotationType.polyline

    @classmethod
    def forward(cls, ann: PolyLine) -> bytes:
        return super().forward(ann)

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[PolyLine, int]:
        shape, offset = super().backward(_bytes, offset)
        return (
            PolyLine(
                points=shape.points,
                id=shape.id,
                attributes=shape.attributes,
                group=shape.group,
                label=shape.label,
                z_order=shape.z_order,
            ),
            offset,
        )


class PolygonMapper(_ShapeMapper):
    ann_type = AnnotationType.polygon

    @classmethod
    def forward(cls, ann: Polygon) -> bytes:
        return super().forward(ann)

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[Polygon, int]:
        shape, offset = super().backward(_bytes, offset)
        return (
            Polygon(
                points=shape.points,
                id=shape.id,
                attributes=shape.attributes,
                group=shape.group,
                label=shape.label,
                z_order=shape.z_order,
            ),
            offset,
        )


class BboxMapper(_ShapeMapper):
    ann_type = AnnotationType.bbox

    @classmethod
    def forward(cls, ann: Bbox) -> bytes:
        return super().forward(ann)

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[Bbox, int]:
        shape, offset = super().backward(_bytes, offset)
        x, y, x2, y2 = shape.points
        return (
            Bbox(
                x,
                y,
                x2 - x,
                y2 - y,
                id=shape.id,
                attributes=shape.attributes,
                group=shape.group,
                label=shape.label,
                z_order=shape.z_order,
            ),
            offset,
        )


class CaptionMapper(AnnotationMapper):
    ann_type = AnnotationType.caption

    @classmethod
    def forward(cls, ann: Caption) -> bytes:
        _bytearray = bytearray()
        _bytearray.extend(super().forward(ann))
        _bytearray.extend(StringMapper.forward(ann.caption))
        return bytes(_bytearray)

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[Caption, int]:
        ann, offset = super().backward(_bytes, offset)
        caption, offset = StringMapper.backward(_bytes, offset)
        return (
            Caption(caption=caption, id=ann.id, attributes=ann.attributes, group=ann.group),
            offset,
        )


class Cuboid3dMapper(AnnotationMapper):
    ann_type = AnnotationType.cuboid_3d

    @classmethod
    def forward(cls, ann: Cuboid3d) -> bytes:
        _bytearray = bytearray()
        _bytearray.extend(super().forward(ann))
        _bytearray.extend(struct.pack("<i", cls.forward_optional_label(ann.label)))
        _bytearray.extend(FloatListMapper.forward(ann._points))
        return bytes(_bytearray)

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[Cuboid3d, int]:
        ann, offset = super().backward(_bytes, offset)
        (label,) = struct.unpack_from("<i", _bytes, offset)
        offset += 4
        points, offset = FloatListMapper.backward(_bytes, offset)
        return (
            Cuboid3d(
                position=points[:3],
                rotation=points[3:6],
                scale=points[6:],
                label=cls.backward_optional_label(label),
                id=ann.id,
                attributes=ann.attributes,
                group=ann.group,
            ),
            offset,
        )


class EllipseMapper(_ShapeMapper):
    ann_type = AnnotationType.ellipse

    @classmethod
    def forward(cls, ann: Ellipse) -> bytes:
        return super().forward(ann)

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[Ellipse, int]:
        shape, offset = super().backward(_bytes, offset)
        x, y, x2, y2 = shape.points
        return (
            Ellipse(
                x,
                y,
                x2,
                y2,
                id=shape.id,
                attributes=shape.attributes,
                group=shape.group,
                label=shape.label,
                z_order=shape.z_order,
            ),
            offset,
        )
