# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from functools import partial
from itertools import chain, repeat
from typing import List, NamedTuple, NewType, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np

from datumaro.util.image import lazy_image, load_image


class UncompressedRle(TypedDict):
    size: Sequence[int]
    counts: bytes


class CompressedRle(TypedDict):
    size: Sequence[int]
    counts: Sequence[int]


Rle = Union[CompressedRle, UncompressedRle]

Polygon = List[int]
"2d polygon with points [x1, y1, x2, y2, ...]"

PolygonGroup = List[Polygon]
"A group of polygons, describing a single object"

BboxCoords = NamedTuple("BboxCoords", [("x", int), ("y", int), ("w", int), ("h", int)])

Segment = Union[PolygonGroup, Rle]

BinaryMask = NewType("BinaryMask", np.ndarray)
IndexMask = NewType("IndexMask", np.ndarray)
ColorMask = NewType("ColorMask", np.ndarray)


def generate_colormap(length=256, *, include_background=True):
    """
    Generates colors using PASCAL VOC algorithm.

    If include_background is True, the result will include the item
    "0: (0, 0, 0)", which is typically used as a background color.
    Otherwise, indices will start from 0, but (0, 0, 0) is not included.

    Returns index -> (R, G, B) mapping.
    """

    def get_bit(number, index):
        return (number >> index) & 1

    colormap = np.zeros((length, 3), dtype=int)

    offset = int(not include_background)
    indices = np.arange(offset, length + offset, dtype=int)

    for j in range(7, -1, -1):
        for c in range(3):
            colormap[:, c] |= get_bit(indices, c) << j
        indices >>= 3

    return {id: tuple(color) for id, color in enumerate(colormap)}


def invert_colormap(colormap):
    return {tuple(a): index for index, a in colormap.items()}


def check_is_mask(mask: np.ndarray) -> bool:
    assert len(mask.shape) in {2, 3}
    if len(mask.shape) == 3:
        assert mask.shape[2] == 1


_default_colormap = generate_colormap()
_default_unpaint_colormap = invert_colormap(_default_colormap)


def unpaint_mask(painted_mask: ColorMask, inverse_colormap=None) -> IndexMask:
    """
    Convert color mask to index mask

    mask: HWC BGR [0; 255]

    colormap: (R, G, B) -> index
    """
    assert len(painted_mask.shape) == 3
    if inverse_colormap is None:
        inverse_colormap = _default_unpaint_colormap

    if callable(inverse_colormap):
        map_fn = lambda a: inverse_colormap((a >> 16) & 255, (a >> 8) & 255, a & 255)
    else:
        map_fn = lambda a: inverse_colormap.get(((a >> 16) & 255, (a >> 8) & 255, a & 255), None)

    painted_mask = painted_mask.astype(int)
    painted_mask = (
        painted_mask[:, :, 0] + (painted_mask[:, :, 1] << 8) + (painted_mask[:, :, 2] << 16)
    )
    uvals, unpainted_mask = np.unique(painted_mask, return_inverse=True)
    palette = []
    for v in uvals:
        class_id = map_fn(v)
        if class_id is None:
            raise KeyError(f"Undeclared color {((v >> 16) & 255, (v >> 8) & 255, v & 255)}")
        palette.append(class_id)
    palette = np.array(palette, dtype=np.min_scalar_type(len(uvals)))
    unpainted_mask = palette[unpainted_mask].reshape(painted_mask.shape[:2])

    return unpainted_mask


def paint_mask(mask: IndexMask, colormap=None) -> ColorMask:
    """
    Applies colormap to index mask

    mask: HW(C) [0; max_index] mask

    colormap: index -> (R, G, B)
    """
    check_is_mask(mask)

    if colormap is None:
        colormap = _default_colormap
    if callable(colormap):
        map_fn = colormap
    else:
        map_fn = lambda c: colormap.get(c, (0, 0, 0))
    palette = np.array([map_fn(c)[::-1] for c in range(256)], dtype=np.uint8)

    mask = mask.astype(np.uint8)
    painted_mask = palette[mask].reshape((*mask.shape[:2], 3))
    return painted_mask


def remap_mask(mask: ColorMask, map_fn) -> ColorMask:
    """
    Changes mask elements from one colormap to another

    # mask: HW(C) [0; max_index] mask
    """
    check_is_mask(mask)

    return np.array([max(0, map_fn(c)) for c in range(256)], dtype=np.uint8)[mask]


def make_index_mask(binary_mask: BinaryMask, index: int, dtype=None) -> IndexMask:
    return binary_mask * np.array([index], dtype=dtype or np.min_scalar_type(index))


def make_binary_mask(mask: Union[BinaryMask, IndexMask]) -> BinaryMask:
    if mask.dtype.kind == "b":
        return mask
    return mask.astype(bool)


def bgr2index(img):
    if img.dtype.kind not in {"b", "i", "u"} or img.dtype.itemsize < 4:
        img = img.astype(np.uint32)
    return (img[..., 0] << 16) + (img[..., 1] << 8) + img[..., 2]


def index2bgr(id_map):
    return np.dstack((id_map >> 16, id_map >> 8, id_map)).astype(np.uint8)


def load_mask(path, inverse_colormap=None):
    mask = load_image(path, dtype=np.uint8)
    if inverse_colormap is not None:
        if len(mask.shape) == 3 and mask.shape[2] != 1:
            mask = unpaint_mask(mask, inverse_colormap)
    return mask


def lazy_mask(path, inverse_colormap=None):
    return lazy_image(path, partial(load_mask, inverse_colormap=inverse_colormap))


def mask_to_rle(binary_mask: BinaryMask) -> CompressedRle:
    # walk in row-major order as COCO format specifies
    bounded = binary_mask.ravel(order="F")

    # add borders to sequence
    # find boundary positions for sequences and compute their lengths
    difs = np.diff(bounded, prepend=[1 - bounded[0]], append=[1 - bounded[-1]])
    (counts,) = np.where(difs != 0)

    # start RLE encoding from 0 as COCO format specifies
    if bounded[0] != 0:
        counts = np.diff(counts, prepend=[0])
    else:
        counts = np.diff(counts)

    return {"counts": counts, "size": list(binary_mask.shape)}


def mask_to_polygons(mask: BinaryMask, area_threshold=1) -> List[Polygon]:
    """
    Convert an instance mask to polygons

    Args:
        mask: a 2d binary mask
        tolerance: maximum distance from original points of
            a polygon to the approximated ones
        area_threshold: minimal area of generated polygons

    Returns:
        A list of polygons like [[x1,y1, x2,y2 ...], [...]]
    """
    import cv2
    from pycocotools import mask as mask_utils

    polygons = []

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_TC89_KCOS
    )

    for contour in contours:
        if len(contour) <= 2:
            continue

        contour = contour.reshape((-1, 2))

        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))  # make polygon closed
        contour = contour.flatten().clip(0)  # [x0, y0, ...]

        # Check if the polygon is big enough
        rle = mask_utils.frPyObjects([contour], mask.shape[0], mask.shape[1])
        area = sum(mask_utils.area(rle))
        if area_threshold <= area:
            polygons.append(contour)
    return polygons


def is_uncompressed_rle(obj: Segment) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("counts"), bytes)


def is_polygon_group(obj: Segment) -> bool:
    return (
        isinstance(obj, list)
        and isinstance(obj[0], list)
        and (len(obj[0]) == 0 or isinstance(obj[0][0], int))
    )


def to_uncompressed_rle(rle: Rle, *, width: int, height: int) -> UncompressedRle:
    if is_uncompressed_rle(rle):
        return rle

    from pycocotools import mask as mask_utils

    return mask_utils.frPyObjects(rle, height, width)


def crop_covered_segments(
    segments: Sequence[Segment],
    width: int,
    height: int,
    iou_threshold: float = 0.0,
    ratio_tolerance: float = 0.001,
    area_threshold: int = 1,
    return_masks: bool = False,
) -> List[Union[Optional[BinaryMask], List[Polygon]]]:
    """
    Find all segments occluded by others and crop them to the visible part only.
    Input segments are expected to be sorted from background to foreground.

    Args:
        segments: 1d list of segment RLEs (in COCO format)
        width: width of the image
        height: height of the image
        iou_threshold: IoU threshold for objects to be counted as intersected
            By default is set to 0 to process any intersected objects
        ratio_tolerance: an IoU "handicap" value for a situation
            when a foreground object is (almost) fully inside of another one,
            and we don't want make a "hole" in the background object.
            If the foreground object is fully or almost fully (iou - this ratio)
            inside the background object, it will be kept.
            The default is to keep tiny (0.1% of IoU) foreground objects.
        area_threshold: minimal area of included segments

    Returns:
        A list of input segments' parts (in the same order as input):

        .. code-block::

            [
                [[x1,y1, x2,y2 ...], ...], # input segment #0 parts
                mask1, # input segment #1 mask (if source segment is mask)
                [], # when source segment is too small
                ...
            ]
    """
    from pycocotools import mask as mask_utils

    # Convert to uncompressed RLEs
    wrapped_segments = [[s] for s in segments]
    input_rles = [
        mask_utils.frPyObjects(s, height, width) if not is_uncompressed_rle(s[0]) else s
        for s in wrapped_segments
    ]

    output_segments = []
    for i, rle_bottom in enumerate(input_rles):
        area_bottom = sum(mask_utils.area(rle_bottom))
        if area_bottom < area_threshold:
            output_segments.append([] if not return_masks else None)
            continue

        rles_top = []
        for j in range(i + 1, len(input_rles)):
            rle_top = input_rles[j]
            iou = sum(mask_utils.iou(rle_bottom, rle_top, [0]))[0]

            if iou <= iou_threshold:
                continue

            area_top = sum(mask_utils.area(rle_top))
            area_ratio = area_top / area_bottom

            # If the top segment is (almost) fully inside the background one,
            # we may need to skip it to avoid making a hole in the background object
            if abs(area_ratio - iou) < ratio_tolerance:
                continue

            rles_top += rle_top

        if not rles_top and is_polygon_group(wrapped_segments[i]) and not return_masks:
            output_segments.append(wrapped_segments[i])
            continue

        rle_bottom = rle_bottom[0]
        bottom_mask = mask_utils.decode(rle_bottom).astype(np.uint8)

        if rles_top:
            rle_top = mask_utils.merge(rles_top)
            top_mask = mask_utils.decode(rle_top).astype(np.uint8)

            bottom_mask -= top_mask
            bottom_mask[bottom_mask != 1] = 0

        if not return_masks and is_polygon_group(wrapped_segments[i]):
            output_segments.append(mask_to_polygons(bottom_mask, area_threshold=area_threshold))
        else:
            if np.sum(bottom_mask) < area_threshold:
                bottom_mask = None
            output_segments.append(bottom_mask)

    return output_segments


def rles_to_mask(rles: Sequence[Union[CompressedRle, Polygon]], width, height) -> BinaryMask:
    from pycocotools import mask as mask_utils

    rles = mask_utils.frPyObjects(rles, height, width)
    rles = mask_utils.merge(rles)
    mask = mask_utils.decode(rles)
    return mask


def find_mask_bbox(mask: BinaryMask) -> BboxCoords:
    cols = np.any(mask, axis=0)
    rows = np.any(mask, axis=1)
    x0, x1 = np.where(cols)[0][[0, -1]]
    y0, y1 = np.where(rows)[0][[0, -1]]
    return BboxCoords(x0, y0, x1 - x0, y1 - y0)


def merge_masks(
    masks: Sequence[Union[IndexMask, Tuple[BinaryMask, int]]], start: Optional[BinaryMask] = None
) -> IndexMask:
    """
    Merges masks into one, mask order is responsible for z order.
    To avoid memory explosion on mask materialization, consider passing
    a generator.

    Inputs: a sequence of index masks or (binary mask, index) pairs

    Outputs: an index mask
    """
    if start is not None:
        masks = chain([start], masks)

    it = iter(masks)

    try:
        merged_mask = next(it)
        if isinstance(merged_mask, tuple) and len(merged_mask) == 2:
            merged_mask = merged_mask[0] * merged_mask[1]
    except StopIteration:
        return None

    for m in it:
        if isinstance(m, tuple) and len(m) == 2:
            merged_mask = np.where(m[0], m[1], merged_mask)
        else:
            merged_mask = np.where(m, m, merged_mask)

    return merged_mask


def close_polygon(p: Polygon) -> Polygon:
    """
    Returns the closed version of the polygon (with the same first and last points),
    or the polygon itself.
    """
    points = np.asarray(p).reshape((-1, 2))

    if len(points) > 0 and not np.all(points[-1] == points[0]):
        points = np.append(points, points[0])

    return points.flatten().tolist()


def simplify_polygon(p: Polygon) -> Polygon:
    "Simplifies the polygon by removing repeated points"

    points = np.asarray(p).reshape((-1, 2))
    updated_points = []

    if len(points) > 0:
        updated_points.append(points[0])

        for point_idx in range(1, len(points)):
            prev_point = points[point_idx - 1]
            point = points[point_idx]
            if not np.all(point == prev_point):
                updated_points.append(point)

        if len(updated_points) < 3:
            updated_points.extend(repeat(updated_points[-1], 3 - len(updated_points)))

    return np.asarray(updated_points).flatten().tolist()
