# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import hashlib
import logging as log
import warnings
from copy import deepcopy
from typing import Callable, Dict, Optional, Set, Tuple
from unittest import TestCase

import cv2
import numpy as np
from attr import attrib, attrs

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.annotations.matcher import LineMatcher, PointsMatcher, match_segments
from datumaro.components.dataset_base import CategoriesInfo, DatasetItem, IDataset
from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image
from datumaro.util import filter_dict, find
from datumaro.util.annotation_util import find_instances, max_bbox
from datumaro.util.attrs_util import default_if_none


def mean_std(dataset: IDataset):
    counter = _MeanStdCounter()

    for item in dataset:
        counter.accumulate(item)

    return counter.get_result()


class _MeanStdCounter:
    """
    Computes unbiased mean and std. dev. for dataset images, channel-wise.
    """

    def __init__(self):
        self._stats = {}  # (id, subset) -> (pixel count, mean vec, std vec)

    def accumulate(self, item: DatasetItem):
        size = item.media.size
        if size is None:
            log.warning(
                "Item %s: can't detect image size, "
                "the image will be skipped from pixel statistics",
                item.id,
            )
            return
        count = np.prod(item.media.size)

        image = item.media.data
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        else:
            image = image[:, :, :3]
        # opencv is much faster than numpy here
        mean, std = cv2.meanStdDev(image.astype(np.double) / 255)

        self._stats[(item.id, item.subset)] = (count, mean, std)

    def get_result(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        n = len(self._stats)

        if n == 0:
            return [0, 0, 0], [0, 0, 0]

        counts = np.empty(n, dtype=np.uint32)
        stats = np.empty((n, 2, 3), dtype=np.double)

        for i, v in enumerate(self._stats.values()):
            counts[i] = v[0]
            stats[i][0] = v[1].reshape(-1)
            stats[i][1] = v[2].reshape(-1)

        mean = lambda i, s: s[i][0]
        var = lambda i, s: s[i][1]

        # make variance unbiased
        np.multiply(np.square(stats[:, 1]), (counts / (counts - 1))[:, np.newaxis], out=stats[:, 1])

        # Use an online algorithm to:
        # - handle different image sizes
        # - avoid cancellation problem
        _, mean, var = self._compute_stats(stats, counts, mean, var)
        return mean * 255, np.sqrt(var) * 255

    # Implements online parallel computation of sample variance
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    @staticmethod
    def _pairwise_stats(count_a, mean_a, var_a, count_b, mean_b, var_b):
        """
        Computes vector mean and variance.

        Needed do avoid catastrophic cancellation in floating point computations

        Returns:
            A tuple (total count, mean, variance)
        """

        # allow long arithmetics
        count_a = int(count_a)
        count_b = int(count_b)

        delta = mean_b - mean_a
        m_a = var_a * (count_a - 1)
        m_b = var_b * (count_b - 1)
        M2 = m_a + m_b + delta**2 * (count_a * count_b / (count_a + count_b))

        return (count_a + count_b, mean_a * 0.5 + mean_b * 0.5, M2 / (count_a + count_b - 1))

    @staticmethod
    def _compute_stats(stats, counts, mean_accessor, variance_accessor):
        """
        Recursively computes total count, mean and variance,
        does O(log(N)) calls.

        Args:
            stats: (float array of shape N, 2 * d, d = dimensions of values)
            count: (integer array of shape N)
            mean_accessor: (function(idx, stats)) to retrieve element mean
            variance_accessor: (function(idx, stats)) to retrieve element variance

        Returns:
            A tuple (total count, mean, variance)
        """

        m = mean_accessor
        v = variance_accessor
        n = len(stats)
        if n == 1:
            return counts[0], m(0, stats), v(0, stats)
        if n == 2:
            return __class__._pairwise_stats(
                counts[0], m(0, stats), v(0, stats), counts[1], m(1, stats), v(1, stats)
            )
        h = n // 2
        return __class__._pairwise_stats(
            *__class__._compute_stats(stats[:h], counts[:h], m, v),
            *__class__._compute_stats(stats[h:], counts[h:], m, v),
        )

    def __len__(self) -> int:
        return len(self._stats)


IMAGE_STATS_SCHEMA = {
    "dataset": {
        "images count": 0,
        "unique images count": 0,
        "repeated images count": 0,
        "repeated images": [],  # [[id1, id2], [id3, id4, id5], ...]
    },
    "subsets": {},
}


def compute_image_statistics(dataset: IDataset):
    if dataset.media_type() != Image:
        raise DatumaroError(
            f"Your dataset's media_type is {dataset.media_type()}, "
            "but only Image media_type is allowed."
        )

    stats = deepcopy(IMAGE_STATS_SCHEMA)

    stats_counter = _MeanStdCounter()
    unique_counter = _ItemMatcher()

    for item in dataset:
        if not isinstance(item.media, Image):
            warnings.warn(
                f"item (id: {item.id}, subset: {item.subset})"
                f" has media_type, {item.media} but only Image media_type is allowed."
            )
            continue

        stats_counter.accumulate(item)
        unique_counter.process_item(item)

    def _extractor_stats(subset_name):
        sub_counter = _MeanStdCounter()
        sub_counter._stats = {
            k: v
            for k, v in stats_counter._stats.items()
            if subset_name and k[1] == subset_name or not subset_name
        }

        available = len(sub_counter._stats) != 0

        stats = {
            "images count": len(sub_counter),
        }

        if available:
            mean, std = sub_counter.get_result()

            stats.update(
                {
                    "image mean": [float(v) for v in mean[::-1]],
                    "image std": [float(v) for v in std[::-1]],
                }
            )
        else:
            stats.update(
                {
                    "image mean": "n/a",
                    "image std": "n/a",
                }
            )
        return stats

    for subset_name in dataset.subsets():
        stats["subsets"][subset_name] = _extractor_stats(subset_name)

    unique_items = unique_counter.get_result()
    repeated_items = [sorted(g) for g in unique_items.values() if 1 < len(g)]

    stats["dataset"].update(
        {
            "images count": len(stats_counter),
            "unique images count": len(unique_items),
            "repeated images count": len(repeated_items),
            "repeated images": repeated_items,  # [[id1, id2], [id3, id4, id5], ...]
        }
    )

    return stats


def compute_ann_statistics(dataset: IDataset):
    labels = dataset.categories().get(AnnotationType.label, LabelCategories())

    def get_label(ann):
        return labels.items[ann.label].name if ann.label is not None else None

    stats = {
        "images count": 0,
        "annotations count": 0,
        "unannotated images count": 0,
        "unannotated images": [],
        "annotations by type": {
            t.name: {
                "count": 0,
            }
            for t in AnnotationType
        },
        "annotations": {},
    }
    by_type = stats["annotations by type"]

    attr_template = {
        "count": 0,
        "values count": 0,
        "values present": set(),
        "distribution": {},  # value -> (count, total%)
    }
    label_stat = {
        "count": 0,
        "distribution": {l.name: [0, 0] for l in labels.items},  # label -> (count, total%)
        "attributes": {},
    }
    stats["annotations"]["labels"] = label_stat
    segm_stat = {
        "avg. area": 0,
        "area distribution": [],  # a histogram with 10 bins
        # (min, min+10%), ..., (min+90%, max) -> (count, total%)
        "pixel distribution": {l.name: [0, 0] for l in labels.items},  # label -> (count, total%)
    }
    stats["annotations"]["segments"] = segm_stat
    segm_areas = []
    pixel_dist = segm_stat["pixel distribution"]
    total_pixels = 0

    for item in dataset:
        if len(item.annotations) == 0:
            stats["unannotated images"].append(item.id)
            continue

        for ann in item.annotations:
            by_type[ann.type.name]["count"] += 1

            if not hasattr(ann, "label") or ann.label is None:
                continue

            if ann.type in {AnnotationType.mask, AnnotationType.polygon, AnnotationType.bbox}:
                area = ann.get_area()
                segm_areas.append(area)
                pixel_dist[get_label(ann)][0] += int(area)

            label_stat["count"] += 1
            label_stat["distribution"][get_label(ann)][0] += 1

            for name, value in ann.attributes.items():
                if name.lower() in {"occluded", "visibility", "score", "id", "track_id"}:
                    continue
                attrs_stat = label_stat["attributes"].setdefault(name, deepcopy(attr_template))
                attrs_stat["count"] += 1
                attrs_stat["values present"].add(str(value))
                attrs_stat["distribution"].setdefault(str(value), [0, 0])[0] += 1

    stats["images count"] = len(dataset)

    stats["annotations count"] = sum(t["count"] for t in stats["annotations by type"].values())
    stats["unannotated images count"] = len(stats["unannotated images"])

    for label_info in label_stat["distribution"].values():
        label_info[1] = label_info[0] / (label_stat["count"] or 1)

    for label_attr in label_stat["attributes"].values():
        label_attr["values count"] = len(label_attr["values present"])
        label_attr["values present"] = sorted(label_attr["values present"])
        for attr_info in label_attr["distribution"].values():
            attr_info[1] = attr_info[0] / (label_attr["count"] or 1)

    # numpy.sum might be faster, but could overflow with large datasets.
    # Python's int can transparently mutate to be of indefinite precision (long)
    total_pixels = sum(int(a) for a in segm_areas)

    segm_stat["avg. area"] = total_pixels / (len(segm_areas) or 1.0)

    for label_info in segm_stat["pixel distribution"].values():
        label_info[1] = label_info[0] / (total_pixels or 1)

    if len(segm_areas) != 0:
        hist, bins = np.histogram(segm_areas)
        segm_stat["area distribution"] = [
            {
                "min": float(bin_min),
                "max": float(bin_max),
                "count": int(c),
                "percent": int(c) / len(segm_areas),
            }
            for c, (bin_min, bin_max) in zip(hist, zip(bins[:-1], bins[1:]))
        ]

    return stats


@attrs
class DistanceComparator:
    iou_threshold = attrib(converter=float, default=0.5)

    def match_annotations(self, item_a, item_b):
        return {t: self._match_ann_type(t, item_a, item_b) for t in AnnotationType}

    def _match_ann_type(self, t, *args):
        # pylint: disable=no-value-for-parameter
        if t == AnnotationType.label:
            return self.match_labels(*args)
        elif t == AnnotationType.bbox:
            return self.match_boxes(*args)
        elif t == AnnotationType.polygon:
            return self.match_polygons(*args)
        elif t == AnnotationType.mask:
            return self.match_masks(*args)
        elif t == AnnotationType.points:
            return self.match_points(*args)
        elif t == AnnotationType.polyline:
            return self.match_lines(*args)
        # pylint: enable=no-value-for-parameter
        else:
            raise NotImplementedError("Unexpected annotation type %s" % t)

    @staticmethod
    def _get_ann_type(t, item):
        return [a for a in item.annotations if a.type == t]

    def match_labels(self, item_a, item_b):
        a_labels = set(a.label for a in self._get_ann_type(AnnotationType.label, item_a))
        b_labels = set(a.label for a in self._get_ann_type(AnnotationType.label, item_b))

        matches = a_labels & b_labels
        a_unmatched = a_labels - b_labels
        b_unmatched = b_labels - a_labels
        return matches, a_unmatched, b_unmatched

    def _match_segments(self, t, item_a, item_b):
        a_boxes = self._get_ann_type(t, item_a)
        b_boxes = self._get_ann_type(t, item_b)
        return match_segments(a_boxes, b_boxes, dist_thresh=self.iou_threshold)

    def match_polygons(self, item_a, item_b):
        return self._match_segments(AnnotationType.polygon, item_a, item_b)

    def match_masks(self, item_a, item_b):
        return self._match_segments(AnnotationType.mask, item_a, item_b)

    def match_boxes(self, item_a, item_b):
        return self._match_segments(AnnotationType.bbox, item_a, item_b)

    def match_points(self, item_a, item_b):
        a_points = self._get_ann_type(AnnotationType.points, item_a)
        b_points = self._get_ann_type(AnnotationType.points, item_b)

        instance_map = {}
        for s in [item_a.annotations, item_b.annotations]:
            s_instances = find_instances(s)
            for inst in s_instances:
                inst_bbox = max_bbox(inst)
                for ann in inst:
                    instance_map[id(ann)] = [inst, inst_bbox]
        matcher = PointsMatcher(instance_map=instance_map)

        return match_segments(
            a_points, b_points, dist_thresh=self.iou_threshold, distance=matcher.distance
        )

    def match_lines(self, item_a, item_b):
        a_lines = self._get_ann_type(AnnotationType.polyline, item_a)
        b_lines = self._get_ann_type(AnnotationType.polyline, item_b)

        matcher = LineMatcher()

        return match_segments(
            a_lines, b_lines, dist_thresh=self.iou_threshold, distance=matcher.distance
        )


def match_items_by_id(a: IDataset, b: IDataset):
    a_items = set((item.id, item.subset) for item in a)
    b_items = set((item.id, item.subset) for item in b)

    matches = a_items & b_items
    matches = [([m], [m]) for m in matches]
    a_unmatched = a_items - b_items
    b_unmatched = b_items - a_items
    return matches, a_unmatched, b_unmatched


def match_items_by_image_hash(a: IDataset, b: IDataset):
    a_hash = find_unique_images(a)
    b_hash = find_unique_images(b)

    a_items = set(a_hash)
    b_items = set(b_hash)

    matches = a_items & b_items
    a_unmatched = a_items - b_items
    b_unmatched = b_items - a_items

    matches = [(a_hash[h], b_hash[h]) for h in matches]
    a_unmatched = set(i for h in a_unmatched for i in a_hash[h])
    b_unmatched = set(i for h in b_unmatched for i in b_hash[h])

    return matches, a_unmatched, b_unmatched


class _ItemMatcher:
    @staticmethod
    def _default_item_hash(item: DatasetItem):
        if not item.media or not item.media.has_data:
            if item.media and hasattr(item.media, "path"):
                return hash(item.media.path)

            log.warning(
                "Item (%s, %s) has no image " "info, counted as unique", item.id, item.subset
            )
            return None

        # Disable B303:md5, because the hash is not used in a security context
        return hashlib.md5(item.media.data.tobytes()).hexdigest()  # nosec

    def __init__(self, item_hash: Optional[Callable] = None):
        self._hash = item_hash or self._default_item_hash

        # hash -> [(id, subset), ...]
        self._unique: Dict[str, Set[Tuple[str, str]]] = {}

    def process_item(self, item: DatasetItem):
        h = self._hash(item)
        if h is None:
            h = str(id(item))  # anything unique

        self._unique.setdefault(h, set()).add((item.id, item.subset))

    def get_result(self):
        return self._unique


def find_unique_images(dataset: IDataset, item_hash: Optional[Callable] = None):
    matcher = _ItemMatcher(item_hash=item_hash)
    for item in dataset:
        matcher.process_item(item)
    return matcher.get_result()


def match_classes(a: CategoriesInfo, b: CategoriesInfo):
    a_label_cat = a.get(AnnotationType.label, LabelCategories())
    b_label_cat = b.get(AnnotationType.label, LabelCategories())

    a_labels = set(c.name for c in a_label_cat)
    b_labels = set(c.name for c in b_label_cat)

    matches = a_labels & b_labels
    a_unmatched = a_labels - b_labels
    b_unmatched = b_labels - a_labels
    return matches, a_unmatched, b_unmatched


@attrs
class ExactComparator:
    match_images: bool = attrib(kw_only=True, default=False)
    ignored_fields = attrib(kw_only=True, factory=set, validator=default_if_none(set))
    ignored_attrs = attrib(kw_only=True, factory=set, validator=default_if_none(set))
    ignored_item_attrs = attrib(kw_only=True, factory=set, validator=default_if_none(set))

    _test: TestCase = attrib(init=False)
    errors: list = attrib(init=False)

    def __attrs_post_init__(self):
        self._test = TestCase()
        self._test.maxDiff = None

    def _match_items(self, a, b):
        if self.match_images:
            return match_items_by_image_hash(a, b)
        else:
            return match_items_by_id(a, b)

    def _compare_categories(self, a, b):
        test = self._test
        errors = self.errors

        try:
            test.assertEqual(sorted(a, key=lambda t: t.value), sorted(b, key=lambda t: t.value))
        except AssertionError as e:
            errors.append({"type": "categories", "message": str(e)})

        if AnnotationType.label in a:
            try:
                test.assertEqual(
                    a[AnnotationType.label].items,
                    b[AnnotationType.label].items,
                )
            except AssertionError as e:
                errors.append({"type": "labels", "message": str(e)})
        if AnnotationType.mask in a:
            try:
                test.assertEqual(
                    a[AnnotationType.mask].colormap,
                    b[AnnotationType.mask].colormap,
                )
            except AssertionError as e:
                errors.append({"type": "colormap", "message": str(e)})
        if AnnotationType.points in a:
            try:
                test.assertEqual(
                    a[AnnotationType.points].items,
                    b[AnnotationType.points].items,
                )
            except AssertionError as e:
                errors.append({"type": "points", "message": str(e)})

    def _compare_annotations(self, a, b):
        ignored_fields = self.ignored_fields
        ignored_attrs = self.ignored_attrs

        a_fields = {k: None for k in a.as_dict() if k in ignored_fields}
        b_fields = {k: None for k in b.as_dict() if k in ignored_fields}
        if "attributes" not in ignored_fields:
            a_fields["attributes"] = filter_dict(a.attributes, ignored_attrs)
            b_fields["attributes"] = filter_dict(b.attributes, ignored_attrs)

        result = a.wrap(**a_fields) == b.wrap(**b_fields)

        return result

    def _compare_items(self, item_a, item_b):
        test = self._test

        a_id = (item_a.id, item_a.subset)
        b_id = (item_b.id, item_b.subset)

        matched = []
        unmatched = []
        errors = []

        try:
            test.assertEqual(
                filter_dict(item_a.attributes, self.ignored_item_attrs),
                filter_dict(item_b.attributes, self.ignored_item_attrs),
            )
        except AssertionError as e:
            errors.append({"type": "item_attr", "a_item": a_id, "b_item": b_id, "message": str(e)})

        b_annotations = item_b.annotations[:]
        for ann_a in item_a.annotations:
            ann_b_candidates = [x for x in item_b.annotations if x.type == ann_a.type]

            ann_b = find(
                enumerate(self._compare_annotations(ann_a, x) for x in ann_b_candidates),
                lambda x: x[1],
            )
            if ann_b is None:
                unmatched.append(
                    {
                        "item": a_id,
                        "source": "a",
                        "ann": str(ann_a),
                    }
                )
                continue
            else:
                ann_b = ann_b_candidates[ann_b[0]]

            b_annotations.remove(ann_b)  # avoid repeats
            matched.append({"a_item": a_id, "b_item": b_id, "a": str(ann_a), "b": str(ann_b)})

        for ann_b in b_annotations:
            unmatched.append({"item": b_id, "source": "b", "ann": str(ann_b)})

        return matched, unmatched, errors

    def compare_datasets(self, a, b):
        self.errors = []
        errors = self.errors

        self._compare_categories(a.categories(), b.categories())

        matched = []
        unmatched = []

        matches, a_unmatched, b_unmatched = self._match_items(a, b)

        if a.categories().get(AnnotationType.label) != b.categories().get(AnnotationType.label):
            return matched, unmatched, a_unmatched, b_unmatched, errors

        _dist = lambda s: len(s[1]) + len(s[2])
        for a_ids, b_ids in matches:
            # build distance matrix
            match_status = {}  # (a_id, b_id): [matched, unmatched, errors]
            a_matches = {a_id: None for a_id in a_ids}
            b_matches = {b_id: None for b_id in b_ids}

            for a_id in a_ids:
                item_a = a.get(*a_id)
                candidates = {}

                for b_id in b_ids:
                    item_b = b.get(*b_id)

                    i_m, i_um, i_err = self._compare_items(item_a, item_b)
                    candidates[b_id] = [i_m, i_um, i_err]

                    if len(i_um) == 0:
                        a_matches[a_id] = b_id
                        b_matches[b_id] = a_id
                        matched.extend(i_m)
                        errors.extend(i_err)
                        break

                match_status[a_id] = candidates

            # assign
            for a_id in a_ids:
                if len(b_ids) == 0:
                    break

                # find the closest, ignore already assigned
                matched_b = a_matches[a_id]
                if matched_b is not None:
                    continue
                min_dist = -1
                for b_id in b_ids:
                    if b_matches[b_id] is not None:
                        continue
                    d = _dist(match_status[a_id][b_id])
                    if d < min_dist and 0 <= min_dist:
                        continue
                    min_dist = d
                    matched_b = b_id

                if matched_b is None:
                    continue
                a_matches[a_id] = matched_b
                b_matches[matched_b] = a_id

                m = match_status[a_id][matched_b]
                matched.extend(m[0])
                unmatched.extend(m[1])
                errors.extend(m[2])

            a_unmatched |= set(a_id for a_id, m in a_matches.items() if not m)
            b_unmatched |= set(b_id for b_id, m in b_matches.items() if not m)

        return matched, unmatched, a_unmatched, b_unmatched, errors
