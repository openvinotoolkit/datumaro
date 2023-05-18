# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from unittest import TestCase

from attr import attrib, attrs

from datumaro.components.annotation import AnnotationType
from datumaro.components.annotations.matcher import LineMatcher, PointsMatcher, match_segments
from datumaro.components.operations import match_items_by_id, match_items_by_image_hash
from datumaro.util import filter_dict, find
from datumaro.util.annotation_util import find_instances, max_bbox
from datumaro.util.attrs_util import default_if_none


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
