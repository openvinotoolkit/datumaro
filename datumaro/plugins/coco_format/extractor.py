# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Any
import json
import logging as log
import os.path as osp

from attrs import define
import pycocotools.mask as mask_utils

from datumaro.components.annotation import (
    AnnotationType, Bbox, Caption, CompiledMask, Label, LabelCategories, Mask,
    Points, PointsCategories, Polygon, RleMask,
)
from datumaro.components.errors import (
    AnnotationImportError, DatumaroError, ItemImportError,
)
from datumaro.components.extractor import (
    DEFAULT_SUBSET_NAME, AnnotationImportErrorAction, DatasetItem,
    ItemImportErrorAction, SourceExtractor,
)
from datumaro.components.media import Image
from datumaro.util import take_by
from datumaro.util.image import lazy_image, load_image
from datumaro.util.mask_tools import bgr2index
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file

from .format import CocoPath, CocoTask


class _CocoExtractor(SourceExtractor):
    """
    Parses COCO annotations written in the following format:
    https://cocodataset.org/#format-data
    """

    def __init__(self, path, task, *,
        merge_instance_polygons=False,
        subset=None,
        keep_original_category_ids=False,
        **kwargs
    ):
        assert osp.isfile(path), path

        if not subset:
            parts = osp.splitext(osp.basename(path))[0].split(task.name + '_',
                maxsplit=1)
            subset = parts[1] if len(parts) == 2 else None
        super().__init__(subset=subset, **kwargs)

        rootpath = ''
        if path.endswith(osp.join(CocoPath.ANNOTATIONS_DIR, osp.basename(path))):
            rootpath = path.rsplit(CocoPath.ANNOTATIONS_DIR, maxsplit=1)[0]
        images_dir = ''
        if rootpath and osp.isdir(osp.join(rootpath, CocoPath.IMAGES_DIR)):
            images_dir = osp.join(rootpath, CocoPath.IMAGES_DIR)
            if osp.isdir(osp.join(images_dir, subset or DEFAULT_SUBSET_NAME)):
                images_dir = osp.join(images_dir, subset or DEFAULT_SUBSET_NAME)
        self._images_dir = images_dir
        self._task = task
        self._rootpath = rootpath

        self._merge_instance_polygons = merge_instance_polygons

        json_data = self._load_json(path)
        self._label_map = {} # coco_id -> dm_id
        self._load_categories(json_data,
            keep_original_ids=keep_original_category_ids,
        )

        if self._task == CocoTask.panoptic:
            self._mask_dir = osp.splitext(path)[0]
        self._items = self._load_items(json_data)

    def __iter__(self):
        yield from self._items.values()

    def _load_categories(self, json_data, *, keep_original_ids):
        if has_meta_file(self._rootpath):
            labels = parse_meta_file(self._rootpath).keys()
            self._categories = {
                AnnotationType.label: LabelCategories.from_iterable(labels)
            }
            # 0 is reserved for no class
            self._label_map = { i + 1: i for i in range(len(labels)) }
            return

        self._categories = {}

        if self._task in [CocoTask.instances, CocoTask.labels,
                CocoTask.person_keypoints, CocoTask.stuff,
                CocoTask.panoptic]:
            self._load_label_categories(json_data['categories'],
                keep_original_ids=keep_original_ids,
            )

        if self._task == CocoTask.person_keypoints:
            self._load_person_kp_categories(json_data['categories'])

    def _load_label_categories(self, json_cat, *, keep_original_ids):
        categories = LabelCategories()
        label_map = {}

        if keep_original_ids:
            for cat in sorted(json_cat, key=lambda cat: cat['id']):
                label_map[cat['id']] = cat['id']

                while len(categories) < cat['id']:
                    categories.add(f"class-{len(categories)}")

                categories.add(cat['name'], parent=cat.get('supercategory'))
        else:
            for idx, cat in enumerate(sorted(json_cat, key=lambda cat: cat['id'])):
                label_map[cat['id']] = idx
                categories.add(cat['name'], parent=cat.get('supercategory'))

        self._categories[AnnotationType.label] = categories
        self._label_map = label_map

    def _load_person_kp_categories(self, json_cat):
        categories = PointsCategories()
        for cat in json_cat:
            label_id = self._label_map[cat['id']]
            categories.add(label_id,
                labels=cat['keypoints'], joints=cat['skeleton']
            )

        self._categories[AnnotationType.points] = categories

    @staticmethod
    def _load_json(path):
        with open(path, 'rb') as f:
            return json.loads(f.read())

    def _load_items(self, json_data):
        items = {}

        img_infos = {}
        for img_info in self._with_progress(json_data['images'],
                desc='Parsing image info'):
            try:
                img_id = img_info['id']
                img_infos[img_id] = img_info

                if img_info.get('height') and img_info.get('width'):
                    image_size = (img_info['height'], img_info['width'])
                else:
                    image_size = None

                items[img_id] = DatasetItem(
                    id='', # osp.splitext(img_info['file_name'])[0],
                    subset=self._subset,
                    image=Image(
                        path=osp.join(self._images_dir, img_info['file_name']),
                        size=image_size),
                    annotations=[],
                    attributes={'id': img_id})
            except Exception as e:
                error_action = self._report_item_error(e, item=img_id)
                if error_action is ItemImportErrorAction.skip_item:
                    continue

        if self._task is not CocoTask.panoptic:
            for ann in self._with_progress(json_data['annotations'],
                    desc='Parsing annotations'):
                try:
                    items[ann['image_id']].annotations += \
                        self._load_annotations(ann, img_infos[ann['image_id']])
                except Exception as e:
                    error_action = self._report_annotation_error(e, item=img_id)
                    if error_action is AnnotationImportErrorAction.skip_item:
                        continue
        else:
            self._load_panoptic_ann(items, json_data)

        return items

    def _load_panoptic_ann(self, items, json_data):
        for ann in self._with_progress(json_data['annotations'],
                desc='Parsing annotations'):
            # For the panoptic task, each annotation struct is a per-image
            # annotation rather than a per-object annotation.
            anns = items[ann['image_id']].annotations
            mask_path = osp.join(self._mask_dir, ann['file_name'])
            mask = lazy_image(mask_path, loader=self._load_pan_mask)
            mask = CompiledMask(instance_mask=mask)
            for segm_info in ann['segments_info']:
                cat_id = self._get_label_id(segm_info)
                segm_id = segm_info['id']
                attributes = { 'is_crowd': bool(segm_info['iscrowd']) }
                anns.append(Mask(image=mask.lazy_extract(segm_id),
                    label=cat_id, id=segm_id,
                    group=segm_id, attributes=attributes))

    @staticmethod
    def _load_pan_mask(path):
        mask = load_image(path)
        mask = bgr2index(mask)
        return mask

    @define
    class _lazy_merged_mask:
        segmentation: Any
        h: int
        w: int

        def __call__(self):
            rles = mask_utils.frPyObjects(self.segmentation, self.h, self.w)
            return mask_utils.merge(rles)

    def _get_label_id(self, ann):
        if not ann['category_id']:
            return None
        return self._label_map[ann['category_id']]

    def _load_annotations(self, ann, image_info=None):
        parsed_annotations = []

        ann_id = ann['id']

        attributes = ann.get('attributes', {})
        if 'score' in ann:
            attributes['score'] = ann['score']

        group = ann_id # make sure all tasks' annotations are merged

        if self._task is CocoTask.instances or \
                self._task is CocoTask.person_keypoints or \
                self._task is CocoTask.stuff:
            label_id = self._get_label_id(ann)

            attributes['is_crowd'] = bool(ann['iscrowd'])

            if self._task is CocoTask.person_keypoints:
                keypoints = ann['keypoints']
                points = []
                visibility = []
                for x, y, v in take_by(keypoints, 3):
                    points.append(x)
                    points.append(y)
                    visibility.append(v)

                parsed_annotations.append(
                    Points(points, visibility, label=label_id,
                        id=ann_id, attributes=attributes, group=group)
                )

            segmentation = ann['segmentation']
            if segmentation and segmentation != [[]]:
                rle = None

                if isinstance(segmentation, list):
                    if not self._merge_instance_polygons:
                        # polygon - a single object can consist of multiple parts
                        for polygon_points in segmentation:
                            parsed_annotations.append(Polygon(
                                points=polygon_points, label=label_id,
                                id=ann_id, attributes=attributes, group=group
                            ))
                    else:
                        # merge all parts into a single mask RLE
                        rle = self._lazy_merged_mask(segmentation,
                            image_info['height'], image_info['width'])
                elif isinstance(segmentation['counts'], list):
                    # uncompressed RLE
                    img_h = image_info['height']
                    img_w = image_info['width']
                    mask_h, mask_w = segmentation['size']
                    if img_h == mask_h and img_w == mask_w:
                        rle = self._lazy_merged_mask(
                            [segmentation], mask_h, mask_w)
                    else:
                        log.warning("item #%s: mask #%s "
                            "does not match image size: %s vs. %s. "
                            "Skipping this annotation.",
                            image_info['id'], ann_id,
                            (mask_h, mask_w), (img_h, img_w)
                        )
                else:
                    # compressed RLE
                    rle = segmentation

                if rle:
                    parsed_annotations.append(RleMask(rle=rle, label=label_id,
                        id=ann_id, attributes=attributes, group=group
                    ))
            else:
                x, y, w, h = ann['bbox']
                parsed_annotations.append(
                    Bbox(x, y, w, h, label=label_id,
                        id=ann_id, attributes=attributes, group=group)
                )
        elif self._task is CocoTask.labels:
            label_id = self._get_label_id(ann)
            parsed_annotations.append(
                Label(label=label_id,
                    id=ann_id, attributes=attributes, group=group)
            )
        elif self._task is CocoTask.captions:
            caption = ann['caption']
            parsed_annotations.append(
                Caption(caption,
                    id=ann_id, attributes=attributes, group=group)
            )
        else:
            raise NotImplementedError()

        return parsed_annotations

class CocoImageInfoExtractor(_CocoExtractor):
    def __init__(self, path, **kwargs):
        kwargs['task'] = CocoTask.image_info
        super().__init__(path, **kwargs)

class CocoCaptionsExtractor(_CocoExtractor):
    def __init__(self, path, **kwargs):
        kwargs['task'] = CocoTask.captions
        super().__init__(path, **kwargs)

class CocoInstancesExtractor(_CocoExtractor):
    def __init__(self, path, **kwargs):
        kwargs['task'] = CocoTask.instances
        super().__init__(path, **kwargs)

class CocoPersonKeypointsExtractor(_CocoExtractor):
    def __init__(self, path, **kwargs):
        kwargs['task'] = CocoTask.person_keypoints
        super().__init__(path, **kwargs)

class CocoLabelsExtractor(_CocoExtractor):
    def __init__(self, path, **kwargs):
        kwargs['task'] = CocoTask.labels
        super().__init__(path, **kwargs)

class CocoPanopticExtractor(_CocoExtractor):
    def __init__(self, path, **kwargs):
        kwargs['task'] = CocoTask.panoptic
        super().__init__(path, **kwargs)

class CocoStuffExtractor(_CocoExtractor):
    def __init__(self, path, **kwargs):
        kwargs['task'] = CocoTask.stuff
        super().__init__(path, **kwargs)
