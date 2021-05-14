
# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
import json
import logging as log
import os.path as osp

from pycocotools.coco import COCO
import pycocotools.mask as mask_utils

from datumaro.components.extractor import (CompiledMask, Mask, SourceExtractor,
    DEFAULT_SUBSET_NAME, DatasetItem,
    AnnotationType, Label, RleMask, Points, Polygon, Bbox, Caption,
    LabelCategories, PointsCategories
)
from datumaro.util.image import Image, lazy_image, load_image
from datumaro.util.mask_tools import bgr2index

from .format import CocoTask, CocoPath


class _CocoExtractor(SourceExtractor):
    def __init__(self, path, task, merge_instance_polygons=False, subset=None):
        assert osp.isfile(path), path

        if not subset:
            subset = osp.splitext(osp.basename(path))[0].rsplit('_', maxsplit=1)
            subset = subset[1] if len(subset) == 2 else None
        super().__init__(subset=subset)

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

        self._merge_instance_polygons = merge_instance_polygons

        if self._task == CocoTask.panoptic:
            #panoptic is not added to pycocotools
            panoptic_config = self._load_panoptic_config(path)
            panoptic_images = osp.splitext(path)[0]

            self._load_panoptic_categories(panoptic_config)
            self._items = list(self._load_panoptic_items(panoptic_config,
                panoptic_images).values())
        else:
            loader = self._make_subset_loader(path)
            self._load_categories(loader)
            self._items = list(self._load_items(loader).values())

    @staticmethod
    def _make_subset_loader(path):
        # COCO API has an 'unclosed file' warning
        coco_api = COCO()
        with open(path, 'r') as f:
            dataset = json.load(f)

        coco_api.dataset = dataset
        coco_api.createIndex()
        return coco_api

    def _load_categories(self, loader):
        self._categories = {}

        if self._task in [CocoTask.instances, CocoTask.labels,
                CocoTask.person_keypoints, CocoTask.stuff]:
            label_categories, label_map = self._load_label_categories(loader)
            self._categories[AnnotationType.label] = label_categories
            self._label_map = label_map

        if self._task == CocoTask.person_keypoints:
            person_kp_categories = self._load_person_kp_categories(loader)
            self._categories[AnnotationType.points] = person_kp_categories

    # pylint: disable=no-self-use
    def _load_label_categories(self, loader):
        catIds = loader.getCatIds()
        cats = loader.loadCats(catIds)

        categories = LabelCategories()
        label_map = {}
        for idx, cat in enumerate(cats):
            label_map[cat['id']] = idx
            categories.add(name=cat['name'], parent=cat.get('supercategory'))

        return categories, label_map
    # pylint: enable=no-self-use

    def _load_person_kp_categories(self, loader):
        catIds = loader.getCatIds()
        cats = loader.loadCats(catIds)

        categories = PointsCategories()
        for cat in cats:
            label_id = self._label_map[cat['id']]
            categories.add(label_id=label_id,
                labels=cat['keypoints'], joints=cat['skeleton']
            )

        return categories

    @staticmethod
    def _load_panoptic_config(path):
        with open(path, 'r') as f:
            return json.load(f)

    def _load_panoptic_categories(self, config):
        label_categories = LabelCategories()
        label_map = {}
        for idx, cat in enumerate(config['categories']):
            label_map[cat['id']] = idx
            label_categories.add(name=cat['name'],
                parent=cat.get('supercategory'))

        self._categories[AnnotationType.label] = label_categories
        self._label_map = label_map

    def _load_items(self, loader):
        items = OrderedDict()

        for img_id in loader.getImgIds():
            image_info = loader.loadImgs(img_id)[0]
            image_path = osp.join(self._images_dir, image_info['file_name'])
            image_size = (image_info.get('height'), image_info.get('width'))
            if all(image_size):
                image_size = (int(image_size[0]), int(image_size[1]))
            else:
                image_size = None
            image = Image(path=image_path, size=image_size)

            anns = loader.getAnnIds(imgIds=img_id)
            anns = loader.loadAnns(anns)
            anns = sum((self._load_annotations(a, image_info) for a in anns), [])

            items[img_id] = DatasetItem(
                id=osp.splitext(image_info['file_name'])[0],
                subset=self._subset, image=image, annotations=anns,
                attributes={'id': img_id})

        return items

    def _load_panoptic_items(self, config, panoptic_images):
        items = OrderedDict()

        imgs_info = {}
        for img in config['images']:
            imgs_info[img['id']] = img

        for ann in config['annotations']:
            img_id = int(ann['image_id'])
            image_path = osp.join(self._images_dir, imgs_info[img_id]['file_name'])
            image_size = (imgs_info[img_id].get('height'),
                imgs_info[img_id].get('width'))
            if all(image_size):
                image_size = (int(image_size[0]), int(image_size[1]))
            else:
                image_size = None
            image = Image(path=image_path, size=image_size)
            anns = []

            mask_path = osp.join(panoptic_images, ann['file_name'])
            mask = lazy_image(mask_path, loader=self._load_pan_mask)
            mask = CompiledMask(instance_mask=mask)
            for segm_info in ann['segments_info']:
                cat_id = self._get_label_id(segm_info)
                segm_id = segm_info['id']
                attributes = { 'is_crowd': bool(segm_info['iscrowd']) }
                anns.append(Mask(image=mask.lazy_extract(segm_id),
                    label=cat_id, id=segm_id,
                    group=segm_id, attributes=attributes))

            items[img_id] = DatasetItem(
                id=osp.splitext(imgs_info[img_id]['file_name'])[0],
                subset=self._subset, image=image,
                annotations=anns, attributes={'id': img_id})
        return items

    @staticmethod
    def _load_pan_mask(path):
        mask = load_image(path)
        mask = bgr2index(mask)
        return mask

    def _get_label_id(self, ann):
        cat_id = ann.get('category_id')
        if cat_id in [0, None]:
            return None
        return self._label_map[cat_id]

    def _load_annotations(self, ann, image_info=None):
        parsed_annotations = []

        ann_id = ann.get('id')

        attributes = {}
        if 'attributes' in ann:
            try:
                attributes.update(ann['attributes'])
            except Exception as e:
                log.debug("item #%s: failed to read annotation attributes: %s",
                    image_info['id'], e)
        if 'score' in ann:
            attributes['score'] = ann['score']

        group = ann_id # make sure all tasks' annotations are merged

        if self._task in [CocoTask.instances, CocoTask.person_keypoints,
            CocoTask.stuff]:
            x, y, w, h = ann['bbox']
            label_id = self._get_label_id(ann)

            is_crowd = bool(ann['iscrowd'])
            attributes['is_crowd'] = is_crowd

            if self._task is CocoTask.person_keypoints:
                keypoints = ann['keypoints']
                points = [p for i, p in enumerate(keypoints) if i % 3 != 2]
                visibility = keypoints[2::3]
                parsed_annotations.append(
                    Points(points, visibility, label=label_id,
                        id=ann_id, attributes=attributes, group=group)
                )

            segmentation = ann.get('segmentation')
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
                        img_h = image_info['height']
                        img_w = image_info['width']
                        rles = mask_utils.frPyObjects(segmentation, img_h, img_w)
                        rle = mask_utils.merge(rles)
                elif isinstance(segmentation['counts'], list):
                    # uncompressed RLE
                    img_h = image_info['height']
                    img_w = image_info['width']
                    mask_h, mask_w = segmentation['size']
                    if img_h == mask_h and img_w == mask_w:
                        rle = mask_utils.frPyObjects(
                            [segmentation], mask_h, mask_w)[0]
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

                if rle is not None:
                    parsed_annotations.append(RleMask(rle=rle, label=label_id,
                        id=ann_id, attributes=attributes, group=group
                    ))
            else:
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
