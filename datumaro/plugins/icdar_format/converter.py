# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp

from datumaro.components.annotation import AnnotationType, CompiledMask
from datumaro.components.converter import Converter
from datumaro.util.image import save_image
from datumaro.util.mask_tools import generate_colormap, paint_mask

from .format import IcdarPath


class IcdarWordRecognitionConverter(Converter):
    DEFAULT_IMAGE_EXT = IcdarPath.IMAGE_EXT

    def apply(self):
        for subset_name, subset in self._extractor.subsets().items():
            annotation = ''
            for item in subset:
                image_filename = self._make_image_filename(item)
                if self._save_images and item.has_image:
                    self._save_image(item, osp.join(self._save_dir,
                        subset_name, IcdarPath.IMAGES_DIR, image_filename))

                annotation += '%s, ' % image_filename
                for ann in item.annotations:
                    if ann.type != AnnotationType.caption:
                        continue
                    annotation += '\"%s\"' % ann.caption
                annotation += '\n'

            if len(annotation):
                anno_file = osp.join(self._save_dir, subset_name, 'gt.txt')
                os.makedirs(osp.dirname(anno_file), exist_ok=True)
                with open(anno_file, 'w', encoding='utf-8') as f:
                    f.write(annotation)

class IcdarTextLocalizationConverter(Converter):
    DEFAULT_IMAGE_EXT = IcdarPath.IMAGE_EXT

    def apply(self):
        for subset_name, subset in self._extractor.subsets().items():
            for item in subset:
                if self._save_images and item.has_image:
                    self._save_image(item,
                        subdir=osp.join(subset_name, IcdarPath.IMAGES_DIR))

                annotation = ''
                for ann in item.annotations:
                    if ann.type == AnnotationType.bbox:
                        annotation += ' '.join(str(p) for p in ann.points)
                        if ann.attributes and 'text' in ann.attributes:
                            annotation += ' \"%s\"' % ann.attributes['text']
                    elif ann.type == AnnotationType.polygon:
                        annotation += ','.join(str(p) for p in ann.points)
                        if ann.attributes and 'text' in ann.attributes:
                            annotation += ',\"%s\"' % ann.attributes['text']
                    annotation += '\n'

                anno_file = osp.join(self._save_dir, subset_name,
                    osp.dirname(item.id), 'gt_' + osp.basename(item.id) + '.txt')
                os.makedirs(osp.dirname(anno_file), exist_ok=True)
                with open(anno_file, 'w', encoding='utf-8') as f:
                    f.write(annotation)

class IcdarTextSegmentationConverter(Converter):
    DEFAULT_IMAGE_EXT = IcdarPath.IMAGE_EXT

    def apply(self):
        for subset_name, subset in self._extractor.subsets().items():
            for item in subset:
                self._save_item(subset_name, subset, item)

    def _save_item(self, subset_name, subset, item):
        if self._save_images and item.has_image:
            self._save_image(item,
                subdir=osp.join(subset_name, IcdarPath.IMAGES_DIR))

        annotation = ''
        colormap = [(255, 255, 255)]
        anns = [a for a in item.annotations if a.type == AnnotationType.mask]
        gen_colormap = generate_colormap(len(anns), False)
        if anns:
            anns = sorted(anns, key=lambda a: int(a.attributes.get('index', 0)))
            group = anns[0].group
            for i, ann in enumerate(anns):
                if ann.group != group or (not ann.group and anns[0].group != 0):
                    annotation += '\n'
                text = ''
                if ann.attributes:
                    if 'text' in ann.attributes:
                        text = ann.attributes['text']
                    if text == ' ':
                        annotation += '#'
                    color = ann.attributes.get('color', '').split()
                    color = tuple(map(int, color)) if len(color) == 3 \
                        else gen_colormap.pop(i)
                    colormap.append(color)
                    annotation += ' '.join(str(p) for p in color)
                    if 'center' in ann.attributes:
                        annotation += ' %s' % ann.attributes['center']
                    else:
                        annotation += ' - -'
                bbox = ann.get_bbox()
                annotation += ' %s %s %s %s' % (bbox[0], bbox[1],
                    bbox[0] + bbox[2], bbox[1] + bbox[3])
                annotation += ' \"%s\"' % text
                annotation += '\n'
                group = ann.group

            mask = CompiledMask.from_instance_masks(anns,
                instance_labels=[m.attributes['index'] + 1 for m in anns])
            mask = paint_mask(mask.class_mask,
                { i: colormap[i] for i in range(len(colormap)) })
            save_image(osp.join(self._save_dir, subset_name,
                item.id + '_GT' + IcdarPath.GT_EXT), mask, create_dir=True)

        anno_file = osp.join(self._save_dir, subset_name,
            item.id + '_GT' + '.txt')
        os.makedirs(osp.dirname(anno_file), exist_ok=True)
        with open(anno_file, 'w', encoding='utf-8') as f:
            f.write(annotation)
