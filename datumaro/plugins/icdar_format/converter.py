# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp

from datumaro.components.converter import Converter
from datumaro.components.extractor import AnnotationType, CompiledMask
from datumaro.util.image import save_image
from datumaro.util.mask_tools import paint_mask

from .format import IcdarPath, IcdarTask


class IcdarWordRecognitionConverter(Converter):
    DEFAULT_IMAGE_EXT = IcdarPath.IMAGE_EXT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task = IcdarTask.word_recognition

    def apply(self):
        for subset_name, subset in self._extractor.subsets().items():
            annotation = ''
            for item in subset:
                if item.has_image and self._save_images:
                    self._save_image(item, osp.join(self._save_dir, subset_name,
                        IcdarPath.IMAGES_DIR, item.id + IcdarPath.IMAGE_EXT))

                annotation += '%s, ' % (item.id + IcdarPath.IMAGE_EXT)
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task = IcdarTask.text_localization

    def apply(self):
        for subset_name, subset in self._extractor.subsets().items():
            for item in subset:
                if item.has_image and self._save_images:
                    self._save_image(item, osp.join(self._save_dir, subset_name,
                        IcdarPath.IMAGES_DIR, item.id + IcdarPath.IMAGE_EXT))

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
                anno_file = osp.join(self._save_dir, subset_name, osp.dirname(item.id),
                    'gt_' + osp.basename(item.id) + '.txt')
                os.makedirs(osp.dirname(anno_file), exist_ok=True)
                with open(anno_file, 'w', encoding='utf-8') as f:
                    f.write(annotation)

class IcdarTextSegmentationConverter(Converter):
    DEFAULT_IMAGE_EXT = IcdarPath.IMAGE_EXT
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task = IcdarTask.text_segmentation

    def apply(self):
        for subset_name, subset in self._extractor.subsets().items():
            for item in subset:
                if item.has_image and self._save_images:
                    self._save_image(item, osp.join(self._save_dir, subset_name,
                        IcdarPath.IMAGES_DIR, item.id + IcdarPath.IMAGE_EXT))

                annotation = ''
                colormap = [(255, 255, 255)]
                anns = [a for a in item.annotations
                    if a.type == AnnotationType.mask]
                if anns:
                    is_not_index = len([p for p in anns if 'index' not in p.attributes])
                    if is_not_index:
                        raise Exception("Item %s: a mask must have"
                            "'index' attribute" % item.id)
                    anns = sorted(anns, key=lambda a: a.attributes['index'])
                    group = anns[0].group
                    for ann in anns:
                        if ann.group != group or (not ann.group and anns[0].group != 0):
                            annotation += '\n'
                        text = ''
                        if ann.attributes:
                            if 'text' in ann.attributes:
                                text = ann.attributes['text']
                            if text == ' ':
                                annotation += '#'
                            if 'color' in ann.attributes and \
                                    len(ann.attributes['color'].split()) == 3:
                                color = ann.attributes['color'].split()
                                colormap.append(
                                    (int(color[0]), int(color[1]), int(color[2])))
                                annotation += ' '.join(p for p in color)
                            else:
                                raise Exception("Item %s: a mask must have "
                                    "an RGB color attribute, e. g. '10 7 50'" % item.id)
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

                anno_file = osp.join(self._save_dir, subset_name, item.id + '_GT' + '.txt')
                os.makedirs(osp.dirname(anno_file), exist_ok=True)
                with open(anno_file, 'w', encoding='utf-8') as f:
                    f.write(annotation)