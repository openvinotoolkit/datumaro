# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp

from datumaro.components.converter import Converter
from datumaro.components.extractor import AnnotationType, CompiledMask
from datumaro.util.image import save_image
from datumaro.util.mask_tools import find_mask_bbox, paint_mask

from .format import IcdarPath, IcdarTask


class _WordRecognitionConverter():
    def __init__(self):
        self.annotations = ''

    def save_categories(self, save_dir, categories):
        pass

    def save_annotations(self, item, categories):
        self.annotations += '%s, ' % (item.id + IcdarPath.IMAGE_EXT)
        for ann in item.annotations:
            if ann.type != AnnotationType.caption:
                continue
            self.annotations += '\"%s\"' % ann.caption
        self.annotations += '\n'

    def write(self, path):
        file = osp.join(path, 'gt.txt')
        os.makedirs(osp.dirname(file), exist_ok=True)
        with open(file, 'w') as f:
            f.write(self.annotations)

    def is_empty(self):
        return len(self.annotations) == 0

class _TextLocalizationConverter():
    def __init__(self):
        self.annotations = {}

    def save_categories(self, save_dir, categories):
        vocabulary_file = osp.join(save_dir,
            IcdarPath.TASK_DIR[IcdarTask.text_localization],
            IcdarPath.VOCABULARY_FILE)
        os.makedirs(osp.dirname(vocabulary_file), exist_ok=True)
        with open(vocabulary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(l.name
                for l in categories[AnnotationType.label])
            )

    def save_annotations(self, item, categories):
        annotation = ''
        for ann in item.annotations:
            if ann.type == AnnotationType.bbox:
                annotation += '%s %s %s %s' % (ann.x, ann.y,
                    ann.x + ann.w, ann.y + ann.h)
                if ann.label is not None:
                    annotation += ' %s' % \
                        categories[AnnotationType.label][ann.label].name
            elif ann.type == AnnotationType.polygon:
                annotation += ','.join(str(p) for p in ann.points)
                if ann.label is not None:
                    annotation += ',\"%s\"' % \
                        categories[AnnotationType.label][ann.label].name
            annotation += '\n'
        self.annotations[item.id] = annotation

    def write(self, path):
        os.makedirs(path, exist_ok=True)
        for item in self.annotations:
            file = osp.join(path, 'gt_' + item + '.txt')
            with open(file, 'w') as f:
                f.write(self.annotations[item])

    def is_empty(self):
        return len(self.annotations) == 0

class _TextSegmentationConverter():
    def __init__(self):
        self.annotations = {}
        self.masks = {}

    def save_categories(self, save_dir, categories):
        pass

    def save_annotations(self, item, categories):
        annotation = ''
        colormap = [(255, 255, 255)]
        group = 0
        for ann in item.annotations:
            if ann.type == AnnotationType.mask:
                char = ''
                if ann.group != group:
                    annotation += '\n'
                    group += 1
                if ann.attributes:
                    if 'char' in ann.attributes:
                        char = ann.attributes['char']
                    if char == ' ':
                        annotation += '#'
                    if 'color' in ann.attributes:
                        colormap.append(ann.attributes['color'])
                        annotation += ' '.join(str(p)
                            for p in ann.attributes['color'])
                    else:
                        annotation += '- - -'
                    if 'center' in ann.attributes:
                        annotation += ' '
                        annotation += ' '.join(str(p)
                            for p in ann.attributes['center'])
                    else:
                        annotation += ' - -'
                image = ann.image
                bbox = find_mask_bbox(image)
                annotation += ' %s %s %s %s' % (bbox[0], bbox[1],
                    bbox[0] + bbox[2], bbox[1] + bbox[3])
                annotation += ' \"%s\"' % char
                annotation += '\n'

        masks = [a for a in item.annotations
            if a.type == AnnotationType.mask]
        compiled_mask = CompiledMask.from_instance_masks(masks)
        mask = paint_mask(compiled_mask.class_mask, {
            i: colormap[i] for i in range(len(colormap))})
        self.annotations[item.id] = annotation
        self.masks[item.id] = mask

    def write(self, path):
        os.makedirs(path, exist_ok=True)
        for item in self.annotations:
            file = osp.join(path, item + '_GT' + '.txt')
            with open(file, 'w') as f:
                f.write(self.annotations[item])
            save_image(osp.join(path, item + '_GT' + IcdarPath.GT_EXT),
                self.masks[item], create_dir=True)

    def is_empty(self):
        return len(self.annotations) == 0


class IcdarConverter(Converter):
    DEFAULT_IMAGE_EXT = IcdarPath.IMAGE_EXT

    _TASK_CONVERTER = {
        IcdarTask.word_recognition: _WordRecognitionConverter,
        IcdarTask.text_localization: _TextLocalizationConverter,
        IcdarTask.text_segmentation: _TextSegmentationConverter,
    }

    def __init__(self, extractor, save_dir, tasks=None, **kwargs):
        super().__init__(extractor, save_dir, **kwargs)

        assert tasks is None or isinstance(tasks, (IcdarTask, list, str))
        if isinstance(tasks, IcdarTask):
            tasks = [tasks]
        elif isinstance(tasks, str):
            tasks = [IcdarTask[tasks]]
        elif tasks:
            for i, t in enumerate(tasks):
                if isinstance(t, str):
                    tasks[i] = IcdarTask[t]
                else:
                    assert t in IcdarTask, t
        self._tasks = tasks

    def _make_task_converter(self, task):
        if task not in self._TASK_CONVERTER:
            raise NotImplementedError()
        return self._TASK_CONVERTER[task]()

    def _make_task_converters(self):
        return { task: self._make_task_converter(task)
            for task in (self._tasks or self._TASK_CONVERTER) }

    def apply(self):
        for subset_name, subset in self._extractor.subsets().items():
            task_converters = self._make_task_converters()
            for task_conv in task_converters.values():
                task_conv.save_categories(self._save_dir,
                    self._extractor.categories())
            for item in subset:
                for task_conv in task_converters.values():
                    if item.has_image and self._save_images:
                        self._save_image(item, osp.join(self._save_dir, subset_name,
                            IcdarPath.IMAGES_DIR, item.id + IcdarPath.IMAGE_EXT))
                    task_conv.save_annotations(item,
                        self._extractor.categories())

            for task, task_conv in task_converters.items():
                if task_conv.is_empty() and not self._tasks:
                    continue
                task_conv.write(osp.join(self._save_dir,
                    IcdarPath.TASK_DIR[task], subset_name))

class IcdarWordRecognitionConverter(IcdarConverter):
    def __init__(self, *args, **kwargs):
        kwargs['tasks'] = IcdarTask.word_recognition
        super().__init__(*args, **kwargs)

class IcdarTextLocalizationConverter(IcdarConverter):
    def __init__(self, *args, **kwargs):
        kwargs['tasks'] = IcdarTask.text_localization
        super().__init__(*args, **kwargs)

class IcdarTextSegmentationConverter(IcdarConverter):
    def __init__(self, *args, **kwargs):
        kwargs['tasks'] = IcdarTask.text_segmentation
        super().__init__(*args, **kwargs)
