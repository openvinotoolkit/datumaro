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


class _WordRecognitionConverter:
    def __init__(self):
        self.annotations = ''

    def save_annotations(self, item, path):
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

class _TextLocalizationConverter:
    def __init__(self):
        self.annotations = {}

    def save_annotations(self, item, path):
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
        self.annotations[item.id] = annotation

    def write(self, path):
        os.makedirs(path, exist_ok=True)
        for item in self.annotations:
            file = osp.join(path, 'gt_' + item + '.txt')
            with open(file, 'w') as f:
                f.write(self.annotations[item])

    def is_empty(self):
        return len(self.annotations) == 0

class _TextSegmentationConverter:
    def __init__(self):
        self.annotations = {}

    def save_annotations(self, item, path):
        annotation = ''
        colormap = [(255, 255, 255)]
        anns = [a for a in item.annotations
            if a.type == AnnotationType.mask]
        if anns:
            is_not_index = len([p for p in anns
                if 'index' not in p.attributes])
            if is_not_index:
                raise Exception("Item %s: mask must have"
                    "index attribute" % item.id)
            anns = sorted(anns, key=lambda a: a.attributes['index'])
            group = anns[0].group
            for ann in anns:
                if ann.group != group or \
                        (ann.group == 0 and anns[0].group != 0):
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
                        colormap.append((int(color[0]), int(color[1]), int(color[2])))
                        annotation += ' '.join(p for p in color)
                    else:
                        raise Exception("Item %s: mask must have "
                            "a three-digit color attribute" % item.id)
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
            save_image(osp.join(path, item.id + '_GT' + IcdarPath.GT_EXT),
                mask, create_dir=True)
        self.annotations[item.id] = annotation

    def write(self, path):
        os.makedirs(path, exist_ok=True)
        for item in self.annotations:
            file = osp.join(path, item + '_GT' + '.txt')
            with open(file, 'w') as f:
                f.write(self.annotations[item])

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
            for item in subset:
                for task, task_conv in task_converters.items():
                    if item.has_image and self._save_images:
                        self._save_image(item, osp.join(
                            self._save_dir, subset_name, IcdarPath.IMAGES_DIR,
                            item.id + IcdarPath.IMAGE_EXT))
                    task_conv.save_annotations(item, osp.join(self._save_dir,
                        IcdarPath.TASK_DIR[task], subset_name))

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
