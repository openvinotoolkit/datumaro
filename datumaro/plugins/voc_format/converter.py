# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict, defaultdict
from enum import Enum, auto
from itertools import chain
import logging as log
import os
import os.path as osp

# Disable B410: import_lxml - the library is used for writing
from lxml import etree as ET  # nosec

from datumaro.components.annotation import (
    AnnotationType, CompiledMask, LabelCategories,
)
from datumaro.components.converter import Converter
from datumaro.components.dataset import ItemStatus
from datumaro.components.extractor import DatasetItem
from datumaro.util import find, str_to_bool
from datumaro.util.annotation_util import make_label_id_mapping
from datumaro.util.image import save_image
from datumaro.util.mask_tools import paint_mask, remap_mask
from datumaro.util.meta_file_util import has_meta_file

from .format import (
    VocInstColormap, VocPath, VocTask, make_voc_categories, make_voc_label_map,
    parse_label_map, parse_meta_file, write_label_map, write_meta_file,
)


def _convert_attr(name, attributes, type_conv, default=None):
    d = object()
    value = attributes.get(name, d)
    if value is d:
        return default

    try:
        return type_conv(value)
    except Exception as e:
        log.warning("Failed to convert attribute '%s'='%s': %s" % \
            (name, value, e))
        return default

def _write_xml_bbox(bbox, parent_elem):
    x, y, w, h = bbox
    bbox_elem = ET.SubElement(parent_elem, 'bndbox')
    ET.SubElement(bbox_elem, 'xmin').text = str(x)
    ET.SubElement(bbox_elem, 'ymin').text = str(y)
    ET.SubElement(bbox_elem, 'xmax').text = str(x + w)
    ET.SubElement(bbox_elem, 'ymax').text = str(y + h)
    return bbox_elem


class LabelmapType(Enum):
    voc = auto()
    source = auto()

class VocConverter(Converter):
    DEFAULT_IMAGE_EXT = VocPath.IMAGE_EXT
    BUILTIN_ATTRS = {'difficult', 'pose', 'truncated', 'occluded' }

    @staticmethod
    def _split_tasks_string(s):
        return [VocTask[i.strip()] for i in s.split(',')]

    @staticmethod
    def _get_labelmap(s):
        if osp.isfile(s):
            return s
        try:
            return LabelmapType[s].name
        except KeyError:
            import argparse
            raise argparse.ArgumentTypeError()

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)

        parser.add_argument('--apply-colormap', type=str_to_bool, default=True,
            help="Use colormap for class and instance masks "
                "(default: %(default)s)")
        parser.add_argument('--label-map', type=cls._get_labelmap, default=None,
            help="Labelmap file path or one of %s" % \
                ', '.join(t.name for t in LabelmapType))
        parser.add_argument('--allow-attributes',
            type=str_to_bool, default=True,
            help="Allow export of attributes (default: %(default)s)")
        parser.add_argument('--keep-empty',
            type=str_to_bool, default=False,
            help="Write subset lists even if they are empty "
                "(default: %(default)s)")
        parser.add_argument('--tasks', type=cls._split_tasks_string,
            help="VOC task filter, comma-separated list of {%s} "
                "(default: all)" % ', '.join(t.name for t in VocTask))

        return parser

    def __init__(self, extractor, save_dir,
            tasks=None, apply_colormap=True, label_map=None,
            allow_attributes=True, keep_empty=False, **kwargs):
        super().__init__(extractor, save_dir, **kwargs)

        assert tasks is None or isinstance(tasks, (VocTask, list, set))
        if tasks is None:
            tasks = set(VocTask)
        elif isinstance(tasks, VocTask):
            tasks = {tasks}
        else:
            tasks = set(t if t in VocTask else VocTask[t] for t in tasks)
        self._tasks = tasks

        self._apply_colormap = apply_colormap
        self._allow_attributes = allow_attributes
        self._keep_empty = keep_empty

        if label_map is None:
            label_map = LabelmapType.source.name
        assert isinstance(label_map, (str, dict)), label_map
        self._load_categories(label_map)

        self._patch = None

    def apply(self):
        self.make_dirs()
        self.save_subsets()
        self.save_label_map()

    def make_dirs(self):
        save_dir = self._save_dir
        subsets_dir = osp.join(save_dir, VocPath.SUBSETS_DIR)
        cls_subsets_dir = osp.join(subsets_dir,
            VocPath.TASK_DIR[VocTask.classification])
        action_subsets_dir = osp.join(subsets_dir,
            VocPath.TASK_DIR[VocTask.action_classification])
        layout_subsets_dir = osp.join(subsets_dir,
            VocPath.TASK_DIR[VocTask.person_layout])
        segm_subsets_dir = osp.join(subsets_dir,
            VocPath.TASK_DIR[VocTask.segmentation])
        ann_dir = osp.join(save_dir, VocPath.ANNOTATIONS_DIR)
        img_dir = osp.join(save_dir, VocPath.IMAGES_DIR)
        segm_dir = osp.join(save_dir, VocPath.SEGMENTATION_DIR)
        inst_dir = osp.join(save_dir, VocPath.INSTANCES_DIR)
        images_dir = osp.join(save_dir, VocPath.IMAGES_DIR)

        os.makedirs(subsets_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(segm_dir, exist_ok=True)
        os.makedirs(inst_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        self._subsets_dir = subsets_dir
        self._cls_subsets_dir = cls_subsets_dir
        self._action_subsets_dir = action_subsets_dir
        self._layout_subsets_dir = layout_subsets_dir
        self._segm_subsets_dir = segm_subsets_dir
        self._ann_dir = ann_dir
        self._img_dir = img_dir
        self._segm_dir = segm_dir
        self._inst_dir = inst_dir
        self._images_dir = images_dir

    def get_label(self, label_id):
        return self._extractor. \
            categories()[AnnotationType.label].items[label_id].name

    def save_subsets(self):
        for subset_name, subset in self._extractor.subsets().items():
            class_lists = OrderedDict()
            clsdet_list = OrderedDict()
            action_list = OrderedDict()
            layout_list = OrderedDict()
            segm_list = OrderedDict()

            for item in subset:
                log.debug("Converting item '%s'", item.id)

                image_filename = self._make_image_filename(item)
                if self._save_images:
                    if item.has_image and item.image.has_data:
                        self._save_image(item,
                            osp.join(self._images_dir, image_filename))
                    else:
                        log.debug("Item '%s' has no image", item.id)

                labels = []
                bboxes = []
                masks = []
                for a in item.annotations:
                    if a.type == AnnotationType.label:
                        labels.append(a)
                    elif a.type == AnnotationType.bbox:
                        bboxes.append(a)
                    elif a.type == AnnotationType.mask:
                        masks.append(a)

                if self._tasks & {VocTask.detection, VocTask.person_layout,
                        VocTask.action_classification}:
                    objects_with_parts, objects_with_actions = \
                        self._write_xml_objects(
                            item, image_filename, bboxes, bool(masks))

                    clsdet_list[item.id] = True

                    if objects_with_parts:
                        layout_list[item.id] = objects_with_parts

                    if objects_with_actions:
                        action_list[item.id] = objects_with_actions

                for label_ann in labels:
                    label = self.get_label(label_ann.label)
                    if not self._is_label(label):
                        continue
                    class_list = class_lists.get(item.id, set())
                    class_list.add(label_ann.label)
                    class_lists[item.id] = class_list

                    clsdet_list[item.id] = True

                if masks and VocTask.segmentation in self._tasks:
                    compiled_mask = CompiledMask.from_instance_masks(masks,
                        instance_labels=[self._label_id_mapping(m.label)
                            for m in masks])

                    self.save_segm(
                        osp.join(self._segm_dir, item.id + VocPath.SEGM_EXT),
                        compiled_mask.class_mask)
                    self.save_segm(
                        osp.join(self._inst_dir, item.id + VocPath.SEGM_EXT),
                        compiled_mask.instance_mask,
                        colormap=VocInstColormap)

                    segm_list[item.id] = True
                elif not masks and self._patch:
                    cls_mask_path = osp.join(self._segm_dir,
                        item.id + VocPath.SEGM_EXT)
                    if osp.isfile(cls_mask_path):
                        os.remove(cls_mask_path)

                    inst_mask_path = osp.join(self._inst_dir,
                        item.id + VocPath.SEGM_EXT)
                    if osp.isfile(inst_mask_path):
                        os.remove(inst_mask_path)

                if len(item.annotations) == 0:
                    clsdet_list[item.id] = None
                    layout_list[item.id] = None
                    action_list[item.id] = None
                    segm_list[item.id] = None

            if self._tasks & {VocTask.classification, VocTask.detection,
                    VocTask.action_classification, VocTask.person_layout}:
                self.save_clsdet_lists(subset_name, clsdet_list)
                if self._tasks & {VocTask.classification}:
                    self.save_class_lists(subset_name, class_lists)
            if self._tasks & {VocTask.action_classification}:
                self.save_action_lists(subset_name, action_list)
            if self._tasks & {VocTask.person_layout}:
                self.save_layout_lists(subset_name, layout_list)
            if self._tasks & {VocTask.segmentation}:
                self.save_segm_lists(subset_name, segm_list)

    def _write_xml_objects(self, item, image_filename, bboxes, has_masks):
        root_elem = ET.Element('annotation')
        if '_' in item.id:
            folder = item.id[ : item.id.find('_')]
        else:
            folder = ''
        ET.SubElement(root_elem, 'folder').text = folder
        ET.SubElement(root_elem, 'filename').text = image_filename

        source_elem = ET.SubElement(root_elem, 'source')
        ET.SubElement(source_elem, 'database').text = 'Unknown'
        ET.SubElement(source_elem, 'annotation').text = 'Unknown'
        ET.SubElement(source_elem, 'image').text = 'Unknown'

        if item.has_image and item.image.has_size:
            h, w = item.image.size
            size_elem = ET.SubElement(root_elem, 'size')
            ET.SubElement(size_elem, 'width').text = str(w)
            ET.SubElement(size_elem, 'height').text = str(h)
            ET.SubElement(size_elem, 'depth').text = ''

        ET.SubElement(root_elem, 'segmented').text = str(int(has_masks))

        objects_with_parts = []
        objects_with_actions = defaultdict(dict)

        main_bboxes = []
        layout_bboxes = []
        for bbox in bboxes:
            label = self.get_label(bbox.label)
            if self._is_part(label):
                layout_bboxes.append(bbox)
            elif self._is_label(label):
                main_bboxes.append(bbox)

        for new_obj_id, obj in enumerate(main_bboxes):
            attr = obj.attributes

            obj_elem = ET.SubElement(root_elem, 'object')

            obj_label = self.get_label(obj.label)
            ET.SubElement(obj_elem, 'name').text = obj_label

            if 'pose' in attr:
                ET.SubElement(obj_elem, 'pose').text = str(attr['pose'])

            ET.SubElement(obj_elem, 'truncated').text = \
                '%d' % _convert_attr('truncated', attr, int, 0)
            ET.SubElement(obj_elem, 'occluded').text = \
                '%d' % _convert_attr('occluded', attr, int, 0)
            ET.SubElement(obj_elem, 'difficult').text = \
                '%d' % _convert_attr('difficult', attr, int, 0)

            bbox = obj.get_bbox()
            if bbox is not None:
                _write_xml_bbox(bbox, obj_elem)

            for part_bbox in filter(
                lambda x: obj.group and obj.group == x.group, layout_bboxes,
            ):
                part_elem = ET.SubElement(obj_elem, 'part')
                ET.SubElement(part_elem, 'name').text = \
                    self.get_label(part_bbox.label)
                _write_xml_bbox(part_bbox.get_bbox(), part_elem)

                objects_with_parts.append(new_obj_id)

            label_actions = self._get_actions(obj_label)
            actions_elem = ET.Element('actions')
            for action in label_actions:
                present = 0
                if action in attr:
                    present = _convert_attr(action, attr,
                        lambda v: int(v is True), 0)
                    ET.SubElement(actions_elem, action).text = '%d' % present

                objects_with_actions[new_obj_id][action] = present
            if len(actions_elem) != 0:
                obj_elem.append(actions_elem)

            if self._allow_attributes:
                native_attrs = set(self.BUILTIN_ATTRS)
                native_attrs.update(label_actions)

                attrs_elem = ET.Element('attributes')
                for k, v in attr.items():
                    if k in native_attrs:
                        continue
                    attr_elem = ET.SubElement(attrs_elem, 'attribute')
                    ET.SubElement(attr_elem, 'name').text = str(k)
                    ET.SubElement(attr_elem, 'value').text = str(v)
                if len(attrs_elem):
                    obj_elem.append(attrs_elem)

        if self._tasks & {VocTask.detection, VocTask.person_layout,
                            VocTask.action_classification}:
            ann_path = osp.join(self._ann_dir, item.id + '.xml')
            os.makedirs(osp.dirname(ann_path), exist_ok=True)
            with open(ann_path, 'w', encoding='utf-8') as f:
                f.write(ET.tostring(root_elem,
                    encoding='unicode', pretty_print=True))

        return objects_with_parts, objects_with_actions

    @staticmethod
    def _get_filtered_lines(path, patch, subset, items=None):
        lines = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line_parts = line.split(maxsplit=1)
                if len(line_parts) < 2:
                    line_parts.append('')
                item, text = line_parts
                if not patch or patch.updated_items.get((item, subset)) != \
                        ItemStatus.removed:
                    lines.setdefault(item, []).append(text)
        if items is not None:
            items.update((k, True) for k in lines)
        return lines

    def save_action_lists(self, subset_name, action_list):
        os.makedirs(self._action_subsets_dir, exist_ok=True)

        ann_file = osp.join(self._action_subsets_dir, subset_name + '.txt')
        items = {k: True for k in action_list}
        if self._patch and osp.isfile(ann_file):
            self._get_filtered_lines(ann_file, self._patch, subset_name, items)

        if items or self._keep_empty:
            with open(ann_file, 'w', encoding='utf-8') as f:
                for item in items:
                    f.write('%s\n' % item)
        elif osp.isfile(ann_file):
            os.remove(ann_file)

        if not items and not self._patch and not self._keep_empty:
            return

        def _write_item(f, item, objs, action):
            if not objs:
                return
            for obj_id, obj_actions in objs.items():
                presented = obj_actions[action]
                f.write('%s %s % d\n' % \
                    (item, 1 + obj_id, 1 if presented else -1))

        all_actions = {
            act: osp.join(self._action_subsets_dir,
                '%s_%s.txt' % (act, subset_name))
            for act in chain(*(self._get_actions(l) for l in self._label_map))
        }
        for action, ann_file in all_actions.items():
            if not items and not self._keep_empty:
                if osp.isfile(ann_file):
                    os.remove(ann_file)
                continue

            lines = {}
            if self._patch and osp.isfile(ann_file):
                lines = self._get_filtered_lines(ann_file, None, subset_name)

            with open(ann_file, 'w', encoding='utf-8') as f:
                for item in items:
                    if item in action_list:
                        _write_item(f, item, action_list[item], action)
                    elif item in lines:
                        print(item, *lines[item], file=f)

    def save_class_lists(self, subset_name, class_lists):
        def _write_item(f, item, item_labels):
            if not item_labels:
                return
            item_labels = [self.get_label(l) for l in item_labels]
            presented = label in item_labels
            f.write('%s % d\n' % (item, 1 if presented else -1))

        os.makedirs(self._cls_subsets_dir, exist_ok=True)

        for label in self._label_map:
            ann_file = osp.join(self._cls_subsets_dir,
                '%s_%s.txt' % (label, subset_name))
            items = {k: True for k in class_lists}
            lines = {}
            if self._patch and osp.isfile(ann_file):
                lines = self._get_filtered_lines(ann_file, self._patch,
                    subset_name, items)

            if not items and not self._keep_empty:
                if osp.isfile(ann_file):
                    os.remove(ann_file)
                continue

            with open(ann_file, 'w', encoding='utf-8') as f:
                for item in items:
                    if item in class_lists:
                        _write_item(f, item, class_lists[item])
                    elif item in lines:
                        print(item, *lines[item], file=f)

    def save_clsdet_lists(self, subset_name, clsdet_list):
        os.makedirs(self._cls_subsets_dir, exist_ok=True)

        ann_file = osp.join(self._cls_subsets_dir, subset_name + '.txt')
        items = {k: True for k in clsdet_list}
        if self._patch and osp.isfile(ann_file):
            self._get_filtered_lines(ann_file, self._patch, subset_name, items)

        if items or self._keep_empty:
            with open(ann_file, 'w', encoding='utf-8') as f:
                for item in items:
                    f.write('%s\n' % item)
        elif osp.isfile(ann_file):
            os.remove(ann_file)

    def save_segm_lists(self, subset_name, segm_list):
        os.makedirs(self._segm_subsets_dir, exist_ok=True)

        ann_file = osp.join(self._segm_subsets_dir, subset_name + '.txt')
        items = {k: True for k in segm_list}
        if self._patch and osp.isfile(ann_file):
            self._get_filtered_lines(ann_file, self._patch, subset_name, items)

        if items or self._keep_empty:
            with open(ann_file, 'w', encoding='utf-8') as f:
                for item in items:
                    f.write('%s\n' % item)
        elif osp.isfile(ann_file):
            os.remove(ann_file)

    def save_layout_lists(self, subset_name, layout_list):
        def _write_item(f, item, item_layouts):
            if 1 < len(item.split()):
                item = '\"' + item + '\"'
            if item_layouts:
                for obj_id in item_layouts:
                    f.write('%s % d\n' % (item, 1 + obj_id))
            else:
                f.write('%s\n' % item)

        os.makedirs(self._layout_subsets_dir, exist_ok=True)

        ann_file = osp.join(self._layout_subsets_dir, subset_name + '.txt')
        items = {k: True for k in layout_list}
        lines = {}
        if self._patch and osp.isfile(ann_file):
            self._get_filtered_lines(ann_file, self._patch, subset_name, items)

        if not items and not self._keep_empty:
            if osp.isfile(ann_file):
                os.remove(ann_file)
            return

        with open(ann_file, 'w', encoding='utf-8') as f:
            for item in items:
                if item in layout_list:
                    _write_item(f, item, layout_list[item])
                elif item in lines:
                    print(item, *lines[item], file=f)

    def save_segm(self, path, mask, colormap=None):
        if self._apply_colormap:
            if colormap is None:
                colormap = self._categories[AnnotationType.mask].colormap
            mask = paint_mask(mask, colormap)
        save_image(path, mask, create_dir=True)

    def save_label_map(self):
        if self._save_dataset_meta:
            write_meta_file(self._save_dir, self._label_map)
        else:
            path = osp.join(self._save_dir, VocPath.LABELMAP_FILE)
            write_label_map(path, self._label_map)

    def _load_categories(self, label_map_source):
        if label_map_source == LabelmapType.voc.name:
            # use the default VOC colormap
            label_map = make_voc_label_map()

        elif label_map_source == LabelmapType.source.name and \
                AnnotationType.mask not in self._extractor.categories():
            # generate colormap for input labels
            labels = self._extractor.categories() \
                .get(AnnotationType.label, LabelCategories())
            label_map = OrderedDict((item.name, [None, [], []])
                for item in labels.items)

        elif label_map_source == LabelmapType.source.name and \
                AnnotationType.mask in self._extractor.categories():
            # use source colormap
            labels = self._extractor.categories()[AnnotationType.label]
            colors = self._extractor.categories()[AnnotationType.mask]
            label_map = OrderedDict()
            for idx, item in enumerate(labels.items):
                color = colors.colormap.get(idx)
                if color is not None:
                    label_map[item.name] = [color, [], []]

        elif isinstance(label_map_source, dict):
            label_map = OrderedDict(
                sorted(label_map_source.items(), key=lambda e: e[0]))

        elif isinstance(label_map_source, str) and osp.isfile(label_map_source):
            if has_meta_file(label_map_source):
                label_map = parse_meta_file(label_map_source)
            else:
                label_map = parse_label_map(label_map_source)

        else:
            raise Exception("Wrong labelmap specified: '%s', "
                "expected one of %s or a file path" % \
                (label_map_source, ', '.join(t.name for t in LabelmapType)))

        bg_label = find(label_map.items(), lambda x: x[1][0] == (0, 0, 0))
        if bg_label is None:
            bg_label = 'background'
            if bg_label not in label_map:
                has_colors = any(v[0] is not None for v in label_map.values())
                color = (0, 0, 0) if has_colors else None
                label_map[bg_label] = [color, [], []]
            label_map.move_to_end(bg_label, last=False)

        self._categories = make_voc_categories(label_map)

        # Update colors with assigned values
        colormap = self._categories[AnnotationType.mask].colormap
        for label_id, color in colormap.items():
            label_desc = label_map[
                self._categories[AnnotationType.label].items[label_id].name]
            label_desc[0] = color

        self._label_map = label_map
        self._label_id_mapping = self._make_label_id_map()

    def _is_label(self, s):
        return self._label_map.get(s) is not None

    def _is_part(self, s):
        for label_desc in self._label_map.values():
            if s in label_desc[1]:
                return True
        return False

    def _is_action(self, label, s):
        return s in self._get_actions(label)

    def _get_actions(self, label):
        label_desc = self._label_map.get(label)
        if not label_desc:
            return []
        return label_desc[2]

    def _make_label_id_map(self):
        map_id, id_mapping, src_labels, dst_labels = make_label_id_mapping(
            self._extractor.categories().get(AnnotationType.label),
            self._categories[AnnotationType.label])

        void_labels = [src_label for src_id, src_label in src_labels.items()
            if src_label not in dst_labels]
        if void_labels:
            log.warning("The following labels are remapped to background: %s" %
                ', '.join(void_labels))
        log.debug("Saving segmentations with the following label mapping: \n%s" %
            '\n'.join(["#%s '%s' -> #%s '%s'" %
                (
                    src_id, src_label, id_mapping[src_id],
                    self._categories[AnnotationType.label] \
                        .items[id_mapping[src_id]].name
                )
                for src_id, src_label in src_labels.items()
            ])
        )

        return map_id

    def _remap_mask(self, mask):
        return remap_mask(mask, self._label_id_mapping)

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        conv = cls(patch.as_dataset(dataset), save_dir=save_dir, **kwargs)
        conv._patch = patch
        conv.apply()

        for filename in os.listdir(conv._cls_subsets_dir):
            if '_' not in filename or not filename.endswith('.txt'):
                continue

            label, subset = osp.splitext(filename)[0].split('_', maxsplit=1)
            if label not in conv._label_map or subset not in dataset.subsets():
                os.remove(osp.join(conv._cls_subsets_dir, filename))

        # Find images that need to be removed
        # images from different subsets are stored in the common directory
        # Avoid situations like:
        # (a, test): added
        # (a, train): removed
        # where the second line removes images from the first.
        ids_to_remove = {}
        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or not item.has_image):
                ids_to_remove[item_id] = (item, False)
            else:
                ids_to_remove.setdefault(item_id, (item, True))

        for item, to_remove in ids_to_remove.values():
            if not to_remove:
                continue

            if conv._tasks & {VocTask.detection,
                    VocTask.action_classification, VocTask.person_layout}:
                ann_path = osp.join(conv._ann_dir, item.id + '.xml')
                if osp.isfile(ann_path):
                    os.remove(ann_path)

            image_path = osp.join(conv._images_dir,
                conv._make_image_filename(item))
            if osp.isfile(image_path):
                os.unlink(image_path)

            if not [a for a in item.annotations
                    if a.type is AnnotationType.mask]:
                path = osp.join(save_dir, VocPath.SEGMENTATION_DIR,
                    item.id + VocPath.SEGM_EXT)
                if osp.isfile(path):
                    os.unlink(path)

                path = osp.join(save_dir, VocPath.INSTANCES_DIR,
                    item.id + VocPath.SEGM_EXT)
                if osp.isfile(path):
                    os.unlink(path)

class VocClassificationConverter(VocConverter):
    def __init__(self, *args, **kwargs):
        kwargs['tasks'] = VocTask.classification
        super().__init__(*args, **kwargs)

class VocDetectionConverter(VocConverter):
    def __init__(self, *args, **kwargs):
        kwargs['tasks'] = VocTask.detection
        super().__init__(*args, **kwargs)

class VocLayoutConverter(VocConverter):
    def __init__(self, *args, **kwargs):
        kwargs['tasks'] = VocTask.person_layout
        super().__init__(*args, **kwargs)

class VocActionConverter(VocConverter):
    def __init__(self, *args, **kwargs):
        kwargs['tasks'] = VocTask.action_classification
        super().__init__(*args, **kwargs)

class VocSegmentationConverter(VocConverter):
    def __init__(self, *args, **kwargs):
        kwargs['tasks'] = VocTask.segmentation
        super().__init__(*args, **kwargs)
