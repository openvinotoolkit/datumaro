# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import csv
import os.path as osp

import numpy as np
from datumaro.components.converter import Converter
from datumaro.components.extractor import (AnnotationType, DatasetItem,
    Importer, Label, LabelCategories, SourceExtractor)

class MnistCsvPath:
    IMAGE_SIZE = 28

class MnistCsvExtractor(SourceExtractor):
    def __init__(self, path, subset=None):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        if not subset:
            file_name = osp.splitext(osp.basename(path))[0]
            subset = file_name.rsplit('_', maxsplit=1)[-1]

        super().__init__(subset=subset)
        self._dataset_dir = osp.dirname(path)

        self._categories = self._load_categories()

        self._items = list(self._load_items(path).values())

    def _load_categories(self):
        label_cat = LabelCategories()

        labels_file = osp.join(self._dataset_dir,
            'labels_%s.txt' % self._subset)
        if osp.isfile(labels_file):
            with open(labels_file, encoding='utf-8') as f:
                for label in f:
                    label_cat.add(label.strip())
        else:
            for i in range(10):
                label_cat.add(str(i))

        return { AnnotationType.label: label_cat }

    def _load_items(self, path):
        items = {}
        with open(path, 'r', encoding='utf-8') as f:
            annotation_table = f.readlines()
        
        metafile = osp.join(self._dataset_dir, 'meta_%s.csv' % self._subset)
        meta = []
        if osp.isfile(metafile):
            with open(metafile, 'r', encoding='utf-8') as f:
                meta = f.readlines()
        
        for i, data in enumerate(annotation_table):
            data = data.split(',')
            item_anno = []
            try:
                label = int(data[0])
            except ValueError:
                label = None
            if label != None:
                item_anno.append(Label(label))
            
            if 0 < len(meta):
                meta[i] = meta[i].strip().split(',')
            
            image = None
            if 1 < len(data):
                if 0 < len(meta) and 1 < len(meta[i]):
                    image = np.array([int(pix) for pix in data[1:]],
                        dtype='uint8').reshape(int(meta[i][-2]), int(meta[i][-1]))
                else:
                    image = np.array([int(pix) for pix in data[1:]],
                        dtype='uint8').reshape(28, 28)

            if 0 < len(meta) and len(meta[i]) in [1, 3]:
                i = meta[i][0]

            items[i] = DatasetItem(id=i, subset=self._subset,
                image=image, annotations=item_anno)
        return items

class MnistCsvImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.csv', 'mnist_csv',
            file_filter=lambda p: not osp.basename(p).startswith('meta'))

class MnistCsvConverter(Converter):
    DEFAULT_IMAGE_EXT = '.png'

    def apply(self):
        for subset_name, subset in self._extractor.subsets().items():
            data = []
            item_ids = {}
            image_sizes = {}
            for item in subset:
                anns = [a.label for a in item.annotations
                    if a.type == AnnotationType.label]
                label = None
                if anns:
                    label = anns[0]

                if item.has_image and self._save_images:
                    image = item.image
                    if not image.has_data:
                        data.append([label, None])
                    else:
                        if image.data.shape[0] != MnistCsvPath.IMAGE_SIZE or \
                                image.data.shape[1] != MnistCsvPath.IMAGE_SIZE:
                            image_sizes[len(data)] = [image.data.shape[0], image.data.shape[1]]
                        image = image.data.reshape(-1).astype(np.uint8).tolist()
                        image.insert(0, label)
                        data.append(image)
                else:
                    data.append([label])
                
                if item.id != str(len(data) - 1):
                    item_ids[len(data) - 1] = item.id
                
            anno_file = osp.join(self._save_dir, 'mnist_%s.csv' % subset_name)
            with open(anno_file, 'w', encoding='utf-8') as f:
                for anno in data:
                    f.write(','.join([str(pix) for pix in anno]) + "\n")
                
            if len(item_ids) or len(image_sizes):
                meta = []
                if len(item_ids) and len(image_sizes):
                    size = [MnistCsvPath.IMAGE_SIZE, MnistCsvPath.IMAGE_SIZE]
                    for i in range(len(data)):
                        w, h = image_sizes.get(i, size)
                        meta.append([item_ids.get(i, i), w, h])

                elif len(item_ids):
                    for i in range(len(data)):
                        meta.append([item_ids.get(i, i)])

                elif len(image_sizes):
                    size = [MnistCsvPath.IMAGE_SIZE, MnistCsvPath.IMAGE_SIZE]
                    for i in range(len(data)):
                        meta.append(image_sizes.get(i, size))

                metafile = osp.join(self._save_dir, 'meta_%s.csv' % subset_name)
                with open(metafile, 'w', encoding='utf-8') as f:
                    for anno in meta:
                        f.write(','.join([str(p) for p in anno]) + "\n")
        
        labels_file = osp.join(self._save_dir, 'labels_%s.txt' % subset_name)
        with open(labels_file, 'w', encoding='utf-8') as f:
            f.writelines(l.name + '\n'
                for l in self._extractor.categories().get(
                    AnnotationType.label, LabelCategories())
            )
