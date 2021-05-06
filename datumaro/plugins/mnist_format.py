# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import gzip
import os.path as osp

import numpy as np
from datumaro.components.converter import Converter
from datumaro.components.extractor import (AnnotationType, DatasetItem,
    Importer, Label, LabelCategories, SourceExtractor)


class MnistPath:
    TEST_LABELS_FILE = 't10k-labels-idx1-ubyte.gz'
    TEST_IMAGES_FILE = 't10k-images-idx3-ubyte.gz'
    LABELS_FILE = '-labels-idx1-ubyte.gz'
    IMAGES_FILE = '-images-idx3-ubyte.gz'
    IMAGE_SIZE = 28
    NONE_LABEL = 255

class MnistExtractor(SourceExtractor):
    def __init__(self, path, subset=None):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        if not subset:
            file_name = osp.splitext(osp.basename(path))[0]
            if file_name.startswith('t10k'):
                subset = 'test'
            else:
                subset = file_name.split('-', maxsplit=1)[0]

        super().__init__(subset=subset)
        self._dataset_dir = osp.dirname(path)

        self._categories = self._load_categories()

        self._items = list(self._load_items(path).values())

    def _load_categories(self):
        label_cat = LabelCategories()

        labels_file = osp.join(self._dataset_dir,
            'labels-%s.txt' % self._subset)
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
        with gzip.open(path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        meta = []
        metafile = osp.join(self._dataset_dir, self._subset + '-meta.gz')
        if osp.isfile(metafile):
            with gzip.open(metafile, 'rb') as f:
                meta = np.frombuffer(f.read(), dtype='<U32')
            meta = meta.reshape(len(labels), int(len(meta) / len(labels)))

        images = None
        images_file = osp.join(self._dataset_dir,
            osp.basename(path).replace('labels-idx1', 'images-idx3'))
        if osp.isfile(images_file):
            with gzip.open(images_file, 'rb') as imgpath:
                images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
                if len(meta) == 0 or len(meta[0]) < 2:
                    images = images.reshape(len(labels), MnistPath.IMAGE_SIZE, MnistPath.IMAGE_SIZE)

        pix_num = 0
        for i, annotation in enumerate(labels):
            annotations = []
            label = annotation
            if label != MnistPath.NONE_LABEL:
                annotations.append(Label(label))

            image = None
            if images is not None:
                if 0 < len(meta) and 1 < len(meta[i]):
                    image = images[pix_num : pix_num + int(meta[i][-2]) * int(meta[i][-1])].reshape(int(meta[i][-2]), int(meta[i][-1]))
                    pix_num += int(meta[i][-2]) * int(meta[i][-1])
                else:
                    image = images[i].reshape(MnistPath.IMAGE_SIZE, MnistPath.IMAGE_SIZE)

            if 0 < len(meta) and (len(meta[i]) == 1 or len(meta[i]) == 3):
                i = meta[i][0]

            items[i] = DatasetItem(id=i, subset=self._subset,
                image=image, annotations=annotations)
        return items


class MnistImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.gz', 'mnist',
            file_filter=lambda p: osp.basename(p).split('-')[1] == 'labels')

class MnistConverter(Converter):
    DEFAULT_IMAGE_EXT = '.png'

    def apply(self):
        label_categories = self._extractor.categories()[AnnotationType.label]

        for subset_name, subset in self._extractor.subsets().items():
            labels = []
            images = np.array([])
            item_ids = {}
            image_sizes = {}
            for item in subset:
                anns = [a.label for a in item.annotations
                    if a.type == AnnotationType.label]
                label = 255
                if anns:
                    label = anns[0]
                labels.append(label)

                if item.id != str(len(labels) - 1):
                    item_ids[len(labels) - 1] = item.id

                if item.has_image and self._save_images:
                    image = item.image
                    if not image.has_data:
                        image_sizes[len(images) - 1] = [0, 0]
                    else:
                        image = image.data
                        if image.shape[0] != MnistPath.IMAGE_SIZE or \
                                image.shape[1] != MnistPath.IMAGE_SIZE:
                            image_sizes[len(labels) - 1] = [image.shape[0], image.shape[1]]
                        images = np.append(images, image.reshape(-1).astype(np.uint8))

            if subset_name == 'test':
                labels_file = osp.join(self._save_dir,
                    MnistPath().TEST_LABELS_FILE)
            else:
                labels_file = osp.join(self._save_dir,
                    subset_name + MnistPath.LABELS_FILE)
            with gzip.open(labels_file, 'wb') as f:
                f.write(np.array([0x0801, len(labels)], dtype='>i4').tobytes())
                f.write(np.array(labels, dtype='uint8').tobytes())

            if 0 < len(images):
                if subset_name == 'test':
                    images_file = osp.join(self._save_dir,
                        MnistPath.TEST_IMAGES_FILE)
                else:
                    images_file = osp.join(self._save_dir,
                        subset_name + MnistPath.IMAGES_FILE)
                with gzip.open(images_file, 'wb') as f:
                    f.write(np.array([0x0803, len(images), MnistPath.IMAGE_SIZE,
                        MnistPath.IMAGE_SIZE], dtype='>i4').tobytes())
                    f.write(np.array(images, dtype='uint8').tobytes())
            
            if len(item_ids) or len(image_sizes):
                meta = []
                if len(item_ids) and len(image_sizes):
                    size = [MnistPath.IMAGE_SIZE, MnistPath.IMAGE_SIZE]
                    for i in range(len(labels)):
                        w, h = image_sizes.get(i, size)
                        meta.append([item_ids.get(i, i), w, h])

                elif len(item_ids):
                    for i in range(len(labels)):
                        meta.append([item_ids.get(i, i)])

                elif len(image_sizes):
                    size = [MnistPath.IMAGE_SIZE, MnistPath.IMAGE_SIZE]
                    for i in range(len(labels)):
                        meta.append(image_sizes.get(i, size))

                metafile = osp.join(self._save_dir, subset_name + '-meta.gz')
                with gzip.open(metafile, 'wb') as f:
                    f.write(np.array(meta, dtype='<U32').tobytes())

        labels_file = osp.join(self._save_dir, 'labels-%s.txt' % subset_name)
        with open(labels_file, 'w', encoding='utf-8') as f:
            f.writelines(l.name + '\n'
                for l in self._extractor.categories().get(
                    AnnotationType.label, LabelCategories())
            )
