# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import pickle

import numpy as np
from datumaro.components.converter import Converter
from datumaro.components.extractor import (AnnotationType, DatasetItem,
    Importer, Label, LabelCategories, SourceExtractor)
from datumaro.util.image import find_images


class CifarPath:
    BATCHES_META = 'batches.meta'
    TRAIN_ANNOTATION_FILE = 'data_batch_'
    IMAGES_DIR = 'images'
    IMAGE_SIZE = 32

CifarLabel = ['airplane', 'automobile', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class CifarExtractor(SourceExtractor):
    def __init__(self, path, subset=None):
        if not subset:
            file_name = osp.splitext(osp.basename(path))[0]
            if file_name.startswith(CifarPath.TRAIN_ANNOTATION_FILE):
                subset = 'train_%s' % file_name.split('_')[-1]
            else:
                subset = file_name.split('_')[0]

        super().__init__(subset=subset)

        batches_meta_file = osp.join(osp.dirname(path), CifarPath.BATCHES_META)
        self._categories = self._load_categories(batches_meta_file)

        self._items = list(self._load_items(path).values())

    def _load_categories(self, path):
        label_cat = LabelCategories()

        if osp.isfile(path):
            # num_cases_per_batch: 1000
            # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer',
            #               'dog', 'frog', 'horse', 'ship', 'truck']
            # num_vis: 3072
            with open(path, 'rb') as fo:
                data = pickle.load(fo, encoding='latin1')
            for label in data['label_names']:
                label_cat.add(label)
        else:
            for label in CifarLabel:
                label_cat.add(label)

        return { AnnotationType.label: label_cat }

    def _load_items(self, path):
        items = {}

        image_dir = osp.join(osp.dirname(path), CifarPath.IMAGES_DIR)
        if osp.isdir(image_dir):
            images = { osp.splitext(osp.relpath(p, image_dir))[0]: p
                for p in find_images(image_dir, recursive=True) }
        else:
            images = {}

        # 'batch_label': 'training batch 1 of 5'
        # 'data': ndarray
        # 'filenames': list
        # 'labels': list
        with open(path, 'rb') as fo:
            annotation_dict = pickle.load(fo, encoding='latin1')

        labels = annotation_dict.get('labels')
        filenames = annotation_dict.get('filenames')
        images_data = annotation_dict.get('data')
        size = annotation_dict.get('image_sizes')

        for i, (filename, label, image_data) in \
                enumerate(zip(filenames, labels, images_data)):
            item_id = osp.splitext(filename)[0]
            annotations = []
            if label != None:
                annotations.append(Label(label))

            image = images.get(item_id)
            if not image:
                image = image_data
                if size and image.any():
                    image = image.reshape(size[i][0],
                        size[i][1], 3).astype(np.uint8)
                elif image.any():
                    image = image.reshape(CifarPath.IMAGE_SIZE,
                        CifarPath.IMAGE_SIZE, 3).astype(np.uint8)

            items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                image=image, annotations=annotations)

        return items


class CifarImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '', 'cifar',
            file_filter=lambda p: osp.basename(p) != CifarPath.BATCHES_META and \
                osp.basename(p) != CifarPath.IMAGES_DIR)


class CifarConverter(Converter):
    DEFAULT_IMAGE_EXT = '.png'

    def apply(self):
        label_categories = self._extractor.categories()[AnnotationType.label]

        label_names = []
        for label in label_categories:
            label_names.append(label.name)
        labels_dict = { 'label_names': label_names }
        batches_meta_file = osp.join(self._save_dir, CifarPath.BATCHES_META)
        with open(batches_meta_file, 'wb') as handle:
            pickle.dump(labels_dict, handle)

        for subset_name, subset in self._extractor.subsets().items():
            filenames = []
            labels = []
            data = []
            image_sizes = {}
            for item in subset:
                filenames.append(item.id + self._find_image_ext(item))

                if item.has_image and self._save_images:
                    self._save_image(item, osp.join(self._save_dir, CifarPath.IMAGES_DIR,
                        self._make_image_filename(item)))

                anns = [a.label for a in item.annotations
                    if a.type == AnnotationType.label]
                label = None
                if anns:
                    label = anns[0]
                labels.append(label)

                image = item.image.data
                if not image.any():
                    data.append(image)
                else:
                    data.append(image.reshape(image.shape[0] * image.shape[1] * \
                        image.shape[2]).astype(np.uint8))
                    if image.shape[0] != CifarPath.IMAGE_SIZE or \
                            image.shape[1] != CifarPath.IMAGE_SIZE:
                        image_sizes[len(data) - 1] = (image.shape[0], image.shape[1])

            annotation_dict = {}
            annotation_dict['filenames'] = filenames
            annotation_dict['labels'] = labels
            annotation_dict['data'] = np.array(data)
            if len(image_sizes):
                size = (CifarPath.IMAGE_SIZE, CifarPath.IMAGE_SIZE)
                # 'image_sizes' isn't included in the standart format,
                # needed for different image sizes
                annotation_dict['image_sizes'] = [image_sizes.get(p, size)
                                                    for p in range(len(data))]

            filename = '%s_batch' % subset_name
            batch_label = None
            if subset_name.startswith('train_'):
                num = subset_name.split('_')[1]
                filename = CifarPath.TRAIN_ANNOTATION_FILE + num
                batch_label = 'training batch %s of 5' % num
            if subset_name == 'test':
                batch_label = 'testing batch 1 of 1'
            if batch_label:
                annotation_dict['batch_label'] = batch_label

            annotation_file = osp.join(self._save_dir, filename)
            with open(annotation_file, 'wb') as handle:
                pickle.dump(annotation_dict, handle)
