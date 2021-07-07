# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import pickle  # nosec - disable B403:import_pickle check

import numpy as np

from datumaro.components.converter import Converter
from datumaro.components.extractor import (
    AnnotationType, DatasetItem, Importer, Label, LabelCategories,
    SourceExtractor,
)
from datumaro.util import cast


class CifarPath:
    BATCHES_META = 'batches.meta'
    TRAIN_ANNOTATION_FILE = 'data_batch_'
    IMAGES_DIR = 'images'
    IMAGE_SIZE = 32

CifarLabel = ['airplane', 'automobile', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Support for Python version CIFAR-10/100

class CifarExtractor(SourceExtractor):
    def __init__(self, path, subset=None):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        if not subset:
            file_name = osp.splitext(osp.basename(path))[0]
            if file_name.startswith(CifarPath.TRAIN_ANNOTATION_FILE):
                subset = 'train_%s' % file_name.split('_')[-1]
            else:
                subset = file_name.rsplit('_', maxsplit=1)[0]

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
            with open(path, 'rb') as labels_file:
                data = pickle.load(labels_file) # nosec - disable B301:pickle check
            for label in data['label_names']:
                label_cat.add(label)
        else:
            for label in CifarLabel:
                label_cat.add(label)

        return { AnnotationType.label: label_cat }

    def _load_items(self, path):
        items = {}

        # 'batch_label': 'training batch 1 of 5'
        # 'data': ndarray
        # 'filenames': list
        # 'labels': list
        with open(path, 'rb') as anno_file:
            annotation_dict = pickle.load(anno_file, encoding='latin1') # nosec - disable B301:pickle check

        labels = annotation_dict.get('labels', [])
        filenames = annotation_dict.get('filenames', [])
        images_data = annotation_dict.get('data')
        size = annotation_dict.get('image_sizes')

        if len(labels) != len(filenames):
            raise Exception("The sizes of the arrays 'filenames', " \
                "'labels' don't match.")

        if 0 < len(images_data) and len(images_data) != len(filenames):
            raise Exception("The sizes of the arrays 'data', " \
                "'filenames', 'labels' don't match.")

        for i, (filename, label) in enumerate(zip(filenames, labels)):
            item_id = osp.splitext(filename)[0]
            annotations = []
            if label != None:
                annotations.append(Label(label))

            image = None
            if 0 < len(images_data):
                image = images_data[i]
                if size is not None and image is not None:
                    image = image.reshape(3, size[i][0],
                        size[i][1]).astype(np.uint8)
                    image = np.transpose(image, (1, 2, 0))
                elif image is not None:
                    image = image.reshape(3, CifarPath.IMAGE_SIZE,
                        CifarPath.IMAGE_SIZE).astype(np.uint8)
                    image = np.transpose(image, (1, 2, 0))

            items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                image=image, annotations=annotations)

        return items


class CifarImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '', 'cifar',
            file_filter=lambda p: osp.basename(p) not in
                {CifarPath.BATCHES_META, CifarPath.IMAGES_DIR})


class CifarConverter(Converter):
    DEFAULT_IMAGE_EXT = '.png'

    def apply(self):
        os.makedirs(self._save_dir, exist_ok=True)

        label_categories = self._extractor.categories()[AnnotationType.label]
        label_names = []
        for label in label_categories:
            label_names.append(label.name)
        labels_dict = { 'label_names': label_names }
        batches_meta_file = osp.join(self._save_dir, CifarPath.BATCHES_META)
        with open(batches_meta_file, 'wb') as labels_file:
            pickle.dump(labels_dict, labels_file)

        for subset_name, subset in self._extractor.subsets().items():
            filenames = []
            labels = []
            data = []
            image_sizes = {}
            for item in subset:
                filenames.append(item.id + self._find_image_ext(item))

                anns = [a.label for a in item.annotations
                    if a.type == AnnotationType.label]
                label = None
                if anns:
                    label = anns[0]
                labels.append(label)

                if item.has_image and self._save_images:
                    image = item.image
                    if not image.has_data:
                        data.append(None)
                    else:
                        image = image.data
                        data.append(np.transpose(image,
                            (2, 0, 1)).reshape(-1).astype(np.uint8))
                        if image.shape[0] != CifarPath.IMAGE_SIZE or \
                                image.shape[1] != CifarPath.IMAGE_SIZE:
                            image_sizes[len(data) - 1] = (image.shape[0], image.shape[1])

            annotation_dict = {}
            annotation_dict['filenames'] = filenames
            annotation_dict['labels'] = labels
            annotation_dict['data'] = np.array(data, dtype=object)
            if len(image_sizes):
                size = (CifarPath.IMAGE_SIZE, CifarPath.IMAGE_SIZE)
                # 'image_sizes' isn't included in the standard format,
                # needed for different image sizes
                annotation_dict['image_sizes'] = [image_sizes.get(p, size)
                    for p in range(len(data))]

            filename = '%s_batch' % subset_name
            batch_label = None
            if subset_name.startswith('train_') and \
                    cast(subset_name.split('_')[1], int) is not None:
                num = subset_name.split('_')[1]
                filename = CifarPath.TRAIN_ANNOTATION_FILE + num
                batch_label = 'training batch %s of 5' % (num, )
            if subset_name == 'test':
                batch_label = 'testing batch 1 of 1'
            if batch_label:
                annotation_dict['batch_label'] = batch_label

            annotation_file = osp.join(self._save_dir, filename)
            with open(annotation_file, 'wb') as labels_file:
                pickle.dump(annotation_dict, labels_file)
