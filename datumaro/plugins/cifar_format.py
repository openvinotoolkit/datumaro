# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
import os
import os.path as osp
import pickle  # nosec - disable B403:import_pickle check - fixed

import numpy as np
import numpy.core.multiarray

from datumaro.components.annotation import (
    AnnotationType, Label, LabelCategories,
)
from datumaro.components.converter import Converter
from datumaro.components.dataset import ItemStatus
from datumaro.components.errors import MediaTypeError
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.components.media import Image
from datumaro.util import cast
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy.core.multiarray" and \
                name in PickleLoader.safe_numpy:
            return getattr(numpy.core.multiarray, name)
        elif module == 'numpy' and name in PickleLoader.safe_numpy:
            return getattr(numpy, name)
        raise pickle.UnpicklingError("Global '%s.%s' is forbidden"
            % (module, name))

class PickleLoader():
    safe_numpy = {
        'dtype',
        'ndarray',
        '_reconstruct',
    }

    def restricted_load(s):
        return RestrictedUnpickler(s, encoding='latin1').load()

class CifarPath:
    META_10_FILE = 'batches.meta'
    META_100_FILE = 'meta'
    TRAIN_FILE_PREFIX = 'data_batch_'
    USELESS_FILE = 'file.txt~'
    IMAGE_SIZE = 32

Cifar10Label = ['airplane', 'automobile', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Support for Python version CIFAR-10/100

class CifarExtractor(SourceExtractor):
    def __init__(self, path, subset=None):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]

        super().__init__(subset=subset)

        self._categories = self._load_categories(osp.dirname(path))
        self._items = list(self._load_items(path).values())

    def _load_categories(self, path):
        if has_meta_file(path):
            return { AnnotationType.label: LabelCategories.
                from_iterable(parse_meta_file(path).keys()) }

        label_cat = LabelCategories()

        meta_file = osp.join(path, CifarPath.META_10_FILE)
        if not osp.isfile(meta_file):
            meta_file = osp.join(path, CifarPath.META_100_FILE)
        if osp.isfile(meta_file):
            # CIFAR-10:
            # num_cases_per_batch: 1000
            # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer',
            #               'dog', 'frog', 'horse', 'ship', 'truck']
            # num_vis: 3072
            # CIFAR-100:
            # fine_label_names: ['apple', 'aquarium_fish', 'baby', ...]
            # coarse_label_names: ['aquatic_mammals', 'fish', 'flowers', ...]
            with open(meta_file, 'rb') as labels_file:
                data = PickleLoader.restricted_load(labels_file)
            labels = data.get('label_names')
            if labels is not None:
                for label in labels:
                    label_cat.add(label)
            else:
                labels = data.get('fine_label_names')
                self._coarse_labels = data.get('coarse_label_names', [])
                if labels is not None:
                    for label in labels:
                        label_cat.add(label)
        else:
            for label in Cifar10Label:
                label_cat.add(label)

        return { AnnotationType.label: label_cat }

    def _load_items(self, path):
        items = {}
        label_cat = self._categories[AnnotationType.label]

        # 'batch_label': 'training batch 1 of 5'
        # 'data': ndarray
        # 'filenames': list
        # CIFAR-10: 'labels': list
        # CIFAR-100: 'fine_labels': list
        #            'coarse_labels': list

        with open(path, 'rb') as anno_file:
            annotation_dict = PickleLoader.restricted_load(anno_file)

        labels = annotation_dict.get('labels', [])
        coarse_labels = annotation_dict.get('coarse_labels', [])
        if len(labels) == 0:
            labels = annotation_dict.get('fine_labels', [])
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
            if label is not None:
                annotations.append(Label(label))
                if 0 < len(coarse_labels) and coarse_labels[i] is not None and \
                        label_cat[label].parent == '':
                    label_cat[label].parent = \
                        self._coarse_labels[coarse_labels[i]]

            image = None
            if 0 < len(images_data):
                image = images_data[i]
                if size is not None and image is not None:
                    image = image.astype(np.uint8) \
                        .reshape(3, size[i][0], size[i][1])
                    image = np.transpose(image, (1, 2, 0))
                elif image is not None:
                    image = image.astype(np.uint8) \
                        .reshape(3, CifarPath.IMAGE_SIZE, CifarPath.IMAGE_SIZE)
                    image = np.transpose(image, (1, 2, 0))

            if image is not None:
                image = Image(data=image)

            items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                media=image, annotations=annotations)

        return items


class CifarImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '', 'cifar',
            file_filter=lambda p: \
                # subset files have no extension in the format
                not osp.splitext(osp.basename(p))[1] and \
                osp.basename(p) not in {
                    CifarPath.META_10_FILE, CifarPath.META_100_FILE,
                    CifarPath.USELESS_FILE
                }
        )


class CifarConverter(Converter):
    DEFAULT_IMAGE_EXT = '.png'

    def apply(self):
        if not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(self._save_dir, exist_ok=True)

        if self._save_dataset_meta:
            self._save_meta_file(self._save_dir)

        label_categories = self._extractor.categories()[AnnotationType.label]
        label_names = []
        coarse_label_names = []
        for label in label_categories:
            label_names.append(label.name)
            if label.parent and (label.parent not in coarse_label_names):
                coarse_label_names.append(label.parent)
        coarse_label_names.sort()

        if coarse_label_names:
            labels_dict = { 'fine_label_names': label_names,
                            'coarse_label_names': coarse_label_names }
            coarse_label_names = OrderedDict((name, i)
                for i, name in enumerate(coarse_label_names))
            meta_file = osp.join(self._save_dir, CifarPath.META_100_FILE)
        else:
            labels_dict = { 'label_names': label_names }
            meta_file = osp.join(self._save_dir, CifarPath.META_10_FILE)
        with open(meta_file, 'wb') as f:
            pickle.dump(labels_dict, f)

        for subset_name, subset in self._extractor.subsets().items():
            filenames = []
            labels = []
            coarse_labels = []
            data = []
            image_sizes = {}
            for item in subset:
                filenames.append(self._make_image_filename(item))

                anns = [a for a in item.annotations
                    if a.type == AnnotationType.label]
                if anns:
                    labels.append(anns[0].label)
                    if coarse_label_names:
                        superclass = label_categories[anns[0].label].parent
                        coarse_labels.append(coarse_label_names[superclass])
                else:
                    labels.append(None)
                    coarse_labels.append(None)

                if self._save_media and item.media:
                    image = item.media
                    if not image.has_data:
                        data.append(None)
                    else:
                        image = image.data
                        data.append(np.transpose(image, (2, 0, 1)) \
                            .reshape(-1).astype(np.uint8))
                        if image.shape[0] != CifarPath.IMAGE_SIZE or \
                                image.shape[1] != CifarPath.IMAGE_SIZE:
                            image_sizes[len(data) - 1] = \
                                (image.shape[0], image.shape[1])

            annotation_dict = {}

            annotation_dict['filenames'] = filenames
            if labels and (len(labels) == len(coarse_labels)):
                annotation_dict['fine_labels'] = labels
                annotation_dict['coarse_labels'] = coarse_labels
            else:
                annotation_dict['labels'] = labels
            annotation_dict['data'] = np.array(data, dtype=object)

            if image_sizes:
                size = (CifarPath.IMAGE_SIZE, CifarPath.IMAGE_SIZE)
                # 'image_sizes' isn't included in the standard format,
                # needed for different image sizes
                annotation_dict['image_sizes'] = [image_sizes.get(p, size)
                    for p in range(len(data))]

            batch_label = None
            if subset_name.startswith(CifarPath.TRAIN_FILE_PREFIX):
                num = subset_name[len(CifarPath.TRAIN_FILE_PREFIX):]
                if cast(num, int) is not None:
                    batch_label = 'training batch %s of 5' % num
            elif subset_name == 'test':
                batch_label = 'testing batch 1 of 1'

            if batch_label:
                annotation_dict['batch_label'] = batch_label

            annotation_file = osp.join(self._save_dir, subset_name)

            if self._patch and subset_name in self._patch.updated_subsets and \
                    not annotation_dict['filenames']:
                if osp.isfile(annotation_file):
                    # Remove subsets that became empty
                    os.remove(annotation_file)
                continue

            with open(annotation_file, 'wb') as labels_file:
                pickle.dump(annotation_dict, labels_file)

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        for subset in patch.updated_subsets:
            conv = cls(dataset.get_subset(subset), save_dir=save_dir, **kwargs)
            conv._patch = patch
            conv.apply()

        for subset, status in patch.updated_subsets.items():
            if status != ItemStatus.removed:
                continue

            subset_file = osp.join(save_dir, subset)
            if osp.isfile(subset_file):
                os.remove(subset_file)
