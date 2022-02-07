# Copyright (C) 2021-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import (
    Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union,
)
import itertools
import logging as log
import os.path as osp

from attrs import field, frozen

from datumaro.components.annotation import (
    AnnotationType, Bbox, Label, LabelCategories,
)
from datumaro.components.extractor import (
    CategoriesInfo, DatasetItem, IExtractor,
)
from datumaro.components.media import ByteImage
from datumaro.util.tf_util import import_tf

try:
    tf = import_tf()
    import tensorflow_datasets as tfds
except ImportError:
    log.debug("Unable to import TensorFlow or TensorFlow Datasets. " \
        "Dataset downloading via TFDS is disabled.")
    TFDS_EXTRACTOR_AVAILABLE = False
else:
    TFDS_EXTRACTOR_AVAILABLE = True

@frozen
class TfdsDatasetMetadata:
    default_converter_name: str

@frozen
class _TfdsAdapter:
    category_transformers: Sequence[
        Callable[['tfds.core.DatasetBuilder', CategoriesInfo], None]]
    data_transformers: Sequence[Callable[[Any, DatasetItem], None]]
    id_generator: Callable[[Any], str] = field(default=None, kw_only=True)

    metadata: TfdsDatasetMetadata

    def transform_categories(self,
        tfds_builder: tfds.core.DatasetBuilder, categories: CategoriesInfo,
    ) -> None:
        for t in self.category_transformers:
            t(tfds_builder, categories)

    def transform_data(self, tfds_example: Any, item: DatasetItem) -> None:
        for t in self.data_transformers:
            t(tfds_example, item)

@frozen
class _SetLabelCategoriesFromClassLabelFeature:
    feature_path: Union[str, Tuple[str, ...]]

    def __call__(self,
        tfds_builder: tfds.core.DatasetBuilder, categories: CategoriesInfo,
    ) -> None:
        assert AnnotationType.label not in categories
        if isinstance(self.feature_path, str):
            feature_connector = tfds_builder.info.features[self.feature_path]
        else:
            feature_connector = tfds_builder.info.features

            for segment in self.feature_path:
                assert isinstance(feature_connector, (
                    tfds.features.FeaturesDict, tfds.features.Sequence,
                ))

                if isinstance(feature_connector, tfds.features.Sequence):
                    assert segment == 'feature'
                    feature_connector = feature_connector.feature
                else:
                    feature_connector = feature_connector[segment]

        assert isinstance(feature_connector, tfds.features.ClassLabel)
        categories[AnnotationType.label] = LabelCategories.from_iterable(
            feature_connector.names)

@frozen
class _SetImageFromImageFeature:
    feature_name: str
    filename_feature_name: Optional[str] = field(default=None)

    def __call__(self, tfds_example: Any, item: DatasetItem) -> None:
        if self.filename_feature_name:
            filename = tfds_example[self.filename_feature_name].numpy() \
                .decode('UTF-8')
        else:
            filename = None

        item.image = ByteImage(data=tfds_example[self.feature_name].numpy(),
            path=filename)

@frozen
class _AddLabelFromClassLabelFeature:
    feature_name: str

    def __call__(self, tfds_example: Any, item: DatasetItem) -> None:
        item.annotations.append(
            Label(tfds_example[self.feature_name].numpy()),
        )

@frozen
class _AddObjectsFromFeature:
    feature_name: str
    bbox_member: str
    label_member: Optional[str] = field(default=None, kw_only=True)

    def __call__(self, tfds_example: Any, item: DatasetItem) -> None:
        tfds_objects = tfds_example[self.feature_name]
        tfds_bboxes = tfds_objects[self.bbox_member]
        num_objects = tfds_bboxes.shape[0]

        tfds_labels = None
        if self.label_member is not None:
            tfds_labels = tfds_objects[self.label_member]
            assert tfds_labels.shape[0] == num_objects

        for i in range(num_objects):
            norm_ymin, norm_xmin, norm_ymax, norm_xmax = tfds_bboxes[i].numpy()
            item.annotations.append(Bbox(
                x=norm_xmin * item.image.size[1],
                y=norm_ymin * item.image.size[0],
                w=(norm_xmax - norm_xmin) * item.image.size[1],
                h=(norm_ymax - norm_ymin) * item.image.size[0],
            ))
            if tfds_labels is not None:
                item.annotations[-1].label = tfds_labels[i].numpy()

@frozen
class _GenerateIdFromTextFeature:
    feature_name: str

    def __call__(self, tfds_example: Any) -> str:
        return tfds_example[self.feature_name].numpy().decode('UTF-8')

@frozen
class _GenerateIdFromFilenameFeature:
    feature_name: str

    def __call__(self, tfds_example: Any) -> str:
        file_name = tfds_example[self.feature_name].numpy().decode('UTF-8')
        return osp.splitext(file_name)[0]

_MNIST_ADAPTER = _TfdsAdapter(
    category_transformers=[_SetLabelCategoriesFromClassLabelFeature('label')],
    data_transformers=[
        _SetImageFromImageFeature('image'),
        _AddLabelFromClassLabelFeature('label'),
    ],
    metadata=TfdsDatasetMetadata(default_converter_name='mnist'),
)

_CIFAR_ADAPTER = _TfdsAdapter(
    category_transformers=[_SetLabelCategoriesFromClassLabelFeature('label')],
    data_transformers=[
        _SetImageFromImageFeature('image'),
        _AddLabelFromClassLabelFeature('label'),
    ],
    id_generator=_GenerateIdFromTextFeature('id'),
    metadata=TfdsDatasetMetadata(default_converter_name='cifar'),
)

_IMAGENET_ADAPTER = _TfdsAdapter(
    category_transformers=[_SetLabelCategoriesFromClassLabelFeature('label')],
    data_transformers=[
        _SetImageFromImageFeature('image', filename_feature_name='file_name'),
        _AddLabelFromClassLabelFeature('label'),
    ],
    id_generator=_GenerateIdFromFilenameFeature('file_name'),
    metadata=TfdsDatasetMetadata(default_converter_name='imagenet_txt'),
)

_VOC_ADAPTER = _TfdsAdapter(
    category_transformers=[_SetLabelCategoriesFromClassLabelFeature(
        ('objects', 'feature', 'label'))],
    data_transformers=[
        _SetImageFromImageFeature('image',
            filename_feature_name='image/filename'),
        _AddObjectsFromFeature('objects', 'bbox', label_member='label')
    ],
    id_generator=_GenerateIdFromFilenameFeature('image/filename'),
    metadata=TfdsDatasetMetadata(default_converter_name='voc'),
)

_TFDS_ADAPTERS = {
    'cifar10': _CIFAR_ADAPTER,
    'cifar100': _CIFAR_ADAPTER,
    'imagenet_v2': _IMAGENET_ADAPTER,
    'mnist': _MNIST_ADAPTER,
    'voc/2012': _VOC_ADAPTER,
}

class _TfdsSplitExtractor(IExtractor):
    def __init__(self, parent: _TfdsExtractor,
        tfds_split: tf.data.Dataset,
        tfds_split_info: tfds.core.SplitInfo,
    ):
        self._parent = parent
        self._tfds_split = tfds_split
        self._tfds_split_info = tfds_split_info

    def __len__(self) -> int:
        return self._tfds_split_info.num_examples

    def __iter__(self) -> Iterator[DatasetItem]:
        for example_index, tfds_example in enumerate(self._tfds_split):
            if self._parent._adapter.id_generator:
                item_id = self._parent._adapter.id_generator(tfds_example)
            else:
                item_id = str(example_index)

            dm_item = DatasetItem(id=item_id, subset=self._tfds_split_info.name)
            self._parent._adapter.transform_data(tfds_example, dm_item)

            yield dm_item

    def categories(self) -> CategoriesInfo:
        return self._parent.categories()

    def subsets(self) -> Dict[str, IExtractor]:
        return {self._tfds_split_info.name: self}

    def get_subset(self, name) -> IExtractor:
        assert name == self._tfds_split_info.name
        return self

    def get(self, id, subset=None) -> Optional[DatasetItem]:
        if subset is not None and subset != self._tfds_split_info.name:
            return None

        for item in self:
            if item.id == id:
                return item

        return None

class _TfdsExtractor(IExtractor):
    _categories: CategoriesInfo

    def __init__(self, tfds_ds_name: str) -> None:
        self._adapter = _TFDS_ADAPTERS[tfds_ds_name]
        tfds_builder = tfds.builder(tfds_ds_name)
        tfds_ds_info = tfds_builder.info

        self._categories = {}
        self._adapter.transform_categories(tfds_builder, self._categories)

        tfds_decoders = {}
        for tfds_feature_name, tfds_fc in tfds_ds_info.features.items():
            if isinstance(tfds_fc, tfds.features.Image):
                tfds_decoders[tfds_feature_name] = tfds.decode.SkipDecoding()

        tfds_builder.download_and_prepare()
        self._tfds_ds = tfds_builder.as_dataset(decoders=tfds_decoders)

        self._split_extractors = {
            split_name: _TfdsSplitExtractor(
                self, split, tfds_ds_info.splits[split_name])
            # Since dicts in Python 3.7+ (and de facto in 3.6+) are
            # order-preserving, sort the splits by name so that we always
            # iterate over them in alphabetical order.
            for split_name, split in sorted(self._tfds_ds.items())
        }

    def __len__(self) -> int:
        return sum(len(ex) for ex in self._split_extractors.values())

    def __iter__(self) -> Iterator[DatasetItem]:
        return itertools.chain.from_iterable(self._split_extractors.values())

    def categories(self) -> CategoriesInfo:
        return self._categories

    def subsets(self) -> Dict[str, IExtractor]:
        return self._split_extractors

    def get_subset(self, name) -> IExtractor:
        return self._split_extractors[name]

    def get(self, id, subset=None) -> Optional[DatasetItem]:
        if subset is None:
            for ex in self._split_extractors.values():
                item = ex.get(id)
                if item is not None: return item
            return None

        if subset not in self._split_extractors:
            return None
        return self._split_extractors[subset].get(id)

AVAILABLE_TFDS_DATASETS: Mapping[str, TfdsDatasetMetadata] = {
    name: adapter.metadata for name, adapter in _TFDS_ADAPTERS.items()
}

def make_tfds_extractor(tfds_ds_name: str) -> IExtractor:
    return _TfdsExtractor(tfds_ds_name)
