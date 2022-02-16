from __future__ import annotations
import argparse
from typing import Iterable, Optional, Tuple
from datumaro.components.extractor import DatasetItem, IExtractor, ItemTransform
from datumaro.util import filter_dict


class DeleteImageTransform(ItemTransform):
    """
    Allows to delete specific images from dataset by their ids.|n
    |n
    Can be useful to clean the dataset from broken or unnecessary images.
    """
    @staticmethod
    def _parse_id(s):
        full_id = s.split(':')
        if len(full_id) != 2:
            raise argparse.ArgumentError(message="Invalid id format of '%s'. "
                                         "Expected a 'name:subset' pair." % s)
        return full_id

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('--id', dest='ids', type=cls._parse_id,
                            action='append', required=True,
                            help="Image id to remove. Id is 'name:subset' pair (repeatable)")
        return parser

    def __init__(self, extractor: IExtractor, ids: Iterable[Tuple[str, str]]):
        super().__init__(extractor)
        self._ids = set(ids or [])

    def transform_item(self, item):
        if (item.id, item.subset) in self._ids:
            return None
        return item


class DeleteAnnotationTransform(ItemTransform):
    """
    Allows to delete annotations on specific images from dataset.|n
    |n
    Can be useful to clean the dataset from broken or unnecessary annotations.
    """
    @staticmethod
    def _parse_id(s):
        full_id = s.split(':')
        if len(full_id) != 2:
            raise argparse.ArgumentError(message="Invalid id format of '%s'. "
                                         "Expected a 'name:subset' pair." % s)
        return full_id

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('--id', dest='ids', type=cls._parse_id,
                            action='append',
                            help="Image id to clean from annotations. "
                            "Id is 'name:subset' pair. If not specified, removes "
                            "all annotations (repeatable)")
        return parser

    def __init__(self, extractor: IExtractor,
                 ids: Optional[Iterable[Tuple[str, str]]] = None):
        super().__init__(extractor)
        self._ids = ids

    def transform_item(self, item: DatasetItem):
        if not self._ids or (item.id, item.subset) in self._ids:
            return item.wrap(annotations=[])
        return item


class DeleteAttributeTransform(ItemTransform):
    """
    Allows to delete attributes on specific images from dataset.|n
    |n
    Can be useful to clean the dataset from broken or unnecessary attributes.
    """
    @staticmethod
    def _parse_id(s):
        full_id = s.split(':')
        if len(full_id) != 2:
            raise argparse.ArgumentError(message="Invalid id format of '%s'. "
                                         "Expected a 'name:subset' pair." % s)
        return full_id

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('--id', dest='ids', type=cls._parse_id,
                            action='append',
                            help="Image id to clean from annotations. "
                            "Id is 'name:subset' pair. If not specified, "
                            "affects all images and annotations (repeatable)")
        parser.add_argument('-a', '--attr', action='append', dest='attributes',
                            help="Attribute name to be removed. If not specified, "
                            "removes all attributes (repeatable)")
        return parser

    def __init__(self, extractor: IExtractor,
                 ids: Optional[Iterable[Tuple[str, str]]] = None,
                 attributes: Optional[Iterable[str]] = None):
        super().__init__(extractor)
        self._ids = ids
        self._attributes = attributes

    def _filter_attrs(self, attrs):
        if not self._attributes:
            return None
        else:
            return filter_dict(attrs, exclude_keys=self._attributes)

    def transform_item(self, item: DatasetItem):
        if not self._ids or (item.id, item.subset) in self._ids:
            filtered_annotations = []
            for ann in item.annotations:
                filtered_annotations.append(ann.wrap(
                    attributes=self._filter_attrs(ann.attributes)))
            return item.wrap(attributes=self._filter_attrs(item.attributes),
                             annotations=filtered_annotations)
        return item
