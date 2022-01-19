# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

# We need to release the opened resources somehow to release file handles and
# close the program normally.
#
# Options:
#
# A. Require converter to open all the media resources.
# - Dataset (IExtractor) just provides media access metainfo
# - Dataset (IExtractor) must provide the list of all media resources
# - Each resource has to provide means for loading and releasing
# - All converters require changes and special handling for different media
# sources.
# - Resource management is explicit
# - Resources are managed safely and effectively
#
# Problems:
# - Too much burden on plugins. Media reporting and resource management takes
# too much efforts in this solution. Extractors and Converters all need to
# bother with this.
#
#
# B. Introduce Media Resource Manager, which contains all the opened
# media resources.
# - No code modifications in converters
# - All (or specific) resources are released by request
# - The system can manage the number or opened resources to control memory load
# (maybe, just extend Image Cache?)
# - Resource management is implicit for the user
#
# Problems:
# - The moment we need to release resources is debatable and needs
# investigation for each operation (however, it's just about the caching,
# so it's unlikely to make the system unstable)

from collections import OrderedDict
import sys

_instance = None

DEFAULT_CAPACITY = 2

class MediaManager:
    @staticmethod
    def get_instance():
        global _instance
        if _instance is None:
            _instance = MediaManager()
        return _instance

    def __init__(self, capacity=DEFAULT_CAPACITY):
        self.capacity = int(capacity)
        self.items = OrderedDict()

    def push(self, key, media):
        if self.capacity <= len(self.items):
            _, v = self.items.popitem(last=True)
            if hasattr(v, 'close') and sys.getrefcount(v) <= 2:
                v.close()
        self.items[key] = media

    def get(self, key):
        default = object()
        item = self.items.get(key, default)
        if item is default:
            return None

        self.items.move_to_end(key, last=False) # naive splay tree
        return item

    def size(self):
        return len(self.items)

    def clear(self):
        for item in self.items.values():
            if hasattr(item, 'close'):
                item.close()
        self.items.clear()
