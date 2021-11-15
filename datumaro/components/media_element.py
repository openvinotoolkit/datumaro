# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp


class MediaElement:
    def __init__(self, path: str) -> None:
        self._path = path

    @property
    def path(self) -> str:
        """Path to the media file"""
        return self._path

    @property
    def ext(self) -> str:
        """Media file extension"""
        return osp.splitext(osp.basename(self.path))[1]

    def __eq__(self, other: object) -> bool:
        # We need to compare exactly with this type
        if type(other) is not __class__: # pylint: disable=unidiomatic-typecheck
            return False
        return self._path == other._path
