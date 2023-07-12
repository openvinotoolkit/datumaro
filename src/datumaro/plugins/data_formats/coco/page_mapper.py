# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterator, List, Tuple

from datumaro.util import parse_json

__all__ = ["COCOPageMapper"]


@dataclass
class AnnotationPageMap:
    offsets: List[int] = field(default_factory=list)
    sizes: List[int] = field(default_factory=list)
    prev_pts: List[int] = field(default_factory=list)


@dataclass
class ItemPage:
    offset: int
    size: int
    ann_last_pt: int


@dataclass
class BracketStatus:
    """Struct to manage status for the brackets ([, ]) during constructing a page map

    If `flush` is True, it means that we have extracted all items from the section.

    "section": [        <- BracketStatus.flush = False and BracketStatus.level = 1
        item_0,
        item_1,
        ...
        item_n,
    ]                   <- BracketStatus.flush = True and BracketStatus.level = 0
    """

    flush: bool = False
    level: int = 0
    get_started: bool = False


@dataclass
class BraceStatus:
    """Struct to manage status for the braces ({, }) during constructing a page map

    The braces set the bounds of the items to be extracted.
    If `flush` is True, it flushes the bytes in its `buffer` which corresponds to the item.

    {                               <- BraceStatus.flush = False and BraceStatus.level = 1
        "attr1": 0,
        "attr2": {                  <- BraceStatus.level = 2
            "nested_attr": 1
        }                           <- BraceStatus.level = 1
    }                               <- BraceStatus.flush = True and BraceStatus.level = 0
    """

    flush: bool = False
    level: int = 0
    buffer: bytes = b""
    start: int = -1
    end: int = -1

    def reset(self):
        self.flush = False
        self.level = 0
        self.buffer = b""
        self.start = -1
        self.end = -1

    def to_dict(self):
        return parse_json(self.buffer)


class COCOSection(Enum):
    IMAGES = b'"images"'
    ANNOTATIONS = b'"annotations"'


class FileReaderWithCache:
    _n_chars = 1024 * 1024 * 16  # 16MB

    def __init__(self, path: str):
        self.fp = open(path, "rb")
        self.update_buffer(0, self._n_chars)

    def close(self):
        self.fp.close()

    def __del__(self):
        self.close()

    def read(self, offset: int, size: int) -> str:
        if offset < self.offset or self.offset + len(self.buffer) <= offset + size:
            self.update_buffer(offset, size)

        _from = offset - self.offset
        _to = _from + size
        return self.buffer[_from:_to]

    def update_buffer(self, offset: int, size: int) -> None:
        self.fp.seek(offset)
        self.buffer = self.fp.read(max(size, self._n_chars))
        self.offset = offset


class COCOPageMapper:
    """Construct page maps for items and annotations from the JSON file,
    which are used for the stream importer.

    It also provides __iter__() to produce item and annotation dictionaries
    in stream manner after constructing the page map.
    """

    _n_chars = 1024 * 1024 * 16  # 16MB
    cnt = 0

    def __init__(self, path: str):
        self.path = path
        self.section_offsets = {
            section: self._find_section_offset(section)
            for section in [COCOSection.IMAGES, COCOSection.ANNOTATIONS]
        }

        self.item_page_map = {}
        self.ann_page_map = AnnotationPageMap()

        self._create_page_map(COCOSection.IMAGES, self._flush_item)
        self._create_page_map(COCOSection.ANNOTATIONS, self._flush_ann)

    @staticmethod
    def _gen_char_and_cursor(fp, n_chars: int = 65536) -> Iterator[Tuple[bytes, int]]:
        while (out := fp.read(n_chars)) != "":
            cursor = fp.tell() - len(out)
            for idx, (c,) in enumerate(struct.iter_unpack("c", out)):
                yield c, cursor + idx

    def _find_section_offset(self, section: COCOSection) -> int:
        pattern = section.value
        len_pattern = len(pattern)
        dst = -1
        buffer = b""
        with open(self.path, "rb") as fp:
            while (out_bytes := fp.read(self._n_chars)) != "":
                cursor = fp.tell()

                buffer = buffer + out_bytes
                found = buffer.find(pattern)

                if found >= 0:
                    dst = cursor - len(buffer) + found
                    break

                buffer = buffer[-len_pattern:]

        if dst < 0:
            raise RuntimeError()

        return dst

    def _flush_item(self, brace: BraceStatus):
        data = brace.to_dict()
        img_id = data.get("id")

        if img_id is None:
            raise ValueError("Cannot find image id.")

        self.item_page_map[img_id] = ItemPage(brace.start, brace.end - brace.start, -1)

    def _flush_ann(self, brace: BraceStatus):
        data = brace.to_dict()

        img_id = data.get("image_id")
        if img_id is None:
            return
        if img_id not in self.item_page_map:
            return

        self.ann_page_map.offsets.append(brace.start)
        self.ann_page_map.sizes.append(brace.end - brace.start)

        item_page = self.item_page_map[img_id]
        curr_pt = len(self.ann_page_map.prev_pts)
        prev_pt = item_page.ann_last_pt

        self.ann_page_map.prev_pts.append(prev_pt)
        item_page.ann_last_pt = curr_pt

    def _create_page_map(
        self, section: COCOSection, flush_callback: Callable[[BraceStatus], None]
    ) -> None:
        section_offset = self.section_offsets[section]
        braket = BracketStatus()
        brace = BraceStatus()

        cnt = 0
        with open(self.path, "rb") as fp:
            fp.seek(section_offset, 0)

            for c, cursor in self._gen_char_and_cursor(fp):
                # Continue to the root brace
                if not braket.get_started and c != b"[":
                    continue

                # Start the root brace
                braket.get_started = True

                self.update(braket, brace, c, cursor)

                if brace.flush:
                    flush_callback(brace)
                    brace.reset()
                    cnt += 1
                if braket.flush:
                    break

        if not braket.get_started:
            raise ValueError(f"Cannot find the list from the section={section}.")
        if (
            brace.buffer.decode("utf-8").replace(os.linesep, "").replace("\t", "").replace(" ", "")
            != ""
        ):
            raise ValueError(
                f"The input has a dictionary with no terminating curly braces. Remaining buffer={brace.buffer}"
            )

    @staticmethod
    def update(braket, brace, c, cursor):
        if c == b"[":
            if braket.level == 0:
                braket.flush = False
            else:
                brace.buffer += b"["
            braket.level += 1
        elif c == b"{":
            if brace.level == 0:
                brace.flush = False
                brace.buffer = b"{"
                brace.start = cursor
            else:
                brace.buffer += b"{"
            brace.level += 1
        elif c == b"}":
            brace.level -= 1
            brace.buffer += b"}"
            if brace.level == 0:
                brace.flush = True
                brace.end = cursor + 1
        elif c == b"]":
            braket.level -= 1
            if braket.level == 0:
                braket.flush = True
            else:
                brace.buffer += b"]"
        else:
            brace.buffer += c

    def __iter__(self) -> Iterator[Tuple[Dict, Dict]]:
        reader = FileReaderWithCache(self.path)

        for item_page in self.item_page_map.values():
            data = reader.read(item_page.offset, item_page.size)
            item_dict = parse_json(data)
            anns_dict = []

            curr_pt = item_page.ann_last_pt

            to_read = []
            while curr_pt >= 0:
                ann_offset = self.ann_page_map.offsets[curr_pt]
                ann_size = self.ann_page_map.sizes[curr_pt]
                curr_pt = self.ann_page_map.prev_pts[curr_pt]
                to_read += [(ann_offset, ann_size)]

            for ann_offset, ann_size in reversed(to_read):
                data = reader.read(ann_offset, ann_size)
                ann_dict = parse_json(data)
                anns_dict.append(ann_dict)

            yield item_dict, anns_dict

        reader.close()

    def __len__(self) -> int:
        return len(self.item_page_map)
