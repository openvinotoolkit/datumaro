# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

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
class BraceStatus:
    flush: bool = False
    level: int = 0
    get_started: bool = False


@dataclass
class CurlyStatus:
    flush: bool = False
    level: int = 0
    buffer: str = ""
    start: int = -1
    end: int = -1

    def reset(self):
        self.flush = 0
        self.level = 0
        self.buffer = ""
        self.start = -1
        self.end = -1

    def to_dict(self):
        return parse_json(self.buffer)


class COCOSection(Enum):
    IMAGES = '"images"'
    ANNOTATIONS = '"annotations"'


class FileReaderWithCache:
    _n_chars = 1024 * 1024 * 16  # 16MB

    def __init__(self, path: str):
        self.fp = open(path, "r")
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
            section: self.find_section_offset(section)
            for section in [COCOSection.IMAGES, COCOSection.ANNOTATIONS]
        }

        self.item_page_map = {}
        self.ann_page_map = AnnotationPageMap()

        self.create_page_map(COCOSection.IMAGES, self.flush_item)
        self.create_page_map(COCOSection.ANNOTATIONS, self.flush_ann)

    @staticmethod
    def gen_char_and_cursor(fp, n_chars: int = 65536):
        while (out := fp.read(n_chars)) != "":
            cursor = fp.tell() - len(out)
            for idx, c in enumerate(out):
                yield c, cursor + idx

    def find_section_offset(self, section: COCOSection) -> int:
        pattern = section.value
        len_pattern = len(pattern)
        dst = -1
        buffer = ""
        with open(self.path, "r", encoding="utf-8") as fp:
            while (out := fp.read(self._n_chars)) != "":
                cursor = fp.tell()

                buffer = buffer + out
                found = buffer.find(pattern)

                if found >= 0:
                    dst = cursor - len(buffer) + found
                    break

                buffer = buffer[-len_pattern:]

        if dst < 0:
            raise RuntimeError()

        return dst

    def flush_item(self, curly: CurlyStatus):
        data = curly.to_dict()
        img_id = data.get("id")

        if img_id is None:
            raise ValueError("Cannot find image id.")

        self.item_page_map[img_id] = ItemPage(curly.start, curly.end - curly.start, -1)

    def flush_ann(self, curly: CurlyStatus):
        data = curly.to_dict()

        img_id = data.get("image_id")
        if img_id is None:
            return
        if img_id not in self.item_page_map:
            return

        self.ann_page_map.offsets.append(curly.start)
        self.ann_page_map.sizes.append(curly.end - curly.start)

        item_page = self.item_page_map[img_id]
        curr_pt = len(self.ann_page_map.prev_pts)
        prev_pt = item_page.ann_last_pt

        self.ann_page_map.prev_pts.append(prev_pt)
        item_page.ann_last_pt = curr_pt

    def create_page_map(
        self, section: COCOSection, flush_callback: Callable[[CurlyStatus], None]
    ) -> None:
        section_offset = self.section_offsets[section]
        brace = BraceStatus(flush=False, level=0, get_started=False)
        curly = CurlyStatus(flush=False, level=0, buffer="", start=-1, end=-1)

        cnt = 0
        with open(self.path, "r", encoding="utf-8") as fp:
            fp.seek(section_offset, 0)

            gen_char_and_cursor = self.gen_char_and_cursor(fp)

            for c, cursor in gen_char_and_cursor:
                # Continue to the root brace
                if not brace.get_started and c != "[":
                    continue

                # Start the root brace
                brace.get_started = True

                self.update(brace, curly, c, cursor)

                if curly.flush:
                    flush_callback(curly)
                    curly.reset()
                    cnt += 1
                if brace.flush:
                    break

        if not brace.get_started:
            raise ValueError(f"Cannot find the list from the section={section}.")
        if curly.buffer.replace("\n", "").replace("\t", "").replace(" ", "") != "":
            raise ValueError(
                f"The input has a dictionary with no terminating curly braces. Remaining buffer={curly.buffer}"
            )

    @staticmethod
    def update(brace, curly, c, cursor):
        if c == "[":
            if brace.level == 0:
                brace.flush = False
            else:
                curly.buffer += "["
            brace.level += 1
        elif c == "{":
            if curly.level == 0:
                curly.flush = False
                curly.buffer = "{"
                curly.start = cursor
            else:
                curly.buffer += "{"
            curly.level += 1
        elif c == "}":
            curly.level -= 1
            curly.buffer += "}"
            if curly.level == 0:
                curly.flush = True
                curly.end = cursor + 1
        elif c == "]":
            brace.level -= 1
            if brace.level == 0:
                brace.flush = True
            else:
                curly.buffer += "]"
        else:
            curly.buffer += c

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
