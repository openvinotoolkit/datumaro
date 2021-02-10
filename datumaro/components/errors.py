# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from attr import attr, attrib, attrs


class DatumaroError(Exception):
    pass

@attrs
class DatasetError(DatumaroError):
    item_id = attrib()

@attrs
class RepeatedItemError(DatasetError):
    def __str__(self):
        return "Item %s is repeated in the source sequence." % (self.item_id)

@attrs
class MismatchingImageInfoError(DatasetError):
    a = attrib()
    b = attrib()

    def __str__(self):
        return "Item %s: mismatching image size info: %s vs %s" % \
            (self.item_id, self.a, self.b)

@attrs
class QualityError(DatasetError):
    pass

@attrs
class AnnotationsTooCloseError(QualityError):
    a = attrib()
    b = attrib()
    distance = attrib()

    def __str__(self):
        return "Item %s: annotations are too close: %s, %s, distance = %s" % \
            (self.item_id, self.a, self.b, self.distance)

@attrs
class WrongGroupError(QualityError):
    found = attrib(converter=set)
    expected = attrib(converter=set)
    group = attrib(converter=list)

    def __str__(self):
        return "Item %s: annotation group has wrong labels: " \
            "found %s, expected %s, group %s" % \
            (self.item_id, self.found, self.expected, self.group)

@attrs
class MergeError(DatasetError):
    sources = attrib(converter=set)

@attrs
class NoMatchingAnnError(MergeError):
    ann = attrib()

    def __str__(self):
        return "Item %s: can't find matching annotation " \
            "in sources %s, annotation is %s" % \
            (self.item_id, self.sources, self.ann)

@attrs
class NoMatchingItemError(MergeError):
    def __str__(self):
        return "Item %s: can't find matching item in sources %s" % \
            (self.item_id, self.sources)

@attrs
class FailedLabelVotingError(MergeError):
    votes = attrib()
    ann = attrib(default=None)

    def __str__(self):
        return "Item %s: label voting failed%s, votes %s, sources %s" % \
            (self.item_id, 'for ann %s' % self.ann if self.ann else '',
            self.votes, self.sources)

@attrs
class FailedAttrVotingError(MergeError):
    attr = attrib()
    votes = attrib()
    ann = attrib()

    def __str__(self):
        return "Item %s: attribute voting failed " \
            "for ann %s, votes %s, sources %s" % \
            (self.item_id, self.ann, self.votes, self.sources)