# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from attr import attrib, attrs


class DatumaroError(Exception):
    pass

class VcsError(DatumaroError):
    pass

@attrs
class SourceExistsError(VcsError):
    name = attrib()

    def __str__(self):
        return "Source %s already exists" % (self.name, )

class ReadonlyProjectError(VcsError):
    pass

class DetachedProjectError(VcsError):
    pass

@attrs
class DatasetError(DatumaroError):
    item_id = attrib()

@attrs
class RepeatedItemError(DatasetError):
    def __str__(self):
        return "Item %s is repeated in the source sequence." % (self.item_id, )

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

@attrs
class DatasetValidationError(DatumaroError):
    severity = attrib()

    def to_dict(self):
        return {
            'anomaly_type': self.__class__.__name__,
            'description': str(self),
            'severity': self.severity.name,
        }

@attrs
class DatasetItemValidationError(DatasetValidationError):
    item_id = attrib()
    subset = attrib()

    def to_dict(self):
        dict_repr = super().to_dict()
        dict_repr['item_id'] = self.item_id
        dict_repr['subset'] = self.subset
        return dict_repr

@attrs
class MissingLabelCategories(DatasetValidationError):
    def __str__(self):
        return "Metadata (ex. LabelCategories) should be defined" \
            " to validate a dataset."

@attrs
class MissingLabelAnnotation(DatasetItemValidationError):
    def __str__(self):
        return "Item needs a label, but not found."

@attrs
class MultiLabelAnnotations(DatasetItemValidationError):
    def __str__(self):
        return 'Item needs a single label but multiple labels are found.'

@attrs
class MissingAttribute(DatasetItemValidationError):
    label_name = attrib()
    attr_name = attrib()

    def __str__(self):
        return f"Item needs the attribute '{self.attr_name}' " \
            f"for the label '{self.label_name}'."

@attrs
class UndefinedLabel(DatasetItemValidationError):
    label_name = attrib()

    def __str__(self):
        return f"Item has the label '{self.label_name}' which " \
            "is not defined in metadata."

@attrs
class UndefinedAttribute(DatasetItemValidationError):
    label_name = attrib()
    attr_name = attrib()

    def __str__(self):
        return f"Item has the attribute '{self.attr_name}' for the " \
            f"label '{self.label_name}' which is not defined in metadata."

@attrs
class LabelDefinedButNotFound(DatasetValidationError):
    label_name = attrib()

    def __str__(self):
        return f"The label '{self.label_name}' is defined in " \
                "metadata, but not found in the dataset."

@attrs
class AttributeDefinedButNotFound(DatasetValidationError):
    label_name = attrib()
    attr_name = attrib()

    def __str__(self):
        return f"The attribute '{self.attr_name}' for the label " \
            f"'{self.label_name}' is defined in metadata, but not " \
            "found in the dataset."

@attrs
class OnlyOneLabel(DatasetValidationError):
    label_name = attrib()

    def __str__(self):
        return f"The dataset has only one label '{self.label_name}'."

@attrs
class OnlyOneAttributeValue(DatasetValidationError):
    label_name = attrib()
    attr_name = attrib()
    value = attrib()

    def __str__(self):
        return "The dataset has the only attribute value " \
            f"'{self.value}' for the attribute '{self.attr_name}' for the " \
            f"label '{self.label_name}'."

@attrs
class FewSamplesInLabel(DatasetValidationError):
    label_name = attrib()
    count = attrib()

    def __str__(self):
        return f"The number of samples in the label '{self.label_name}'" \
            f" might be too low. Found '{self.count}' samples."

@attrs
class FewSamplesInAttribute(DatasetValidationError):
    label_name = attrib()
    attr_name = attrib()
    attr_value = attrib()
    count = attrib()

    def __str__(self):
        return "The number of samples for attribute = value " \
            f"'{self.attr_name} = {self.attr_value}' for the label " \
            f"'{self.label_name}' might be too low. " \
            f"Found '{self.count}' samples."

@attrs
class ImbalancedLabels(DatasetValidationError):
    def __str__(self):
        return 'There is an imbalance in the label distribution.'

@attrs
class ImbalancedAttribute(DatasetValidationError):
    label_name = attrib()
    attr_name = attrib()

    def __str__(self):
        return "There is an imbalance in the distribution of attribute" \
            f" '{self. attr_name}' for the label '{self.label_name}'."

@attrs
class ImbalancedBboxDistInLabel(DatasetValidationError):
    label_name = attrib()
    prop = attrib()

    def __str__(self):
        return f"Values of bbox '{self.prop}' are not evenly " \
                f"distributed for '{self.label_name}' label."

@attrs
class ImbalancedBboxDistInAttribute(DatasetValidationError):
    label_name = attrib()
    attr_name = attrib()
    attr_value = attrib()
    prop = attrib()

    def __str__(self):
        return f"Values of bbox '{self.prop}' are not evenly " \
            f"distributed for '{self.attr_name}' = '{self.attr_value}' for " \
            f"the '{self.label_name}' label."

@attrs
class MissingBboxAnnotation(DatasetItemValidationError):
    def __str__(self):
        return 'Item needs one or more bounding box annotations, ' \
            'but not found.'

@attrs
class NegativeLength(DatasetItemValidationError):
    ann_id = attrib()
    prop = attrib()
    val = attrib()

    def __str__(self):
        return f"Bounding box annotation '{self.ann_id}' in " \
            "the item should have a positive value of " \
            f"'{self.prop}' but got '{self.val}'."

@attrs
class InvalidValue(DatasetItemValidationError):
    ann_id = attrib()
    prop = attrib()

    def __str__(self):
        return f"Bounding box annotation '{self.ann_id}' in " \
            'the item has an inf or a NaN value of ' \
            f"bounding box '{self.prop}'."

@attrs
class FarFromLabelMean(DatasetItemValidationError):
    label_name = attrib()
    ann_id = attrib()
    prop = attrib()
    mean = attrib()
    val = attrib()

    def __str__(self):
        return f"Bounding box annotation '{self.ann_id}' in " \
            f"the item has a value of bounding box '{self.prop}' that " \
            "is too far from the label average. (mean of " \
            f"'{self.label_name}' label: {self.mean}, got '{self.val}')."

@attrs
class FarFromAttrMean(DatasetItemValidationError):
    label_name = attrib()
    ann_id = attrib()
    attr_name = attrib()
    attr_value = attrib()
    prop = attrib()
    mean = attrib()
    val = attrib()

    def __str__(self):
        return f"Bounding box annotation '{self.ann_id}' in the " \
            f"item has a value of bounding box '{self.prop}' that " \
            "is too far from the attribute average. (mean of " \
            f"'{self.attr_name}' = '{self.attr_value}' for the " \
            f"'{self.label_name}' label: {self.mean}, got '{self.val}')."
