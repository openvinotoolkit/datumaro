# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Any, Tuple

from attrs import define, field


class ImmutableObjectError(Exception):
    def __str__(self):
        return "Cannot set value of immutable object"


class DatumaroError(Exception):
    pass


class VcsError(DatumaroError):
    pass


class ReadonlyDatasetError(VcsError):
    def __str__(self):
        return "Can't update a read-only dataset"


class ReadonlyProjectError(VcsError):
    def __str__(self):
        return "Can't change a read-only project"


@define(auto_exc=False)
class UnknownRefError(VcsError):
    ref = field()

    def __str__(self):
        return f"Can't parse ref '{self.ref}'"


class MissingObjectError(VcsError):
    pass


class MismatchingObjectError(VcsError):
    pass


@define(auto_exc=False)
class UnsavedChangesError(VcsError):
    paths = field()

    def __str__(self):
        return "There are some uncommitted changes: %s" % ", ".join(self.paths)


class ForeignChangesError(VcsError):
    pass


class EmptyCommitError(VcsError):
    pass


class PathOutsideSourceError(VcsError):
    pass


class SourceUrlInsideProjectError(VcsError):
    def __str__(self):
        return "Source URL cannot point inside the project"


class UnexpectedUrlError(VcsError):
    pass


class MissingSourceHashError(VcsError):
    pass


class PipelineError(DatumaroError):
    pass


class InvalidPipelineError(PipelineError):
    pass


class EmptyPipelineError(InvalidPipelineError):
    pass


class MultiplePipelineHeadsError(InvalidPipelineError):
    pass


class MissingPipelineHeadError(InvalidPipelineError):
    pass


class InvalidStageError(InvalidPipelineError):
    pass


class UnknownStageError(InvalidStageError):
    pass


class MigrationError(DatumaroError):
    pass


class OldProjectError(DatumaroError):
    def __str__(self):
        return """
            The project you're trying to load was
            created by the old Datumaro version. Try to migrate the
            project with 'datum project migrate' and then reload.
            """


@define(auto_exc=False)
class ProjectNotFoundError(DatumaroError):
    path = field()

    def __str__(self):
        return f"Can't find project at '{self.path}'"


@define(auto_exc=False)
class ProjectAlreadyExists(DatumaroError):
    path = field()

    def __str__(self):
        return f"Can't create project: a project already exists " f"at '{self.path}'"


@define(auto_exc=False)
class UnknownSourceError(DatumaroError):
    name = field()

    def __str__(self):
        return f"Unknown source '{self.name}'"


@define(auto_exc=False)
class UnknownTargetError(DatumaroError):
    name = field()

    def __str__(self):
        return f"Unknown target '{self.name}'"


@define(auto_exc=False)
class UnknownFormatError(DatumaroError):
    format = field()

    def __str__(self):
        return (
            f"Unknown source format '{self.format}'. To make it "
            "available, add the corresponding Extractor implementation "
            "to the environment"
        )


@define(auto_exc=False)
class SourceExistsError(DatumaroError):
    name = field()

    def __str__(self):
        return f"Source '{self.name}' already exists"


class DatasetExportError(DatumaroError):
    pass


@define(auto_exc=False)
class ItemExportError(DatasetExportError):
    """
    Represents additional item error info. The error itself is supposed to be
    in the `__cause__` member.
    """

    item_id: Tuple[str, str]

    def __str__(self):
        return "Failed to export item %s" % (self.item_id,)


class AnnotationExportError(ItemExportError):
    pass


class DatasetImportError(DatumaroError):
    pass


@define(auto_exc=False)
class ItemImportError(DatasetImportError):
    """
    Represents additional item error info. The error itself is supposed to be
    in the `__cause__` member.
    """

    item_id: Tuple[str, str]

    def __str__(self):
        return "Failed to import item %s" % (self.item_id,)


class AnnotationImportError(ItemImportError):
    pass


@define(auto_exc=False)
class DatasetNotFoundError(DatasetImportError):
    path = field()

    def __str__(self):
        return f"Failed to find dataset at '{self.path}'"


@define(auto_exc=False)
class MultipleFormatsMatchError(DatasetImportError):
    formats = field()

    def __str__(self):
        return (
            "Failed to detect dataset format automatically:"
            " data matches more than one format: %s" % ", ".join(self.formats)
        )


class NoMatchingFormatsError(DatasetImportError):
    def __str__(self):
        return "Failed to detect dataset format automatically: " "no matching formats found"


class DatasetError(DatumaroError):
    pass


class MediaTypeError(DatumaroError):
    pass


class CategoriesRedefinedError(DatasetError):
    def __str__(self):
        return "Categories can only be set once for a dataset"


@define(auto_exc=False)
class RepeatedItemError(DatasetError):
    item_id = field()

    def __str__(self):
        return f"Item {self.item_id} is repeated in the source sequence."


class DatasetQualityError(DatasetError):
    pass


@define(auto_exc=False)
class AnnotationsTooCloseError(DatasetQualityError):
    item_id = field()
    a = field()
    b = field()
    distance = field()

    def __str__(self):
        return "Item %s: annotations are too close: %s, %s, distance = %s" % (
            self.item_id,
            self.a,
            self.b,
            self.distance,
        )


@define(auto_exc=False)
class WrongGroupError(DatasetQualityError):
    item_id = field()
    found = field(converter=set)
    expected = field(converter=set)
    group = field(converter=list)

    def __str__(self):
        return "Item %s: annotation group has wrong labels: " "found %s, expected %s, group %s" % (
            self.item_id,
            self.found,
            self.expected,
            self.group,
        )


@define(auto_exc=False, init=False)
class DatasetMergeError(DatasetError):
    sources = field(converter=set, factory=set, kw_only=True)

    def _my__init__(self, msg=None, *, sources=None):
        super().__init__(msg)
        self.__attrs_init__(sources=sources or set())


# Pylint will raise false positive warnings for derived classes,
# when __init__ is defined directly
setattr(DatasetMergeError, "__init__", DatasetMergeError._my__init__)


@define(auto_exc=False)
class MismatchingImageInfoError(DatasetMergeError):
    item_id: Tuple[str, str]
    a: Tuple[int, int]
    b: Tuple[int, int]

    def __str__(self):
        return "Item %s: mismatching image size info: %s vs %s" % (self.item_id, self.a, self.b)


@define(auto_exc=False)
class MismatchingMediaPathError(DatasetMergeError):
    item_id: Tuple[str, str]
    a: str
    b: str

    def __str__(self):
        return "Item %s: mismatching media path info: %s vs %s" % (self.item_id, self.a, self.b)


@define(auto_exc=False)
class MismatchingMediaError(DatasetMergeError):
    item_id: Tuple[str, str]
    a: Any
    b: Any

    def __str__(self):
        return "Item %s: mismatching media info: %s vs %s" % (self.item_id, self.a, self.b)


@define(auto_exc=False)
class MismatchingAttributesError(DatasetMergeError):
    item_id: Tuple[str, str]
    key: str
    a: Any
    b: Any

    def __str__(self):
        return "Item %s: mismatching image attribute %s: %s vs %s" % (
            self.item_id,
            self.key,
            self.a,
            self.b,
        )


class ConflictingCategoriesError(DatasetMergeError):
    pass


@define(auto_exc=False)
class NoMatchingAnnError(DatasetMergeError):
    item_id = field()
    ann = field()

    def __str__(self):
        return "Item %s: can't find matching annotation " "in sources %s, annotation is %s" % (
            self.item_id,
            self.sources,
            self.ann,
        )


@define(auto_exc=False)
class NoMatchingItemError(DatasetMergeError):
    item_id = field()

    def __str__(self):
        return "Item %s: can't find matching item in sources %s" % (self.item_id, self.sources)


@define(auto_exc=False)
class FailedLabelVotingError(DatasetMergeError):
    item_id = field()
    votes = field()
    ann = field(default=None)

    def __str__(self):
        return "Item %s: label voting failed%s, votes %s, sources %s" % (
            self.item_id,
            "for ann %s" % self.ann if self.ann else "",
            self.votes,
            self.sources,
        )


@define(auto_exc=False)
class FailedAttrVotingError(DatasetMergeError):
    item_id = field()
    attr = field()
    votes = field()
    ann = field()

    def __str__(self):
        return "Item %s: attribute voting failed " "for ann %s, votes %s, sources %s" % (
            self.item_id,
            self.ann,
            self.votes,
            self.sources,
        )


@define(auto_exc=False)
class VideoMergeError(DatasetMergeError):
    item_id = field()

    def __str__(self):
        return "Item %s: video merging is not possible" % (self.item_id,)


@define(auto_exc=False)
class DatasetValidationError(DatumaroError):
    severity = field()

    def to_dict(self):
        return {
            "anomaly_type": self.__class__.__name__,
            "description": str(self),
            "severity": self.severity.name,
        }


@define(auto_exc=False)
class DatasetItemValidationError(DatasetValidationError):
    item_id = field()
    subset = field()

    def to_dict(self):
        dict_repr = super().to_dict()
        dict_repr["item_id"] = self.item_id
        dict_repr["subset"] = self.subset
        return dict_repr


@define(auto_exc=False)
class MissingLabelCategories(DatasetValidationError):
    def __str__(self):
        return "Metadata (ex. LabelCategories) should be defined" " to validate a dataset."


@define(auto_exc=False)
class MissingAnnotation(DatasetItemValidationError):
    ann_type = field()

    def __str__(self):
        return f"Item needs '{self.ann_type}' annotation(s), " "but not found."


@define(auto_exc=False)
class MultiLabelAnnotations(DatasetItemValidationError):
    def __str__(self):
        return "Item needs a single label but multiple labels are found."


@define(auto_exc=False)
class MissingAttribute(DatasetItemValidationError):
    label_name = field()
    attr_name = field()

    def __str__(self):
        return f"Item needs the attribute '{self.attr_name}' " f"for the label '{self.label_name}'."


@define(auto_exc=False)
class UndefinedLabel(DatasetItemValidationError):
    label_name = field()

    def __str__(self):
        return f"Item has the label '{self.label_name}' which " "is not defined in metadata."


@define(auto_exc=False)
class UndefinedAttribute(DatasetItemValidationError):
    label_name = field()
    attr_name = field()

    def __str__(self):
        return (
            f"Item has the attribute '{self.attr_name}' for the "
            f"label '{self.label_name}' which is not defined in metadata."
        )


@define(auto_exc=False)
class LabelDefinedButNotFound(DatasetValidationError):
    label_name = field()

    def __str__(self):
        return (
            f"The label '{self.label_name}' is defined in "
            "metadata, but not found in the dataset."
        )


@define(auto_exc=False)
class AttributeDefinedButNotFound(DatasetValidationError):
    label_name = field()
    attr_name = field()

    def __str__(self):
        return (
            f"The attribute '{self.attr_name}' for the label "
            f"'{self.label_name}' is defined in metadata, but not "
            "found in the dataset."
        )


@define(auto_exc=False)
class OnlyOneLabel(DatasetValidationError):
    label_name = field()

    def __str__(self):
        return f"The dataset has only one label '{self.label_name}'."


@define(auto_exc=False)
class OnlyOneAttributeValue(DatasetValidationError):
    label_name = field()
    attr_name = field()
    value = field()

    def __str__(self):
        return (
            "The dataset has the only attribute value "
            f"'{self.value}' for the attribute '{self.attr_name}' for the "
            f"label '{self.label_name}'."
        )


@define(auto_exc=False)
class FewSamplesInLabel(DatasetValidationError):
    label_name = field()
    count = field()

    def __str__(self):
        return (
            f"The number of samples in the label '{self.label_name}'"
            f" might be too low. Found '{self.count}' samples."
        )


@define(auto_exc=False)
class FewSamplesInAttribute(DatasetValidationError):
    label_name = field()
    attr_name = field()
    attr_value = field()
    count = field()

    def __str__(self):
        return (
            "The number of samples for attribute = value "
            f"'{self.attr_name} = {self.attr_value}' for the label "
            f"'{self.label_name}' might be too low. "
            f"Found '{self.count}' samples."
        )


@define(auto_exc=False)
class ImbalancedLabels(DatasetValidationError):
    def __str__(self):
        return "There is an imbalance in the label distribution."


@define(auto_exc=False)
class ImbalancedAttribute(DatasetValidationError):
    label_name = field()
    attr_name = field()

    def __str__(self):
        return (
            "There is an imbalance in the distribution of attribute"
            f" '{self. attr_name}' for the label '{self.label_name}'."
        )


@define(auto_exc=False)
class ImbalancedDistInLabel(DatasetValidationError):
    label_name = field()
    prop = field()

    def __str__(self):
        return (
            f"Values of '{self.prop}' are not evenly " f"distributed for '{self.label_name}' label."
        )


@define(auto_exc=False)
class ImbalancedDistInAttribute(DatasetValidationError):
    label_name = field()
    attr_name = field()
    attr_value = field()
    prop = field()

    def __str__(self):
        return (
            f"Values of '{self.prop}' are not evenly "
            f"distributed for '{self.attr_name}' = '{self.attr_value}' for "
            f"the '{self.label_name}' label."
        )


@define(auto_exc=False)
class NegativeLength(DatasetItemValidationError):
    ann_id = field()
    prop = field()
    val = field()

    def __str__(self):
        return (
            f"Annotation '{self.ann_id}' in "
            "the item should have a positive value of "
            f"'{self.prop}' but got '{self.val}'."
        )


@define(auto_exc=False)
class InvalidValue(DatasetItemValidationError):
    ann_id = field()
    prop = field()

    def __str__(self):
        return (
            f"Annotation '{self.ann_id}' in "
            "the item has an inf or a NaN value of "
            f"'{self.prop}'."
        )


@define(auto_exc=False)
class FarFromLabelMean(DatasetItemValidationError):
    label_name = field()
    ann_id = field()
    prop = field()
    mean = field()
    val = field()

    def __str__(self):
        return (
            f"Annotation '{self.ann_id}' in "
            f"the item has a value of '{self.prop}' that "
            "is too far from the label average. (mean of "
            f"'{self.label_name}' label: {self.mean}, got '{self.val}')."
        )


@define(auto_exc=False)
class FarFromAttrMean(DatasetItemValidationError):
    label_name = field()
    ann_id = field()
    attr_name = field()
    attr_value = field()
    prop = field()
    mean = field()
    val = field()

    def __str__(self):
        return (
            f"Annotation '{self.ann_id}' in the "
            f"item has a value of '{self.prop}' that "
            "is too far from the attribute average. (mean of "
            f"'{self.attr_name}' = '{self.attr_value}' for the "
            f"'{self.label_name}' label: {self.mean}, got '{self.val}')."
        )
