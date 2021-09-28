# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from attr import attrib, attrs


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

@attrs
class UnknownRefError(VcsError):
    ref = attrib()

    def __str__(self):
        return f"Can't parse ref '{self.ref}'"

class MissingObjectError(VcsError):
    pass

class MismatchingObjectError(VcsError):
    pass

@attrs
class UnsavedChangesError(VcsError):
    paths = attrib()

    def __str__(self):
        return "There are some uncommitted changes: %s" % ', '.join(self.paths)

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


@attrs
class ProjectNotFoundError(DatumaroError):
    path = attrib()

    def __str__(self):
        return f"Can't find project at '{self.path}'"

@attrs
class ProjectAlreadyExists(DatumaroError):
    path = attrib()

    def __str__(self):
        return f"Can't create project: a project already exists " \
            f"at '{self.path}'"

@attrs
class UnknownSourceError(DatumaroError):
    name = attrib()

    def __str__(self):
        return f"Unknown source '{self.name}'"

@attrs
class UnknownTargetError(DatumaroError):
    name = attrib()

    def __str__(self):
        return f"Unknown target '{self.name}'"

@attrs
class UnknownFormatError(DatumaroError):
    format = attrib()

    def __str__(self):
        return f"Unknown source format '{self.format}'. To make it " \
            "available, add the corresponding Extractor implementation " \
            "to the environment"

@attrs
class SourceExistsError(DatumaroError):
    name = attrib()

    def __str__(self):
        return f"Source '{self.name}' already exists"


class DatasetImportError(DatumaroError):
    pass

@attrs
class DatasetNotFoundError(DatasetImportError):
    path = attrib()

    def __str__(self):
        return f"Failed to find dataset at '{self.path}'"

@attrs
class MultipleFormatsMatchError(DatasetImportError):
    formats = attrib()

    def __str__(self):
        return "Failed to detect dataset format automatically:" \
            " data matches more than one format: %s" % \
            ', '.join(self.formats)

class NoMatchingFormatsError(DatasetImportError):
    def __str__(self):
        return "Failed to detect dataset format automatically: " \
            "no matching formats found"

class DatasetError(DatumaroError):
    pass

class CategoriesRedefinedError(DatasetError):
    def __str__(self):
        return "Categories can only be set once for a dataset"

@attrs
class RepeatedItemError(DatasetError):
    item_id = attrib()

    def __str__(self):
        return f"Item {self.item_id} is repeated in the source sequence."


class DatasetQualityError(DatasetError):
    pass

@attrs
class AnnotationsTooCloseError(DatasetQualityError):
    item_id = attrib()
    a = attrib()
    b = attrib()
    distance = attrib()

    def __str__(self):
        return "Item %s: annotations are too close: %s, %s, distance = %s" % \
            (self.item_id, self.a, self.b, self.distance)

@attrs
class WrongGroupError(DatasetQualityError):
    item_id = attrib()
    found = attrib(converter=set)
    expected = attrib(converter=set)
    group = attrib(converter=list)

    def __str__(self):
        return "Item %s: annotation group has wrong labels: " \
            "found %s, expected %s, group %s" % \
            (self.item_id, self.found, self.expected, self.group)


@attrs(init=False)
class DatasetMergeError(DatasetError):
    sources = attrib(converter=set, factory=set, kw_only=True)

    def _my__init__(self, msg=None, *, sources=None):
        super().__init__(msg)
        self.__attrs_init__(sources=sources or set())

# Pylint will raise false positive warnings for derived classes,
# when __init__ is defined directly
setattr(DatasetMergeError, '__init__', DatasetMergeError._my__init__)

@attrs
class MismatchingImageInfoError(DatasetMergeError):
    item_id = attrib()
    a = attrib()
    b = attrib()

    def __str__(self):
        return "Item %s: mismatching image size info: %s vs %s" % \
            (self.item_id, self.a, self.b)

class ConflictingCategoriesError(DatasetMergeError):
    pass

@attrs
class NoMatchingAnnError(DatasetMergeError):
    item_id = attrib()
    ann = attrib()

    def __str__(self):
        return "Item %s: can't find matching annotation " \
            "in sources %s, annotation is %s" % \
            (self.item_id, self.sources, self.ann)

@attrs
class NoMatchingItemError(DatasetMergeError):
    item_id = attrib()

    def __str__(self):
        return "Item %s: can't find matching item in sources %s" % \
            (self.item_id, self.sources)

@attrs
class FailedLabelVotingError(DatasetMergeError):
    item_id = attrib()
    votes = attrib()
    ann = attrib(default=None)

    def __str__(self):
        return "Item %s: label voting failed%s, votes %s, sources %s" % \
            (self.item_id, 'for ann %s' % self.ann if self.ann else '',
            self.votes, self.sources)

@attrs
class FailedAttrVotingError(DatasetMergeError):
    item_id = attrib()
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
class MissingAnnotation(DatasetItemValidationError):
    ann_type = attrib()

    def __str__(self):
        return f"Item needs '{self.ann_type}' annotation(s), " \
            "but not found."

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
class ImbalancedDistInLabel(DatasetValidationError):
    label_name = attrib()
    prop = attrib()

    def __str__(self):
        return f"Values of '{self.prop}' are not evenly " \
                f"distributed for '{self.label_name}' label."

@attrs
class ImbalancedDistInAttribute(DatasetValidationError):
    label_name = attrib()
    attr_name = attrib()
    attr_value = attrib()
    prop = attrib()

    def __str__(self):
        return f"Values of '{self.prop}' are not evenly " \
            f"distributed for '{self.attr_name}' = '{self.attr_value}' for " \
            f"the '{self.label_name}' label."

@attrs
class NegativeLength(DatasetItemValidationError):
    ann_id = attrib()
    prop = attrib()
    val = attrib()

    def __str__(self):
        return f"Annotation '{self.ann_id}' in " \
            "the item should have a positive value of " \
            f"'{self.prop}' but got '{self.val}'."

@attrs
class InvalidValue(DatasetItemValidationError):
    ann_id = attrib()
    prop = attrib()

    def __str__(self):
        return f"Annotation '{self.ann_id}' in " \
            'the item has an inf or a NaN value of ' \
            f"'{self.prop}'."

@attrs
class FarFromLabelMean(DatasetItemValidationError):
    label_name = attrib()
    ann_id = attrib()
    prop = attrib()
    mean = attrib()
    val = attrib()

    def __str__(self):
        return f"Annotation '{self.ann_id}' in " \
            f"the item has a value of '{self.prop}' that " \
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
        return f"Annotation '{self.ann_id}' in the " \
            f"item has a value of '{self.prop}' that " \
            "is too far from the attribute average. (mean of " \
            f"'{self.attr_name}' = '{self.attr_value}' for the " \
            f"'{self.label_name}' label: {self.mean}, got '{self.val}')."
