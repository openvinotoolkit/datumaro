errors module
-------------

.. automodule:: datumaro.components.errors

   .. autoclass:: ImmutableObjectError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: DatumaroError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: VcsError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: ReadonlyDatasetError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: ReadonlyProjectError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: UnknownRefError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: ref = attrib()

   .. autoclass:: MissingObjectError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: MismatchingObjectError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: UnsavedChangesError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: paths = attrib()

   .. autoclass:: ForeignChangesError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: EmptyCommitError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: PathOutsideSourceError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: SourceUrlInsideProjectError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: UnexpectedUrlError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: MissingSourceHashError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: PipelineError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: InvalidPipelineError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: EmptyPipelineError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: MultiplePipelineHeadsError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: MissingPipelineHeadError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: InvalidStageError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: UnknownStageError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: MigrationError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: OldProjectError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: ProjectNotFoundError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: path = attrib()

   .. autoclass:: ProjectAlreadyExists
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: path = attrib()

   .. autoclass:: UnknownSourceError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: name = attrib()

   .. autoclass:: UnknownTargetError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: name = attrib()

   .. autoclass:: UnknownFormatError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: format = attrib()

   .. autoclass:: SourceExistsError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: name = attrib()

   .. autoclass:: DatasetImportError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: DatasetNotFoundError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: path = attrib()

   .. autoclass:: MultipleFormatsMatchError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: formats = attrib()

   .. autoclass:: NoMatchingFormatsError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: DatasetError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: CategoriesRedefinedError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: RepeatedItemError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: item_id = attrib()

   .. autoclass:: DatasetQualityError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: AnnotationsTooCloseError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: item_id = attrib()
                     a = attrib()
                     b = attrib()
                     distance = attrib()

   .. autoclass:: WrongGroupError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: item_id = attrib()
                     found = attrib(converter=set)
                     expected = attrib(converter=set)
                     group = attrib(converter=list)

   .. autoclass:: DatasetMergeError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: sources = attrib(converter=set, factory=set, kw_only=True)

   .. autoclass:: MismatchingImageInfoError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: item_id = attrib()
                     a = attrib()
                     b = attrib()

   .. autoclass:: ConflictingCategoriesError
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: NoMatchingAnnError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: item_id = attrib()
                     ann = attrib()

   .. autoclass:: NoMatchingItemError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: item_id = attrib()

   .. autoclass:: FailedLabelVotingError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: item_id = attrib()
                     votes = attrib()
                     ann = attrib(default=None)

   .. autoclass:: FailedAttrVotingError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: item_id = attrib()
                     attr = attrib()
                     votes = attrib()
                     ann = attrib()

   .. autoclass:: DatasetValidationError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: severity = attrib()

   .. autoclass:: DatasetItemValidationError
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: item_id = attrib()
                     subset = attrib()

   .. autoclass:: MissingLabelCategories
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: ann_type = attrib()

   .. autoclass:: MissingAnnotation
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: MultiLabelAnnotations
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: MissingAttribute
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: label_name = attrib()
                     attr_name = attrib()

   .. autoclass:: UndefinedLabel
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: label_name = attrib()

   .. autoclass:: UndefinedAttribute
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: label_name = attrib()
                   attr_name = attrib()

   .. autoclass:: LabelDefinedButNotFound
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: label_name = attrib()

   .. autoclass:: AttributeDefinedButNotFound
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: label_name = attrib()
                     attr_name = attrib()

   .. autoclass:: OnlyOneLabel
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: label_name = attrib()

   .. autoclass:: OnlyOneAttributeValue
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: label_name = attrib()
                     attr_name = attrib()
                     value = attrib()

   .. autoclass:: FewSamplesInLabel
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: label_name = attrib()
                     count = attrib()

   .. autoclass:: FewSamplesInAttribute
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: label_name = attrib()
                     attr_name = attrib()
                     attr_value = attrib()
                     count = attrib()

   .. autoclass:: ImbalancedLabels
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: ImbalancedAttribute
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: label_name = attrib()
                     attr_name = attrib()

   .. autoclass:: ImbalancedDistInLabel
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: label_name = attrib()
                     prop = attrib()

   .. autoclass:: ImbalancedDistInAttribute
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: label_name = attrib()
                     attr_name = attrib()
                     attr_value = attrib()
                     prop = attrib()

   .. autoclass:: NegativeLength
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: ann_id = attrib()
                     prop = attrib()
                     val = attrib()

   .. autoclass:: InvalidValue
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: ann_id = attrib()
                     prop = attrib()

   .. autoclass:: FarFromLabelMean
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: label_name = attrib()
                     ann_id = attrib()
                     prop = attrib()
                     mean = attrib()
                     val = attrib()

   .. autoclass:: FarFromAttrMean
      :members:
      :undoc-members:
      :show-inheritance:

      .. py:data:: label_name = attrib()
                     ann_id = attrib()
                     attr_name = attrib()
                     attr_value = attrib()
                     prop = attrib()
                     mean = attrib()
                     val = attrib()
