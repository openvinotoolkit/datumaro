operations module
-----------------

.. automodule:: datumaro.components.operations

   .. autofunction:: get_ann_type

   .. autofunction:: match_annotations_equal

   .. autofunction:: merge_annotations_equal

   .. autofunction:: merge_categories

   .. autoclass:: MergingStrategy
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: ExactMerge
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. automethod:: merge

      .. automethod:: merge_items

      .. automethod:: merge_images

      .. automethod:: merge_anno

      .. automethod:: merge_categories

   .. autoclass:: IntersectMerge
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. autoclass:: Conf
         .. py:data:: pairwise_dist = attrib(converter=float, default=0.5)
                        sigma = attrib(converter=list, factory=list)

         .. py:data:: output_conf_thresh = attrib(converter=float, default=0)
                        quorum = attrib(converter=int, default=0)
                        ignored_attributes = attrib(converter=set, factory=set)

      **Error trackers:**

      .. py:data:: attrib

      .. automethod:: add_item_error

      **Indexes:**

      .. py:data:: _dataset_map

         id(dataset) -> (dataset, index)

      .. py:data:: _item_map

         id(item) -> (item, id(dataset))

      .. py:data:: _ann_map

         id(ann) -> (ann, id(item))

      .. py:data:: _item_id
      .. py:data:: _item

      **Misc.**

      .. py:data:: _categories = attrib(init=False)  merged categories

      .. autofunction:: get_ann_source

      .. autofunction:: merge_items

      .. autofunction:: merge_annotations

      .. autofunction:: match_items

   .. autoclass:: AnnotationMatcher

   .. autoclass:: LabelMatcher

   .. autoclass:: _ShapeMatcher
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. py:data:: pairwise_dist = attrib(converter=float, default=0.9)
                     cluster_dist = attrib(converter=float, default=-1.0)

      .. autofunction:: match_annotations

      .. autofunction:: distance

      .. autofunction:: label_matcher

   .. autoclass:: BboxMatcher
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: PolygonMatcher
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: MaskMatcher
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: PointsMatcher
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. py:data:: sigma: Optional[list] = attrib(default=None)
                     instance_map = attrib(converter=dict)

   .. autoclass:: LineMatcher
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: CaptionsMatcher
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: Cuboid3dMatcher
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: AnnotationMerger
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: LabelMerger
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. py:data:: quorum = attrib(converter=int, default=0)

   .. autoclass:: _ShapeMerger
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. py:data:: quorum = attrib(converter=int, default=0)

   .. autoclass:: BboxMerger
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: PolygonMerger
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: MaskMerger
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: PointsMerger
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: LineMerger
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: CaptionsMerger
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: Cuboid3dMerger
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autofunction:: match_segments

      .. py:data:: a_matches = -np.ones(len(a_segms), dtype=int)

         indices of b_segms matched to a bboxes

      .. py:data:: b_matches = -np.ones(len(b_segms), dtype=int)

         indices of a_segms matched to b bboxes

      .. py:data:: matches = []

         matches: boxes we succeeded to match completely

      .. py:data:: mispred = []

         mispred: boxes we succeeded to match, having label mismatch

   .. autoclass:: mean_std
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: _MeanStdCounter
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. automethod:: _pairwise_stats

      .. automethod:: _compute_stats

   .. autoclass:: StatsCounter
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      Implements online parallel computation of sample variance
      https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
      Needed do avoid catastrophic cancellation in floating point computations

   .. autofunction:: compute_image_statistics

   .. autofunction:: compute_ann_statistics

      .. py:function:: get_label

      .. py:function:: total_pixels

         numpy.sum might be faster, but could overflow with large datasets.
         Python's int can transparently mutate to be of indefinite precision (long)

   .. autoclass:: DistanceComparator
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. py:data:: iou_threshold = attrib(converter=float, default=0.5)


   .. autofunction:: match_items_by_id
   .. autofunction:: match_items_by_image_hash
   .. autofunction:: find_unique_images

   .. autofunction:: match_classes

   .. autoclass:: ExactComparator
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. code-block::

         match_images: bool = attrib(kw_only=True, default=False)
         ignored_fields = attrib(kw_only=True,
            factory=set, validator=default_if_none(set))
         ignored_attrs = attrib(kw_only=True,
            factory=set, validator=default_if_none(set))
         ignored_item_attrs = attrib(kw_only=True,
            factory=set, validator=default_if_none(set))
