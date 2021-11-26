operations module
-----------------

.. automodule:: datumaro.components.operations

   .. autofunction:: match_annotations_equal

   .. autofunction:: merge_annotations_equal

   .. autoclass:: ExactMerge
      :members:
      :show-inheritance:

      .. py:function:: merge

      .. py:function:: merge_items

      .. py:function:: merge_images

      .. py:function:: merge_anno

      .. py:function:: merge_categories

   .. autoclass:: IntersectMerge
      :members:
      :show-inheritance:

         **Error trackers:**

         .. py:data:: attrib

         .. py:function:: add_item_error

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

      .. py:function:: _find_cluster_groups(clusters)

         Find segment groups in the cluster group.
         And also find adjacent clusters after all the segment groups
         in this cluster group have been found.
         Annotation without a group will be skipped.

   .. autofunction:: match_segments

   .. autoclass:: _ShapeMatcher(AnnotationMatcher)
      :members:
      :show-inheritance:

      .. autofunction:: match_annotations(self, sources)

         Match segments in sources, pairwise.
         Join all segments into matching clusters.

   .. autoclass:: LineMatcher(_ShapeMatcher)
      :members:
      :show-inheritance:

      Compute inter-line area, normalize by common bbox

   .. autoclass:: mean_std
      :members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: StatsCounter
      :members:
      :private-members:
      :special-members:
      :show-inheritance:

      Implements online parallel computation of sample variance
      https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
      Needed do avoid catastrophic cancellation in floating point computations

   .. autofunction:: compute_image_statistics

   .. autofunction:: compute_ann_statistics

   .. autoclass:: DistanceComparator
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: ExactComparator
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autofunction:: match_classes

   .. autofunction:: find_unique_images

   .. autofunction:: match_items_by_image_hash

   .. autofunction:: match_items_by_id

   .. autofunction:: merge_categories
