operations module
-----------------

.. automodule:: datumaro.components.operations

   .. py:class:: IntersectMerge

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

   .. py:class:: _ShapeMatcher(AnnotationMatcher)

      .. py:function:: match_annotations(self, sources)

         Match segments in sources, pairwise.
         Join all segments into matching clusters.

   .. py:class:: LineMatcher(_ShapeMatcher)

      Compute inter-line area, normalize by common bbox

   .. autoclass:: mean_std
      :members:
      :private-members:
      :special-members:

   .. py:class:: StatsCounter

      Implements online parallel computation of sample variance
      https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
      Needed do avoid catastrophic cancellation in floating point computations
