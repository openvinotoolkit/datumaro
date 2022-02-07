dataset module
--------------

.. automodule:: datumaro.components.dataset

   .. autodata:: IDataset

   .. autoclass:: DatasetItemStorage
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: DatasetItemStorageDatasetView
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: ItemStatus
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: DatasetPatch
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      The purpose of this class is to indicate that the input dataset is
      a patch and autofill patch info in Converter

   .. autoclass:: DatasetSubset
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: DatasetStorage
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. automethod:: _iter_init_cache

         Merges the source, source transforms and patch, caches the result
         and provides an iterator for the resulting item sequence.

         If iterated in parallel, the result is undefined.
         If storage is changed during iteration, the result is undefined.

         Cases:
            1. Has source and patch
            2. Has source, transforms and patch
               a. Transforms affect only an item (i.e. they are local)
               b. Transforms affect whole dataset

         The patch is always applied on top of the source / transforms stack.

   .. autoclass:: Dataset
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autofunction:: eager_mode
