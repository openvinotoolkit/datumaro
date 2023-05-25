
Dataset Handler
###############

Datumaro provides the dataset import and export functionalities.

When importing multiple datasets, Datumaro helps to manipulate and merge them into a single
dataset. Since the manipulations such as reidentification, label redefinition, or filtration are
mostly the topic of transformation, we here describe how to merge two heterogeneous datasets
through `IntersectMerge`.

.. toctree::
   :maxdepth: 1
   :hidden:

   notebooks/01_merge_multiple_datasets_for_classification
   notebooks/02_merge_heterogeneous_datasets_for_detection
   notebooks/09_encrypt_dataset

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card::

      .. button-ref:: notebooks/01_merge_multiple_datasets_for_classification
         :color: primary
         :outline:
         :expand:

   .. grid-item-card::

      .. button-ref:: notebooks/02_merge_heterogeneous_datasets_for_detection
         :color: primary
         :outline:
         :expand:

   .. grid-item-card::

      .. button-ref:: notebooks/09_encrypt_dataset
         :color: primary
         :outline:
         :expand:
