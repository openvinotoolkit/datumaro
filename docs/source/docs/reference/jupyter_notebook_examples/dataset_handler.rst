Dataset handler
===============

Datumaro provides the dataset import and export functionalities.

When importing multiple datasets, Datumaro helps to manipulate and merge them into a single
dataset. Since the manipulations such as reidentification, label redefinition, or filtration are
mostly the topic of transformation, we here describe how to merge two heterogeneous datasets
through `IntersectMerge`.

Jupyter Notebook Example
------------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   notebooks/01_merge_multiple_datasets_for_classification
   notebooks/02_merge_heterogeneous_datasets_for_detection
