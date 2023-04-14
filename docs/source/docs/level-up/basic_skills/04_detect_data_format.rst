===================================================
Level 4: Detect Data Format from an Unknown Dataset
===================================================

Datumaro provides a function to detect the format of a dataset before importing data. This can be
useful in cases where information about the original format of the data has been lost or is unclear.
With this function, users can easily identify the format and proceed with appropriate data
handling processes.

Detect data format
==================

.. tabbed:: CLI

  .. code-block:: bash

    datum detect-format <path/to/data>

  The printed format can be utilized as `format` argument when importing a dataset as following the
  :ref:`previous level <Level 3: Data Import and Export>`.

.. tabbed:: Python

  .. code-block:: python

      from datumaro.components.environment import Environment

      data_path = '/path/to/data'

      env = Environment()

      detected_formats = env.detect_dataset(data_path)


  (Optional) With the detected format, we can import the dataset as below.

  .. code-block:: python

      from datumaro.components.dataset import Dataset

      dataset = Dataset.import_from(data_path, detected_formats[0])
