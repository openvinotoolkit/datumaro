=============
Level 8: Dataset Refinement with Validation Report
=============

Datumaro aims to refine data

``` bash
datum create -o <project/dir>
datum import -p <project/dir> -f image_dir <directory/path/>
```

or, if you work with Datumaro API:

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
