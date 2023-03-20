NYU Depth Dataset V2
====================

Format specification
--------------------

The original NYU Depth Dataset V2 is available
`here <https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html>`_.

Supported annotation types:

- ``DepthAnnotation``

Import NYU Depth Dataset V2
---------------------------

The NYU Depth Dataset V2 is available for free `download <http://datasets.lids.mit.edu/nyudepthv2/>`_.

A Datumaro project with a NYU Depth Dataset V2 source can be created in the following way:

.. code-block::

    datum create
    datum import --format nyu_depth_v2 <path/to/dataset>

It is also possible to import the dataset using Python API:

.. code-block::

    import datumaro as dm

    dataset = dm.Dataset.import_from('<path/to/dataset>', 'nyu_depth_v2')

NYU Depth Dataset V2 directory should have the following structure:

Dataset/
    ├── 1.h5
    ├── 2.h5
    ├── 3.h5
    └── ...

To make sure that the selected dataset has been added to the project, you can
run ``datum project info``, which will display the project information.

Examples
--------

Examples of using this format from the code can be found in
`the format tests <https://github.com/openvinotoolkit/datumaro/tree/develop/tests/unit/test_nyu_depth_v2_format.py>`_
