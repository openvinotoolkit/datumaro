.. Datumaro API documentation documentation master file, created by
   sphinx-quickstart on Fri Oct 22 20:11:30 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Datumaro API documentation!
======================================

.. toctree::
   :numbered:
   :maxdepth: 5
   :glob:

   Main documentation <https://openvinotoolkit.github.io/datumaro/docs/>
   *
   
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Installation
------------

To install datumaro, use setur.py
The `setuptools.setup` contains options for installing Datumaro.

.. autoclass:: setuptools.setup
   :members:

The version is determined using the function ``find_version``. The version is taken from the file `datumaro/version.py`

.. autofunction:: find_version
   :members:
