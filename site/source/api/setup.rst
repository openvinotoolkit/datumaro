Setup
=====

.. py:function:: setup.find_version(project_dir=None)

   Return a version of the Datumaro

.. py:function:: setup.parse_requirements

   Parsing a list from a text file

.. py:data:: DEFAULT_REQUIREMENTS

   Variable containing a list of default required packages

.. py:data:: CORE_REQUIREMENTS

   Variable containing a list of core required packages

.. py:function:: setup.open

   Sets the :data:`long_description` variable to contain the path to the package description file

.. py:function:: setup.setuptools.setup

   Installation Datumaro with use module :mod:`setuptools.setup`

   :param name: name of the package.

   :param version: sets with use :func:`setup.find_version`.
   :param author: project author.
   :param author_email: author email.
   :param description: description of the package.
   :param long_description: readme file opened :func:`setup.open`.
   :param long_description_content_type: type of the long decription.
   :param url: main package url.
   :param packages:
      specifies the list of packages using :mod:`setuptools.find_packages`.
   :param classifiers:
      sets the package classification
      by `Programming Language`, `License`, `Operating System`.
   :param python_requires: sets the required python version.
   :param install_requires:
      sets the list of the third-party required modules сan be
      set to a variable :data:`CORE_REQUIREMENTS`.
   :param extras_require:
      sets the require by `tf`, `tf-gpu`, `default`. Parameter `default`
      сan be set to a variable :data:`DEFAULT_REQUIREMENTS`.
   :param entry_points: point in a program where the execution of a program begins,
      and where the program has access to command line arguments.
   :param include_package_data: include package data

Description of third-party modules
----------------------------------

.. automodule:: setuptools
   :members: setup, find_packages