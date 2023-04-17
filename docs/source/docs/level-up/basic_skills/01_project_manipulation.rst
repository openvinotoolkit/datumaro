=============================
Level 1: Project Manipulation
=============================

Projects are intended for complex use of Datumaro CLI commands.
If you want to know more about the concept and meaning of the project, please see :ref:`Projects`.
You might be able to skip this page, if you plan to do various dataset management tasks using the Datumaro Python API rather than CLI commands.
However, even for the Python API, you have to get back to the project if you need dataset versioning.

Create a project
================

.. tab-set::

  .. tab-item:: ProjectCLI

    You can create a project using the following command.

    .. code-block:: bash

      datum project create [-h] [-o DST_DIR] [--overwrite]

    The default command create a project in the current directory,
    but you can specify a specific directory path with ``-o DST_DIR``.
    Use ``--overwrite`` if you need to overwrite an existing project.

Import a dataset to the project
===============================

.. tab-set::

  .. tab-item:: ProjectCLI

    .. code-block:: bash

      datum project import [-h] [-n NAME] -f FORMAT [-r RPATH] [--no-check] [-p PROJECT_DIR] url ...

    ``-n NAME`` is the name of dataset you can assign. if you don't give it, Datumaro generates it automatically
    (``source-0``, ``source-1``, ...). ``-f FORMAT`` is the format of the dataset to be imported to the project.
    The default behavior adds the dataset to the project in the current working directory,
    but You can specify a specific directory path of the project with ``-p PROJECT_DIR``.
    ``url`` should be given for the path of the dataset to import.

    For example, the following command is an example of importing the :ref:`Cityscapes` dataset to the project with naming it as ``my-dataset``.

    .. code-block:: bash

      datum project import -n my-dataset --format cityscapes -p <path/to/project> <path/to/cityscapes>

Remove a dataset from the project
=================================

.. tab-set::

  .. tab-item:: ProjectCLI

    .. code-block:: bash

      datum project remove [-h] [--force] [--keep-data] [-p PROJECT_DIR] names [names ...]

    ``names`` are the names of the imported dataset you want to remove it from the project.
    If you want to remove only the metadata and keep the actual data, use ``--keep-data``.

    For example, the following command is an example of removing the :ref:`Cityscapes` dataset from the previous step.

    .. code-block:: bash

      datum project remove my-dataset

Print dataset information
=========================

.. tab-set::

  .. tab-item:: ProjectCLI

    Print dataset information added in the project.

    .. code-block:: bash

      datum project dinfo [-h] [--all] [-p PROJECT_DIR] [revpath]

    ``revpath`` is either a dataset path or a revision path. For more information on the revision, please see
    :ref:`Level 11: Project Versioning`. ``--all`` directive shows all the dataset information of your project.

    For example, the following command is an example of printing information about the :ref:`Cityscapes` dataset from the previous step.

    .. code-block:: bash

      datum project dinfo my-dataset

Add model to project
====================

.. tab-set::

  .. tab-item:: ProjectCLI

    .. code-block:: bash

      datum model add [-h] [-n NAME] -l LAUNCHER [--copy] [--no-check] [-p PROJECT_DIR] ...

    Add an AI model into a project. The model requires an inference launcher for its model format.
    Currently, we only support `OpenVINO™ <https://github.com/openvinotoolkit/openvino>`_ launcher.
    Each launcher has its own options, which are passed after the ``--`` separator, pass ``-- -h`` for more info.
    To copy the model files into the project directory, you can use ``--copy`` argument.

    Here is an example to add an `OpenVINO™ <https://github.com/openvinotoolkit/openvino>`_ model to the project.

    .. code-block:: bash

      datum model add -n my-model -l openvino -- -d <path/to/model.xml> -w <path/to/model.bin> -i <path/to/interpreter.py>

    .. note::
      In addition to entering the path to the model weights (``-w WEIGHTS``) and metafiles (``-d DESCRIPTION``),
      you must enter the interpreter file path (``-i INTERPRETER``) written in Python to interpret that model output as well.

      .. collapse:: An example of the interpreter (``ssd_mobilenet_coco_detection_interp.py``)

        .. literalinclude:: ../../../../../datumaro/plugins/openvino_plugin/samples/ssd_mobilenet_coco_detection_interp.py
          :language: python

Remove model from project
=========================

.. tab-set::

  .. tab-item:: ProjectCLI

    .. code-block:: bash

      datum model remove [-h] [-p PROJECT_DIR] name

    To remove the model added in your project, you can use this command. It requires a name of the added model.

    For example, the model added in the previous step has its name as ``my-model``. We can remove this model as follows.

    .. code-block:: bash

      datum model add -n my-model -l openvino -- -d <path/to/model.xml> -w <path/to/model.bin> -i <path/to/interpreter.py>

Print project information
=========================

.. tab-set::

  .. tab-item:: ProjectCLI

    Print an overall information of the project.

    .. code-block:: bash

      datum project pinfo [-h] [-p PROJECT_DIR] [revision]

    ``revision`` means the version of you project (:ref:`Level 11: Project Versioning`).
    If it is not given, the latest revision of the project is shown.
