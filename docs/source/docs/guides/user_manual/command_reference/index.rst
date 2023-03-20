Command reference
=================

The command line is split into the separate _commands_ and command _contexts_.
Contexts group multiple commands related to a specific topic, e.g.
project operations, data source operations etc. Almost all the commands
operate on projects, so the ``project`` context and commands without a context
are mostly the same. By default, commands look for a project in the current
directory. If the project you're working on is located somewhere else, you
can pass the ``-p/--project <path>`` argument to the command.

> **Note**: command behavior is subject to change, so this text might be
  outdated,
> **always check the ``--help`` output of the specific command**
<!-- markdownlint-disable-line MD028 -->
> **Note**: command parameters must be passed prior to the positional arguments.

Datumaro functionality is available with the ``datum`` command.

Usage:

.. code-block::

    datum [-h] [--version] [--loglevel LOGLEVEL] [command] [command args]

Parameters:

- ``--loglevel`` (string) - Logging level, one of
  ``debug``, ``info``, ``warning``, ``error``, ``critical`` (default: ``info``)
- ``--version`` - Print the version number and exit.
- ``-h, --help`` - Print the help message and exit.

.. toctree::
   :maxdepth: 1

   checkout
   commit
   convert
   create
   describe-downloads
   detect-format
   diff
   download
   explain
   export
   filter
   generate
   info
   log
   merge
   models
   patch
   projects
   sources
   stats
   status
   transform
   util
   validate
