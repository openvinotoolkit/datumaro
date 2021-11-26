project module
--------------

.. automodule:: datumaro.components.project

   .. autoclass:: IgnoreMode
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: ProjectSources
      :members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: BuildStageType
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: Pipeline
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. automethod:: _get_subgraph

   .. autoclass:: ProjectBuilder
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

      .. automethod:: _init_pipeline

   .. autoclass:: ProjectBuildTargets
      :members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: GitWrapper
      :members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. autoclass:: Revision
      :members:
      :undoc-members:
      :private-members:
      :special-members:
      :show-inheritance:

   .. py:data:: Revision = NewType('Revision', str)

      a commit hash or a named reference

   .. py:data:: ObjectId = NewType('ObjectId', str)

      a commit or an object hash

   .. autoclass:: Tree
      :members:
      :private-members:
      :special-members:
      :show-inheritance:

      can be:
         - attached to the work dir
         - attached to a revision

   .. autoclass:: DiffStatus
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: Project
      :members:
      :private-members:
      :special-members:
      :show-inheritance:
