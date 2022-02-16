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
      :show-inheritance:

      .. autofunction:: __init__

   .. autoclass:: BuildStageType
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: Pipeline
      :members:
      :undoc-members:
      :private-members:
      :show-inheritance:

      .. automethod:: _get_subgraph

      .. autofunction:: __init__

   .. autoclass:: ProjectBuilder
      :members:
      :undoc-members:
      :private-members:
      :show-inheritance:

      .. automethod:: _init_pipeline

   .. autoclass:: ProjectBuildTargets
      :members:
      :private-members:
      :show-inheritance:

   .. autoclass:: GitWrapper
      :members:
      :private-members:
      :show-inheritance:

   .. autoclass:: Tree
      :members:
      :private-members:
      :show-inheritance:

      can be:
         - attached to the work dir
         - attached to a revision

   .. autoclass:: DiffStatus
      :members:
      :undoc-members:
      :show-inheritance:

   .. autoclass:: Revision

      a commit hash or a named reference

   .. autodata:: ObjectId

      a commit or an object hash

   .. autoclass:: Project
      :members:
      :private-members:
      :special-members:
      :show-inheritance:
