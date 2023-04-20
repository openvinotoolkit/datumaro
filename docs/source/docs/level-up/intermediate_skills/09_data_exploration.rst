=====================================================
Level 9: Dataset Explorartion from a Query Image/Text
=====================================================


Datumaro support exploration feature to find out similar data for query among dataset. With query, the exploration result includes top-k similar data among dataset.
Through this feature, you could figure out dataset property. You could check the visualization result of exploration using `Visualizer`.

More detailed descriptions about explorer are given by :doc:`Explore <../../command-reference/context_free/explorer>`
The Python example for the usage of explorer is described in :doc:`here <../../jupyter_notebook_examples/notebooks/07_data_explorer>`.


.. tab-set::

    .. tab-item:: Python

        With Python API, we can explore similar items as below

        .. code-block:: python

            from datumaro.components.dataset import Dataset
            from datumaro.components.environment import Environment
            from datumaro.componenets.explorer import Explorer

            data_path = '/path/to/data'

            env = Environment()
            detected_formats = env.detect_dataset(data_path)

            dataset = Dataset.import_from(data_path, detected_formats[0])

            explorer = Explorer(dataset)
            query = '/path/to/image/file'
            topk = 20
            topk_result = explorer.explore_topk(query, topk)

    .. tab-item:: ProjectCLI

        With the project-based CLI, we first require to ``create`` a project by

        .. code-block:: bash

            datum project create -o <path/to/project>

        We now ``import`` data in to project through

        .. code-block:: bash

            datum project import --project <path/to/project> <path/to/data>

        We can ``explore`` similar items for the query

        .. code-block:: bash

            datum explore -q QUERY -topk TOPK_NUM -p <path/to/project>

        ``QUERY`` could be image file path, text description, list of both of them

        ``TOPK_NUM`` is an integer that you want to find the number of similar results for query

        Exploration result would be printed by log and visualized result would be saved by ``explorer.png``
