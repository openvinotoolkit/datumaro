# Project Versioning

Datumaro aims to manipulate the project for both model and dataset versioning.

``` bash
datum create -o <project/dir>
datum import -p <project/dir> -f image_dir <directory/path/>
```

or, if you work with Datumaro API:

- for using with a project:

  ```python
  from datumaro.project import Project

  project = Project.init()
  project.import_source('source1', format='image_dir', url='directory/path/')
  dataset = project.working_tree.make_dataset()
  ```

- for using as a dataset:

  ```python
  from datumaro import Dataset

  dataset = Dataset.import_from('directory/path/', 'image_dir')
  ```
