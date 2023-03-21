search
======

## Search Datasets

This command search similar data results for query on dataset. The result includes top-k similar data among target dataset and the visualization of result is saved as png file. This tool is supposed to help users to figure out dataset property.

Currently, hash of each data is computed based on the CLIP ([article](https://arxiv.org/abs/2103.00020)), which could support both image and text modality. Supported model format is Openvino IR and those are uploaded in [openvinotoolkit storage](https://storage.openvinotoolkit.org/repositories/datumaro/models/). When you call `Searcher` class, hash of whole dataset is started to compute. Through CLIP, we extracted feature of data, converted to binary value and pack the elements into bits. Each hash information is saved as `HashKey` in annotations.

To search similar data in dataset, you need to set query first. Query could be image, text, list of images and list of texts. The query does not need to be an image that exists in the dataset. You can put in any data that you want to search similar dataset. And you need to set top-k that how much you want to find similar data. For single query, we computed hamming distance of hash between whole dataset and query. And we sorted those distance and select top-k data which have short distance. For list query, we repeated computing distance for each query and select top-k data based on distance among all dataset.

The command can be applied to a dataset. And if you want to use multiple dataset as database, you could use merged dataset. The current project (`-p/--project`) is also used a context for plugins, so it can be useful for dataset paths having custom formats. When not specified, the current project's working tree is used. To save visualized result (`-s/--save`) is turned on as default. This visualized result is based on [Visualizer](https://openvinotoolkit.github.io/datumaro/docs/python-api/python-api-examples/visualizer/).

Usage:
``` bash
datum search [-q <path/to/image.jpg> or <text_query>] [-topk TOPK]
```

Parameters:
- `-q, --query` (string) - Image path or text to use as query.
- `-topk` (int) - Number how much you want to find similar data.
- `-p, --project` (string) - Directory of the project to operate on (default: current directory).
- `-s, --save` (bool) - Save visualized result of similar dataset.

Examples:
- Use image query
```bash
datum create <...>
datum search -q path/to/image.jpg -topk 10
```
- Use text query
```bash
datum create <...>
datum search -q elephant -topk 10
```
- Use list of images query
```bash
datum create <...>
datum search -q path/to/image1.jpg path/to/image2.jpg path/to/image3.jpg -topk 50
```
- Use list of texts query
```bash
datum create <...>
datum search -q motorcycle bus train -topk 50
```
