---
title: 'Plugin example'
linkTitle: 'Plugin example'
description: ''
weight: 10
---

<!--lint disable fenced-code-flag-->

```
datumaro/plugins/
- my_plugin1/file1.py
- my_plugin1/file2.py
- my_plugin2.py
```

<!--lint enable fenced-code-flag-->

`my_plugin1/file2.py` contents:

```python
from datumaro.components.extractor import Transform, CliPlugin
from .file1 import something, useful

class MyTransform(Transform, CliPlugin):
    NAME = "custom_name" # could be generated automatically

    """
    Some description. The text will be displayed in the command line output.
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('-q', help="Very useful parameter")
        return parser

    def __init__(self, extractor, q):
        super().__init__(extractor)
        self.q = q

    def transform_item(self, item):
        return item
```

`my_plugin2.py` contents:

```python
from datumaro.components.extractor import Extractor

class MyFormat: ...
class _MyFormatConverter(Converter): ...
class MyFormatExtractor(Extractor): ...

exports = [MyFormat] # explicit exports declaration
# MyFormatExtractor and _MyFormatConverter won't be exported
```
