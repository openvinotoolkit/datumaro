---
title: 'Writing a plugin'
linkTitle: 'Writing a plugin'
description: ''
weight: 9
---

A plugin is a Python module with any name, which exports some symbols. Symbols,
starting with `_` are not exported by default. To export a symbol,
inherit it from one of the special classes:

```python
from datumaro.components.extractor import Importer, Extractor, Transform
from datumaro.components.launcher import Launcher
from datumaro.components.converter import Converter
```

The `exports` list of the module can be used to override default behaviour:
```python
class MyComponent1: ...
class MyComponent2: ...
exports = [MyComponent2] # exports only MyComponent2
```

There is also an additional class to modify plugin appearance in command line:

```python
from datumaro.components.cli_plugin import CliPlugin

class MyPlugin(Converter, CliPlugin):
  """
    Optional documentation text, which will appear in command-line help
  """

  NAME = 'optional_custom_plugin_name'

  def build_cmdline_parser(self, **kwargs):
    parser = super().build_cmdline_parser(**kwargs)
    # set up argparse.ArgumentParser instance
    # the parsed args are supposed to be used as invocation options
    return parser
```
