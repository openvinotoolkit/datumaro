---
title: 'Command reference'
linkTitle: 'Command reference'
description: ''
weight: 6
---

<div class="text-center large-scheme">

```mermaid
flowchart LR
  d(("#0009; datum #0009;")):::mainclass
  s(source):::nofillclass
  m(model):::nofillclass

  d===s
    s===id1[add]:::hideclass
    s===id2[remove]:::hideclass
    s===id3[info]:::hideclass
  d===m
    m===id4[add]:::hideclass
    m===id5[remove]:::hideclass
    m===id6[run]:::hideclass
    m===id7[info]:::hideclass
  d====str1[create]:::filloneclass
  d====str2[add]:::filloneclass
  d====str3[remove]:::filloneclass
  d====str4[export]:::filloneclass
  d====str5[info]:::filloneclass
  d====str6[transform]:::filltwoclass
  d====str7[filter]:::filltwoclass
  d====str8[diff]:::fillthreeclass
  d====str9[merge]:::fillthreeclass
  d====str10[validate]:::fillthreeclass
  d====str11[explain]:::fillthreeclass
  d====str12[stats]:::fillthreeclass
  d====str13[commit]:::fillfourclass
  d====str14[checkout]:::fillfourclass
  d====str15[status]:::fillfourclass
  d====str16[log]:::fillfourclass

  classDef nofillclass fill-opacity:0;
  classDef hideclass fill-opacity:0,stroke-opacity:0;
  classDef filloneclass fill:#CCCCFF,stroke-opacity:0;
  classDef filltwoclass fill:#FFFF99,stroke-opacity:0;
  classDef fillthreeclass fill:#CCFFFF,stroke-opacity:0;
  classDef fillfourclass fill:#CCFFCC,stroke-opacity:0;
```

</div>

The command line is split into the separate *commands* and command *contexts*.
Contexts group multiple commands related to a specific topic, e.g.
project operations, data source operations etc. Almost all the commands
operate on projects, so the `project` context and commands without a context
are mostly the same. By default, commands look for a project in the current
directory. If the project you're working on is located somewhere else, you
can pass the `-p/--project <path>` argument to the command.

> **Note**: command behavior is subject to change and might be outdated,
> **always check the `--help` output of the specific command**

> **Note**: command parameters must be passed prior to the positional arguments.

Datumaro functionality is available with the `datum` command.

Usage:
``` bash
datum [-h] [--version] [--loglevel LOGLEVEL] [command] [command args]
```

Parameters:
- `--loglevel` (string) - Logging level, one of
  `debug`, `info`, `warning`, `error`, `critical` (default: `info`)
- `--version` - Print the version number and exit.
- `-h, --help` - Print the help message and exit.
