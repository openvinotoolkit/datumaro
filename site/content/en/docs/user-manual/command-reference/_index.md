---
title: 'Command reference'
linkTitle: 'Command reference'
description: ''
weight: 6
---

<div class="text-center large-scheme-two">

```mermaid
%%{init { 'theme':'neutral' }}%%
flowchart LR
  d(("#0009; datum #0009;")):::mainclass
  m(model):::nofillclass
  p(project):::nofillclass
  s(source):::nofillclass

  d===m
    m===m_add[add]:::hideclass
    m===m_info[info]:::hideclass
    m===m_remove[remove]:::hideclass
    m===m_run[run]:::hideclass
  d===p
    p===p_info[info]:::hideclass
    p===p_migrate[migrate]:::hideclass
  d===s
    s===s_add[add]:::hideclass
    s===s_info[info]:::hideclass
    s===s_remove[remove]:::hideclass
  d====_add[add]:::filloneclass
  d====_create[create]:::filloneclass
  d====_describe_downloads[describe-downloads]:::filloneclass
  d====_detect_format[detect-format]:::filloneclass
  d====_download[download]:::filloneclass
  d====_export[export]:::filloneclass
  d====_import[import]:::filloneclass
  d====_info[info]:::filloneclass
  d====_remove[remove]:::filloneclass
  d====_generate[generate]:::filloneclass
  d====_filter[filter]:::filltwoclass
  d====_transform[transform]:::filltwoclass
  d====_diff[diff]:::fillthreeclass
  d====_explain[explain]:::fillthreeclass
  d====_merge[merge]:::fillthreeclass
  d====_patch[patch]:::fillthreeclass
  d====_stats[stats]:::fillthreeclass
  d====_validate[validate]:::fillthreeclass
  d====_checkout[checkout]:::fillfourclass
  d====_commit[commit]:::fillfourclass
  d====_log[log]:::fillfourclass
  d====_status[status]:::fillfourclass

  classDef nofillclass fill-opacity:0;
  classDef hideclass fill-opacity:0,stroke-opacity:0;
  classDef filloneclass fill:#CCCCFF,stroke-opacity:0;
  classDef filltwoclass fill:#FFFF99,stroke-opacity:0;
  classDef fillthreeclass fill:#CCFFFF,stroke-opacity:0;
  classDef fillfourclass fill:#CCFFCC,stroke-opacity:0;
```

</div>

The command line is split into the separate _commands_ and command _contexts_.
Contexts group multiple commands related to a specific topic, e.g.
project operations, data source operations etc. Almost all the commands
operate on projects, so the `project` context and commands without a context
are mostly the same. By default, commands look for a project in the current
directory. If the project you're working on is located somewhere else, you
can pass the `-p/--project <path>` argument to the command.

> **Note**: command behavior is subject to change, so this text might be
  outdated,
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
