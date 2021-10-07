---
title: 'How to control telemetry data collection'
linkTitle: 'How to control telemetry data collection'
description: ''
weight: 30
---

To enable the collection of telemetry data, the ISIP consent file
must exist and contain "1", otherwise telemetry will be disabled.
The ISIP file can be created/modified by OpenVINO installer
or manually and used by other OpenVINO tools.

The location of the ISIP consent file depends on the OS:
- Windows: `%localappdata%\Intel Corporation\isip`,
- Linux, MacOS: `$HOME/intel/isip`.
