---
title: 'Documentation'
linkTitle: 'Documentation'
weight: 20
no_list: true
menu:
  main:
    weight: 20
tags: [ 'Schemes', ]
---

## Dataset Management Framework (Datumaro)

A framework and CLI tool to build, transform, and analyze datasets.

<div class="text-center">

```mermaid
flowchart LR
    datasets[(VOC dataset<br/>+<br/>COCO datset<br/>+<br/>CVAT annotation)]
    datumaro{Datumaro}
    dataset[dataset]
    annotation[Annotation tool]
    training[Model training]
    publication[Publication, statistics etc]
    datasets-->datumaro
    datumaro-->dataset
    dataset-->annotation & training & publication
```

</div>


<section id="docs">

{{< blocks/section color="docs" >}}

{{% blocks/feature icon="fa-sign-in-alt" title="[Getting started](/docs/getting_started/)" %}}

Basic information and sections needed for a quick start.

{{% /blocks/feature %}}

{{% blocks/feature icon="fa-book" title="[User Manual](/docs/user-manual)" %}}

This section contains documents for Datumaro users.

{{% /blocks/feature %}}

{{% blocks/feature icon="fa-code" title="[Developer Guide](/docs/developer-guide/)" %}}

Documentation for Datumaro developers.

{{% /blocks/feature %}}

{{< /blocks/section >}}

</section>
