---
title: 'Dataset Management Framework Documentation'
linkTitle: 'Documentation'
weight: 20
no_list: true
---

Welcome to the documentation for the Dataset Management Framework (Datumaro).

The Datumaro is a free framework and CLI tool for building, transforming,
and analyzing datasets.
It is developed and used by Intel to build, transform, and analyze annotations
and datasets in a large number of [supported formats](/docs/user-manual/supported-formats/).

Our documentation provides information for AI researchers, developers,
and teams, who are working with datasets and annotations.

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

<!--lint disable maximum-line-length-->
<section id="docs">

{{< blocks/section color="docs" >}}

{{% blocks/feature icon="fa-sign-in-alt" title="[Getting started](/docs/getting_started/)" %}}

Basic information and sections needed for a quick start.

{{% /blocks/feature %}}

{{% blocks/feature icon="fa-book" title="[User Manual](/docs/user-manual/)" %}}

This section contains documents for Datumaro users.

{{% /blocks/feature %}}

{{% blocks/feature icon="fa-code" title="[Developer Manual](/docs/developer_manual/)" %}}

Documentation for Datumaro developers.

{{% /blocks/feature %}}

{{< /blocks/section >}}

</section>
<!--lint enable maximum-line-length-->
