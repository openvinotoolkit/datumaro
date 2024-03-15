# Transform

If you wish to apply transformations to two datasets, simply click on the transform tab. Upon clicking the transform tab, you'll be presented with the following screen:

![Transform Tab](../../../../images/gui/multiple/transform_tab.png)

You can apply transformations separately to each dataset. Simply select the desired columns of the dataset to apply the transformations.
You can check the subsets and category info of each dataset in the 'Transform' tab as follows. After applying each transform, you can verify if it has been applied correctly using this information.

DatumaroApp offers the following types of transform manipulations:

**Category Management**:
- **_[Label Remapping](#label-remapping)_**:  Renames dataset items by regular expression.

**Subset Management**:
- **_[Aggregation](#aggregation)_**: Aggregates subsets into one subset.
- **_[Split](#split)_**: Splits the dataset into subsets for classification, detection, segmentation, or re-identification.
- **_[Subset Rename](#rename)_**: Renames and removes subsets.

**Item Management**:
- **_[Reindexing](#rename)_**: Renames dataset items with numbers.
- **_[Filtration](#filtration)_**: Extract a sub-dataset from a dataset through some condition.
- **_[Remove](#remove)_**: Removes specific images or annotations.
- **_[Auto-correction](#auto-correction)_**: Correct the dataset from a validation report.

## Category Management
### Label Remapping
![Label Remapping](../../../../images/gui/multiple/transform_label_remapping.png)

Renames labels in the dataset. For more details, please refer to the [remap_labels](../../command-reference/context_free/transform.md/#remap_labels) CLI.

To remap labels in the dataset, follow these steps:
1. **Identify the Label**: If there is no information available for label remapping between the two datasets, a notification will appear. In such cases, please first navigate to the **_'Compare'_** tab to review the label remapping between the two datasets.
2. **Initiate Remapping**: Once the labels for remapping are prepared, confirm them and click the **_"Do Label Remap"_** button to proceed.
3. **Remove Unselected Labels**: If you wish to remove all labels except for the selected ones, please toggle the **_"Delete unselected labels"_** option.

## Subset Management
### Aggregation
![Aggregation](../../../../images/gui/multiple/transform_aggregation.png)

Merges subsets of the dataset into one. For more details, please refer to [data_aggregation](../../level-up/intermediate_skills/05_data_aggregation.rst).

To aggregate subsets in the dataset, adhere to these steps:
1. **Choose Subsets**: Select the subsets you wish to aggregate.
2. **Name the Aggregated Subset**: Provide a name for the aggregated subset in the designated field.
3. **Initiate Aggregation**: After configuring the subsets and naming the aggregated subset, click the **_"Do aggregation"_** button to proceed.

### Split
![Split](../../../../images/gui/multiple/transform_split.png)
![Add Subset](../../../../images/gui/multiple/transform_split_add_subset.png)

Combines all subsets of the dataset into one and then divides them into multiple subsets. For more details, please refer to the [random_split](../../command-reference/context_free/transform.md#random_split) CLI.

To split the dataset into multiple subsets, follow these steps:
1. **Add Subsets**: Add subsets to split by specifying their names and ratios. Ensure that the total ratio of subsets equals 1 for the split to proceed. The default values are set to 0.5, 0.2, and 0.3 for train, val, and test subsets respectively.
2. **Initiate Split**: After configuring the subsets and their ratios, click the **_"Do split"_** button to proceed with the split.

### Rename
![Subset Rename](../../../../images/gui/multiple/transform_subset_rename.png)

Renames the subsets of the dataset. For more details, please refer to [map_subsets](../../command-reference/context_free/transform.md#map_subsets) CLI.

To rename a subset, follow these steps:
1. **Select Subset**: Choose the subset you want to rename.
2. **Specify New Name**: Enter the new name for the selected subset.
3. **Initiate Rename**: After configuring the subset name, click the **_"Do Subset Rename"_** button to proceed.

### Reindex
![Reindex](../../../../images/gui/multiple/transform_reindexing.png)

Reindexing involves changing the IDs of items in the dataset. For detailed information, please refer to the [reindex](../../command-reference/context_free/transform.md#reindex) CLI.

To reindex items, follow these steps:
1. **Specify Starting Index**: Enter the starting index for reindexing. The default is 0. If you've entered a different value, press the **_"Set IDs from 0"_** button to apply it.
2. **Reindex Based on Media Name**: If you want to reindex based on the media name, press the **_"Set IDs with media name"_** button.
3. **Verify Reindexing**: Check the item ID table to ensure that the reindexing has been done correctly.

Please review the item ID table below to ensure that the reindexing has been done correctly.
![Reindex from 0](../../../../images/gui/multiple/transform_reindexing_set_ids_from_0.png)

If you prefer to reindex based on media name, click the **_"Set IDs with media name"_** button.

### Filtration
![Filtration](../../../../images/gui/multiple/transform_filtration.png)

Filtering involves extracting sub-datasets from the dataset using filters. For detailed information, please refer to the [filter](../../command-reference/context_free/filter.md) CLI.

DatumaroApp offers three filtering modes: `items`, `annotations`, and `item+annotations`. After selecting the desired mode, you can write an XML expression describing how the filtering should be performed.

For example:
- If you want to extract images where the height is greater than the width, you can use the expression `/item[image/width < image/height]`.
- If you want to extract only the images from the 'train' subset, you can use the expression `item[subset="train"]`.
- If you want to extract images with annotations labeled as 'cat', you can switch the filtering mode to 'annotation' and use the expression `/item/annotation[label="cat"]`.
Feel free to customize the filter expression as needed.

If you wish to examine the detailed information of each item in the dataset before writing the filter, you can toggle the 'Show XML Representation' to access the detailed information of each item.

![Show XML Representation](../../../../images/gui/multiple/transform_filtration_show_xml.png)

Once you have prepared the filter expression, click the **_"Filter Dataset"_** button to proceed.

### Remove
![Remove](../../../../images/gui/multiple/transform_remove.png)
Removing involves deleting specific items or annotations within the dataset. For detailed information, please refer to the [remove_images](../../command-reference/context_free/transform.md#remove_images) and [remove_annotations](../../command-reference/context_free/transform.md#remove_annotations) CLI.

To remove items or annotations, follow these steps:
1. **Select a Subset**: Choose the subset where you want to apply the removal.
2. **Choose an Item**: Select the item you wish to remove from the subset.
3. **View Annotation Information**: Use the visualizer below to view information about the annotations for the selected item. By default, all annotations are displayed. To view specific annotations, simply select them.
4. **Remove Item**: If you want to remove the entire item, press the **_"Remove item"_** button.
5. **Remove Annotation**: If you want to remove a specific annotation, select it and press the **_"Remove annotation"_** button.
    ![Remove Annotation](../../../../images/gui/multiple/transform_remove_annotation.png)

### Auto Correction
![Auto Correction](../../../../images/gui/multiple/transform_auto_correction.png)

Correcting involves assessing the characteristics of the dataset and automatically correcting any inaccuracies. The dataset is refined based on the validation report, rejecting undefined labels, missing annotations, and outliers.

For detailed information on validation, please refer to the [validate](../../command-reference/context_free/validate.md) CLI.

To correct the dataset, follow these steps:
1. **Select a Task**: Choose the task for which you want to validate the dataset. The validation report will be displayed below.
2. **Correct the Dataset**: If you want to correct the dataset based on the report, press the **_"Correct a dataset"_** button.

![Correct Dataset](../../../../images/gui/multiple/transform_auto_correction_correct_dataset.png)
You can see how the dataset has been refined. In this example, corrections have been made for items classified as UndefinedAttribute, FarFromLabelMean, and MissingAnnotation.

For more detailed functionalities of transformations, please follow the instructions provided [here](../../command-reference/context_free/transform.md) to utilize the CLI.
