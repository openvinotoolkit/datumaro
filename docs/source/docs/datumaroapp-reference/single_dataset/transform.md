# Transform

If you want to apply transformations to the dataset, you can press the transform tab. Pressing the transform tab will display the following screen:

![Transform Tab](../../../../images/gui/single/transform_tab.png)

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
- **_[Near Duplicate Removal](#near-duplicate-removal)_**: Removes near-duplicated images in subset.

## Category Management
### Label Remapping
![Label Remapping](../../../../images/gui/single/transform_label_remapping.png)

This involves renaming labels in the dataset. For detailed instructions, refer to the [remap_labels](../../command-reference/context_free/transform.md/#remap_labels) CLI.

To remap labels in the dataset, follow these steps:
1. **Identify the Label**: The table presents the dataset's labels along with their counts. Locate the label you wish to remap.
2. **Update the Label**: In the **dst** column, replace the existing label name with the desired new name. Ensure to check the checkbox corresponding to the label.
3. **View Empty Labels**: Toggle the **_"Hide empty labels"_** switch to display all empty labels, if necessary.
4. **Initiate Remapping**: After configuring the desired label changes, click the **_"Do Label Remap"_** button to proceed.
5. **Remove Unselected Labels**: If you intend to delete all unselected labels, activate the **_"Delete unselected labels"_** toggle.
6. **Verify Changes**: Confirm that the changes have been applied by checking the Category info.

## Subset Management
### Aggregation
![Aggregation](../../../../images/gui/single/transform_aggregation.png)

This involves combining subsets of the dataset into one. For detailed instructions, refer to the [data_aggregation](../../level-up/intermediate_skills/05_data_aggregation.rst).

To aggregate subsets in the dataset, adhere to these steps:
1. **Choose Subsets**: Select the subsets you wish to aggregate.
2. **Name the Aggregated Subset**: Provide a name for the aggregated subset in the designated field.
3. **Initiate Aggregation**: After configuring the subsets and naming the aggregated subset, click the **_"Do aggregation"_** button to proceed.
4. **Verify Changes**: Confirm that the changes have been applied by checking the Subset info.

### Split
![Split](../../../../images/gui/single/transform_split.png)
![Add Subset](../../../../images/gui/single/transform_split_add_subset.png)

This involves splitting all subsets of the dataset into several subsets. For detailed instructions, refer to the [random_split](../../command-reference/context_free/transform.md#random_split) CLI.

To split the dataset into multiple subsets, follow these steps:
1. **Add Subsets**: Add subsets to split by specifying their names and ratios. Ensure that the total ratio of subsets equals 1 for the split to proceed. The default values are set to 0.5, 0.2, and 0.3 for train, val, and test subsets respectively.
2. **Initiate Split**: After configuring the subsets and their ratios, click the **_"Do split"_** button to proceed with the split.
3. **Verify Changes**: Confirm that the changes have been applied by checking the Subset info.

### Rename
![Subset Rename](../../../../images/gui/single/transform_subset_rename.png)

This involves renaming subsets of the dataset. For detailed instructions, refer to the [map_subsets](../../command-reference/context_free/transform.md#map_subsets) CLI.

To rename a subset, follow these steps:

1. **Select Subset**: Choose the subset you want to rename.
2. **Specify New Name**: Enter the new name for the selected subset.
3. **Initiate Rename**: After configuring the subset name, click the **_"Do Subset Rename"_** button to proceed.
4. **Verify Changes**: Confirm that the changes have been applied by checking the Subset info.

## Item Management
### Reindex
![Reindex](../../../../images/gui/single/transform_reindexing.png)

This involves renaming item IDs in the dataset. For detailed instructions, refer to the [reindex](../../command-reference/context_free/transform.md#reindex) CLI.

To reindex items, follow these steps:
1. **Specify Starting Index**: Enter the starting index for reindexing. The default is 0. If you've entered a different value, press the **_"Set IDs from 0"_** button to apply it.
2. **Reindex Based on Media Name**: If you want to reindex based on the media name, press the **_"Set IDs with media name"_** button.
3. **Verify Reindexing**: Check the item ID table to ensure that the reindexing has been done correctly.
    ![Reindex from 0](../../../../images/gui/single/transform_reindexing_set_ids_from_0.png)

### Filtration
![Filtration](../../../../images/gui/single/transform_filtration.png)

Filtering involves extracting sub-datasets from the dataset using filters. For detailed information, please refer to the [filter](../../command-reference/context_free/filter.md) CLI.

DatumaroApp offers three filtering modes: `items`, `annotations`, and `item+annotations`. After selecting the desired mode, you can write an XML expression describing how the filtering should be performed.

For example:
- If you want to extract images where the height is greater than the width, you can use the expression `/item[image/width < image/height]`.
- If you want to extract only the images from the 'train' subset, you can use the expression `item[subset="train"]`.
- If you want to extract images with annotations labeled as 'cat', you can switch the filtering mode to 'annotation' and use the expression `/item/annotation[label="cat"]`.
Feel free to customize the filter expression as needed.

![Show XML Representation](../../../../images/gui/single/transform_filteration_show_xml.png)
If you wish to examine the detailed information of each item in the dataset before writing the filter, you can toggle the **_"Show XML Representation"_** to access the detailed information of each item.
Once everything is set up, press the **_"Filter Dataset_"** button to proceed.

### Remove
![Remove](../../../../images/gui/single/transform_remove.png)

This involves removing specific items or annotations from the dataset. For detailed instructions, refer to the [remove_images](../../command-reference/context_free/transform.md#remove_images) and [remove_annotations](../../command-reference/context_free/transform.md#remove_annotations) CLI.

To remove items or annotations, follow these steps:
1. **Select a Subset**: Choose the subset where you want to apply the removal.
2. **Choose an Item**: Select the item you wish to remove from the subset.
3. **View Annotation Information**: Use the visualizer below to view information about the annotations for the selected item. By default, all annotations are displayed. To view specific annotations, simply select them.
4. **Remove Item**: If you want to remove the entire item, press the **_"Remove item"_** button.
5. **Remove Annotation**: If you want to remove a specific annotation, select it and press the **_"Remove annotation"_** button.

### Auto Correction
![Auto Correction](../../../../images/gui/single/transform_auto_correction.png)

This involves correcting dataset features based on the validation report. The dataset is refined based on the validation report, rejecting undefined labels, missing annotations, and outliers. For detailed instructions, refer to the [validate](../../command-reference/context_free/validate.md) CLI.

To correct the dataset, follow these steps:
1. **Select a Task**: Choose the task for which you want to validate the dataset. The validation report will be displayed below.
2. **Correct the Dataset**: If you want to correct the dataset based on the report, press the **_"Correct a dataset"_** button.

![Correct Dataset](../../../../images/gui/single/transform_auto_correction_correct_dataset.png)
You can see how the dataset has been refined. In this example, corrections have been made for items classified as like UndefinedAttribute, FarFromAttrMean, FarFromLabelMean, and MissingAnnotation.

### Near Duplicate Removal
![Near Duplicate Removal](../../../../images/gui/single/transform_near_duplicate_removal.png)

This involves removing near-duplicated images from the dataset subset. For detailed instructions, refer to the [ndr](../../command-reference/context_free/transform.md#ndr) CLI.

To remove near-duplicated images from a dataset subset, follow these steps:
1. **Select a subset to apply NDR**: Choose the subset from which you want to remove near-duplicated images.
2. **Specify a name for the subset with removed data**: Enter the name you want to assign to the subset after near-duplicated images are removed in the "_Subset name for the removed data_" field.
3. **Advanced options**: If you want to provide more detailed options, toggle the **_"Advanced option"_**.
    ![Advanced Option](../../../../images/gui/single/transform_near_duplicate_removal_advanced_option.png)

    - Set the maximum output dataset size: Enter the maximum size you want the output dataset to be.
    - Specify the maximum output dataset size and choose the oversample and undersample policies. These policies help refine the selection process based on whether the number of images to be removed exceeds or is less than the length of the result after removal.

        For _oversample_ policy, choose between random and similarity methods.
        - The **random** method randomly selects the data to be removed.
        - The **similarity** method selects data based on their similarity values, preferring higher similarity values.

        For _undersample_ policy, choose between uniform and inverse methods.
        - The **uniform** method samples data based on a uniform distribution.
        - The **inverse** method samples data based on the reciprocal of the number of items with the same similarity.

4. **Click "_Find Near Duplicate_"** to start the process of identifying near-duplicated images. If duplicated images are found, the **_"Remove Near Duplicate"_** button will be enabled.

5. If near-duplicated images are detected, click **_"Remove Near Duplicate"_** to proceed with their removal.

For more detailed functionalities of transformations, please follow the instructions provided [here](../../command-reference/context_free/transform.md) to utilize the CLI.
