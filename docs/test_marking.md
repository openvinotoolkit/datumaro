# Source code elements related to pytest
## Contents
- [Test case description](#Test_case_description)
  - [Test marking](#Test_marking)
    - [Requirements](#Requirements)
    - [Type](#Type)
    - [Priority](#Priority)
    - [Component](#Component)
    - [Skip](#Skip)
    - [Bug(s)](#Bug)
    - [Parameters](#Parameters)
    - [Fixtures](#Fixtures)
  - [Test naming convention](#TestNaming)
    - [Test name](#TestName)
  - [Test documentation](#TestDoc)
    - [Docstring](#Docstring)
    - [Steps in test body](#Step)
          
<a id="Test_case_description"></a>
## Test case description
<a id="Test_marking"></a>
### Test marking
<a id="Requirements"></a>
#### Requirements 
Fully defined in GitHub issues, e.g.
```
    @pytest.mark.reqids(Requirements.DATUM_244, Requirements.DATUM_333)
```
Requirements constants need to be added to datumaro/tests/constants/requirements.py for example:
```
# [DATUMARO] Add SNYK scan integration
DATUM_244 = "DATUM-244 Add Snyk integration"
```  
Recommended notation for requirements 

```python
@pytest.mark.requids(Requirements.DATUM_123)
``` 
where DATUM is a keyword for datumaro indication, and number is a datumaro github issue number.

<a id="Type"></a>
#### Type
Test types: gui_smoke, gui_regression, manual, gui_other, gui_long, api, component, unit e.g.
```python
    @pytest.mark.component
    @pytest.mark.unit
```

<a id="Priority"></a>
#### Priority 
Test priorities: low, medium, high, e.g.,
```python
     @pytest.mark.priority_low
     @pytest.mark.priority_medium
     @pytest.mark.priority_high
```
<a id="Component"></a>
#### Component

Component marking used for indication of different system components e.g.
```python
    @pytest.mark.components(DatumaroComponent.Datumaro)
```
<a id="Skip"></a>
#### Skip 

For marking tests, which should be skipped for example for not yet tests, e.g.,
```python
   @pytest.mark.skip(reason=SkipMessages.NOT_IMPLEMENTED)
```
<a id="Bug"></a>
#### Bug(s):

In case of test failure, bug should be entered into github issues, and test can be marked e.g.
```python
    @pytest.mark.bugs("DATUM-219 - Return format is not uniform")
```
<a id="Parameters"></a>
#### Parameters: 

Parameters are used for running the same test with different parameters e.g. 
```python
@pytest.mark.parametrize("numpy_array, batch_size", [  
        (np.zeros([2]), 0),  
        (np.zeros([2]), 1),
        (np.zeros([2]), 2),
        (np.zeros([2]), 5),
        (np.zeros([5]), 2),
    ])
```
(be aware that test parametrization is supported in pytest, and is not supported in unittests) 

<a id="Fixtures"></a>
#### Fixtures 

If needed pytest fixtures can be used - for more details see pytest documentation <br>
https://docs.pytest.org/en/6.2.x/contents.html <br>
(be aware that fixtures are supported in pytest, and they are not supported in unittests) 

<a id="TestNaming"></a>
### Test naming convention

<a id="TestName"></a>
#### Test name:

Test method should be prefixed with "test_" prefix e.g.   
```python
    test_*()
```
"test_" prefix is an indication for pytest to treat method as a test for execution. 
It's important not to start other methods with the "test_" prefix.

<a id="DestDoc"></a>
### Test documentation

<a id="Docstring"></a>
#### Docstring 

Tests are documented with Docstring. Every test method documentation string should contain: Description, Expected results 
and Steps. These fields are required but, not limited e.g.
```python
        def test_can_convert_polygons_to_mask(self):
        """
        <b>Description:</b>
        Ensure that the dataset polygon segmentation mode can be properly converted into dataset mask segmentation mode.

        <b>Expected results:</b>
        Dataset segmentation mask converted from dataset segmentation polygon is equal to expected mask.

        <b>Steps:</b>
        1. Prepare dataset with polygon segmentation mode (source dataset)
        2. Prepare dataset with expected mask segmentation mode (target dataset)
        3. Convert source dataset to target, with segmentation mode changed from polygon to mask and verify that result
        segmentation mask is equal to expected mask.

        """

```
<a id="Steps"></a>
#### Steps in test body: 

Steps description in test body are placed as a code comment lines e.g.
```python
      # 1.Prepare dataset with polygon segmentation mode (source dataset)")
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.zeros((6, 10, 3)),
                annotations=[
                    Polygon([0, 0, 4, 0, 4, 4],
                        label=3, id=4, group=4),
                    Polygon([5, 0, 9, 0, 5, 5],
                        label=3, id=4, group=4),
                ]
            ),
        ], categories=[str(i) for i in range(10)])

        # 2. Prepare dataset with expected mask segmentation mode (target dataset)
        target_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.zeros((6, 10, 3)),
                annotations=[
                    Mask(np.array([
                            [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                            # only internal fragment (without the border),
                            # but not everywhere...
                        ),
                        attributes={ 'is_crowd': True },
                        label=3, id=4, group=4),
                ], attributes={'id': 1}
            ),
        ], categories=[str(i) for i in range(10)])

        # 3. Convert source dataset to target, with segmentation mode changed from polygon to mask and verify that
        #    result segmentation mask is equal to expected mask.
        with TempTestDir() as test_dir:
            self._test_save_and_load(source_dataset,
                partial(CocoInstancesConverter.convert, segmentation_mode='mask'),
                test_dir, target_dataset=target_dataset)

```
