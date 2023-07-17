# Tabular

## Format specification

Tabular dataset generally refers to table data with multiple rows and columns. <br>
`.csv` files are the most common format, and OpenML uses `.arff` as the official format.

Datumaro only supports tabular data in `.csv` format where the first row is a header with unique column names.
It's because other formats can be converted to `.csv` easily as shown below.

```python
# convert '.arff' to '.csv'
from scipy.io.arff import loadarff
import pandas as pd
data = loadarff("dataset.arff")
df = pd.DataFrame(data[0])
categorical = [col for col in df.columns if df[col].dtype=="O"]
df[categorical] = df[categorical_columns].apply(lambda x: x.str.decode('utf8'))
df.to_csv("arff.csv", index=False)

# convert '.parquet', '.feather', '.hdf5', '.pickle' to '.csv'.
pd.read_parquet("dataset.parquet").to_csv('parquet.csv', index=False)
pd.read_feather("dataset.feather").to_csv('feather.csv', index=False)
pd.read_hdf("dataset.hdf5").to_csv('hdf5.csv', index=False)
pd.read_pickle("dataset.pickle").to_csv('pickle.csv', index=False)

# convert '.jay' to '.csv'
import datatable as dt
data = dt.fread("dataset.jay")
data.to_csv("jay.csv")
```

A tabular dataset can be one of the following:
- a single file with a `.csv` extension
- a directory contains `.csv` files (supports only 1 depth).
    <!--lint disable fenced-code-flag-->
    ```
    dataset/
    ├── aaa.csv
    ├── ...
    └── zzz.csv
    ```

Supported annotation types:
- `Tabular`

## Import tabular dataset

A Datumaro project with a tabular source can be created in the following way:

```bash
datum project create
datum project import --format tabular <path/to/dataset>
```

It is also possible to import the dataset using Python API:

```python
import datumaro as dm
dataset = dm.Dataset.import_from('<path/to/dataset>', 'tabular')
```

Datumaro stores the imported table as media (a list of `TableRow`) and annotates the target columns.
The last column is regarded as the target column,
which can be specified by the user when importing the dataset as shown below.

```bash
datum project create
datum project import --format tabular <path/to/buddy/dataset> -- --target breed_category,pet_category
datum project import --format tabular <path/to/electricity/dataset> -- --target class
```

```python
import datumaro as dm
dataset = dm.Dataset.import_from('<path/to/buddy/dataset>', 'tabular', target=["breed_category", "pet_category"])
dataset = dm.Dataset.import_from('<path/to/electricity/dataset>', 'tabular', target="class")
```

As shown, the target can be a single column name or a comma-separated list of columns.

Note that each tabular file is considered as a subset.

## Export tabular dataset

Datumaro supports exporting a tabular dataset using CLI or python API.
Each subset will be saved to a separate `.csv` file.

```bash
datum project create
datum project import -f tabular <path/to/dataset>
datum project export -f tabular -o <output/dir>
```

```python
import datumaro as dm
dataset = dm.Dataset.import_from('<path/to/dataset>', 'tabular')
dataset.export('<path/to/output/dir>', 'tabular')
```

Note that converting a tabular dataset into other formats and vice versa is not supproted.

## Examples

Examples of using this format from the code can be found in
[the format tests](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/unit/test_tabular_format.py)

Examples below are [Electricity](https://www.openml.org/d/44156) dataset.

|     date |   day |   period |   nswprice |   nswdemand |   vicprice |   vicdemand |   transfer | class   |
|---------:|------:|---------:|-----------:|------------:|-----------:|------------:|-----------:|:--------|
| 0.425556 |     5 | 0.340426 |   0.076108 |    0.392889 |   0.003467 |    0.422915 |   0.414912 | UP      |
| 0.425512 |     4 | 0.617021 |   0.060376 |    0.483041 |   0.003467 |    0.422915 |   0.414912 | DOWN    |
| 0.013982 |     4 | 0.042553 |   0.061967 |    0.521125 |   0.003467 |    0.422915 |   0.414912 | DOWN    |
| 0.907349 |     3 | 0.06383  |   0.080581 |    0.331003 |   0.00538  |    0.47566  |   0.441228 | DOWN    |
| 0.889341 |     0 | 0.361702 |   0.027141 |    0.379649 |   0.001624 |    0.248317 |   0.69386  | DOWN    |
