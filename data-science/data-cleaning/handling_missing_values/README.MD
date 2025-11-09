from sympy import nfloat

## I. Preparation 

### 1. Dataset(From Kaggle):

```bash
kaggle kernels output alexisbcook/handling-missing-values -p /path/to/dest
```

### 2. Import necessary package and the dataset 

```python
# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")

# set seed for reproducibility
np.random.seed(0) 
```

### 3. Check dataset 

```python
nfl_data.head()
```

### 4. Check missing data points do we have?

```python

# get the number of missing data points per column 
missing_values_count = nfl_data_isnull().sum()

# look at the # of missing points in the first ten columns
print("missing count follow by column: ")
print(missing_values_count[0:10])


# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing 
percent_missing = (total_missing/total_cells)
print(percent_missing)

```

### 5. Figure out why the data is missing 

#### a. Drop missing values 

```jupyterpython
nfl_data.dropna()
```

- Check columns 

```python
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
```

- just how much data did we lose?

```python
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Column with na's dropped: %d" % columns_with_na_dropped.shape[1])
```

#### b. Filling in missing values automatically 

```python
subset_nfl_data = nfl_data.lloc[:, 'EPA':'Season'].head()
subset_nfl_data
```

- replace all NA's with 0

```python
subset_nfl_data.fillna(0)
```

- I could also be a bit more savvy and replace missing values 

```python
subset_nfl_data.fillna(method='bfill', axis=0).fillna(0)
```



