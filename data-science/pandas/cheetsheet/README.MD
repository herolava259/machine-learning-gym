# Cheat Sheet for Data Science 

## I. Some useful functions 
### 1. pd.qcut/cut
- Sử dung cho field có giá trị liên tục phân mảnh thành các bucket/bin

### 2. pd.melt 
- Biến đổi DataFrame thành dạng liệt kê của biến id_vars nào đó với các giá trị của value_vars. Dạng liệt kê này sẽ có là một biến index id_vars sẽ gặp giá trị cột là kiểu 
cột value_var(col name của cột value_vars) gì và có giá trị của cột là bao nhiêu

```shell
>>> df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
...                    'B': {0: 1, 1: 3, 2: 5},
...                    'C': {0: 2, 1: 4, 2: 6}})
>>> df
   A  B  C
0  a  1  2
1  b  3  4
2  c  5  6

>>> pd.melt(df, id_vars=['A'], value_vars=['B'])
   A variable  value
0  a        B      1
1  b        B      3
2  c        B      5

>>> pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])
   A variable  value
0  a        B      1
1  b        B      3
2  c        B      5
3  a        C      2
4  b        C      4
5  c        C      6
```

### 3. pd.DataFrame.unstack 

- Xoay (Pivot) một index phức hợp (MultiIndex) thành một columns
- **Mark Down**: [parameter] level: chỉ định số thứ tự của index đơn trong tập index phức hợp. Lưu ý: -1 là số thứ tụ đầu, 0 là cuối index phức hợp 

```shell
>>> index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'),
...                                    ('two', 'a'), ('two', 'b')])
>>> s = pd.Series(np.arange(1.0, 5.0), index=index)
>>> s
one  a   1.0
     b   2.0
two  a   3.0
     b   4.0
dtype: float64

>>> s.unstack(level=-1)
     a   b
one  1.0  2.0
two  3.0  4.0
```

### 4. pd.DataFrame.crosstab

- Nó sẽ thực hiện 2 việc 
    1. Nhóm các row có cùng giá trị trong các cột chỉ định
    2. Thực hiện một aggrgation function trên các giá trị của những row tròng mỗi nhóm. Nêu khoog chỉ định *aggfunc* mặc định nó tính phần trăm số số row trong nhóm.


```shell
>>> a = np.array(["foo", "foo", "foo", "foo", "bar", "bar",
...               "bar", "bar", "foo", "foo", "foo"], dtype=object)
>>> b = np.array(["one", "one", "one", "two", "one", "one",
...               "one", "two", "two", "two", "one"], dtype=object)
>>> c = np.array(["dull", "dull", "shiny", "dull", "dull", "shiny",
...               "shiny", "dull", "shiny", "shiny", "shiny"],
...              dtype=object)
>>> pd.crosstab(a, [b, c], rownames=['a'], colnames=['b', 'c'])
b   one        two
c   dull shiny dull shiny
a
bar    1     2    1     0
foo    2     2    1     2


>>> foo = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
>>> bar = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
>>> pd.crosstab(foo, bar)
col_0  d  e
row_0
a      1  0
b      0  1
>>> pd.crosstab(foo, bar, dropna=False)
col_0  d  e  f
row_0
a      1  0  0
b      0  1  0
c      0  0  0

>>> foo = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
>>> bar = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
>>> pd.crosstab(foo, bar)
col_0  d  e
row_0
a      1  0
b      0  1
>>> pd.crosstab(foo, bar, dropna=False)
col_0  d  e  f
row_0
a      1  0  0
b      0  1  0
c      0  0  0
```