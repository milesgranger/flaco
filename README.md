## flaco

[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![CI](https://github.com/milesgranger/flaco/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/milesgranger/flaco/actions/workflows/CI.yml)
[![PyPI](https://img.shields.io/pypi/v/flaco.svg)](https://pypi.org/project/flaco)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/flaco)
[![Downloads](https://pepy.tech/badge/flaco/month)](https://pepy.tech/project/flaco)

Perhaps the fastest and most memory efficient way to
pull data from PostgreSQL into [pandas](https://pandas.pydata.org/) 
and [numpy](https://numpy.org/doc/stable/index.html). 🚀

Have a gander at the initial [benchmarks](./benchmarks) 🏋

flaco tends to use nearly ~3-5x less memory than standard `pandas.read_sql` 
and about ~3x faster. However, it's probably 50x less stable at the moment. 😜

To wet your appetite, here's a memory profile between flaco and pandas `read_sql` 
on a table with 2M rows with columns of various types. (see [test_benchmarks.py](benchmarks/test_benchmarks.py))
*If the test data has null values, you can expect a ~3x saving, instead of the ~5x 
you see here; therefore (hot tip 🔥), supply fill values in your queries where possible via `coalesce`.
```bash
Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
    99    140.5 MiB    140.5 MiB           1   @profile
   100                                         def memory_profile():
   101    140.5 MiB      0.0 MiB           1       stmt = "select * from test_table"
   102                                         
   103                                             
   104    140.9 MiB      0.4 MiB           1       with Database(DB_URI) as con:
   105    441.8 MiB    300.9 MiB           1           data = read_sql(stmt, con)
   106    441.8 MiB      0.0 MiB           1           _flaco_df = pd.DataFrame(data, copy=False)
   107                                         
   108                                             
   109    441.8 MiB      0.0 MiB           1       engine = create_engine(DB_URI)
   110   2091.5 MiB   1649.7 MiB           1       _pandas_df = pd.read_sql(stmt, engine)
```

---

### Example

```python
from flaco.io import read_sql, Database


uri = "postgresql://postgres:postgres@localhost:5432/postgres"
stmt = "select * from my_big_table"

with Database(uri) as con:
    data = read_sql(stmt, con)  # dict of column name to numpy array

# If you have pandas installed, you can create a DataFrame
# with zero copying like this:
import pandas as pd
df = pd.DataFrame(data, copy=False)

# If you know the _exact_ rows which will be returned
# you can supply n_rows to perform a single array 
# allocation without needing to resize during query reading.
with Database(uri) as con:
    data = read_sql(stmt, con, 1_000)
```

---

# Notes

> Is this a drop in replacement for `pandas.read_sql`?

No. It varies in a few ways:
- It will return a `dict` of `str` ➡ `numpy.ndarray` objects. But this 
  can be passed with _zero_ copies to  `pandas.DataFrame`
- When querying integer columns, if a null is encountered, the array will be 
  converted to `dtype=object` and nulls from PostgreSQL will be `None`. 
  Whereas pandas will convert the underlying array to a float type; where nulls
  from postgres are basically `numpy.nan` types.
- It lacks basically all of the options `pandas.read_sql` has.


Furthermore, while it's pretty neat this lib can allow faster and less resource
intensive use of numpy/pandas against PostgreSQL, it's in early 
stages of development and you're likely to encounter some sharp edges
which include, but not limited to:

- 📝 Poor/non-existant error messages
- 💩 Core dumps
- 🚰 Memory leaks (although I think most are handled now)
- 🦖 Almost complete lack of exception handling from underlying Rust/C interface
- 📍 PostgreSQL `numeric` type should ideally be converted to `decimal.Decimal`
     but uses `f64` for now; potentially loosing precision. Note, this
     is exactly what `pandas.read_sql` does. 
- ❗ Might not handle all or custom arbitrary PostgreSQL types. If you encounter
   such types, either convert them to a supported type like text/json/jsonb 
   (ie `select my_field::text ...`), or open an issue if a standard type is not 
   supported.

---

### License

> _Why did you choose such lax licensing? Could you change to a copy left license, please?_

...just kidding, no one would ask that. This is dual licensed under 
[Unlicense](LICENSE) and [MIT](LICENSE-MIT). 