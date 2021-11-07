## flaco

[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![CI](https://github.com/milesgranger/flaco/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/milesgranger/flaco/actions/workflows/CI.yml)
[![PyPI](https://img.shields.io/pypi/v/flaco.svg)](https://pypi.org/project/flaco)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/flaco)
[![Downloads](https://pepy.tech/badge/flaco/month)](https://pepy.tech/project/flaco)

Perhaps the fastest* and most memory efficient way to
pull data from PostgreSQL into [pandas](https://pandas.pydata.org/) 
and [numpy](https://numpy.org/doc/stable/index.html). 🚀

Have a gander at the initial [benchmarks](./benchmarks) 🏋

flaco tends to use nearly ~3-6x less memory than standard `pandas.read_sql` 
and about ~2-3x faster. However, it's probably 50x less stable at the moment. 😜

To whet your appetite, here's a memory profile between flaco, [connectorx](https://github.com/sfu-db/connector-x) 
and `pandas.read_sql` on a table with 1M rows with columns of various types. 
(see [test_benchmarks.py](benchmarks/test_benchmarks.py)) *If the data, 
specifically integer types, has null values, you can expect a bit lower savings
what you see here; therefore (hot tip 🔥), supply fill values in your queries 
where possible via `coalesce`.

```bash
Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
   118     97.9 MiB     97.9 MiB           1   @profile
   119                                         def memory_profile():
   120     97.9 MiB      0.0 MiB           1       stmt = "select * from test_table"
   121                                         
   122    354.9 MiB    257.0 MiB           1       _cx_df = cx.read_sql(DB_URI, stmt, return_type="pandas")
   123                                         
   124    354.9 MiB      0.0 MiB           1       with Database(DB_URI) as con:
   125    533.9 MiB    178.9 MiB           1           data = read_sql(stmt, con)
   126    541.2 MiB      7.3 MiB           1           _flaco_df = pd.DataFrame(data, copy=False)
   127                                         
   128    545.3 MiB      4.1 MiB           1       engine = create_engine(DB_URI)
   129   1680.9 MiB   1135.5 MiB           1       _pandas_df = pd.read_sql(stmt, engine)
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
  can be passed with zero copies to  `pandas.DataFrame`
- When querying integer columns, if a null is encountered, the array will be 
  converted to `dtype=object` and nulls from PostgreSQL will be `None`. 
  Whereas pandas will convert the underlying array to a float type; where nulls
  from postgres are basically `numpy.nan` types.
- It lacks basically all of the options `pandas.read_sql` has.


> How does this compare with [connectorx](https://github.com/sfu-db/connector-x)?

Connectorx is an _exceptionally_ impressive library, and more mature than flaco. 
They have much wider support for a range of data sources, while flaco only 
supports postgres for now.

Performance wise, benchmarking seems to indicate flaco is generally more performant
in terms of memory, but connectorx is faster when temporal data types (time, timestamp, etc)
are used. If it's pure numeric dtypes, flaco is faster and more memory efficient.

Connectorx [will make precheck queries](https://github.com/sfu-db/connector-x#how-does-connectorx-download-the-data)
to the source before starting to download data. Depending on your application,
this can be a significant bottleneck. Specially, some data warehousing queries
are very expensive and even doing  a `LIMIT 1`, will cause significant load on
the source database.

Flaco will not run _any_ precheck queries. _However_, you can supply either
`n_rows` or `size_hint` to `flaco.io.read_sql` to give either exact, or a
hint to reduce the number of allocations/resizing of arrays during data loading.

**When in doubt, it's likely you should choose connectorx as it's more mature and
offers great performance.**

# Words of caution

While it's pretty neat this lib can allow faster and less resource
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