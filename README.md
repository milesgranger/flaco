## flaco

[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![CI](https://github.com/milesgranger/flaco/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/milesgranger/flaco/actions/workflows/CI.yml)
[![PyPI](https://img.shields.io/pypi/v/flaco.svg)](https://pypi.org/project/flaco)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/flaco)
[![Downloads](https://pepy.tech/badge/flaco/month)](https://pepy.tech/project/flaco)


The easiest and perhaps most memory efficient way to get PostgreSQL data (more flavors to come?)
into `pyarrow.Table`, `pandas.DataFrame` or Arrow (IPC/Feather) and Parquet files. 

Since [Arrow](https://github.com/apache/arrow) supports efficient and even larger-than-memory processing,
as with [dask](https://github.com/dask/dask), [duckdb](https://duckdb.org/), or others.
Just getting data onto disk is sometimes the hardest part; this aims to make that easier. 

API:
`flaco.read_sql_to_file`: Read SQL query into Feather or Parquet file.
`flaco.read_sql_to_pyarrow`: Read SQL query into a pyarrow table.

NOTE:
This is still a WIP. I intend to generalize it more to be
useful towards a wider audience. Issues and pull requests welcome!

---

### Example

```bash
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
   122    147.9 MiB    147.9 MiB           1   @profile
   123                                         def memory_profile():
   124    147.9 MiB      0.0 MiB           1       stmt = "select * from test_table"
   125
   126                                             # Read SQL to file
   127    150.3 MiB      2.4 MiB           1       flaco.read_sql_to_file(DB_URI, stmt, 'result.feather', flaco.FileFormat.Feather)
   128    150.3 MiB      0.0 MiB           1       with pa.memory_map('result.feather', 'rb') as source:
   129    150.3 MiB      0.0 MiB           1           table1 = pa.ipc.open_file(source).read_all()
   130    408.1 MiB    257.8 MiB           1           table1_df1 = table1.to_pandas()
   131
   132                                             # Read SQL to pyarrow.Table
   133    504.3 MiB     96.2 MiB           1       table2 = flaco.read_sql_to_pyarrow(DB_URI, stmt)
   134    644.1 MiB    139.8 MiB           1       table2_df = table2.to_pandas()
   135
   136                                             # Pandas
   137    648.8 MiB      4.7 MiB           1       engine = create_engine(DB_URI)
   138   1335.4 MiB    686.6 MiB           1       _pandas_df = pd.read_sql(stmt, engine)
```

---

### License

> _Why did you choose such lax licensing? Could you change to a copy left license, please?_

...just kidding, no one would ask that. This is dual licensed under 
[Unlicense](LICENSE) or [MIT](LICENSE-MIT), at your discretion.
