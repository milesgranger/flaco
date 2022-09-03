## flaco

[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![CI](https://github.com/milesgranger/flaco/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/milesgranger/flaco/actions/workflows/CI.yml)
[![PyPI](https://img.shields.io/pypi/v/flaco.svg)](https://pypi.org/project/flaco)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/flaco)
[![Downloads](https://pepy.tech/badge/flaco/month)](https://pepy.tech/project/flaco)


The easiest and perhaps most memory efficient way to get PostgreSQL data (more flavors to come?)
into Arrow (IPC/Feather) or Parquet files. 

If you're trying to load data directly into Pandas then you may find that evan a 'real' 100MB can cause
bloat upwards of 1GB. Expanding this can cause significant bottle necks in processing data efficiently.

Since [Arrow](https://github.com/apache/arrow) supports efficient and even larger-than-memory processing,
as with [dask](https://github.com/dask/dask), [duckdb](https://duckdb.org/), or others.
Just getting data onto disk is sometimes the hardest part; this aims to make that easier.
---

### Example

```python
from flaco import read_sql_to_file, FileFormat


uri = "postgresql://postgres:postgres@localhost:5432/postgres"
stmt = "select * from my_big_table"

read_sql_to_file(uri, stmt, 'output.data', FileFormat.Parquet)

# Then with pandas...
import pandas as pd
df = pd.read_parquet('output.data')

# pyarrow... (memory mapped file, where potentially larger than memory)
import pyarrow as pa
with pa.memory_map('output.data', 'rb') as source:
  table = pa.ipc.open_file(source).read_all()  # mmap pyarrow.Table

# DuckDB...
import duckdb
cur = duckdb.connect()
cur.execute("select * from read_parquet('output.data')")

# Or anything else which works with Arrow and/or Parquet files
```

---

### License

> _Why did you choose such lax licensing? Could you change to a copy left license, please?_

...just kidding, no one would ask that. This is dual licensed under 
[Unlicense](LICENSE) or [MIT](LICENSE-MIT), at your discretion.
