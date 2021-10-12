## flaco

[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![CI](https://github.com/milesgranger/flaco/workflows/CI/badge.svg?branch=master)](https://github.com/milesgranger/flaco/actions?query=branch=master)
[![PyPI](https://img.shields.io/pypi/v/flaco.svg)](https://pypi.org/project/flaco)
[![Downloads](https://pepy.tech/badge/flaco/month)](https://pepy.tech/project/flaco)

Perhaps the fastest and most memory efficient way to
pull data from PostgreSQL into [pandas](https://pandas.pydata.org/) 
and [numpy](https://numpy.org/doc/stable/index.html). 🚀

---

### Example

```python
from flaco.io import read_sql, Database

con = Database("postgresql://postgres:postgres@localhost:5432/postgres")

stmt = "select * from my_big_table"
data = read_sql(stmt, con)  # dict of column name to numpy array

# If you have pandas installed, you can create a DataFrame
# with zero copying like this:
import pandas as pd
df = pd.DataFrame(data, copy=False)


# If you know the _exact_ rows which will be returned
# you can supply n_rows to perform a single array 
# allocation without needing to resize during query reading.
data = read_sql(stmt, con, 1_000)
```

---

# Notes

While it's neat this lib can allow faster and less resource
intensive use of numpy/pandas against PostgreSQL, it is currently 
has a prioritization on  performance, in terms of speed and memory 
efficiency. It's likely you'll experience some rough edges, to 
include, but not limited to:

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
