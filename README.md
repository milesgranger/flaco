## flaco

Perhaps the fastest and most memory efficient way to
pull data from PostgreSQL into [pandas](https://pandas.pydata.org/) 
and [numpy](https://numpy.org/doc/stable/index.html). 🚀

---

### Example

```python
from flaco import Connection

# normal install, returns dict of col name -> array
from flaco.numpy import read_sql as read_sql_numpy

# pip install flaco[pandas] for pandas support
# reads sql queries into dataframe
from flaco.pandas import read_sql as read_sql_pandas

con = Connection("postgresql://postgres:postgres@localhost:5432/postgres")

stmt = "select * from my_big_table"

numpy_arrays = read_sql_numpy(stmt, con)
pandas_df = read_sql_pandas(stmt, con)

# If you know the _exact_ number of rows being
# returned from the query, we can do a single 
# allocation, and will result in pandas.array types
# for flaco.pandas.read_sql (no need to do .convert_dtypes())
# this is the most efficient possible way to load
# as it requires no resizing of output arrays
df = read_sql_pandas(stmt, con, n_rows=1_000_00)
```

---

# Notes

While it's neat this lib can allow faster and less resource
intensive use of pandas/numpy against PostgreSQL, it is currently focused on 
performance, in terms of speed and memory efficiency. It's likely you'll
experience some rough edges, to include, but not limited to:

- 📝 Poor/non-existant error messages
- 💩 Core dumps
- 🚰 Memory leaks (although I think most are handled now)
- 🦖 Almost complete lack of exception handling from underlying Rust/C interface

Please keep this in mind when creating new issues or requesting features.
