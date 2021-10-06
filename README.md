## flaco

Perhaps the fastest and most memory efficient way to
pull data from PostgreSQL into [pandas](https://pandas.pydata.org/) 
and [numpy](https://numpy.org/doc/stable/index.html). 🚀

---

### Example

```python
from flaco.io import read_sql, Connection

con = Connection("postgresql://postgres:postgres@localhost:5432/postgres")

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

Please keep this in mind when creating new issues or requesting features.
