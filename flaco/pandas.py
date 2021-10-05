import pandas as pd
from .io import read_sql as _read_sql, Connection

def read_sql(stmt: str, con: Connection):
    data = _read_sql(stmt, con)
    return pd.DataFrame(data, copy=False)
