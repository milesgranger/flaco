import timeit
import numpy as np
import pandas as pd

from flaco.io import read_sql, Connection


def test_simple_table(postgresdb, postgresdb_connection_uri):
    df = pd.DataFrame()
    df["col1"] = range(10)
    df["col2"] = df.col1.astype(str) + "-hello"

    df.to_sql("test_simple_table", index=False, con=postgresdb)

    engine = Connection(postgresdb_connection_uri)
    columns, data = read_sql("select * from test_simple_table", engine)
    assert set(columns) == {"col1", "col2"}
    assert data[0].sum() == df.col1.sum()
    assert (data[1] == df.col2.to_numpy()).all()


def test_large_table(postgresdb, postgresdb_connection_uri):

    size = 1_000_000

    df = pd.DataFrame()
    df["col1"] = np.arange(0, size)
    df["col2"] = df.col1.astype(str) + '-hello'
    df["col3"] = np.random.random(size=size)

    df.to_sql("test_large_table", con=postgresdb, index=False)
    engine = Connection(postgresdb_connection_uri)

    scope = locals()
    scope["pd"] = pd
    scope["read_sql"] = read_sql

    t1 = timeit.timeit("pd.read_sql('select * from test_large_table', con=postgresdb)", globals=scope, number=2)
    print(t1)

    t2 = timeit.timeit('''read_sql("select * from test_large_table", engine)''', globals=scope, number=2)
    print(t2)

    # faster than pandas by at least 1/3rd
    assert t1 > t2 * 0.66
