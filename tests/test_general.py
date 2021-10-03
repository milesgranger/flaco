import pandas as pd

from flaco import read_sql, Engine


def test_simple_table(postgresdb, postgresdb_connection_uri):
    df = pd.DataFrame()
    df["col1"] = range(10)
    df["col2"] = df.col1.astype(str) + "-hello"

    df.to_sql("test_simple_table", index=False, con=postgresdb)

    engine = Engine(postgresdb_connection_uri)
    columns, data = read_sql("select * from test_simple_table", engine)
    assert set(columns) == {"col1", "col2"}
    assert data[0].sum() == df.col1.sum()
    assert (data[1] == df.col2.to_numpy()).all()
