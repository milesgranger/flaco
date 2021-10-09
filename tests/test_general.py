import timeit
import datetime as dt
import numpy as np
import pandas as pd
from hypothesis import strategies as st, given, settings
from hypothesis.extra.pandas import data_frames, column

from flaco.io import read_sql, Database


@given(
    df_in=data_frames(
        columns=[
            column("col1", elements=st.text(max_size=300)),
            column("col2", elements=st.binary(max_size=300)),
            column("col3", elements=st.integers()),
            column("col4", elements=st.floats()),
        ]
    )
)
@settings(max_examples=10, deadline=dt.timedelta(seconds=5))
def test_property_values(df_in, postgresdb, postgresdb_connection_uri):
    if df_in.empty:
        return
    df_in = df_in.loc[:5_000, :]  # avoid huge loads into db
    df_in.to_sql("foo", index=False, con=postgresdb, if_exists="replace")

    db = Database(postgresdb_connection_uri)
    db.connect()
    data = read_sql("select * from foo", db)
    db.disconnect()
    df_out = pd.DataFrame(data, copy=False)
    if not df_in.empty:
        assert df_in["col3"].sum() == df_out["col3"].sum()
    else:
        assert df_out.columns.size == 0


def test_large_table(postgresdb, postgresdb_connection_uri):

    size = 1_000_000

    df = pd.DataFrame()
    df["col1"] = np.arange(0, size)
    df["col2"] = df.col1.astype(str) + "-hello"
    df["col3"] = np.random.random(size=size)

    df.to_sql("test_large_table", con=postgresdb, index=False)
    engine = Database(postgresdb_connection_uri)

    scope = locals()
    scope["pd"] = pd
    scope["read_sql"] = read_sql

    t1 = timeit.timeit(
        "pd.read_sql('select * from test_large_table', con=postgresdb)",
        globals=scope,
        number=2,
    )
    print(t1)

    t2 = timeit.timeit(
        """read_sql("select * from test_large_table", engine)""",
        globals=scope,
        number=2,
    )
    print(t2)

    # faster than pandas by at least 1/3rd
    assert t1 > t2 * 0.66
