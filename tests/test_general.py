import timeit
import pytest
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from flaco.io import read_sql, Database


@pytest.mark.skip(reason="docker not implemented on ci")
@pytest.mark.parametrize(
    "table",
    (
        "actor",
        "address",
        "category",
        "city",
        "country",
        "customer",
        # "film",  # unsuppoted custom `mpaa_rating` TODO: support arbitrary types by serializing to string?
        "film_actor",
        "film_category",
        "inventory",
        "language",
        "payment",
        "rental",
        "staff",
        "store",
    ),
)
def test_basic_select_all_tables(postgresdb_connection_uri, table):

    query = f"select * from {table}"

    df1 = pd.read_sql_table(table, con=create_engine(postgresdb_connection_uri))

    db = Database(postgresdb_connection_uri)
    db.connect()
    data = read_sql(query, db)
    db.disconnect()
    df2 = pd.DataFrame(data, copy=False)

    assert set(df1.columns) == set(df2.columns)
    assert len(df1.columns) == len(df2.columns)


@pytest.mark.skip(reason="docker not implemented on ci")
def test_large_table(postgresdb_connection_uri):

    size = 1_000_000

    df = pd.DataFrame()
    df["col1"] = np.arange(0, size)
    df["col2"] = df.col1.astype(str) + "-hello"
    df["col3"] = np.random.random(size=size)

    df.to_sql(
        "test_large_table", con=create_engine(postgresdb_connection_uri), index=False
    )
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
