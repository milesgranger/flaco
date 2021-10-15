import timeit
import pytest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from flaco.io import read_sql, Database


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

    with Database(postgresdb_connection_uri) as db:
        data = read_sql(query, db)
    df2 = pd.DataFrame(data, copy=False)

    assert set(df1.columns) == set(df2.columns)
    assert len(df1.columns) == len(df2.columns)


def test_simple_timing(postgresdb_engine, postgresdb_connection_uri, simple_table):

    scope = dict(pd=pd, read_sql=read_sql, postgresdb_engine=postgresdb_engine)

    t1 = timeit.timeit(
        f"pd.read_sql('select * from {simple_table}', con=postgresdb_engine)",
        globals=scope,
        number=5,
    )
    print(t1)

    with Database(postgresdb_connection_uri) as con:
        scope["con"] = con
        t2 = timeit.timeit(
            f"""read_sql("select * from {simple_table}", con)""",
            globals=scope,
            number=5,
        )
    print(t2)

    # faster than pandas by at least 1/3rd
    assert t1 > t2 * 0.66


def test_simple_group_by(postgresdb_engine, postgresdb_connection_uri, simple_table):
    df1 = pd.read_sql_table(simple_table, con=postgresdb_engine)

    with Database(postgresdb_connection_uri) as con:
        data = read_sql(f"select * from {simple_table}", con)
    df2 = pd.DataFrame(data, copy=False)

    # just numeric columns
    df1_group = df1.groupby("col1").sum()
    df2_group = df2.groupby("col1").sum()
    assert df1_group.equals(df2_group)

    # max will include string columns
    df1_group = df1.groupby("col1").max()
    df2_group = df2.groupby("col1").max()
    assert df1_group.equals(df2_group)


def test_mixed_types_and_nulls(postgresdb_engine, postgresdb_connection_uri):
    table = "test_table"
    n_rows = 5_000
    engine = postgresdb_engine

    df = pd.DataFrame()
    df["col1"] = np.random.randint(0, 1_000, size=n_rows).astype(np.int32)
    df["col2"] = df.col1.astype(np.uint32)
    df["col3"] = df.col1.astype(np.float32)
    df["col4"] = df.col1.astype(np.float64)
    df["col5"] = df.col1.astype(str) + "-hello"
    df["col6"] = df.col5.astype(bytes)
    df.to_sql(table, index=False, con=engine, if_exists="replace")
    df = df[:2]
    df.loc[:, :] = None
    df.to_sql(table, index=False, con=engine, if_exists="append")

    # This should work fine
    with Database(postgresdb_connection_uri) as con:
        data = read_sql(f"select * from {table}", con)

    # Last two rows should all be None, but nothing else
    df = pd.DataFrame(data)
    assert df.loc[len(df) - 2 :, :].isna().all().all()
    assert not df.loc[: len(df) - 2, :].isna().all().all()

    # int columns were converted to object dtype, but non-null vals are 'int'
    assert isinstance(df.col1[0], int)
    assert isinstance(df.col2[0], int)

    # floats are still the same numpy type
    assert isinstance(df.col3[0], np.float32)
    assert isinstance(df.col4[0], np.float64)
