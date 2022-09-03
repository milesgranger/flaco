import timeit
import pytest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from flaco import read_sql_to_file, FileFormat, FlacoException


@pytest.mark.parametrize(
    "table",
    (
        "actor",
        "address",
        "category",
        "city",
        "country",
        "customer",
        # "film",  # unsupported custom `mpaa_rating` TODO: support arbitrary types by serializing to string?
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
@pytest.mark.parametrize("format", (FileFormat.Feather, FileFormat.Parquet))
def test_basic_select_all_tables(postgresdb_connection_uri, tmpdir, table, format):

    query = f"select * from {table}"

    df1 = pd.read_sql_table(table, con=create_engine(postgresdb_connection_uri))

    out = str(tmpdir / "out.data")
    read_sql_to_file(postgresdb_connection_uri, query, out, format)
    if format == FileFormat.Feather:
        df2 = pd.read_feather(out)
    elif format == FileFormat.Parquet:
        df2 = pd.read_parquet(out)

    assert set(df1.columns) == set(df2.columns)
    assert len(df1.columns) == len(df2.columns)
    assert len(df1) == len(df2)


def test_simple_group_by(
    postgresdb_engine, postgresdb_connection_uri, simple_table, tmpdir
):
    df1 = pd.read_sql_table(simple_table, con=postgresdb_engine)

    out = str(tmpdir / "out.feather")
    read_sql_to_file(
        postgresdb_connection_uri,
        f"select * from {simple_table}",
        out,
        FileFormat.Feather,
    )
    df2 = pd.read_feather(out)

    # just numeric columns
    df1_group = df1.groupby("col1").sum()
    df2_group = df2.groupby("col1").sum()
    assert df1_group.equals(df2_group)

    # max will include string columns
    df1_group = df1.groupby("col1").max()
    df2_group = df2.groupby("col1").max()
    assert df1_group.equals(df2_group)


def test_mixed_types_and_nulls(postgresdb_engine, postgresdb_connection_uri, tmpdir):
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
    out = str(tmpdir / "out.feather")
    read_sql_to_file(
        postgresdb_connection_uri, f"select * from test_table", out, FileFormat.Feather
    )
    df = pd.read_feather(out).convert_dtypes()

    # Last two rows should all be None, but nothing else
    assert df.loc[len(df) - 2 :, :].isna().all().all()
    assert not df.loc[: len(df) - 2, :].isna().all().all()


def test_query_error(postgresdb_connection_uri):
    with pytest.raises(FlacoException):
        read_sql_to_file(
            postgresdb_connection_uri, "select fail", "out", FileFormat.Feather
        )


def test_bad_connection_error():
    with pytest.raises(FlacoException):
        read_sql_to_file("baduri", "select fail", "out", FileFormat.Feather)
