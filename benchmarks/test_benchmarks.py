import pytest
import numpy as np
import pandas as pd
from memory_profiler import profile
from sqlalchemy import create_engine
from flaco.io import Database, read_sql


DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres"
DB_TABLES = (
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
)


@pytest.mark.parametrize("loader", ("pandas", "flaco"))
def test_basic(benchmark, loader: str):
    def _read_all_tables(func, sql, con):
        for table in DB_TABLES:
            # flaco outputs dict of str -> numpy arrays; so wrap it in a
            # DataFrame just for fairness, although it shouldn't cost anything
            # as numpy arrays default to no copy.
            if loader == "pandas":
                func(sql.format(table=table), con)
            else:
                pd.DataFrame(func(sql.format(table=table), con))

    if loader == "pandas":
        engine = create_engine(DB_URI)
        benchmark(
            _read_all_tables, pd.read_sql, "select * from {table}", engine,
        )
    else:
        with Database(DB_URI) as con:
            benchmark(_read_all_tables, read_sql, "select * from {table}", con)


@pytest.mark.parametrize("loader", ("pandas", "flaco"))
@pytest.mark.parametrize(
    "n_rows", np.arange(100_000, 1_000_000, 100_000), ids=lambda val: f"rows={val}"
)
def test_incremental_size(benchmark, loader: str, n_rows: int):
    n_cols = 5
    table = "test_table"
    engine = create_engine(DB_URI)

    data = np.random.randint(0, 100_000, size=n_rows * n_cols).reshape((-1, n_cols))
    pd.DataFrame(data).to_sql(
        table, index=False, con=engine, chunksize=10_000, if_exists="replace"
    )

    if loader == "pandas":
        engine = create_engine(DB_URI)
        benchmark(lambda *args: pd.read_sql(*args), f"select * from {table}", engine)
    else:
        with Database(DB_URI) as con:
            benchmark(
                lambda *args: pd.DataFrame(read_sql(*args)),
                f"select * from {table}",
                con,
            )


def _table_setup(n_rows: int = 1_000_000, include_nulls: bool = False):
    table = "test_table"
    engine = create_engine(DB_URI)

    df = pd.DataFrame()
    df["col1"] = np.random.randint(0, 1000, size=n_rows).astype(np.int32)
    df["col2"] = df.col1.astype(np.uint32)
    df["col3"] = df.col1.astype(np.float32)
    df["col4"] = df.col1.astype(np.float64)
    df["col5"] = df.col1.astype(str) + "-hello"
    df["col6"] = df.col5.astype(bytes)
    df.to_sql(table, index=False, con=engine, chunksize=10_000, if_exists="replace")

    if include_nulls:
        df = df[:20]
        df.loc[:, :] = None
        df.to_sql(table, index=False, con=engine, if_exists="append")


@profile
def memory_profile():
    stmt = "select * from test_table"

    # ~145MB
    with Database(DB_URI) as con:
        data1 = read_sql(stmt, con)
        _flaco_df1 = pd.DataFrame(data1, copy=False)

    # ~260MB
    engine = create_engine(DB_URI)
    _pandas_df1 = pd.read_sql(stmt, engine)


if __name__ == "__main__":
    _table_setup(n_rows=500_000, include_nulls=True)
    memory_profile()
