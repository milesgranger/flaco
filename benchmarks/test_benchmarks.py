import pytest
import numpy as np
import pandas as pd
import flaco
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as pf
import pyarrow.dataset as ds
from memory_profiler import profile
from sqlalchemy import create_engine


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


@pytest.mark.parametrize("loader", ("pandas", "flaco", "connectorx"))
def test_basic(benchmark, loader: str):
    def _read_all_tables(func, sql, con):
        for table in DB_TABLES:
            # flaco outputs dict of str -> numpy arrays; so wrap it in a
            # DataFrame just for fairness, although it shouldn't cost anything
            # as numpy arrays default to no copy.
            if loader == "pandas":
                func(sql.format(table=table), con)
            elif loader == "connectorx":
                func(DB_URI, sql.format(table=table))
            else:
                pd.DataFrame(func(sql.format(table=table), con), copy=False)

    if loader == "pandas":
        engine = create_engine(DB_URI)
        benchmark(
            _read_all_tables, pd.read_sql, "select * from {table}", engine,
        )
    elif loader == "connectorx":
        benchmark(
            _read_all_tables, cx.read_sql,"select * from {table}", DB_URI
        )
    else:
        with Database(DB_URI) as con:
            benchmark(_read_all_tables, read_sql, "select * from {table}", con)


@pytest.mark.parametrize("loader", ("pandas", "flaco", "connectorx"))
@pytest.mark.parametrize(
    "n_rows", np.arange(100_000, 1_000_000, 100_000), ids=lambda val: f"rows={val}"
)
def test_incremental_size(benchmark, loader: str, n_rows: int):
    table = _table_setup(n_rows=n_rows, include_nulls=False)

    if loader == "pandas":
        engine = create_engine(DB_URI)
        benchmark(lambda *args: pd.read_sql(*args), f"select * from {table}", engine)
    elif loader == "connectorx":
        benchmark(cx.read_sql, DB_URI, f"select * from {table}", return_type="pandas")
    else:
        with Database(DB_URI) as con:
            benchmark(
                lambda *args: pd.DataFrame(read_sql(*args), copy=False),
                f"select * from {table}",
                con,
            )


def _table_setup(n_rows: int = 1_000_000, include_nulls: bool = False):
    table = "test_table"
    engine = create_engine(DB_URI)

    engine.execute(f"drop table if exists {table}")
    engine.execute(f"""
        create table {table} (
            col1 int, 
            col2 int8, 
            col3 float8, 
            col4 float4, 
            col5 text, 
            col6 bytea,
            col7 date,
            col8 timestamp without time zone,
            col9 timestamp with time zone,
            col10 time
        )
    """)

    df = pd.DataFrame()
    df["col1"] = np.random.randint(0, 1000, size=n_rows).astype(np.int32)
    df["col2"] = df.col1.astype(np.uint32)
    df["col3"] = df.col1.astype(np.float32)
    df["col4"] = df.col1.astype(np.float64)
    df["col5"] = df.col1.astype(str) + "-hello"
    df["col6"] = df.col1.astype(bytes)
    df["col7"] = pd.date_range('2000-01-01', '2001-01-01', periods=len(df))
    df["col8"] = pd.to_datetime(df.col7)
    df["col9"] = pd.to_datetime(df.col7, utc=True)
    df["col10"] = df.col9.dt.time
    df.to_sql(table, index=False, con=engine, chunksize=50_000, if_exists="replace")

    if include_nulls:
        df = df[:20]
        df.loc[:, :] = None
        df.to_sql(table, index=False, con=engine, if_exists="append")
    return table

@profile
def memory_profile():
    stmt = "select * from test_table"

    # Read SQL to file
    flaco.read_sql_to_file(DB_URI, stmt, 'result.feather', flaco.FileFormat.Feather)
    with pa.memory_map('result.feather', 'rb') as source:
        table1 = pa.ipc.open_file(source).read_all()
        table1_df1 = table1.to_pandas()

    # Read SQL to pyarrow.Table
    table2 = flaco.read_sql_to_pyarrow(DB_URI, stmt)
    table2_df = table2.to_pandas()
    
    # Pandas
    engine = create_engine(DB_URI)
    _pandas_df = pd.read_sql(stmt, engine)


if __name__ == "__main__":
    #_table_setup(n_rows=1_000_000, include_nulls=True)
    memory_profile()
