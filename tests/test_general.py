import timeit
import pytest
import pandas as pd
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

    db = Database(postgresdb_connection_uri)
    db.connect()
    data = read_sql(query, db)
    db.disconnect()
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
