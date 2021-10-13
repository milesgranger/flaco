import pytest
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


@pytest.fixture(scope="session")
def postgresdb_connection_uri():
    return "postgresql://postgres:postgres@localhost:5432/postgres"


@pytest.fixture
def postgresdb_engine(postgresdb_connection_uri):
    return create_engine(postgresdb_connection_uri)


@pytest.fixture
def simple_table(postgresdb_engine):
    size = 10_000
    df = pd.DataFrame()
    df["col1"] = np.random.randint(0, 100, size=size)
    df["col2"] = df.col1.astype(str) + "-hello"
    df["col3"] = np.random.random(size=size)
    con = postgresdb_engine.connect()
    df.to_sql("simple_table", con=con, index=False, if_exists="replace")
    return "simple_table"
