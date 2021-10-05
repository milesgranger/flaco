import pandas as pd
from sqlalchemy import create_engine
from flaco import Engine, read_sql
from memory_profiler import profile

@profile
def run():
    stmt = "select * from test_large_table"
    engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")
    df1 = pd.read_sql(stmt, con=engine)

    con = Engine("postgresql://postgres:postgres@localhost:5432/postgres")
    data = read_sql(stmt, con)
    df2 = pd.DataFrame(data, copy=False)

if __name__ == '__main__':
    run()
