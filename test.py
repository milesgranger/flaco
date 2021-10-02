from flaco.io import read_sql, Engine
engine = Engine('postgres://postgres:postgres@localhost:5432/postgres')

for i in range(3):
    print(f"Run {i}\n")
    read_sql("select * from foo", engine)

