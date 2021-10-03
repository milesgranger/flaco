import pytest
import docker
import sqlalchemy


@pytest.fixture(scope="session")
def postgresdb_connection_uri():
    return "postgresql://postgres:postgres@localhost:5432/postgres"


@pytest.yield_fixture(scope="session")
def postgresdb(postgresdb_connection_uri):
    """
    Run a postgres container for integration type tests which need
    access to a database.
    """
    client = docker.from_env()
    postgres = client.containers.run(
        image="docker.io/postgres:12-alpine",
        environment={"POSTGRES_USER": "postgres", "POSTGRES_PASSWORD": "postgres"},
        ports={"5432/tcp": "5432"},
        remove=True,
        detach=True,
    )
    engine = sqlalchemy.create_engine(postgresdb_connection_uri)

    # Wait for ability to connect...
    while True:
        try:
            conn = engine.connect()
            conn.close()
        except:
            continue
        else:
            break

    try:
        yield engine
    finally:
        postgres.kill()
