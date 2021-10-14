load-data:
	PGPASSWORD=postgres psql -f pagila/pagila-schema.sql  -h localhost -p 5432 -U postgres -w
	PGPASSWORD=postgres psql -f pagila/pagila-data.sql  -h localhost -p 5432 -U postgres -w

bench:
	python -m pytest -v --benchmark-sort name --benchmark-only benchmarks
