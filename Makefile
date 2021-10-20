default:
	cargo build --release
	pip install -e .

load-data:
	PGPASSWORD=postgres psql -f pagila/pagila-schema.sql  -h localhost -p 5432 -U postgres -w
	PGPASSWORD=postgres psql -f pagila/pagila-data.sql  -h localhost -p 5432 -U postgres -w

bench:
	python -m pytest -v --benchmark-sort name --benchmark-only benchmarks

memory-profile-rust:
	cargo build --example memory_profile --release
	chmod +x target/release/examples/memory_profile
	mprof run ./target/release/examples/memory_profile
	mprof plot

memory-profile-python:
	python -m memory_profiler benchmarks/test_benchmarks.py

heaptrack:
	cargo build --example memory_profile --release
	chmod +x target/release/examples/memory_profile
	heaptrack ./target/release/examples/memory_profile

clean:
	rm ./mprofile_*
	rm ./heaptrack*
