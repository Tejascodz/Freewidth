.PHONY: install dev lint test smoke serve-mock bench-mock clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

lint:
	ruff check freewidth tests

test:
	python -m pytest -q

smoke:
	python -m freewidth smoke --out runs/smoke

serve-mock:
	python -m freewidth serve --config configs/engine_mock.yaml --port 8000

bench-mock:
	python -m freewidth bench --config configs/engine_mock.yaml --dataset toy --limit 50 --out runs/bench_mock

clean:
	rm -rf runs .pytest_cache
	find . -name __pycache__ -prune -exec rm -rf {} +
