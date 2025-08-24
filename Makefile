.PHONY: sync fmt lint test clean all install dev

# Default target
all: sync lint test

# Environment setup
sync:
	uv sync

install: sync

dev: sync
	uv sync --group dev

# Code formatting and linting
fmt:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run mypy lldc

# Testing
test:
	uv run pytest -q

test-cov:
	uv run pytest --cov=lldc --cov-report=term-missing

test-verbose:
	uv run pytest -v

# Running experiments
run-exp:
	uv run python -m lldc.scripts.sweeps $(ARGS)

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

# Combined targets
check: lint test

ci: sync lint test
