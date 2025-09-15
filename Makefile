# Makefile for International Bond System
# Provides common development tasks and automation

.PHONY: help install install-dev test test-unit test-integration test-performance lint format type-check security-check clean build docs serve-docs docker-build docker-run deploy-local

# Default target
help:
	@echo "International Bond System - Development Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  install-all      Install all dependencies (dev, docs, ml, web)"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance tests only"
	@echo "  test-coverage    Run tests with coverage report"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint             Run all linting tools"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run mypy type checking"
	@echo "  security-check   Run bandit security analysis"
	@echo "  pre-commit       Run pre-commit hooks"
	@echo ""
	@echo "Build Commands:"
	@echo "  clean            Clean build artifacts"
	@echo "  build            Build package"
	@echo "  build-wheel      Build wheel package"
	@echo "  build-sdist      Build source distribution"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs             Build documentation"
	@echo "  serve-docs       Serve documentation locally"
	@echo "  docs-clean       Clean documentation build"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo "  docker-clean     Clean Docker images"
	@echo ""
	@echo "Deployment Commands:"
	@echo "  deploy-local     Deploy locally for testing"
	@echo "  deploy-staging   Deploy to staging environment"
	@echo "  deploy-prod      Deploy to production environment"

# Setup Commands
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

# Testing Commands
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/ -v --tb=short -m "unit"

test-integration:
	pytest tests/ -v --tb=short -m "integration"

test-performance:
	pytest tests/ -v --tb=short -m "performance"

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml

test-watch:
	pytest-watch tests/ -- -v --tb=short

# Code Quality Commands
lint: flake8 mypy bandit

flake8:
	flake8 src/ tests/

mypy:
	mypy src/

bandit:
	bandit -r src/ -f json -o bandit-report.json || bandit -r src/

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

type-check:
	mypy src/

security-check:
	bandit -r src/

pre-commit:
	pre-commit run --all-files

pre-commit-install:
	pre-commit install

# Build Commands
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

build-wheel:
	python -m build --wheel

build-sdist:
	python -m build --sdist

# Documentation Commands
docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

serve-docs:
	cd docs/_build/html && python -m http.server 8000

docs-live:
	sphinx-autobuild docs docs/_build/html --host 0.0.0.0 --port 8000

# Docker Commands
docker-build:
	docker build -t international-bond-system:latest .

docker-run:
	docker run -p 8080:8080 international-bond-system:latest

docker-clean:
	docker system prune -f
	docker image prune -f

# Development Commands
dev-setup: install-dev pre-commit-install
	@echo "Development environment setup complete!"

dev-test: format lint test
	@echo "Development testing complete!"

dev-check: format-check lint type-check security-check test
	@echo "All development checks passed!"

# Performance Commands
benchmark:
	python -m pytest tests/ -m "performance" --benchmark-only

profile:
	python -m cProfile -o profile.stats main.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Database Commands
db-init:
	python -c "from src.data.database import init_database; init_database()"

db-migrate:
	python -c "from src.data.database import migrate_database; migrate_database()"

db-reset:
	python -c "from src.data.database import reset_database; reset_database()"

# Data Commands
data-download:
	python scripts/download_data.py

data-update:
	python scripts/update_data.py

data-validate:
	python scripts/validate_data.py

# Deployment Commands
deploy-local: build
	pip install dist/*.whl --force-reinstall

deploy-staging:
	@echo "Deploying to staging environment..."
	# Add staging deployment commands here

deploy-prod:
	@echo "Deploying to production environment..."
	# Add production deployment commands here

# Monitoring Commands
logs:
	tail -f logs/application.log

monitor:
	python scripts/monitor_system.py

health-check:
	python scripts/health_check.py

# Utility Commands
requirements-update:
	pip-compile requirements.in
	pip-compile requirements-dev.in

requirements-sync:
	pip-sync requirements.txt requirements-dev.txt

version-bump-patch:
	bump2version patch

version-bump-minor:
	bump2version minor

version-bump-major:
	bump2version major

# CI/CD Commands
ci-test: dev-check test-coverage
	@echo "CI testing complete!"

ci-build: clean build
	@echo "CI build complete!"

ci-deploy: ci-test ci-build
	@echo "CI deployment ready!"