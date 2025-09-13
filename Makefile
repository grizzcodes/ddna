# Makefile for DDNA - Screenplay DNA System

.PHONY: help install test clean run docker-build docker-run format lint

# Variables
PYTHON := python3
PIP := pip3
VENV := venv
PROJECT := ddna

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make dev-install  - Install with dev dependencies"
	@echo "  make test        - Run tests"
	@echo "  make format      - Format code with black"
	@echo "  make lint        - Run linting"
	@echo "  make clean       - Clean temporary files"
	@echo "  make run         - Run the application"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run  - Run Docker container"

install:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install --upgrade pip
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt
	. $(VENV)/bin/activate && $(PIP) install -e .
	@echo "Installation complete. Activate venv with: source $(VENV)/bin/activate"

dev-install: install
	. $(VENV)/bin/activate && $(PIP) install -e ".[dev]"
	. $(VENV)/bin/activate && pre-commit install

test:
	. $(VENV)/bin/activate && pytest tests/ -v --cov=$(PROJECT) --cov-report=html

format:
	. $(VENV)/bin/activate && black $(PROJECT) tests

lint:
	. $(VENV)/bin/activate && flake8 $(PROJECT) tests
	. $(VENV)/bin/activate && mypy $(PROJECT)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true

run:
	. $(VENV)/bin/activate && python -m $(PROJECT).cli

docker-build:
	docker build -t $(PROJECT):latest .

docker-run:
	docker run -it --rm \
		-v $(PWD)/input:/app/input \
		-v $(PWD)/output:/app/output \
		-v $(PWD)/models:/app/models \
		--gpus all \
		$(PROJECT):latest

# Development shortcuts
dev: dev-install
	@echo "Development environment ready!"

quick-test:
	. $(VENV)/bin/activate && pytest tests/ -v -x

serve:
	. $(VENV)/bin/activate && uvicorn $(PROJECT).api:app --reload --host 0.0.0.0 --port 8000