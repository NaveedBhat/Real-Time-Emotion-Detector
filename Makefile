VENV   := venv
PYTHON := ./$(VENV)/bin/python3
PIP    := ./$(VENV)/bin/pip

.PHONY: run serve test lint freeze install clean help

## Install venv and all dependencies
install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

## Run the emotion detector
run:
	$(PYTHON) main.py

## Run the multi-face web dashboard
serve:
	$(PYTHON) server.py

## Run with debug logging
debug:
	$(PYTHON) main.py --debug

## Run unit tests
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

## Lint with flake8 (max line length 100)
lint:
	$(PYTHON) -m flake8 src/ main.py --max-line-length=100 --ignore=E501

## Freeze all installed packages to requirements.lock
freeze:
	$(PIP) freeze > requirements.lock
	@echo "Locked dependency tree written to requirements.lock"

## Remove Python cache files
clean:
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "./$(VENV)/*" -delete 2>/dev/null || true
	@echo "Cache cleared."

## Show available commands
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  install  Create venv and install requirements"
	@echo "  run      Run the OpenCV app (main.py)"
	@echo "  serve    Run the web dashboard (server.py)"
	@echo "  debug    Run with debug logging"
	@echo "  test     Run unit tests with pytest"
	@echo "  lint     Run flake8 linter"
	@echo "  clean    Remove cache files"
	@echo "  freeze   Update requirements.txt with exact versions"
	@echo "  help     Show this help message"
