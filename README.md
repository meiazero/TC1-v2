# Real Estate Valuation Pipeline

ML pipeline for regression on the Real Estate Valuation Dataset.

## Setup

This project uses **Poetry** for dependency management. If you don't have Poetry installed, follow the instructions at https://python-poetry.org/docs/#installation

```bash
# Install dependencies
poetry install

# (Optional) Activate the virtual environment
poetry env activate
```

## Usage

Run the pipeline with:
```bash
python main.py
```

## Code Formatting and Linting

- **Black** for formatting: `poetry run black .`
- **isort** for import sorting: `poetry run isort .`
- **Flake8** for linting: `poetry run flake8`
- **Mypy** for static type checking: `poetry run mypy`