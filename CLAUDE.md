# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT:** Do not include attribution to Claude in the commit message or co-author attribution. Ever.

## Build Commands
- Install: `poetry install`
- Run CLI: `recsys-lite --help`
- Test: `pytest` (single test: `pytest path/to/test.py::test_function`)
- Lint: `ruff .`
- Format: `black .`
- Sort imports: `isort .`
- Type check: `mypy .`

## Code Style Guidelines
- Python 3.11+ with static typing
- Use Black for code formatting (line length 88)
- Sort imports with isort: standard library, third-party, local
- Type annotations for all functions and variables
- Function/variable names: snake_case
- Class names: PascalCase
- Constants: UPPER_SNAKE_CASE
- Use context managers for resource handling
- Handle errors explicitly with appropriate try/except blocks
- Document public functions with docstrings
- Keep functions small and focused on a single responsibility
- Prefer composition over inheritance
- Use PathLib for file operations