# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT:** Do not include attribution to Claude in the commit message or co-author attribution. Ever.

**IMPORTANT:** Never commit changes to git unless explicitly asked. Only make git commits when the user specifically requests it.

**IMPORTANT:** Never disable tests, linting, or type checking in CI scripts without explicit user permission.

## Build Commands
- Install: `poetry install`
- Run CLI: `recsys-lite --help`
- Test: `pytest` (single test: `pytest path/to/test.py::test_function`)
- Lint: `ruff .`
- Format: `ruff format .`
- Check formatting: `ruff check .`
- Type check: `mypy .`

## CI Commands
- Run all CI checks locally: `./run_ci_locally.sh`
- Install pre-push hook: `cp pre-push.hook .git/hooks/pre-push && chmod +x .git/hooks/pre-push`

## Code Style Guidelines
- Python 3.11+ with static typing
- Use Black for code formatting (line length 120)
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