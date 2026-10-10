# Contributing to ds-agent-playbook

Thank you for contributing! We welcome contributions ranging from adding new recipes to enhancing statistical diagnostics and performance.

## Ways to Contribute
1. **Submit New Recipes**: Add real-world domain recipes in `recipes/` (e.g., time-series feature engineering, LLM dataset validation).
2. **Improve Cleaner Heuristics**: Expand `CleanerAgent` to support categorical encodings, outlier clamping, or correlation pruning.
3. **Benchmark Engines**: Add benchmarks comparing Polars/DuckDB execution speeds against Pandas.

## Development Workflow
1. Fork and clone the repository.
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Run test suite:
   ```bash
   uv run pytest tests/
   ```
4. Create a descriptive branch (`git checkout -b recipe/time-series-leakage`).
5. Open a Pull Request referencing any related issues.
