# Contributing to biosqi

Thanks for your interest in improving biosqi. Bug reports, documentation fixes,
new validation datasets, and code contributions are all welcome.

## Reporting issues

- Search [existing issues](https://github.com/akshajbachu/biosqi/issues) first.
- For bugs, include the biosqi version
  (`python -c "import biosqi; print(biosqi.__version__)"`), your Python version
  and OS, the command or code you ran, and the full error output.
- If the problem depends on a recording, attach a small synthetic or public
  example. Never upload identifiable patient data.
- Report security vulnerabilities privately; see [SECURITY.md](SECURITY.md).

## Development setup

biosqi uses [uv](https://docs.astral.sh/uv/) for Python and npm for the
dashboard.

```bash
git clone https://github.com/akshajbachu/biosqi.git
cd biosqi
uv sync --extra dev

cd frontend
npm ci
cd ..
```

## Checks

Every pull request must pass the same checks CI runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/biosqi
uv run pytest

cd frontend
npm run lint
npm test -- --run
npm run build
```

The test suite enforces at least 90% coverage. Add or update tests alongside
any behavior change.

If you change an indicator, the fusion model, or calibration, rerun the
benchmarks and commit the regenerated results together with the code:

```bash
uv run python benchmarks/run_all.py
```

Benchmark numbers in the README are generated, never hand-edited.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep each pull request focused on one change.
3. Update documentation in `docs/` and add an entry to the `Unreleased`
   section of [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
4. Make sure all checks pass, then open the pull request and describe what
   changed and why.

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE) and that you will follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
