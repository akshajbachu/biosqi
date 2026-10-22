# Release checklist

## Before tagging

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src/biosqi
uv run pytest
uv run python benchmarks/run_all.py
uv build
```

Confirm that `benchmarks/results/results.json`, both SVG plots, the README
metric block, and `src/biosqi/models/default.json` changed in the same run.
Inspect the generated model name and provenance before release.

Build the dashboard and verify the bundled assets:

```bash
cd frontend
npm ci
npm run test:run
npm run build
cd ..
biosqi view
```

## Cutting a release

1. Bump `version` in `pyproject.toml`, `__version__` in
   `src/biosqi/__init__.py`, and `version`/`date-released` in `CITATION.cff`.
2. Move the `Unreleased` entries in `CHANGELOG.md` under the new version.
3. Commit, then tag and push:

   ```bash
   git tag v0.2.0
   git push origin main v0.2.0
   ```

The `Release` workflow then:

1. checks that the tag matches the package version,
2. builds the sdist and wheel,
3. publishes them to [PyPI](https://pypi.org/p/biosqi) via trusted publishing
   (GitHub environment `pypi`; no API token is stored), and
4. creates the GitHub release with generated notes and the built files.

[Zenodo](https://zenodo.org) archives every published GitHub release and mints
a version DOI. The concept DOI in the README always resolves to the latest
version.

Do not upload releases to Zenodo manually; the GitHub integration already
creates a record for each release, and a manual upload would produce a
separate, unrelated record.
