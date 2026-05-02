# methylcurate

![PyPI version](https://img.shields.io/pypi/v/methylcurate.svg)

Agentic-AI Tool for GEO dataset retrieval, metadata harmonization, and aging clock evaluation.

* [GitHub](https://github.com/travyse/methylcurate/) | [PyPI](https://pypi.org/project/methylcurate/) | [Documentation](https://travyse.github.io/methylcurate/)
* Created by [Travyse Anthony Edwards](https://travyse.github.io) | GitHub [@travyse](https://github.com/travyse) | PyPI [@travyse](https://pypi.org/user/travyse/)
* MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://travyse.github.io/methylcurate/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/methylcurate.git
cd methylcurate

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `methylcurate`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

methylcurate was created in 2026 by Travyse Anthony Edwards.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
