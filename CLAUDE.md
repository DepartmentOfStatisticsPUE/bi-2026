# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Educational repository for "Badania Internetowe 2025/26" (Internet Research) at Poznań University of Economics, taught by dr Maciej Beręsewicz. Covers survey estimation for non-probability samples: representativeness measures, inverse probability weighting (IPW), calibration estimators, and web scraping. All course content is in Polish.

## Architecture

Materials exist in parallel across four formats with a consistent `NN-topic` naming scheme:

| Format | Directory | Role |
|--------|-----------|------|
| Quarto (.qmd) | `codes/qmd/` | **Primary source** — narrative + executable R code |
| R scripts (.R) | `codes/R/` | **Generated** from .qmd via `knitr::purl()` |
| Python (.py) | `codes/python/` | Standalone Python implementations |
| Jupyter (.ipynb) | `codes/ipynb/` | Python notebooks (may be hand-edited) |

The Quarto files are the canonical source for R code. R scripts in `codes/R/` are derived artifacts.

Other directories:
- `slides/` — LaTeX Beamer presentation (`bi.tex`) with figures in `fig-*` subdirectories
- `project/` — Student project description, templates (`.qmd`, `.ipynb`), and example report
- `homeworks/` — Problem sets in `.qmd` and `.ipynb` format
- `codes/data/` — CSV datasets used in examples

## Key Commands

**Render a Quarto notebook to HTML:**
```bash
quarto render codes/qmd/03-ipw-1.qmd
```

**Extract R script from Quarto source:**
```bash
Rscript -e 'knitr::purl("codes/qmd/03-ipw-1.qmd", output="codes/R/03-ipw-1.R")'
```

**Run an R script:**
```bash
Rscript codes/R/03-ipw-1.R
```

**Run a Python script:**
```bash
python3 codes/python/03-ipw-1.py
```

**Convert Jupyter notebook to HTML:**
```bash
jupyter nbconvert --to html codes/ipynb/03-ipw-1.ipynb
```

## Conventions

- **Naming**: Files follow `NN-topic` pattern (e.g., `02-reprezentatywnosc`, `03-ipw-1`)
- **Chunk labels**: All code chunks in `.qmd` files must have a `#| label:` — use descriptive kebab-case names (e.g., `a-pakiety-r`, `zad-b1-py`, `b-przyklad-mle-r`)
- **Language**: All prose, variable names in data, and comments are in Polish
- **R packages**: `survey`, `sampling`, `nonprobsvy` (survey estimation); `rvest` (scraping)
- **Python packages**: `samplics`, `pandas`, `statsmodels`, `scikit-learn`, `selenium`
- **Homework submission**: Students submit a single HTML file via Moodle
- **Git-ignored directories**: `additional/`, `zaliczenie/`, `rozwiazania/` (solutions, student lists)
