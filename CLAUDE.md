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

Numbered topics (`00`–`07`) cover: `survey`/`sampling`/`samplics` packages → web scraping → representativeness measures → IPW (intro, calibrated, exercise) → mass imputation (NN/PMM, GLM). `codes/qmd/00-wykresy-wyklady.qmd` is a separate notebook used to generate figures for the lecture slides (not student-facing material) — leave it out of the `readme.md` notebook table.

Other directories:
- `slides/` — LaTeX Beamer presentation (`bi.tex`, compiles with `pdflatex` from inside `slides/`); figures are pre-generated PNGs in `fig-*` subdirectories
- `project/` — Student project description (`opis-projektu.md`), templates (`szablon.qmd`, `szablon.ipynb`), and example report (`projekt-przyklad.qmd` → `projekt-przyklad.html`)
- `homeworks/` — Problem sets in `.qmd` and `.ipynb` format
- `codes/data/` — CSV datasets used in examples (`admin.csv`, `jvs.csv`, `g2016.csv`)

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
- **Homework submission**: Students submit a single HTML file via Moodle — keep `embed-resources: true` (or `self-contained: true` on older `.qmd` files) in YAML headers so renders are portable
- **Git-ignored directories**: `additional/`, `zaliczenie/`, `rozwiazania/` (solutions, student lists)
- **HTML output location**: notebooks 00–02 render to `codes/`, 03+ render to `codes/qmd/` — follow the existing convention per topic when updating `readme.md` links

## Quarto with Python chunks (reticulate)

When a `.qmd` file mixes R and Python chunks (`panel-tabset`), reticulate by default uses `~/.virtualenvs/r-reticulate/bin/python` which lacks `pandas`/`scipy`/`statsmodels`. Pin the Anaconda Python via a hidden setup chunk at the top:

```r
#| label: setup
#| include: false
library(reticulate)
use_python("/opt/anaconda3/bin/python3", required = TRUE)
```

For exercise/homework `.qmd` files where students fill in code, mark each placeholder chunk individually with `#| eval: false` (rather than setting it document-wide) so example chunks still execute and the document compiles end-to-end.

## IPW estimators in `nonprobsvy`

R's `nonprob()` defaults to the **IPW 1 (Horvitz-Thompson)** estimator: $\hat{\mu} = \sum w_i y_i / \hat{N}$ where $\hat{N} = \sum_{i \in S_B} d_i^B = $ `sum(weights(svydesign))`. It is **not** the Hájek estimator ($\sum w_i y_i / \sum w_i$). To replicate R results in Python:

- Compute `N_pop = jvs["weight"].sum()` and use it as the denominator
- For MLE: solve the pseudo-likelihood score equation `U(γ) = Σ_{S_A} x_i − Σ_{S_B} d_i^B π(x_i, γ) x_i = 0` via `scipy.optimize.fsolve` — `statsmodels.GLM` with `freq_weights` fits a different objective and gives different estimates
- For GEE: weights are calibrated so `sum(w) == N_pop`; HT and Hájek coincide

## Data quirks (`nonprobsvy::admin` and `nonprobsvy::jvs`)

- `single_shift` (target) exists **only in `admin`**, not in `jvs` — there is no JVS-based "true value" benchmark
- `region` is stored as zero-padded character (`"02"`, `"04"`, …, `"32"`); when reading from the CSV in Python, convert with `df["region"].astype(str).str.zfill(2)` so dummy encoding matches R's factor levels
- Treat `size`, `nace`, `region` as categorical (dummy-encode); `private` as numeric (0/1)
