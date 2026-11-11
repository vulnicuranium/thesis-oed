# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Statistics honors thesis, "Certifying Optimal Experimental Design for Battery Aging" (proposal: `proposals/hu_thesis_proposal_v3.tex`, compile with `pdflatex` twice). The repo is an analysis workspace, not a software package: there is no build system, test suite, or linter. Work happens in Jupyter notebooks that build on the public Stroebl et al. (2024) battery aging dataset (figshare 25975315 v1; 279 cells, 71 test points, calendar "k" and cycle "z" sub-studies, two stages, designs `cLH` / `FF` / `pi`, where `pi` is the OED arm). `papers/` holds reference PDFs (git-ignored).

## Thesis context
- C1: pre-hoc admissibility diagnostic (operationalizes Stroebl R1–R3 from pilot data)
- C2: simulation coverage study of bootstrap / Fisher / profile-likelihood / mixed-effects intervals (the "4σ phenomenon")
- C3: robust/compromise designs evaluated on Severson/MATR, Sandia, and Stroebl data
- Deadlines: good draft Mar 9, 2027; final draft Apr 16, 2027

## Rules
- Never modify or delete anything in data/raw/.

## Setup and commands

- Environment: `conda env create -f environment.yml` (env `thesis-oed`, Python 3.11; VS Code is configured to use conda). Run notebooks with JupyterLab from the repo root or `notebooks/`.
- Download raw data: `bash src/download_stroebl.sh` (needs `jq`, macOS `md5`). It wipes and recreates `data/raw/stroebl/`, and writes `_metadata.json` there.
- Raw data and `*.parquet` outputs are git-ignored (`data/raw/`, `data/processed/`); regenerate them by running the notebooks.

## Pipeline architecture

Notebooks are numbered and must be run in order, because later ones read parquet files written by earlier ones. All paths are relative to `notebooks/` (`../data/...`).

1. `00_explore.ipynb`: initial exploration of the raw data.
2. `01_metadata.ipynb`: loads `experiments_meta.csv`, strips column whitespace, and builds the `tp_id` test-point key (`type_s<stage>_<sampling>_<tp>`). Confirms 279 cells and 71 test points.
3. `02_checkup.ipynb`: parses the check-up (`*_CU.csv`, etc.) files read straight out of each cell's zip (no extraction) to get capacity per test. It writes `data/processed/capacity_rpt.parquet`.
4. `03_eda.ipynb`: reads that parquet and produces the thesis figures in `figs/` (Fig 1 calendar aging, Figs 2 and 2b cycle aging by arm and C-rate, Fig 3 replicate noise). It also computes cumulative throughput (Ah, EFC) from the large `*_ZYK.csv` cycling blocks and attaches it to the table.

## Data gotchas (hard-won; keep these invariants)

- Cell identity: the authors' `serial_internal` is reused across cells, so it is not a unique key. Use `cell_id = file_name + "_s" + stage` (e.g. `TP_k01_02_s1`). The stage is inferred from the first test date (before 2022-06-01 is stage 1, otherwise stage 2), then joined to the metadata on `(file_name, stage)` with `validate="one_to_one"`.
- Merging on serial caused duplicate rows (fixed in commit 820d02b). A test is uniquely identified by `(zip, cu, kind)`, so check `cap.duplicated(subset=["zip","cu","kind"])` after any merge.
- Metadata column names have stray whitespace (e.g. `" volt_meter"`), so strip them after reading.
- Regular check-ups are `kind in ["CU","exCU"]`. Entry and end tests use a different protocol. Also filter `n_dis_rows > 0` to drop tests with no capacity discharge.
- Relative capacity `Q_rel` is `Q_Ah` divided by the cell's first check-up. Replicate alignment across cells at a test point uses the check-up's order within the cell, not the date.
- Dates in the raw files are `dd.mm.yyyy` or `dd.mm.yy` and include typos, so parse defensively and yield NaT. Failures are logged rather than raised.

## Conventions

- Notebook cells are written to be self-contained (reload the parquet from disk rather than depending on earlier kernel state) and carry a header comment saying what the cell tests.
- Comments: concise and professional, describing what the code does. Comment non-obvious logic and statistical choices, not every line. Never reference the conversation (no "the table you just built").
- Figures are saved to `figs/` and referenced from the LaTeX proposal.
