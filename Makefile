# Reproducible workflow for the neuron-alignment experiments. See README.md.
#   make env         create .venv with the pinned dependencies
#   make results     figures + tables from the committed checkpoint in artifacts/runs (< 1 min)
#   make reproduce   everything from raw data: download MNIST, all training, figures, tables
#   make test        run the test suite

PYTHON ?= python3
VENV   := .venv
PY     := $(VENV)/bin/python
PATCH  ?= 14

.PHONY: help env check-env reproduce pipeline data base sweeps mnist results test test-fast clean

help:  ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-10s %s\n", $$1, $$2}'

env:  ## create .venv and install requirements.txt plus this package
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install --no-deps -e .

check-env:
	@test -x $(PY) || { echo "error: $(PY) not found; run 'make env' first" >&2; exit 1; }
	@$(PY) -c "import numpy, matplotlib, alignment" 2>/dev/null || \
		{ echo "error: dependencies missing from $(VENV); run 'make env'" >&2; exit 1; }

reproduce: pipeline results  ## full run from raw data (about 20-30 min on a laptop)

pipeline: data base sweeps mnist  ## regenerate data/processed and artifacts/runs

data: check-env  ## download MNIST (MD5-checked) and extract digits 0 and 1
	$(PY) -m pipeline.prepare_mnist

base: check-env  ## base run (about 10 s)
	$(PY) -m pipeline.run_base

sweeps: check-env  ## four parameter sweeps, 5 seeds each (about 15-25 min)
	$(PY) -m pipeline.run_sweeps

mnist: check-env  ## MNIST 0 vs 1 run (about 1 min)
	$(PY) -m pipeline.run_mnist --patch $(PATCH)

results: check-env  ## figures and tables in results/ from artifacts/runs
	$(PY) -m analysis.fig_base
	$(PY) -m analysis.fig_ablation
	$(PY) -m analysis.fig_mnist --patch $(PATCH)
	$(PY) -m analysis.make_tables --patch $(PATCH)

test: check-env  ## all tests, including the full base-run regression test (about 15 s)
	$(PY) -m pytest

test-fast: check-env  ## tests without the slow regression test
	$(PY) -m pytest -m "not slow"

clean:  ## delete generated figures and tables (artifacts and data are kept)
	rm -rf results/figures/* results/tables/*
