# Reproduce the study. `make verify` needs nothing but the repo; the rest needs data.
PYTHON  ?= python3.12
PY      := ./.venv/bin/python
GHSL_DIR ?= data/raw/ghsl
ORIG_PROCESSED ?=
export GHSL_DIR
PYSRC   := PYTHONPATH=src $(PY) -W ignore
XR      := external/cross-region
XR_COMMIT := 649e9ad
GDAL_NET := GDAL_HTTP_TIMEOUT=60 GDAL_HTTP_CONNECTTIMEOUT=20 GDAL_HTTP_MAX_RETRY=6 GDAL_HTTP_RETRY_DELAY=3 \
            GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif PYTHONUNBUFFERED=1
R8      := south_asia ssa east_asia andes mena sea eeca oceania

.PHONY: env test verify pilot external processed landsat r3-matrix features embeddings analysis figure all

env:                ## Python 3.11+ virtual environment
	$(PYTHON) -m venv .venv && $(PY) -m pip install -r requirements.txt

test:               ## method checks on synthetic inputs (no data)
	$(PY) -m pytest -q tests/

verify:             ## re-derive headline numbers from committed results (no data)
	$(PY) -W ignore verify.py

pilot:              ## outcome-side pilots run before the pre-registration
	$(PY) -W ignore -I pilot/pilot_decomp.py data/receipts > pilot/outputs/pilot_decomp.txt
	$(PY) -W ignore -I pilot/pilot_outcome_only.py data/receipts > pilot/outputs/pilot_outcome_only.txt
	$(PY) -W ignore pilot/pilot_ceilings_power.py > pilot/outputs/pilot_ceilings_power.txt

external:           ## cross-region training code at the pinned commit, plus the R3 rerun script
	test -d $(XR)/.git || git clone https://github.com/rohanbalixz/Cross-Region-Source-Invariance-in-Earth-Observation $(XR)
	git -C $(XR) checkout -q $(XR_COMMIT)
	cp rerun/landsat_matrix_dump.py $(XR)/scripts/eval/landsat_matrix_dump.py

processed:          ## per-city GHSL rasters from the global GHSL R2023A files in $(GHSL_DIR)
	$(PYSRC) -m geotransfer.build_processed --out $(XR)/data/processed
	test -z "$(ORIG_PROCESSED)" || $(PYSRC) -m geotransfer.build_processed --out $(XR)/data/processed \
	    --check-against $(ORIG_PROCESSED)

landsat:            ## raw Landsat history, 220 cities, Microsoft Planetary Computer (no credentials)
	cd $(XR) && cut -d' ' -f1 ../../data/cities.txt | $(GDAL_NET) xargs -P 4 -I{} \
	    ../../$(PY) -W ignore -m scripts.acquire.landsat_history --city {}

r3-matrix:          ## train 20 regions x 5 seeds on raw Landsat, city-disjoint scoring
	cd $(XR) && ../../$(PY) -W ignore -m scripts.eval.landsat_matrix_dump
	cp $(XR)/results/metrics/landsat_matrix_dump_multiseed.json data/receipts/
	mkdir -p data/features/landsat && cp $(XR)/results/landsat_dump/features/*.npz data/features/landsat/

features:           ## GHSL tile features for the 8 harmonised regions (needs GHSL_DIR)
	$(PYSRC) -m geotransfer.ghsl_features --regions $(R8) --out data/features/ghsl

embeddings:         ## GeoCLIP and SatCLIP-ResNet50 L40/L10 at tile centres
	$(PYSRC) -m geotransfer.location_embeddings --features data/features/ghsl
	$(PYSRC) -m geotransfer.location_embeddings --features data/features/landsat

analysis:           ## the pre-registered analysis (refuses unless the pre-registration is committed)
	$(PY) -W ignore src/run_analysis.py --regimes R1 R2 R3

figure:
	$(PY) -W ignore src/make_figure.py

all: test features embeddings analysis figure verify
