.PHONY: install test lint smoke clean

install:            ## local dev environment (Kaggle installs from requirements.txt)
	python -m pip install -e ".[]" && python -m pip install pytest ruff

test:
	python -m pytest -q

lint:
	ruff check src scripts tests

smoke:              ## one tiny epoch end-to-end; needs the dataset under data/
	python scripts/train.py --config configs/smoke.yaml

clean:
	rm -rf artifacts .pytest_cache **/__pycache__
