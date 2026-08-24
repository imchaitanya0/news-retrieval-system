.PHONY: data pipeline

data:
	python -m src.data.download

pipeline:
	python -m src.data.build_pipeline
