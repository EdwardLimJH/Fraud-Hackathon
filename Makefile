.ONESHELL:

all: venv dataset features train predict evaluate

OS := $(shell uname)

venv: venv/bin/activate
ifeq ($(OS), Linux)
	. ./venv/bin/activate
else # Windows
	./venv/Scripts/activate.bat
endif

venv/bin/activate: requirements.txt
	python -m venv venv
ifeq ($(OS), Linux)	# Linux or macOS
	chmod +x venv/bin/activate
	. ./venv/bin/activate
else # Windows
	python -m venv venv
	./venv/Scripts/activate.bat
endif
	pip install -r requirements.txt

dataset: 
	python ./src/data/make_dataset.py -i ./data/raw/base.zip -o ./data/raw/

features:
	python ./src/features/build_features.py -i ./data/raw/base.csv -r ./data/interim/reverse_na.csv -g ./data/interim/new_features.csv -n ./data/interim/no_missing.csv -e ./data/interim/processed.csv -xn ./data/processed/X_train_preresampling.csv -m ./data/processed/min_max_scaler.pkl -yn ./data/processed/y_train_preresampling.csv -xtr ./data/processed/X_train_resampled.csv -ytr ./data/processed/y_train_resampled.csv -xt ./data/processed/X_test.csv -yt ./data/processed/y_test.csv

train:
	python ./src/models/train_models.py -ix ./data/processed/X_train_resampled.csv -iy ./data/processed/y_train_resampled.csv -m logisticregression -s ./models/logisticregression.pkl

predict:
	python ./src/models/predict_model.py -s ./models/logisticregression.pkl -o ./data/predictions/logisticregression_predictions.csv -ix ./data/processed/X_test.csv

evaluate:
	python ./src/models/evaluate_model.py -yp ./data/predictions/logisticregression_predictions.csv -o ./data/predictions/logisticregression_eval.csv -yt ./data/processed/y_test.csv


visualize:
	python ./src/visualizations/create_visualizations.py -i ./data/raw/base.csv -o ./reports/