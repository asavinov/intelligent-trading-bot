#!/bin/bash

python -m scripts.download_binance -c config.json
python -m scripts.merge -c config.json
python -m scripts.features -c config.json
python -m scripts.labels -c config.json
python -m scripts.train -c config.json
python -m scripts.signals -c config.json
python -m scripts.train_signals -c config.json
python -m service.server -c config.json
