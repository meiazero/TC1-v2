#!/bin/bash

rm -rf experiments-raw
rm -rf experiments-curated

python src/main.py --config src/config/models.yml --data data/raw/real-estate-valuation-dataset.csv --output experiments-raw --random-state 45

python src/main.py --config src/config/models.yml --data data/raw/real-estate-valuation-dataset.csv --output experiments-curated --random-state 45 --remove-outliers