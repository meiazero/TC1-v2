#!/bin/bash

rm -rf experiments-raw-v2

python src/main.py --config src/config/models.yml --data data/raw/real-estate-valuation-dataset.csv --output experiments-raw-v2 --random-state 45

# python src/main.py --config src/config/models.yml --data data/raw/real-estate-valuation-dataset.csv --output experiments-curated-v2 --random-state 45 --remove-outliers