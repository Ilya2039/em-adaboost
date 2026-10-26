#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
if [ ! -d data/Datasets ]; then
    git clone --depth 1 https://github.com/jbrownlee/Datasets data/Datasets
fi
if [ ! -d data/weka ]; then
    mkdir -p data/weka
    unzip -q data/Datasets/weka-datasets.zip -d data/weka
fi
python datasets.py
