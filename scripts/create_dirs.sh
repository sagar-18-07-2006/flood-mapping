#!/usr/bin/env bash
set -e
mkdir -p data/raw data/interim data/processed data/outputs
touch data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep data/outputs/.gitkeep
echo "Data directories created."
