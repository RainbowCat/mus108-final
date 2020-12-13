#!/usr/bin/env bash

python3 main.py data/Debussy_30.wav data/SpringDay_30.wav 1 1e5 2000 0.1
python3 main.py data/Debussy_30.wav data/SpringDay_30.wav 1e5 1 2000 0.1

python3 main.py data/Debussy_30.wav data/SpringDay_30.wav 1 1e5 2000 1
python3 main.py data/Debussy_30.wav data/SpringDay_30.wav 1e5 1 2000 1