#!/usr/bin/env bash

INPUT="data/SpringDay_30.wav"
OUTPUT="data/HighwayToHell_30.wav"
EPOCHS=10000

python3 main.py $INPUT $OUTPUT 1 1e5 $EPOCHS 1