#!/usr/bin/env bash

INPUT="data/SpringDay_30.wav"
OUTPUT="data/HighwayToHell_30.wav"
EPOCHS=2000

python3 main.py $INPUT $OUTPUT 1 1e2 $EPOCHS 0.2
python3 main.py $INPUT $OUTPUT 1 1e3 $EPOCHS 0.2
python3 main.py $INPUT $OUTPUT 1 1e4 $EPOCHS 0.2
python3 main.py $INPUT $OUTPUT 1 1e5 $EPOCHS 0.2
python3 main.py $INPUT $OUTPUT 1 1e6 $EPOCHS 0.2

python3 main.py $INPUT $OUTPUT 1 1e2 $EPOCHS 1
python3 main.py $INPUT $OUTPUT 1 1e3 $EPOCHS 1
python3 main.py $INPUT $OUTPUT 1 1e4 $EPOCHS 1
python3 main.py $INPUT $OUTPUT 1 1e5 $EPOCHS 1
python3 main.py $INPUT $OUTPUT 1 1e6 $EPOCHS 1

python3 main.py $INPUT $OUTPUT 1 1e2 $EPOCHS 3
python3 main.py $INPUT $OUTPUT 1 1e3 $EPOCHS 3
python3 main.py $INPUT $OUTPUT 1 1e4 $EPOCHS 3
python3 main.py $INPUT $OUTPUT 1 1e5 $EPOCHS 3
python3 main.py $INPUT $OUTPUT 1 1e6 $EPOCHS 3

python3 main.py $INPUT $OUTPUT 1 1e2 $EPOCHS 7
python3 main.py $INPUT $OUTPUT 1 1e3 $EPOCHS 7
python3 main.py $INPUT $OUTPUT 1 1e4 $EPOCHS 7
python3 main.py $INPUT $OUTPUT 1 1e5 $EPOCHS 7
python3 main.py $INPUT $OUTPUT 1 1e6 $EPOCHS 7