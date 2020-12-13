#!/usr/bin/env bash

for N in 1 2 .. 9
do
python3 main.py genres/blues/blues.0000$N.wav genres/blues/blues.0000$N.wav 1 1 1 1
python3 main.py genres/classical/classical.0000$N.wav genres/classical/classical.0000$N.wav 1 1 1 1
python3 main.py genres/country/country.0000$N.wav genres/country/country.0000$N.wav 1 1 1 1
python3 main.py genres/disco/disco.0000$N.wav genres/disco/disco.0000$N.wav 1 1 1 1
python3 main.py genres/hiphop/hiphop.0000$N.wav genres/hiphop/hiphop.0000$N.wav 1 1 1 1
python3 main.py genres/jazz/jazz.0000$N.wav genres/jazz/jazz.0000$N.wav 1 1 1 1
python3 main.py genres/metal/metal.0000$N.wav genres/metal/metal.0000$N.wav 1 1 1 1
python3 main.py genres/pop/pop.0000$N.wav genres/pop/pop.0000$N.wav 1 1 1 1
python3 main.py genres/reggae/reggae.0000$N.wav genres/reggae/reggae.0000$N.wav 1 1 1 1
python3 main.py genres/rock/rock.0000$N.wav genres/rock/rock.0000$N.wav 1 1 1 1
done

for N in 10 11 .. 99
do
python3 main.py genres/blues/blues.000$N.wav genres/blues/blues.000$N.wav 1 1 1 1
python3 main.py genres/classical/classical.000$N.wav genres/classical/classical.000$N.wav 1 1 1 1
python3 main.py genres/country/country.000$N.wav genres/country/country.000$N.wav 1 1 1 1
python3 main.py genres/disco/disco.000$N.wav genres/disco/disco.000$N.wav 1 1 1 1
python3 main.py genres/hiphop/hiphop.000$N.wav genres/hiphop/hiphop.000$N.wav 1 1 1 1
python3 main.py genres/jazz/jazz.000$N.wav genres/jazz/jazz.000$N.wav 1 1 1 1
python3 main.py genres/metal/metal.000$N.wav genres/metal/metal.000$N.wav 1 1 1 1
python3 main.py genres/pop/pop.000$N.wav genres/pop/pop.000$N.wav 1 1 1 1
python3 main.py genres/reggae/reggae.000$N.wav genres/reggae/reggae.000$N.wav 1 1 1 1
python3 main.py genres/rock/rock.000$N.wav genres/rock/rock.000$N.wav 1 1 1 1
done
