#!/usr/bin/env bash

for N in {1..9}
do
python3 genres.py genres/blues/blues.0000$N.wav genres/blues/blues.0000$N.wav 1 1 1 1
python3 genres.py genres/classical/classical.0000$N.wav genres/classical/classical.0000$N.wav 1 1 1 1
python3 genres.py genres/country/country.0000$N.wav genres/country/country.0000$N.wav 1 1 1 1
python3 genres.py genres/disco/disco.0000$N.wav genres/disco/disco.0000$N.wav 1 1 1 1
python3 genres.py genres/hiphop/hiphop.0000$N.wav genres/hiphop/hiphop.0000$N.wav 1 1 1 1
python3 genres.py genres/jazz/jazz.0000$N.wav genres/jazz/jazz.0000$N.wav 1 1 1 1
python3 genres.py genres/metal/metal.0000$N.wav genres/metal/metal.0000$N.wav 1 1 1 1
python3 genres.py genres/pop/pop.0000$N.wav genres/pop/pop.0000$N.wav 1 1 1 1
python3 genres.py genres/reggae/reggae.0000$N.wav genres/reggae/reggae.0000$N.wav 1 1 1 1
python3 genres.py genres/rock/rock.0000$N.wav genres/rock/rock.0000$N.wav 1 1 1 1
done

for N in {10..99}
do
python3 genres.py genres/blues/blues.000$N.wav genres/blues/blues.000$N.wav 1 1 1 1
python3 genres.py genres/classical/classical.000$N.wav genres/classical/classical.000$N.wav 1 1 1 1
python3 genres.py genres/country/country.000$N.wav genres/country/country.000$N.wav 1 1 1 1
python3 genres.py genres/disco/disco.000$N.wav genres/disco/disco.000$N.wav 1 1 1 1
python3 genres.py genres/hiphop/hiphop.000$N.wav genres/hiphop/hiphop.000$N.wav 1 1 1 1
python3 genres.py genres/jazz/jazz.000$N.wav genres/jazz/jazz.000$N.wav 1 1 1 1
python3 genres.py genres/metal/metal.000$N.wav genres/metal/metal.000$N.wav 1 1 1 1
python3 genres.py genres/pop/pop.000$N.wav genres/pop/pop.000$N.wav 1 1 1 1
python3 genres.py genres/reggae/reggae.000$N.wav genres/reggae/reggae.000$N.wav 1 1 1 1
python3 genres.py genres/rock/rock.000$N.wav genres/rock/rock.000$N.wav 1 1 1 1
done
