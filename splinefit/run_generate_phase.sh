#!/bin/sh

seed=1001
bsub -o genphase.o python3 1_generate_phase_drift_rndmwalk_segmented.py $seed

