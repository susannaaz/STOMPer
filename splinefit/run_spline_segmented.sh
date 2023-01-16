#!/bin/sh

seed=1001 
# init, step, end
for i in `seq 0 500000 2520500000`
do echo $i
    bsub -o bsub_spline_runs/${i}.o python3 2_spline_interpolation_segmented.py $i $seed
done




