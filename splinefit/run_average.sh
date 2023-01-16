#!/bin/sh

seed=1001

seg=19
for i in `seq 0 19 5039`
do echo $i
    sum=$(expr $i + $seg)
    bsub -o bsub_outs_linearint/${i}.o python3 3_average_from_spline.py $i $sum $seed
done

