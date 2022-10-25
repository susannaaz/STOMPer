#!/bin/bash
#BSUB -n 24
#BSUB -q p

module load openmpi/2.1.6-gcc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/gcc-9.3.0/lib64:/opt/gcc-9.3.0/lib:/opt/gnu-libs/:/group/cmb/litebird/usr/ysakurai/packages/cfitsio/lib/

## SET DEFAULT VALUE FOR ARGUMENTS
suffix=SUFFIX
work_dir=/group/cmb/litebird/usr/ysakurai/HWP_IP/
bolo=000_001_000_QA_140_T
test=0

mpirun -np 24 --output-filename ${work_dir}/log/log_mpi \
    ./Sanepic \
    -F ${work_dir}data/ \
    -B ${suffix} \
    -f ${work_dir}/Database/samplistLB_143.bi \
    -d 1 \
    -X ${work_dir}/Database/Bolofile_IMo_v1.3.txt \
    -Z ${work_dir}/Pointing/ \
    -P _LB_45_50_LB_v28_19Hz \
    -d 1 \
    -C ${bolo} \
    -p 1 \
    -n 143 \
    -l 4194304 \
    -N 256 \
    -h 1 \
    -w 0.25353203871075525 \
    -O ${work_dir}/Maps/Sanepic \
    -e ${bolo}${suffix} \
    -k ${work_dir}/Noise/iSpf_fknee_0.05Hz_fsamp_19Hz_alpha_1_ \
    -u 4194304
