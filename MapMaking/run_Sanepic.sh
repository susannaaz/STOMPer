#!/bin/sh

work_dir=/group/cmb/litebird/usr/ysakurai/HWP_IP/
suffix=_v28_nside256_CMB_fg_dipole_shwp5
#suffix=_v28_nside256_CMB_fg_dipole_shwp5_fknee_0.05Hz_fsamp_19Hz_alpha_1
sed -i s/SUFFIX/${suffix}/ mpirun_Sanepic.sh
bsub -J "Job_mpi" -o ${work_dir}/log/log_mpi_Sanepic.o < mpirun_Sanepic.sh
sed -i s/${suffix}/SUFFIX/ mpirun_Sanepic.sh

