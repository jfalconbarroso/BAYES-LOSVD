#!/bin/bash
#############################
#SBATCH -J Wings
#SBATCH -p batch
#SBATCH -n 5
#SBATCH -t 1-00:00:00
#SBATCH -o log_Wings-%j.out
#SBATCH -e log_Wings-%j.err
#SBATCH -D .
#############################
module load mpi
echo "EXECUTING Wings!"
export OMP_NUM_THREADS=5
# python bayes-losvd_run.py -n 5 -i 1500 -c 1 -f ../preproc_data/testcases_Wings.hdf5 -t S0
# python bayes-losvd_run.py -n 5 -i 1500 -c 1 -f ../preproc_data/testcases_Wings.hdf5 -t S1
# python bayes-losvd_run.py -n 5 -i 1500 -c 1 -f ../preproc_data/testcases_Wings.hdf5 -t A1
# python bayes-losvd_run.py -n 5 -i 1500 -c 1 -f ../preproc_data/testcases_Wings.hdf5 -t A2
# python bayes-losvd_run.py -n 5 -i 1500 -c 1 -f ../preproc_data/testcases_Wings.hdf5 -t A3
# python bayes-losvd_run.py -n 5 -i 1500 -c 1 -f ../preproc_data/testcases_Wings.hdf5 -t B3
# python bayes-losvd_run.py -n 5 -i 1500 -c 1 -f ../preproc_data/testcases_Wings.hdf5 -t B4
python bayes-losvd_run.py -n 5 -i 1500 -c 1 -f ../preproc_data/testcases_Wings.hdf5 -t A4
