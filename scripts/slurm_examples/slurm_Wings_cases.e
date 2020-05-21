#!/bin/bash
#############################
#SBATCH -J Wings
#SBATCH -p batch
#SBATCH -n 10
#SBATCH -t 1-00:00:00
#SBATCH -o log_Wings-%j.out
#SBATCH -e log_Wings-%j.err
#SBATCH -D .
#############################
module load mpi
echo "EXECUTING Wings!"
export OMP_NUM_THREADS=10
python bayes-losvd_run.py -s 1 -n 5 -i 1500 -c 2 -f ../preproc_data/testcases_Wings.hdf5 -t S0 -b 10,30,60,90,110
python bayes-losvd_run.py -s 1 -n 5 -i 1500 -c 2 -f ../preproc_data/testcases_Wings.hdf5 -t S1 -b 10,30,60,90,110
python bayes-losvd_run.py -s 1 -n 5 -i 1500 -c 2 -f ../preproc_data/testcases_Wings.hdf5 -t A1 -b 10,30,60,90,110
python bayes-losvd_run.py -s 1 -n 5 -i 1500 -c 2 -f ../preproc_data/testcases_Wings.hdf5 -t A2 -b 10,30,60,90,110
python bayes-losvd_run.py -s 1 -n 5 -i 1500 -c 2 -f ../preproc_data/testcases_Wings.hdf5 -t A3 -b 10,30,60,90,110
python bayes-losvd_run.py -s 1 -n 5 -i 1500 -c 2 -f ../preproc_data/testcases_Wings.hdf5 -t B3 -b 10,30,60,90,110
python bayes-losvd_run.py -s 1 -n 5 -i 1500 -c 2 -f ../preproc_data/testcases_Wings.hdf5 -t B4 -b 10,30,60,90,110
