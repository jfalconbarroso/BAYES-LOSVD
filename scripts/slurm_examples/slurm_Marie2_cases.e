#!/bin/bash
#############################
#SBATCH -J Marie2
#SBATCH -p long
#SBATCH -n 92
#SBATCH -t 3-00:00:00
#SBATCH -o log_Marie2-%j.out
#SBATCH -e log_Marie2-%j.err
#SBATCH -D .
#############################
module load mpi
echo "EXECUTING Marie2!"
export OMP_NUM_THREADS=92
python bayes-losvd_run.py -s 1 -n 46 -i 1500 -c 2 -f ../preproc_data/testcases_Marie2.hdf5 -t S0
python bayes-losvd_run.py -s 1 -n 46 -i 1500 -c 2 -f ../preproc_data/testcases_Marie2.hdf5 -t S1
python bayes-losvd_run.py -s 1 -n 46 -i 1500 -c 2 -f ../preproc_data/testcases_Marie2.hdf5 -t A1
python bayes-losvd_run.py -s 1 -n 46 -i 1500 -c 2 -f ../preproc_data/testcases_Marie2.hdf5 -t A2
python bayes-losvd_run.py -s 1 -n 46 -i 1500 -c 2 -f ../preproc_data/testcases_Marie2.hdf5 -t A3
python bayes-losvd_run.py -s 1 -n 46 -i 1500 -c 2 -f ../preproc_data/testcases_Marie2.hdf5 -t B3
python bayes-losvd_run.py -s 1 -n 46 -i 1500 -c 2 -f ../preproc_data/testcases_Marie2.hdf5 -t B4
