#!/bin/bash
#############################
#SBATCH -J Double_Gaussian
#SBATCH -p long
#SBATCH -n 25
#SBATCH -t 3-00:00:00
#SBATCH -o log_Double_Gaussian-%j.out
#SBATCH -e log_Double_Gaussian-%j.err
#SBATCH -D .
#############################
module load mpi
echo "EXECUTING Double_Gaussian!"
export OMP_NUM_THREADS=25
python bayes-losvd_run.py -n 25 -i 1500 -c 1 -f ../preproc_data/testcases_Double_Gaussian.hdf5 -t S0
python bayes-losvd_run.py -n 25 -i 1500 -c 1 -f ../preproc_data/testcases_Double_Gaussian.hdf5 -t S1
python bayes-losvd_run.py -n 25 -i 1500 -c 1 -f ../preproc_data/testcases_Double_Gaussian.hdf5 -t A1
python bayes-losvd_run.py -n 25 -i 1500 -c 1 -f ../preproc_data/testcases_Double_Gaussian.hdf5 -t A2
python bayes-losvd_run.py -n 25 -i 1500 -c 1 -f ../preproc_data/testcases_Double_Gaussian.hdf5 -t A3
python bayes-losvd_run.py -n 25 -i 1500 -c 1 -f ../preproc_data/testcases_Double_Gaussian.hdf5 -t B3
python bayes-losvd_run.py -n 25 -i 1500 -c 1 -f ../preproc_data/testcases_Double_Gaussian.hdf5 -t B4
