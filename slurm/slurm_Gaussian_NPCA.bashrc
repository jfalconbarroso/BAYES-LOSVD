#!/bin/bash
#############################
#SBATCH -J NPCA
#SBATCH -p batch
#SBATCH -n 25
#SBATCH -t 1-00:00:00
#SBATCH -o log_Gaussian_NPCA-%j.out
#SBATCH -e log_Gaussian_NPCA-%j.err
#SBATCH -D .
#############################

echo "EXECUTING NPCA!"
export OMP_NUM_THREADS=25

cd ../scripts
python bayes_losvd_run.py -n 25 -f ../preproc_data/testdata_Gaussian-NPCA2.hdf5  -t RWtest -i 1000
python bayes_losvd_run.py -n 25 -f ../preproc_data/testdata_Gaussian-NPCA5.hdf5  -t RWtest -i 1000
python bayes_losvd_run.py -n 25 -f ../preproc_data/testdata_Gaussian-NPCA10.hdf5 -t RWtest -i 1000