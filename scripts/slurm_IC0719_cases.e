#!/bin/bash
#############################
#SBATCH -J IC0719
#SBATCH -p batch
#SBATCH -n 6
#SBATCH -t 1-00:00:00
#SBATCH -o log_IC0719-%j.out
#SBATCH -e log_IC0719-%j.err
#SBATCH -D .
#############################
module load mpi
echo "EXECUTING IC07019!"
export OMP_NUM_THREADS=6
python bayes-losvd_run.py -n 3 -i 1500 -c 2 -f ../preproc_data/IC0719-test_blue.hdf5 -t S0 -b 0,11,66
python bayes-losvd_run.py -n 3 -i 1500 -c 2 -f ../preproc_data/IC0719-test_blue.hdf5 -t S1 -b 0,11,66
python bayes-losvd_run.py -n 3 -i 1500 -c 2 -f ../preproc_data/IC0719-test_blue.hdf5 -t A1 -b 0,11,66
python bayes-losvd_run.py -n 3 -i 1500 -c 2 -f ../preproc_data/IC0719-test_blue.hdf5 -t A2 -b 0,11,66
python bayes-losvd_run.py -n 3 -i 1500 -c 2 -f ../preproc_data/IC0719-test_blue.hdf5 -t A3 -b 0,11,66
python bayes-losvd_run.py -n 3 -i 1500 -c 2 -f ../preproc_data/IC0719-test_blue.hdf5 -t B3 -b 0,11,66
python bayes-losvd_run.py -n 3 -i 1500 -c 2 -f ../preproc_data/IC0719-test_blue.hdf5 -t B4 -b 0,11,66
#
python bayes-losvd_run.py -n 3 -i 1500 -c 2 -f ../preproc_data/IC0719-test_red.hdf5 -t S0 -b 0,11,66
python bayes-losvd_run.py -n 3 -i 1500 -c 2 -f ../preproc_data/IC0719-test_red.hdf5 -t S1 -b 0,11,66
python bayes-losvd_run.py -n 3 -i 1500 -c 2 -f ../preproc_data/IC0719-test_red.hdf5 -t A1 -b 0,11,66
python bayes-losvd_run.py -n 3 -i 1500 -c 2 -f ../preproc_data/IC0719-test_red.hdf5 -t A2 -b 0,11,66
python bayes-losvd_run.py -n 3 -i 1500 -c 2 -f ../preproc_data/IC0719-test_red.hdf5 -t A3 -b 0,11,66
python bayes-losvd_run.py -n 3 -i 1500 -c 2 -f ../preproc_data/IC0719-test_red.hdf5 -t B3 -b 0,11,66
python bayes-losvd_run.py -n 3 -i 1500 -c 2 -f ../preproc_data/IC0719-test_red.hdf5 -t B4 -b 0,11,66
