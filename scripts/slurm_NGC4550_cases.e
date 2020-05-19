#!/bin/bash
#############################
#SBATCH -J NGC4550
#SBATCH -p batch
#SBATCH -n 6
#SBATCH -t 1-00:00:00
#SBATCH -o log_NGC4550-%j.out
#SBATCH -e log_NGC4550-%j.err
#SBATCH -D .
#############################
module load mpi
echo "EXECUTING NGC4550!"
export OMP_NUM_THREADS=6
python bayes-losvd_run.py -s 1 -n 3 -i 1500 -c 2 -f ../preproc_data/NGC4550_SAURON.hdf5 -t S0 -b 0,169,189
python bayes-losvd_run.py -s 1 -n 3 -i 1500 -c 2 -f ../preproc_data/NGC4550_SAURON.hdf5 -t S1 -b 0,169,189
python bayes-losvd_run.py -s 1 -n 3 -i 1500 -c 2 -f ../preproc_data/NGC4550_SAURON.hdf5 -t A1 -b 0,169,189
python bayes-losvd_run.py -s 1 -n 3 -i 1500 -c 2 -f ../preproc_data/NGC4550_SAURON.hdf5 -t A2 -b 0,169,189
python bayes-losvd_run.py -s 1 -n 3 -i 1500 -c 2 -f ../preproc_data/NGC4550_SAURON.hdf5 -t A3 -b 0,169,189
python bayes-losvd_run.py -s 1 -n 3 -i 1500 -c 2 -f ../preproc_data/NGC4550_SAURON.hdf5 -t B3 -b 0,169,189
python bayes-losvd_run.py -s 1 -n 3 -i 1500 -c 2 -f ../preproc_data/NGC4550_SAURON.hdf5 -t B4 -b 0,169,189
