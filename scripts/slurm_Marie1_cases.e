#!/bin/bash
#############################
#SBATCH -J Marie1
#SBATCH -p batch
#SBATCH -n 3
#SBATCH -t 24:00:00
#SBATCH -o log_Marie1-%j.out
#SBATCH -e log_Marie1-%j.err
#SBATCH -D .
#############################
module load mpi
echo "EXECUTING Marie1!"
export OMP_NUM_THREADS=3
./run_Marie1_testcases.e