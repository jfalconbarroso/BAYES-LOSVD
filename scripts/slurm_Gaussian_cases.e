#!/bin/bash
#############################
#SBATCH -J gaussian
#SBATCH -p batch
#SBATCH -n 3
#SBATCH -t 24:00:00
#SBATCH -o log_gaussian-%j.out
#SBATCH -e log_gaussian-%j.err
#SBATCH -D .
#############################
module load mpi
echo "EXECUTING Gaussian!"
export OMP_NUM_THREADS=3
./run_Gaussian_testcases.e