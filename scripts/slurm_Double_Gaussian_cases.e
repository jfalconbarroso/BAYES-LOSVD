#!/bin/bash
#############################
#SBATCH -J double_gaussian
#SBATCH -p batch
#SBATCH -n 3
#SBATCH -t 24:00:00
#SBATCH -o log_double_gaussian-%j.out
#SBATCH -e log_double_gaussian-%j.err
#SBATCH -D .
#############################
module load mpi
echo "EXECUTING Double Gaussian!"
export OMP_NUM_THREADS=3
./run_Double_Gaussian_testcases.e