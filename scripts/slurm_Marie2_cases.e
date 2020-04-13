#!/bin/bash
#############################
#SBATCH -J Marie2
#SBATCH -p batch
#SBATCH -n 3
#SBATCH -t 24:00:00
#SBATCH -o log_Marie2-%j.out
#SBATCH -e log_Marie2-%j.err
#SBATCH -D .
#############################
module load mpi
echo "EXECUTING Marie2!"
export OMP_NUM_THREADS=3
./run_Marie2_testcases.e