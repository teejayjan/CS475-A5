#!/bin/bash
#SBATCH -J CudaMonteCarlo
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o cudaMonteCarlo.out
#SBATCH -e cudaMonteCarlo.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jant@oregonstate.edu
for b in 16 32 64 128
do
    for n in 2048 4096 8192 16384 32768 64512 128000 256000 512000 1000448 2000896 4000768 6000640 8000512 10000384 
    do
        /usr/local/apps/cuda/cuda-10.1/bin/nvcc -DBLOCKSIZE=$b -DNUMTRIALS=$n -o montecarlo cudaMonteCarlo.cu
        ./montecarlo
    done
done