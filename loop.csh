#!/bin/csh

foreach b (16 32 64 128)
    foreach n (2048 4096 8192 16384 32768 64512 128000 256000 512000 1000448)
        /usr/local/apps/cuda/cuda-10.1/bin/nvcc -DBLOCKSIZE=$b -DNUMTRIALS=$n -o montecarlo cudaMonteCarlo.cu
        ./montecarlo
    end
end