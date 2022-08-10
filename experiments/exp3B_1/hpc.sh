#!/bin/bash
#$ -l h_rt=3:00:00  #time needed
#$ -pe smp 4 #number of cores
#$ -l rmem=4G #Maximum amount (xx) of real memory to be requested per CPU core
#$ -l gpu=1 # Number of GPUs per every CPU core
#$ -o ./output_SeqClassifier.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M dahaniyanarayana1@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory


module load apps/python/conda
# Only needed if we're using GPU* Load the CUDA and cuDNN module
module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176
source activate dis_venv_1

python ./run.py
