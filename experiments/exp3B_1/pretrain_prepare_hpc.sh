#!/bin/bash
#$ -l h_rt=4:00:00  #time needed
#$ -pe smp 6 #number of cores
#$ -l rmem=3G #Maximum amount (xx) of real memory to be requested per CPU core

#$ -o ./output.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M dahaniyanarayana1@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory


module load apps/python/conda
# Only needed if we're using GPU* Load the CUDA and cuDNN module
module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176
source activate dis_venv_1

python ../../exp_helpers/prepare_cc_corpus/processCCNews.py -i ../../data/cc_news_data/ -o ../../data/cc_news_data/cc_processed_option1 -p 10
