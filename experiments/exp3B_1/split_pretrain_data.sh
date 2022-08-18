#!/bin/bash

# Split pretrain data for experiment 3B_1
python ../../exp_helpers/prepare_cc_corpus/preTrainData_splitter.py \
    --all_replace_file=./pretrain_data/all_replace_data.txt \
    --train_ratio=0.99 \
    --random_reduce_ratio=0.8 \
    --output_dir=./pretrain_data_split