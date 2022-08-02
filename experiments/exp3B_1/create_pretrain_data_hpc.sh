#!/bin/bash
# TODO Convert this to HPC script
python ../../exp_helpers/prepare_cc_corpus/createPreTrainData_simplified.py -c ../../data/cc_news_data_option1/output -o ./pretrain_data -d ../../experiments/exp0/tmp -i ../../data/token_files/option1_idioms.csv