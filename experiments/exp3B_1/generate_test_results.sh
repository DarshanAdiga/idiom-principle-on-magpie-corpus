#!/bin/bash
python ../../exp_helpers/produce_test_results.py --pred_file ./checkpoints_SeqClassifier/test_results_None.txt \
    --test_file ./tmp/test.csv \
    --out_dir ./test_report