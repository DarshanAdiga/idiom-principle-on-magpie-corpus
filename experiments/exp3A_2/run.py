# To make the 'exp_helpers' module visible here
import sys
sys.path.append("../../exp_helpers")


import run_glue_f1_macro

exp_model = '../../local_models/bert-base-cased_STR_option2'
exp_seed = 26
exp_dir = './'
model_checkpoint_dir = exp_dir + 'checkpoints'
train_csv = exp_dir + 'tmp/train.csv'
dev_csv = exp_dir + 'tmp/dev.csv'
test_csv = exp_dir + 'tmp/test.csv'

# Run the helper script that trains, saves the sentence classification model
args = f"""
--model_name_or_path {exp_model} \
--do_train \
--do_eval \
--do_predict \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 9 \
--output_dir {model_checkpoint_dir} \
--seed {exp_seed} \
--train_file      {train_csv} \
--validation_file {dev_csv} \
--test_file       {test_csv} \
--test_metrics \
--evaluation_strategy epoch \
--save_strategy epoch  \
--load_best_model_at_end \
--metric_for_best_model f1 \
--save_total_limit 3
"""

# NOTE: This is a bit of a hack, but works for now
args = args.replace("\n", " ").split()
sys.argv = [sys.argv[0]] + args
run_glue_f1_macro.main()
