import re
import os
import sys
import random
random.seed( 42 )

from tqdm          import tqdm
from collections   import defaultdict

import argparse

IDIOM_PATTERN = re.compile(r'.* ?ID[a-z]*ID ?.*')
def process_and_filter_line(line, random_reduce_ratio=0.3):
    """Preprocess the line and, 
      - if 'random_reduce_ratio', ignore the line NOT containing any idiom with a probability of 'random_reduce_ratio'. (Default is 0.3)
    """
    # Remove the trailing newline
    line = line.strip()
    
    if line and len(line) > 20:
        # check if IDIOM_PATTERN matches the line
        if IDIOM_PATTERN.match(line):
            # Go ahead and return the line
            return line
        else:
            # Ignore the line with 'random_reduce_ratio' probability
            if random.random() < random_reduce_ratio:
                return None
            else:
                return line
    else:
        return None


def __test_re_matching():
    """A simple test for regex matching of idioms"""
    test_lines = [
        ("This is a ID wrong ID test", False),
        ("collapse IDinbusinessID activity i", True),
        ("than IDlielowID, proud", True), 
        ("passengers IDonboardID.", True), 
        ("dividends IDinthelongrunID,” Warne", True), 
        ("IDtestID is a test", True),
    ]
    for line,match in test_lines:
        err = f'Error: {line} should match {match}'
        if IDIOM_PATTERN.match(line):
            assert match, err
        else:
            assert not match, err


def train_eval_split(all_replace_file, train_ratio, output_dir, random_reduce_ratio=0.3):
    """
    Split the corpus into train and eval sets.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f'Output directory {output_dir} already exists. Aborting.')
        sys.exit(1)

    # Read all the lines from the file
    with open(all_replace_file, 'r') as f:
        lines = f.readlines()
    # Split the lines into train and eval sets
    train_lines = []
    eval_lines = []
    ignored_count = 0
    for line in tqdm(lines):
        line = process_and_filter_line(line, random_reduce_ratio)
        if line:
            if random.random() < train_ratio:
                train_lines.append(line)
            else:
                eval_lines.append(line)
        else:
            ignored_count += 1
    # Write the train and eval sets to files
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for l in train_lines:
            f.write(l + '\n')
    with open(os.path.join(output_dir, 'eval.txt'), 'w') as f:
        for l in eval_lines:
            f.write(l + '\n')

    print('Done Processing.')
    print(f"Ignored {ignored_count} lines.")
    print(f"Split {len(lines)} lines into {len(train_lines)} train and {len(eval_lines)} eval lines.")
    print(f"Saved train and eval sets to {output_dir} directory.")

if __name__ == '__main__' :

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--all_replace_file', help="Path to 'all_replace_data.txt' file", required=True)
    arg_parser.add_argument('--train_ratio', help='Value between 0 and 1, indicating ratio of train split', required=True, type=float)
    arg_parser.add_argument('--random_reduce_ratio', help='Value between 0 and 1, indicating ratio of NON-idiom lines to be ignored randomly', default=0.3, type=float)
    arg_parser.add_argument('--output_dir', help='Dir to save {train,dev,test}.txt files', required=True)

    args = arg_parser.parse_args()

    # Test the regex matching
    __test_re_matching()

    train_eval_split(args.all_replace_file, args.train_ratio, args.output_dir, args.random_reduce_ratio)
