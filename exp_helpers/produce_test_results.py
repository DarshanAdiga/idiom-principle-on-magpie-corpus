import os
import sys
import argparse

import pandas as pd
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix

TARGET_NAMES = ['0', '1']


def save_overall_classification_report(df_merged, out_dir):

    # Get confusion matrix and save it as well
    print('-'*15, 'Confusion Matrix', '-'*15)
    confusion_report = confusion_matrix(df_merged['label'], df_merged['prediction'], labels = [0, 1])
    act_labels =['actual_'+lb for lb in TARGET_NAMES]
    pred_labels =['pred_'+lb for lb in TARGET_NAMES]
    df_confusion_report = pd.DataFrame(confusion_report, index=act_labels, columns=pred_labels)
    print(df_confusion_report)
    print()

    print('-'*15, 'Classification Report', '-'*15)
    report = classification_report(df_merged['label'], df_merged['prediction'], target_names=TARGET_NAMES)
    print(report)
    print('-'*40)

    # Save the report to a file
    with open(os.path.join(out_dir, 'overall_classification_report.txt'), 'w') as f:
        # Save the confusion matrix
        f.write('Confusion Matrix:\n')
        f.write(str(df_confusion_report))
        f.write('\n')
        f.write('\n')
        # Save the classification report
        f.write(report)
        f.write('\n')
    print(f"Classification report saved to {out_dir}/classification_report.txt")


def get_segregated_reports(df_merged, out_dir):
    """
    TODO: Compute Classification Report for each of the three kinds of PIE segregations
    """
    pass


def main(pred_file, test_file, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print("Output directory already exists. Exiting.")
        sys.exit(1)

    # Should contain tab separated 'index	prediction' columns
    df_preds = pd.read_csv(pred_file, sep='\t')
    # Should contain comma separated 'sentence_0,label' columns
    df_test = pd.read_csv(test_file, sep=',')

    #merge the two dataframes, side-by-side
    df_test['index'] = df_test.index
    df_merged = pd.merge(df_test, df_preds, on='index')
    assert df_merged.shape[0] == df_test.shape[0] and df_merged.shape[0] == df_preds.shape[0], "Error: Merging dataframes seems inconsistent!"

    # Compute the classification report
    save_overall_classification_report(df_merged, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="Path to predictions file containg test predictions. \
        ## Should contain tab separated 'index \t prediction' columns.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to file containing true labels file, should have 'label' column. \
        ## Should contain comma separated 'sentence_0,label' columns.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory to save the test results")

    args = parser.parse_args()
    
    pred_file = args.pred_file
    test_file = args.test_file
    out_dir = args.out_dir

    main(pred_file, test_file, out_dir)