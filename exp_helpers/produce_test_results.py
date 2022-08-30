import os
import sys
import argparse

import pandas as pd
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix

TARGET_NAMES = ['0', '1']


def save_overall_classification_report(y_true, y_pred, out_dir, outfile_prefix):

    # Get confusion matrix and save it as well
    print('-'*15, 'Confusion Matrix', '-'*15)
    confusion_report = confusion_matrix(y_true, y_pred, labels = [0, 1])
    act_labels =['actual_'+lb for lb in TARGET_NAMES]
    pred_labels =['pred_'+lb for lb in TARGET_NAMES]
    df_confusion_report = pd.DataFrame(confusion_report, index=act_labels, columns=pred_labels)
    print(df_confusion_report)
    print()

    print('-'*15, 'Classification Report', '-'*15)
    report = classification_report(y_true, y_pred, target_names=TARGET_NAMES, digits=4)
    print(report)
    print('-'*40)

    # Save the report to a file
    report_file = outfile_prefix + '_classification_report.txt'
    with open(os.path.join(out_dir, report_file), 'w') as f:
        # Save the confusion matrix
        f.write('Confusion Matrix:\n')
        f.write(str(df_confusion_report))
        f.write('\n')
        f.write('\n')
        # Save the classification report
        f.write(report)
        f.write('\n')
    print(f"Classification report saved to {report_file}")


def insert_test_PIE_tokens(df_merged, magpie_token_set):
    """
    Identify those PIE tokens that are present in the df_merged, and add them as a separate column
    Return the updated dataframe
    """
    test_PIE_list = list()
    for i,mrow in df_merged.iterrows():
        test_sent = mrow['sentence_0']
        matched_token = None
        for token in magpie_token_set:
            if token in test_sent:
                matched_token = token
                break
        if matched_token:
            test_PIE_list.append(matched_token)
        else:
            raise Exception("No PIE token found in the sentence!")
    # Add the PIE tokens as a column
    df_merged['test_PIE_tokens'] = test_PIE_list
    return df_merged


def save_segregated_reports(df_merged, df_pie_segregation, out_dir):
    """
    Compute & Save the Classification Reports for each of the three kinds of PIE segregations
    """
    # Join the dataframes
    df_merged_segregated = pd.merge(df_merged, df_pie_segregation, left_on='test_PIE_tokens', \
        right_on='idiom_token', how='inner')
    # print(df_merged_segregated.columns)

    # ------------------------------------
    #  Compute the classification reports

    # 1. Degree of Idiomaticity
    for degree in df_merged_segregated['degree_of_idiomaticity'].unique():
        df_subset = df_merged_segregated[df_merged_segregated['degree_of_idiomaticity'] == degree]
        save_overall_classification_report(df_subset['label'], df_subset['prediction'], out_dir, outfile_prefix='degree_of_idiomaticity_'+degree)

    # 2. CCNews Rarity
    for rarity in df_merged_segregated['ccnews_rarity'].unique():
        df_subset = df_merged_segregated[df_merged_segregated['ccnews_rarity'] == rarity]
        save_overall_classification_report(df_subset['label'], df_subset['prediction'], out_dir, outfile_prefix='ccnews_rarity_'+rarity)

    # 3. Morphology Type
    for morph_type in df_merged_segregated['morphology_type'].unique():
        df_subset = df_merged_segregated[df_merged_segregated['morphology_type'] == morph_type]
        save_overall_classification_report(df_subset['label'], df_subset['prediction'], out_dir, outfile_prefix='morphology_type_'+morph_type)


def main(pred_file, test_file, pie_segregation_file, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print("ERROR: Output directory already exists!! Exiting.")
        sys.exit(1)

    # Should contain tab separated 'index	prediction' columns
    df_preds = pd.read_csv(pred_file, sep='\t')
    # Should contain comma separated 'sentence_0,label' columns
    df_test = pd.read_csv(test_file, sep=',')

    #merge the two dataframes, side-by-side
    df_test['index'] = df_test.index
    df_merged = pd.merge(df_test, df_preds, on='index')
    assert df_merged.shape[0] == df_test.shape[0] and df_merged.shape[0] == df_preds.shape[0], "Error: Merging dataframes seems inconsistent!"

    # Load the PIE segregation dataframe
    df_pie_segregation = pd.read_csv(pie_segregation_file)
    # Remove unimportant columns
    df_pie_segregation = df_pie_segregation[[ 'idiom_token', 'degree_of_idiomaticity', 'ccnews_rarity', 'morphology_type']]
    # Remove duplicates rows, (that were there due to 'label' column)
    df_pie_segregation.drop_duplicates(inplace=True)

    # Compute the classification report
    save_overall_classification_report(df_merged['label'], df_merged['prediction'], out_dir, outfile_prefix='overall')

    # Compute the classification report for each of the three kinds of PIE segregations
    magpie_token_set = set(df_pie_segregation['idiom_token'].unique())
    # Identify the PIE tokens in the test sentences
    df_merged = insert_test_PIE_tokens(df_merged, magpie_token_set)
    save_segregated_reports(df_merged, df_pie_segregation, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="Path to predictions file containg test predictions. \
        ## Should contain tab separated 'index \t prediction' columns.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to file containing true labels file, should have 'label' column. \
        ## Should contain comma separated 'sentence_0,label' columns.")
    parser.add_argument("--pie_segregation_file", type=str, required=True, help="Path to file containing PIE segregation CSV file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory to save the test results")

    args = parser.parse_args()
    
    pred_file = args.pred_file
    test_file = args.test_file
    pie_segregation_file = args.pie_segregation_file
    out_dir = args.out_dir

    main(pred_file, test_file, pie_segregation_file, out_dir)