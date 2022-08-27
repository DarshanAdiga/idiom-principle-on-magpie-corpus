import re
import os
import sys
import csv
import json
import pickle
import random
random.seed( 42 )

from tqdm          import tqdm
from collections   import defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import argparse
from nltk.stem.porter import PorterStemmer


porter_stemmer = PorterStemmer()
nltk.download('punkt')

WORD_OR_SPACE_PATTERN = re.compile('[^\w\s]')
# At least 'MATCHING_SCORE_THRESHOLD' percentage of idiom words should match EXACTLY in the matched phrase
MATCHING_SCORE_THRESHOLD = 0.5

# Define the mechanism that maps the idiom to its single-token
def tokenise_idiom_using_dict( idiom, idiom_token_dict ) :
    return idiom_token_dict[ idiom ]

# New way of matching idioms in sentences!
def create_idiom_re_compiled_list(idiom_list):
    """
        Compile the regular expressions for the idioms.
        Compile a sorted-idiom-words list as well, to be used to score the matched idioms

        Returns a tuple <idiom_re_compiled, sorted_idiom_words>
    """
    idiom_re_compiled = list()
    sorted_idiom_tokens = list()
    for idiom in idiom_list:
        idiom = idiom.strip()
        i_words = [porter_stemmer.stem(idiom_word) for idiom_word in idiom.split()]

        # For scoring purposes
        sorted_idiom_tokens.append(sorted(i_words))

        i_words = [i_word + "([\w'-])*" for i_word in i_words]
        re_pattern = " ?\w* ".join(i_words)
        re_pattern = "(^| )" + re_pattern + "( |$)"
        re_pattern = re.compile(re_pattern, re.IGNORECASE)
        idiom_re_compiled.append(re_pattern)

    return idiom_re_compiled,sorted_idiom_tokens

def get_match_score(matched_phrase, s_idiom_tokens):
    """
        Get a score between 0-1, using the words/tokens.
        s_idiom_tokens contains the words/tokens of the actual idiom in alphabetically sorted order.
    """
    matched_tokens=matched_phrase.strip().lower().split()
    s_matched_tokens = sorted(matched_tokens)
    end = min(len(s_idiom_tokens), len(s_matched_tokens))
    match_count = 0
    for i in range(end):
        # NOTE: The regex-based matches a broad variations of tokens, here we are being strict!
        if s_idiom_tokens[i] == s_matched_tokens[i]:
            match_count += 1
    return match_count/len(s_idiom_tokens)

# New way of matching idioms in sentences!
def match_idioms(idiom_list, idiom_re_compiled, sorted_idiom_tokens, sentence):
    """Tries to match the sequence of words of an idiom in a sentence, with a possibility of
       i. presence of non-idiom words in between.
       ii. variations of idiom words (thus stemming is used)

        Returns a tuple <actual idiom, a list>: 
            - a list of idioms as present in the 'idiom_list'
            - a list of zero or more <start,end> (both inclusive) span pairs of idiom-phrases that are present in the given sentence
    """
    matched_idiom_spans = list()
    actual_idioms = list()
    # Replace every non-word non-space characters with a space 
    sentence = WORD_OR_SPACE_PATTERN.sub(' ', sentence)
    # IMP: The string length should remain the same!!

    # Ignore multiple idioms starting with same index
    already_seen_starts = set()

    for idiom,idiom_re,s_idiom_tokens in zip(idiom_list, idiom_re_compiled, sorted_idiom_tokens):
        match = idiom_re.search(sentence)
        if match:
            matched = match.group()
            match_span = match.span()
            match_span = (match_span[0], match_span[1]-1) # Make End inclusive

            # Strip the spaces, if exists
            if ' ' == matched[0]:
                match_span = (match_span[0]+1, match_span[1])
            if ' ' == matched[-1]:
                match_span = (match_span[0], match_span[1]-1)

            # Get a matching score between the idiom and the matched phrase
            match_score = get_match_score(sentence[match_span[0]:match_span[1]+1], s_idiom_tokens)

            # Proceed further, only if the matching score is above a threshold
            if match_score < MATCHING_SCORE_THRESHOLD:
                continue

            if match_span[0] in already_seen_starts:
                print(f'Warning: Multiple idioms starting at same index! {match}')
                continue
            already_seen_starts.add(match_span[0])

            # Check if there are overlapping idioms
            handled_overlap = False
            for i,other_match_span in enumerate(matched_idiom_spans):
                # If the max of starts is less than the min of ends, then there is an overlap
                if max(other_match_span[0], match_span[0]) < min(other_match_span[1], match_span[1]):
                    print(f'Warning: Overlapping idioms! {match}')
                    # Then just blindly take the shorted match among the two
                    # NOTE: The better way of handling this is to keep the one with highest score & length
                    handled_overlap = True
                    if other_match_span[1] - other_match_span[0] > match_span[1] - match_span[0]:
                        # Replace the old match with the new one
                        matched_idiom_spans[i] = match_span
                        actual_idioms[i] = idiom
                    break
            if handled_overlap:
                continue

            matched_idiom_spans.append(match_span)
            actual_idioms.append(idiom)
            # print(':' + idiom + ':' + str(match) + ': Updated span:' + str(match_span) + ': Score:' + str(match_score))

    return actual_idioms,matched_idiom_spans

def fast_replace(actual_idioms, matched_idiom_spans, idiom_token_dict, sentence):
    """
        Replace the matched idioms with their single-token representations in a single-pass through the string.
        Returns the processed sentence.
    """
    # If nothing to replace
    if len(matched_idiom_spans) == 0:
        return sentence

    # The conventional way of modifying the same 'sentence' in-place doesn't work, because,
    # the span indices are with respect to the unmodified original 'sentence'!!.

    # Hence, let's do it in a single pass
    processed = ''
    # Sort the spans in increasing order of start index, BUT
    # we need to sort (actual_idioms,matched_idiom_spans) together!!!
    sorted_idiom_spans = sorted(zip(matched_idiom_spans, actual_idioms), key=lambda x: x[0])
    matched_idiom_spans, actual_idioms = zip(*sorted_idiom_spans) # Note, the order of things here!

    start = 0
    for actual_idiom,matched_span in zip(actual_idioms, matched_idiom_spans):
        single_token_for_idiom = tokenise_idiom_using_dict(actual_idiom, idiom_token_dict)
        processed += sentence[start: matched_span[0]]
        processed += single_token_for_idiom
        start = matched_span[1]+1
    # Add the pending part of the sentence
    processed += sentence[start:]
    
    return processed

def __test_regex_idiom_matching(idiom_list, idiom_token_dict):

    # Pre-compile the regexes for matching idioms
    idiom_re_compiled,sorted_idiom_tokens = create_idiom_re_compiled_list(idiom_list)

    import time
    start = time.time()
    test_cases = [
	("leaders have been working around the clock to ensure", "leaders have been working IDaroundtheclockID to ensure"),
        ("board and attack next season with", "board and attack next season with"),
        ("proposed five-person board would include", "proposed five-person board would include"),
        ("skill that the end of the day we", "skill that the end of the day we"),

        ("the great sea of college football", "the great sea of college football"),
        ("Cowboys won the game 53", "Cowboys won the game 53"),

        ("its entire history which is a real privacy problem that inhibits real world adoption especially in businesses where they may not want to reveal who their customers are, how much they're receiving and who their suppliers are", \
            "its entire history which is a real privacy problem that inhibits real world adoption especially IDinbusinessID where they may not want to reveal who their customers are, how much they're receiving and who their suppliers are"),
            # "its entire hisIDinbusinessID suppliers are"

        # Good examples
        ("than lie low, proud", "than IDlielowID, proud"),
        ("Lie low, proud", "IDlielowID, proud"),
        ("than Lie low.", "than IDlielowID."),
        ("than Lie low", "than IDlielowID"),
        ("than Lied low", "than IDlielowID"),
        ("than Lie-low", "than IDlielowID"),
        ("than Lie lowest", "than IDlielowID"),

        ("died behind bars in 2017", "died IDbehindbarsID in 2017"),
        ("cities think twice about", "cities IDthinktwiceID about"),
        ("passengers on board.", "passengers IDonboardID."),

        # Multiple idioms
        ("died behind bars in 2017 and Lie low was his", "died IDbehindbarsID in 2017 and IDlielowID was his"),
        ("I reckon it will pay dividends in the long run,” Warne said.",\
            "I reckon it will pay dividends IDinthelongrunID,” Warne said."),
    ]

    for test,actual in test_cases: 
        actual_idioms,matched_idiom_spans = match_idioms(idiom_list, idiom_re_compiled, sorted_idiom_tokens, test)
        # Replace all the matches
        processed = fast_replace(actual_idioms, matched_idiom_spans, idiom_token_dict, test)
        assert processed == actual, "\nTest case: {}\nProcessed: {}\nExpected: {}".format(test, processed, actual)

    print('Test passed!')
    time_taken = time.time()-start
    print(f'Total time taken: {time_taken:0.4f} seconds. Average time per test case: {time_taken/len(test_cases):0.4f} seconds.')

def _load_single_dataset_sents( location ) :

    sents = list()
    with open( location, encoding='utf-8') as csvfile :
        reader = csv.reader( csvfile )
        next(reader, None) # Skip the header <sentence_0,label>
        for row in reader :
            sents.append( row[0] )
    return sents

def _load_dataset_sents( dataset_sents_info ) :

    files = [ 'dev.csv', 'test.csv', 'train.csv' ]

    sents = list()
    for file_name in files :
        file_location = os.path.join( dataset_sents_info, file_name )
        sents += _load_single_dataset_sents( file_location )
        
    return sents 

def _preprocess_dataset_sents(dataset_sents):
    preprocessed_sents = list()
    for dataset_sent in dataset_sents :
        dataset_sent = ''.join( dataset_sent.lower().split() )
        preprocessed_sents.append( dataset_sent )
    return preprocessed_sents

def sent_in_dataset( pre_processed_dataset_sents, sent ) :

    sent = ''.join( sent.lower().split() )
    for dataset_sent in pre_processed_dataset_sents :
        if  dataset_sent in sent or sent in dataset_sent :
            return True
    return False

def filter(cc_data_location, out_location, dataset_sents_info, idioms=None, idiom_token_dict=None, limit_count=None ) :

    dataset_sents = _load_dataset_sents( dataset_sents_info )
    preprocessed_dataset_sents = _preprocess_dataset_sents( dataset_sents )
    

    if idioms is None : 
        print('Error: idioms is None')
        return

    # counts = pickle.load( open( 'data/processCCNNews-status.pk3', 'rb' ) )[ 'counts' ]
    # for idiom in idioms :
    #     print( idiom, "-->", counts [ idiom ] )
    # sys.exit()

    # Pre-compile the regexes for matching idioms
    idiom_re_compiled,sorted_idiom_tokens = create_idiom_re_compiled_list(idioms)


    data_files = [f for f in os.listdir( cc_data_location ) if os.path.isfile(os.path.join(cc_data_location, f))]

    line_number           = 0
    documents_no_replace  = list()
    documents_all_replace = list()
    included_counts       = defaultdict( int ) 
    act_idiom_counts       = defaultdict( int )
    for data_file in tqdm( data_files ) :
        data_file = os.path.join( cc_data_location, data_file )
        data      = open( data_file, 'r', encoding='utf-8', errors='ignore' ).read()
        for doc in data.split( '\n--DocBreak--\n' ) :

            this_doc = list()
            for line in doc.split( '\n' ) :
                line = line.lstrip().rstrip()
                if len( line ) < 5 :
                    continue
                if line[0] == '*' :
                    continue;
                this_doc += [ i for i in sent_tokenize( line ) if len( i ) > 5 and len( i.split() ) > 3 ]
            this_doc = [ i.replace( '**', '' ) for i in this_doc ]
            this_doc = [ i.replace( '_', '' ) for i in this_doc ]
            this_doc = [ i.lstrip().rstrip() for i in this_doc ]
            this_doc = [ re.sub( r'\s+', ' ', i ) for i in this_doc ]

            all_doc         = list()
            replaced_doc    = list()
            replaced        = False
            this_line       = 0
            for sent in this_doc :
                original_sent = sent
                if len( sent.split() ) > 500 :
                    continue

                if sent_in_dataset( preprocessed_dataset_sents, sent ) :
                    print( "Found sent in dataset: ", sent, flush=True )
                    continue

                actual_idioms, matched_idiom_spans = match_idioms(idioms, idiom_re_compiled, sorted_idiom_tokens, sent)

                if len( matched_idiom_spans ) > 0 :
                    replaced = True

                    # Stats
                    for act_idiom in actual_idioms:
                        act_idiom_counts[act_idiom] += 1

                    # Replace all the matches
                    sent = fast_replace(actual_idioms, matched_idiom_spans, idiom_token_dict, sent)

                replaced_doc.append( sent )
                    
                all_doc.append( original_sent )
                    
                this_line += 1
            if replaced :
                documents_no_replace.append( all_doc )
                documents_all_replace.append( replaced_doc )
                line_number += this_line + 1
            assert len( replaced_doc ) == len( all_doc )


    for outfile_name, data in [
            [ os.path.join( out_location, 'no_replace_data.txt' ) , documents_no_replace  ],
            [ os.path.join( out_location, 'all_replace_data.txt' ), documents_all_replace ]
    ] : 
        with open( outfile_name, 'w' ) as outfile :
            # Header
            outfile.write( 'text' + '\n' )
            for doc in data :
                for sent in doc :
                    outfile.write( sent + "\n" )
                outfile.write( "\n" )
        print( "Wrote: ", outfile_name )

    # Save some stats
    stats = {'idiom_counts': act_idiom_counts}
    stats_filename = os.path.join(out_location, 'stats.json')
    with open(stats_filename, 'w') as outfile:
        json.dump(stats, outfile)
        print("Wrote: ", stats_filename)

    return


def load_idiom_token_dict_and_idioms(idioms_csv):
    idiom_token_dict = dict()
    idioms = list()
    with open(idioms_csv) as i_csvfile :
        reader = csv.reader(i_csvfile)
        next(reader, None) # Skip the header <idiom,idiom_token>
        for row in reader :
            idiom = row[0]
            idiom_token = row[1]
            idiom_token_dict[idiom] = idiom_token
            idioms.append(idiom)

    idioms = list( set( idioms ) ) 
    print( "Picked {} idioms.".format( len( idioms ) ) )
    assert len( idioms ) > 0
    return idiom_token_dict, idioms
                                           
    
if __name__ == '__main__' :

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-c', '--cc_data_location', help='Directory containing preprocessed cc-news files', required=True)
    arg_parser.add_argument('-o', '--out_location', help='Dir to write output files', required=True)
    arg_parser.add_argument('-d', '--dataset_sents_info', help='Dir containing {train,dev,test}.csv files for de-duplication', required=True)
    arg_parser.add_argument('-i', '--idioms_csv', help='CSV File containing <idiom,token> *_idioms.csv', required=True)

    args = arg_parser.parse_args()
    
    params = {
        'cc_data_location'   : args.cc_data_location,
        'out_location'       : args.out_location,
        'dataset_sents_info' : args.dataset_sents_info,
        'idioms'             : None, 
        'idiom_token_dict'   : None,
        'limit_count'        : None , 
    }

    from pprint import pprint
    print( "PARAMS: " )
    pprint( params )

    ## Load idiom_token_dict and idioms list
    params['idiom_token_dict'], params['idioms'] = load_idiom_token_dict_and_idioms(args.idioms_csv)
        
    # from pprint import pprint
    # print( "UPDATED PARAMS: " )
    # pprint( params )
        
    # Test the regex matching mechanism
    __test_regex_idiom_matching(params['idioms'], params['idiom_token_dict'])
    
    os.makedirs( params[ 'out_location' ] )
    filter( **params ) 
