'''
Perform Minimum Bayes' Risk decoding , but select best sequence (greedily) between
a set of 'limit' sequences.

algorithm:

a = union (limit sequence edits)
b = intersect (limit sequence edits)

to_add = a - b

best = b
for e in to_add:
    b' = best + e
    if reward(b') is best so far:
        best = b'

Assume each input file aligned and is of the form:
id1 sentence1
id2 sentence2
.
.
.

'''

import sys
import os
import argparse
from tqdm import tqdm
import errant
from functools import reduce
from difflib import SequenceMatcher
from collections import Counter
import spacy

from mbr import edit_agreement, edit_jaccard, edit_f05, edit_prec, edit_rec, return_edits
from max_vote import get_tokens

annotator = errant.load('en')
nlp = spacy.load("en_core_web_sm", exclude=['tagger', 'parser', 'ner', 'lemmatizer', 'textcat', 'custom'])

def difflib_edits_to_output(difflib_edits, source):
    if len(difflib_edits)>0:
        source_tokens = get_tokens(nlp(source))
        final_text = ""
        current_start = 0
        previous_end = 0

        for action in difflib_edits:
            current_start = action[1]
            final_text += "".join(source_tokens[previous_end: current_start])
            if action[0] == 'replace':
                final_text += action[3]
                previous_end = action[2]
            if action[0] == 'delete':
                previous_end = action[2]
            if action[0] == 'insert':
                final_text += action[2]
                previous_end = action[1]

        final_text += "".join(source_tokens[previous_end:])
        return final_text
    else:
        return source


def errant_transform(difflib_edits, input):
    '''
    Transform difflib edits to a set of errant edits
    '''

    # convert difflib edits to output sequence
    final_text = difflib_edits_to_output(difflib_edits, input)

    # get errant edits
    return return_edits(input, final_text)



def difflib_edits(source_sentence, target_sentences=[]):
    source_tokens = get_tokens(nlp(source_sentence))

    target_docs_tokens = [get_tokens(nlp(sent)) for sent in target_sentences]
    all_actions = []

    for i in range(len(target_sentences)):
        target_tokens = target_docs_tokens[i]
        matcher = SequenceMatcher(None, source_tokens, target_tokens)
        raw_diffs = list(matcher.get_opcodes())
        for diff in raw_diffs:
            if diff[0] == 'replace':
                # "source_start_token", "source_end_token", "target_part"
                all_actions.append(
                    ('replace', diff[1], diff[2], "".join(target_tokens[diff[3]: diff[4]]))
                )
            if diff[0] == 'delete':
                # "source_start_token", "source_end_token"
                all_actions.append(
                    ('delete', diff[1], diff[2])
                )
            if diff[0] == 'insert':
                # "source_start_token", "target_part"
                all_actions.append(
                    ('insert', diff[1], "".join(target_tokens[diff[3]: diff[4]]))
                )
    return all_actions



def expected_reward(sel_edit_seq, uni_edit_seqs, scorer):
    '''
    Calulate the expected reward for the selected_edit_sequence as per the scorer
    with respect to all uni edit seqs
    '''
    total = 0
    for uni in uni_edit_seqs:
        score = scorer[args.reward](uni, sel_edit_seq)
        total += score
    return total


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--lims', type=str, nargs='+', required=False, help='path to data outputs with predicted sequences to be selection limits')
    commandLineParser.add_argument('--universe', type=str, nargs='+', required=False, help='path to data outputs with predicted sequences defining universe of sequences to compare against')
    commandLineParser.add_argument('--input', type=str, required=True, help='path to input file with source incorrect sequences')
    commandLineParser.add_argument('--outfile', type=str, required=True, help='path to save final predictions')
    commandLineParser.add_argument('--reward', type=str, default='agreement', choices=['agreement', 'jaccard', 'rec', 'prec', 'f05'], help='reward metric to use')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/mbr_lim.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')


    # load predicted data
    lim_data = []
    for fpath in args.lims:
        with open(fpath, 'r') as f:
            seqs = f.readlines()
        seqs = [s.strip('\n') for s in seqs]
        lim_data.append(seqs)

    uni_data = []
    for fpath in args.universe:
        with open(fpath, 'r') as f:
            seqs = f.readlines()
        seqs = [s.strip('\n') for s in seqs]
        uni_data.append(seqs)
    
    # Load input sequences
    with open(args.input, 'r') as f:
        incs = f.readlines()
    
    
    # reward metric
    scorer = {'agreement':edit_agreement, 'jaccard':edit_jaccard, 'rec':edit_rec, 'prec':edit_prec, 'f05':edit_f05}
    
    # select samples
    selected_sample = []
    for n, (lim_samples, uni_samples) in tqdm(enumerate(zip(zip(*lim_data), zip(*uni_data))), total=len(incs)):
        uni_edits = [return_edits(incs[n], s) for s in uni_samples]

        all_lim_edits = difflib_edits(incs[n], lim_samples) # all lim edits combined

        curr = [k for k, v in Counter(all_lim_edits).items() if v == len(lim_samples)] # intersection of all lim seq edits
        curr.sort(key=lambda x: x[1])
        to_add = [k for k, v in Counter(all_lim_edits).items() if v != len(lim_samples)] # edits to add greedily
        to_add.sort(key=lambda x: x[1])

        curr_score = expected_reward(errant_transform(curr, incs[n]), uni_edits, scorer)
        for e in to_add:
            test = curr[:]
            test.append(e)
            test.sort(key=lambda x: x[1])
            score = expected_reward(errant_transform(test, incs[n]), uni_edits, scorer)
            if score > curr_score:
                curr = test[:]
                curr_score = score
        
        # convert difflib edits to output sequence
        out_seq = difflib_edits_to_output(curr, incs[n])
        selected_sample.append(out_seq.rstrip('\n'))

    
    # save selected samples
    with open(args.outfile, 'w') as f:
        for sample in selected_sample:
            f.write(sample+'\n')