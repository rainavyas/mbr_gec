'''
Perform Minimum Bayes' Risk decoding , but different selection and universe sequences

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
from mbr import edit_agreement, edit_jaccard, edit_f05, edit_prec, edit_rec, return_edits

annotator = errant.load('en')


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--selection', type=str, nargs='+', required=False, help='path to data outputs with predicted sequences to select from')
    commandLineParser.add_argument('--universe', type=str, nargs='+', required=False, help='path to data outputs with predicted sequences defining universe of sequences to compare against')
    commandLineParser.add_argument('--input', type=str, required=True, help='path to input file with source incorrect sequences')
    commandLineParser.add_argument('--outfile', type=str, required=True, help='path to save final predictions')
    commandLineParser.add_argument('--reward', type=str, default='agreement', choices=['agreement', 'jaccard', 'rec', 'prec', 'f05'], help='reward metric to use')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/mbr_diff.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')


    # load predicted data
    sel_data = []
    for fpath in args.selection:
        with open(fpath, 'r') as f:
            seqs = f.readlines()
        seqs = [s.strip('\n') for s in seqs]
        sel_data.append(seqs)

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

    for n, (sel_samples, uni_samples) in tqdm(enumerate(zip(zip(*sel_data), zip(*uni_data))), total=len(incs)):
        sel_edits = [return_edits(incs[n], s) for s in sel_samples]
        uni_edits = [return_edits(incs[n], s) for s in uni_samples]
        best = [None, -1] # [model index, score] 
        for i in range(len(sel_edits)):
            total = 0
            for j in range(len(uni_edits)):
                score = scorer[args.reward](uni_edits[j], sel_edits[i])
                total += score
            if total > best[1]:
                best = [i, total]
        selected_sample.append(sel_samples[best[0]])
    
    # save selected samples
    with open(args.outfile, 'w') as f:
        for sample in selected_sample:
            f.write(sample+'\n')