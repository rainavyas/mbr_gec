'''
Perform Minimum Bayes' Risk decoding 

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



def return_edits(input, prediction):
    '''
    Get edits
    '''
    annotator = errant.load('en')
    input = annotator.parse(input)
    prediction = annotator.parse(prediction)
    alignment = annotator.align(input, prediction)
    edits = annotator.merge(alignment)
    for e in edits:
        e = annotator.classify(e)
    return edits

def edit_agreement(edits1, edits2):
    '''
        Number of matching edits 
    '''
    edits1_str = [e.o_str+' -> '+e.c_str for e in edits1]
    edits2_str = [e.o_str+' -> '+e.c_str for e in edits2]

    matched = 0
    for e1_str in edits1_str:
        if e1_str in edits2_str:
            matched += 1
    return matched

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--pred_files', type=str, nargs='+', required=False, help='path to data outputs with predicted sequences')
    commandLineParser.add_argument('--input', type=str, required=True, help='path to input file with source incorrect sequences')
    commandLineParser.add_argument('--outfile', type=str, required=True, help='path to save final predictions')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/mbr.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')


    # load predicted data
    data = []
    for fpath in args.pred_files:
        with open(fpath, 'r') as f:
            seqs = f.readlines()
        seqs = [s.strip('\n') for s in seqs]
        data.append(seqs)
    
    # Load input sequences
    with open(args.input, 'r') as f:
        incs = f.readlines()
    
    # select samples
    selected_sample = []
    for n, samples in tqdm(enumerate(zip(*data)), total=len(incs)):
        best = [None, -1] # [model index, score] 
        for i in range(len(samples)):
            total = 0
            edits_i = return_edits(incs[n], samples[i])
            for j in range(len(samples)):
                if i == j:
                    continue
                edits_j = return_edits(incs[n], samples[j])
                score = edit_agreement(edits_j, edits_i)
                total += score
            if total > best[1]:
                best = [i, total]
        selected_sample.append(samples[best[0]])
    
    # save selected samples
    with open(args.outfile, 'w') as f:
        for sample in selected_sample:
            f.write(sample+'\n')