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

annotator = errant.load('en')


def return_edits(input, prediction):
    '''
    Get edits
    '''
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

def edit_jaccard(edits1, edits2):
    list1 = [e.o_str+' -> '+e.c_str for e in edits1]
    list2 = [e.o_str+' -> '+e.c_str for e in edits2]

    if len(list1) == 0 and len(list2) == 0:
        return 1

    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection

    return float(intersection) / union

def edit_rec(edits1, edits2):

    # rec estimate based reward
    # edits1: ref 
    # edits2: hyp
    list1 = [e.o_str+' -> '+e.c_str for e in edits1]
    list2 = [e.o_str+' -> '+e.c_str for e in edits2]

    if len(list1) == 0:
        return 1

    intersection = len(list(set(list1).intersection(list2)))
    return float(intersection) / len(list1)

def edit_prec(edits1, edits2):

    # prec estimate based reward
    # edits1: ref 
    # edits2: hyp
    list1 = [e.o_str+' -> '+e.c_str for e in edits1]
    list2 = [e.o_str+' -> '+e.c_str for e in edits2]

    if len(list2) == 0:
        return 1

    intersection = len(list(set(list1).intersection(list2)))
    return float(intersection) / len(list2)

def edit_f05(edits1, edits2):
    # f05 estimate based reward
    # edits1: ref 
    # edits2: hyp
    k = 0.5
    list1 = [e.o_str+' -> '+e.c_str for e in edits1]
    list2 = [e.o_str+' -> '+e.c_str for e in edits2]
    if len(list1 + list2) == 0:
        return 1
    intersection = len(list(set(list1).intersection(list2)))
    # return ((1+(k**2))*intersection)/((k*len(list1)) + len(list2)) # original code implemented incorrectly: thanks to Takumi Goto for spotting this
    return ((1+(k**2))*intersection)/(((k**2)*len(list1)) + len(list2))


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--pred_files', type=str, nargs='+', required=False, help='path to data outputs with predicted sequences')
    commandLineParser.add_argument('--input', type=str, required=True, help='path to input file with source incorrect sequences')
    commandLineParser.add_argument('--outfile', type=str, required=True, help='path to save final predictions')
    commandLineParser.add_argument('--reward', type=str, default='agreement', choices=['agreement', 'jaccard', 'rec', 'prec', 'f05'], help='reward metric to use')
    commandLineParser.add_argument('--upperbound', action='store_true', help='select sample wrt to ref')
    commandLineParser.add_argument('--ref', type=str, default='', required=False, help='path to ref file if upperbound')
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
    
    if args.upperbound:
        # load ref file and align with other files
        with open(args.ref, 'r') as f:
            lines = f.readlines()
        lines = [s.strip('\n') for s in lines]

        id_to_seq = {}
        for l in lines:
            idd = l.split(' ')[0]
            seq = ' '.join(l.split(' ')[1:])
            id_to_seq[idd] = seq
        
        corrs = []
        for l in incs:
            idd = l.split(' ')[0]
            try:
                corrs.append(id_to_seq[idd])
            except:
                corrs.append('no match')

    
    # reward metric
    scorer = {'agreement':edit_agreement, 'jaccard':edit_jaccard, 'rec':edit_rec, 'prec':edit_prec, 'f05':edit_f05}
    
    # select samples
    selected_sample = []
    if args.upperbound:
        for n, samples in tqdm(enumerate(zip(*data)), total=len(incs)):
            edits = [return_edits(incs[n], s) for s in samples]
            corr_edits = return_edits(incs[n], corrs[n])
            best = [None, -1] # [model index, score] 
            for i in range(len(edits)):
                total = scorer[args.reward](corr_edits, edits[i])
                if total > best[1]:
                    best = [i, total]
            selected_sample.append(samples[best[0]])

    else:
        for n, samples in tqdm(enumerate(zip(*data)), total=len(incs)):
            edits = [return_edits(incs[n], s) for s in samples]
            best = [None, -1] # [model index, score] 
            for i in range(len(edits)):
                total = 0
                for j in range(len(edits)):
                    if i == j:
                        continue
                    score = scorer[args.reward](edits[j], edits[i])
                    total += score
                if total > best[1]:
                    best = [i, total]
            selected_sample.append(samples[best[0]])
    
    # save selected samples
    with open(args.outfile, 'w') as f:
        for sample in selected_sample:
            f.write(sample+'\n')
