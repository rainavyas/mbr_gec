How to get Errant MaxMatch metrics:

Assume you have three files with a sentence per line (generate using create_files.py):
file1.inc # original incorrect sentences
file2.pred # Model prediction sentences

There are two stages:

1) Generate the edits files

errant_parallel -orig file1.inc -cor file2.pred -out edits-pred.m2

The data directory should already contain a prepared .m2 file, e.g. data/CoNLL-14/conll14st-test-data/noalt/official-2014.combined.m2

2) Calculate the overlap F-score

errant_compare -hyp edits-pred.m2 -ref above.m2
