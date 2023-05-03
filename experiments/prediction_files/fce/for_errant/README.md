How to get Errant MaxMatch metrics:

Assume you have three files with a sentence per line (generate using my align data script):
file1.inc # original incorrect sentences
file2.pred # Model prediction sentences
file3.corr # True manual labels, reference sentences

There are two stages:

1) Generate the edits files

errant_parallel -orig file1.inc -cor file2.pred -out edits-pred.m2
errant_parallel -orig file1.inc -cor file2.corr -out edits-corr.m2

2) Calculate the overlap F-score

errant_compare -hyp edits-pred.m2 -ref edits-corr.m2
