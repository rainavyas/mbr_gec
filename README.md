This is the official implementation code for the conference paper "Minimum Bayes’ Risk Decoding for System Combination of Grammatical Error Correction System", accepted for publication at the AACL-IJCNLP 2023 conference.

# Minimum Bayes’ Risk Decoding for System Combination of Grammatical Error Correction System
For sequence-to-sequence tasks it is challenging to combine individual system outputs. Further, there is also often a mismatch between the decoding criterion and the one used for assessment. Minimum Bayes' Risk (MBR) decoding can be used to combine system outputs in a manner that encourages better alignment with the final assessment criterion. This paper examines MBR decoding for Grammatical Error Correction (GEC) systems, where performance is usually evaluated in terms of edits and an associated F-score. Hence, we propose a novel MBR loss function directly linked to this form of criterion. Furthermore, an approach to expand the possible set of candidate sentences is described. This builds on a current max-voting combination scheme, as well as individual edit-level selection. Experiments on three popular GEC datasets and with state-of-the-art GEC systems demonstrate the efficacy of the proposed MBR approach. Additionally, the paper highlights how varying reward metrics within the MBR decoding framework can provide control over precision, recall, and the F-score in combined GEC systems.

# Run MBR
1) `predict.py`: generate the prediction file for a specific GEC system
2) `mbr.py`: Generate a new prediction file after performing Minimum Bayes' Risk combination (arguments specificy which form of MBR to do)

## Requirements

`pip install torch errant transformers`
   
