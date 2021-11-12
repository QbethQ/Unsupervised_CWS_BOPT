2nd International Chinese Word Segmentation Bakeoff - Data Release
Release 1, 2005-11-18

* File List

gold/       Contains the gold standard segmentation of the test data
            along with the training data word lists.

scripts/    Contains the scoring script and simple segmenter.

testing/    Contains the unsegmented test data.

training/   Contains the segmented training data.

doc/        Contains the instructions used in the bakeoff.

* Scoring

The script 'score' is used to generate compare two segmentations. The
script takes three arguments:

1. The training set word list
2. The gold standard segmentation
3. The segmented test file

You must not mix character encodings when invoking the scoring
script. For example:

% perl scripts/score gold/cityu_training_words.utf8 \
    gold/cityu_test_gold.utf8 test_segmentation.utf8 > score.ut8

