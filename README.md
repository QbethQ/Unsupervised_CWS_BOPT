# Unsupervised_CWS_BOPT
This is the source code of ***Unsupervised Chinese Word Segmentation with BERT Oriented Probing and Transformation***.

## To run
First download pre-trained BERT model and put in this directory. Add `"num_labels": 2` in `bert-base-chinese-pytorch_model/bert_config.json`.

Run `train.py` to train models. This may take a lot of time.
- Specify a dataset: modify the code in line 44 to `dataset = 'pku'` to train on PKU dataset, and to `dataset = 'msr'` to train on MSR dataset.
- Specify an output directory: modify the code in line 45.
- You can also change some training settings in line 227-232 and 240-246.

Run `evaluation.py` to examine models on development set, which is randomly chosen from training set. Choose the model with highest evaluation_score. (F1-score is just to show that our method is reasonable. It cannot be the standard to choose model)

Run `segmentor.py` to use the model to segment words.
- Modify the code in line 37-38 to specify a model.

Run `score` script in `dataset/scripts/` to see the recall, precision and F1-score. The usage of it is as follows, which is from *2nd International Chinese Word Segmentation Bakeoff*.

> * Scoring
>
> The script 'score' is used to generate compare two segmentations. The
script takes three arguments:
> 
> 1. The training set word list
> 2. The gold standard segmentation
> 3. The segmented test file
> 
> You must not mix character encodings when invoking the scoring
> script. For example:
> 
> % perl scripts/score gold/pku_training_words.utf8 \
>     gold/pku_test_gold.utf8 test_segmentation.utf8 > score.utf8
