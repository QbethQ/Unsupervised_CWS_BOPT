# Unsupervised_CWS_BOPT

This is the source code of ***Unsupervised Chinese Word Segmentation with BERT Oriented Probing and Transformation***.

## To run

First download pre-trained BERT model and put in this directory. Add `"num_labels": 2` in `bert-base-chinese-pytorch_model/bert_config.json`.

Run `train.py` to train models. This may take a lot of time.
- Usage:  `python train.py --dataset {pku,msr} [--gpu_id GPU_ID] [--output_dir OUTPUT_DIR]`.
  - The dataset is either 'pku' or 'msr'.
  - The default GPU ID is 0.
  - The default output directory is `./saved_models`.
- You can also change some training settings in line 233-238 and 246-252.

Run `evaluation.py` to examine models on development set, which is randomly chosen from training set. Choose the model with highest evaluation_score. (F1-score is just to show that our method is reasonable. It cannot be the standard to choose model)
- Usage: `python evaluation.py --dataset {pku,msr} [--gpu_id GPU_ID] [--model_dir MODEL_DIR]`.
  - The dataset is either 'pku' or 'msr'.
  - The default GPU ID is 0.
  - The default output directory is `./saved_models`.

Run `segmentor.py` to use the model to segment words.
- Usage: `python segmentor.py --dataset {pku,msr} [--gpu_id GPU_ID] [--model_dir MODEL_DIR] [--model MODEL]`.
  - The dataset is either 'pku' or 'msr'.
  - The default GPU ID is 0.
  - The default model_dir is `./saved_models`.
  - Use the model number to specify a model. The default model is model_0.

Run `score` script in `dataset/scripts/` to see the recall, precision and F1-score. The usage of it is as follows, which is from *2nd International Chinese Word Segmentation Bakeoff*.

> * Scoring
>
> The script 'score' is used to generate compare two segmentations. The
> script takes three arguments:
>
> 1. The training set word list
> 2. The gold standard segmentation
> 3. The segmented test file
>
> You must not mix character encodings when invoking the scoring
> script. For example:
>
> % perl scripts/score gold/pku_training_words.utf8
> gold/pku_test_gold.utf8 test_segmentation.utf8 > score.utf8
