# bert2dnn
Large Scale BERT Distillation 

Code for paper "BERT2DNN: BERT Distillation with MassiveUnlabeled Data for Online E-Commerce Search"

## TODOs
- [x] BERT2DNN model implement
- [ ] SST/amazon data pipeline
- [ ] BERT/ERNIE finetune

## Requirements
* Python 3
* Tensorflow 1.15

## Quickstart: 
### Traing data
SST-2 dataset is in a tab-seperated format:
| sentence | Label |
| --- | --- | 
| hide new secretions from the parental units | 0 |

After fine-tuning BERT/ERNIE with this data, we obtain the teacher model, which could be used to predict scores on the transfer dat
aset. 
| sentence | Label | logits | prob | prob_t2 |
| --- | --- | --- | --- | --- |
| hide new secretions from the parental units | 0 | -1.2881309986114502 | 0.024137031017202534 | 0.13589785133992555

This script will generate TF examples containing pair of text and label for training. The text is already tokenized with unigram and bigram tokenizer. The label is a soft target with a selected temperature.
```
python gen_tfrecord.py \
--input_file INPUT_TSV_FILE \
--output_file OUTPUT_TFRECORD \
--idx_text 0 --idx_label 3
```

### Model training
```
python run.py --do_train True --do_eval True
```


<!-- ## Steps
1. Download SST dataset
2. Fine tuning BERT-base model for SST classification task
3. Download another large comment sentiment classification dataset: amazon review dataset
4. Use the fine-tuned BERT model to predict labels for amazon review data
5. Train BERT2DNN model with transfer dataset -->


## Transfer Dataset
Our experiment use two public datasets:
1. Stanford Sentiment Treebank: [SST-2](https://nlp.stanford.edu/sentiment/index.html) [download](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8)
2. Amazon review dataset: [download](https://snap.stanford.edu/data/movies.txt.gz)


