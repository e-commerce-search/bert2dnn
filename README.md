# bert2dnn
Large Scale BERT Distillation 

Code for paper "BERT2DNN: BERT Distillation with MassiveUnlabeled Data for Online E-Commerce Search"

## TODOs
- [x] BERT2DNN model implement
- [ ] SST/amazon data pipeline
- [ ] BERT/ERNIE finetune


## Steps
1. Download SST dataset
2. Fine tuning BERT-base model for SST classification task
3. Download another large comment sentiment classification dataset: amazon review dataset
4. Use the fine-tuned BERT model to predict labels for amazon review data
5. Train BERT2DNN model with transfer dataset


## Transfer Dataset
Our experiment use two public datasets:
1. Stanford Sentiment Treebank: [SST-2](https://nlp.stanford.edu/sentiment/index.html) [download](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8)
2. Amazon review dataset: [download](https://snap.stanford.edu/data/movies.txt.gz)

