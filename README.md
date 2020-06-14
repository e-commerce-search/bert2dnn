# bert2dnn
Large Scale BERT Distillation 

Code for paper "BERT2DNN: BERT Distillation with MassiveUnlabeled Data for Online E-Commerce Search"

## TODOs
- [x] BERT2DNN model implement
- [ ] SST data pipeline
- [ ] amazon data pipeline
- [ ] BERT/ERNIE finetune


## Steps
1. Download SST dataset
2. Fine tuning BERT-base model for SST classification task
3. Download another large comment sentiment classification dataset: amazon review dataset
4. Use the fine-tuned BERT model to predict labels for amazon review data
5. Train BERT2DNN model with transfer dataset

