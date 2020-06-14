class Config:
    visible_gpus = "0,1,2,3"
    vocab_file="./dict/words.txt"
    embedding_size = 64
    epoches = 1
    dropout = 0
    hidden_units = [1024, 512,128, 64]
    batch_size = 128
    regularizer_scale = 0.01
    learning_rate = 0.01
    data_dir = "./tfrecords"
    input_files = "part-*[1-9].tfrecord"
    eval_files = "part-*0.tfrecord"
    test_files = "part-00.tfrecord"
    model_path = "models/bert2dnn"
    label_thres = 0.5
    throttle_secs = 30
    optimizer = "adagrad"
