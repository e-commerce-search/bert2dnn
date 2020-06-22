import tensorflow as tf
from config import Config



def dnn_model_fn(features, labels, mode, params):
    feature_columns = params['feature_columns']
    net = tf.feature_column.input_layer(features, feature_columns)
    regularizer = tf.contrib.layers.l2_regularizer(Config.regularizer_scale)
    for units in Config.hidden_units:
        net = tf.layers.dense(
            net, units=units, activation=tf.nn.relu, 
            kernel_regularizer=regularizer
        )
        if Config.dropout > 0.0:
            net = tf.layers.dropout(net, Config.dropout, 
                training=(mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(net, 1, activation=None,
            kernel_regularizer=regularizer)
    probs = tf.nn.sigmoid(logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {}
        predictions['logits'] = logits
        predictions['probabilities'] = probs
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    # compute loss
    # labels = tf.Print(labels, [labels], message="labels before expand = ")
    labels = tf.expand_dims(labels, 1) 
    # labels = tf.Print(labels, [labels], message="labels = ")
    # logits = tf.Print(logits, [logits], message="logits = ")
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)

    # compute metrics
    binary_labels = tf.to_int32(tf.greater(labels, Config.label_thres))
    predicted_class = tf.to_int32(tf.greater(probs, 0.5))

    accuracy = tf.metrics.accuracy(binary_labels, predicted_class)
    auc = tf.metrics.auc(binary_labels, probs, name='auc')
    precision = tf.metrics.precision(binary_labels, predicted_class)
    recall = tf.metrics.recall(binary_labels, predicted_class)
    metrics = {'auc': auc, 
        'acc':accuracy, 
        'precision': precision,
        'recall': recall}
    for k, v in metrics.items():
        tf.summary.scalar(k, v[1])

    # evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        summary_hook = tf.train.SummarySaverHook(
            save_steps=200, output_dir=Config.model_path, 
            summary_op=tf.summary.merge_all())
        eval_hooks = [summary_hook]
        return tf.estimator.EstimatorSpec(mode, loss=loss, 
        evaluation_hooks=eval_hooks,
            eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    if Config.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.learning_rate)
    elif Config.optimizer == "adagrad":
        optimizer = tf.train.AdagradOptimizer(learning_rate=Config.learning_rate)
    else:
        raise Exception('Unsupported optimizer...')
    
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    logging_hook = tf.train.LoggingTensorHook({"loss" : loss, 
        "accuracy" : accuracy[1]}, every_n_iter=200)
    
    return tf.estimator.EstimatorSpec(mode, loss=loss, 
        train_op=train_op, training_hooks=[logging_hook])