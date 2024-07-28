import tensorflow as tf
from data_preparation import SplitRatios, create_dataset, ds_shuffle_split_prefetch
from set_model import ModelConfigs, make_sequential_model
from tensorflow.keras.optimizers import Adam  #type: ignore
from tensorflow.keras.losses import CategoricalCrossentropy #type: ignore
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, TopKCategoricalAccuracy #type: ignore
from tensorflow.keras.models import load_model #type: ignore
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import TensorBoard #type: ignore
import tensorboard
from tensorboard.plugins.hparams import api as hp
from datetime import datetime as dt

class MakeHParams(dict):
    def __init__(self, configs: ModelConfigs):
        super(MakeHParams, self).__init__()
        for key in configs.keys():
            self[key] = hp.HParam(key, hp.Discrete(configs[key]))

def get_dataset(configs: ModelConfigs):
    ds = create_dataset(configs)
    ratios = SplitRatios(0.7, 0.15, 0.15)
    train_ds, val_ds, test_ds = ds_shuffle_split_prefetch(ds, ratios)
    
    return train_ds, val_ds, test_ds

def tune_model_params(train_ds, val_ds, params: MakeHParams):    
    metrics = [CategoricalAccuracy(name = 'accuracy'), TopKCategoricalAccuracy(k = 2, name = 'top_k_acc'), Precision(name = 'precision'),
               Recall(name = 'recall')]

    log_dir = f"Emotions_Detection/tflogs_fit/{dt.now().strftime('%d-%m-%Y_%H%M')}"
    
    hparams = {}
    hparams['image_size'] = params['image_size'].domain.values
    
    epochs = params['epochs']
    for epochs in params['epochs'].domain.values:
        hparams['epochs'] = epochs
        for rate in params['learning_rate'].domain.values:
            hparams['learning_rate'] = rate
            for b_size in params['batch_size'].domain.values:
                hparams['batch_size'] = b_size
                for conv2d_01_filters in params['conv2d_01_filters'].domain.values:
                    hparams['conv2d_01_filters'] = conv2d_01_filters
                    for conv2d_01_kernel in params['conv2d_01_kernel'].domain.values:
                        hparams['conv2d_01_kernel'] = conv2d_01_kernel
                        for conv2d_01_strides in params['conv2d_01_strides'].domain.values:
                            hparams['conv2d_01_strides'] = conv2d_01_strides
                            for conv2d_02_filters in params['conv2d_02_filters'].domain.values:
                                hparams['conv2d_02_filters'] = conv2d_02_filters
                                for conv2d_02_kernel in params['conv2d_02_kernel'].domain.values:
                                    hparams['conv2d_02_kernel'] = conv2d_02_kernel
                                    for conv2d_02_strides in params['conv2d_02_strides'].domain.values:
                                        hparams['conv2d_02_strides'] = conv2d_02_strides
                                        for conv2d_03_filters in params['conv2d_03_filters'].domain.values:
                                            hparams['conv2d_03_filters'] = conv2d_03_filters
                                            for conv2d_03_kernel in params['conv2d_03_kernel'].domain.values:
                                                hparams['conv2d_03_kernel'] = conv2d_03_kernel
                                                for conv2d_03_strides in params['conv2d_03_strides'].domain.values:
                                                    hparams['conv2d_03_strides'] = conv2d_03_strides
                                                    for dense_01 in params['dense_01'].domain.values:
                                                        hparams['dense_01'] = dense_01
                                                        for dense_02 in params['dense_02'].domain.values:
                                                            hparams['dense_02'] = dense_02
                                                            for dense_03 in params['dense_03'].domain.values:
                                                                hparams['dense_03'] = dense_03
                                                                name, loss, categorical_accuracy, topK_categorical_accuracy, precision, recall, hp = make_model(log_dir, metrics, train_ds, val_ds, hparams)
                                                                eval_model(log_dir, name, loss, categorical_accuracy, topK_categorical_accuracy, precision, recall, hp)

def make_model(log_dir: str, metrics: list, train_ds, val_ds, hparams: dict):
    model = make_sequential_model(hparams)
    
    hp = hparams.copy()
    hp.pop('image_size')
    
    loss = CategoricalCrossentropy()
    
    name = ''
    for key in hparams:
        name = name + f'{key}_'
    
    board = TensorBoard(log_dir = f'{log_dir}/{name}', histogram_freq = 1, write_graph = True, update_freq = 'epoch')
    
    model.compile(optimizer = Adam(learning_rate = hparams['learning_rate']), loss = loss, metrics = metrics)
    model.fit(train_ds, validation_data = val_ds, epochs = hparams['epochs'], verbose = 1)
    
    loss, categorical_accuracy, topK_categorical_accuracy, precision, recall = model.evaluate(val_ds)
    
    return name, loss, categorical_accuracy, topK_categorical_accuracy, precision, recall, hp

def eval_model(log_dir, name, loss, categorical_accuracy, topK_categorical_accuracy, precision, recall, hparams):    
    file_writer = tf.summary.create_file_writer(f'{log_dir}/fw/{name}')
    with file_writer.as_default():
        hp.hparams(hparams)

        tf.summary.scalar('Loss', loss, step = 1)
        tf.summary.scalar('Categorical_Accuracy', categorical_accuracy, step = 1)
        tf.summary.scalar('TopK_Categorical_Accuracy', topK_categorical_accuracy, step = 1)
        tf.summary.scalar('Precision', precision, step = 1)
        tf.summary.scalar('Recall', recall, step = 1)
         
def config_values():
    image_size = (100, 100)
    learning_rate = [0.01, 0.001]
    epochs = [3, 5]
    batch_size = [6]
    conv2d_01_filters = [12]
    conv2d_01_kernel = [4]
    conv2d_01_strides = [1]
    conv2d_02_filters = [6]
    conv2d_02_kernel = [3]
    conv2d_02_strides = [1]
    conv2d_03_filters = [4]
    conv2d_03_kernel = [2]
    conv2d_03_strides = [1]
    dense_01 = [40]
    dense_02 = [20]
    dense_03 = [3]
            
    configs = ModelConfigs (image_size = image_size, learning_rate = learning_rate, epochs = epochs,
                            batch_size = batch_size, conv2d_01_filters = conv2d_01_filters, conv2d_01_kernel = conv2d_01_kernel,
                            conv2d_01_strides = conv2d_01_strides, conv2d_02_filters = conv2d_02_filters, conv2d_02_kernel = conv2d_02_kernel,
                            conv2d_02_strides = conv2d_02_strides, conv2d_03_filters = conv2d_03_filters, conv2d_03_kernel = conv2d_03_kernel,
                            conv2d_03_strides = conv2d_03_strides, dense_01 = dense_01, dense_02 = dense_02, dense_03 = dense_03)
    
    return configs

def save_model(model, name: str):
    model.save(f'Emotions_Detection/keras_models/{name}.keras')

def transform_test(test_ds):
    test_labels = []
    test_data = []
    
    for batch in test_ds.as_numpy_iterator():
        data_batch, label_batch = batch
        for data, label in zip(data_batch, label_batch):
            test_labels.append(label)
            test_data.append(data)

    test_labels = np.array(test_labels)
    test_data = np.array(test_data)
        
    return test_data, test_labels
  
                                        
if __name__ == '__main__':
    configs = config_values()
    configs.pop('name')
    configs.pop('method')
    configs.pop('metric')
        
    params = MakeHParams(configs)
    train_ds, val_ds, test_ds = get_dataset(configs)
    tune_model_params(train_ds, val_ds, params)

    # model = load_model('Emotions_Detection/keras_models/test_01.keras')


