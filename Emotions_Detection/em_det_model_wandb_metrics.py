import tensorflow as tf
from data_preparation import SplitRatios, create_dataset, ds_shuffle_split_prefetch, encode_image
from set_model import ModelConfigs, make_sequential_wandb
from tensorflow.keras.optimizers import Adam  #type: ignore
from tensorflow.keras.losses import CategoricalCrossentropy #type: ignore
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, TopKCategoricalAccuracy #type: ignore
from tensorflow.keras.models import load_model #type: ignore
import pandas as pd
import numpy as np
import wandb
from wandb.integration.keras import WandbCallback
from wandb_info import wandb_login_info
from datetime import datetime as dt

class ImgWandbCb(WandbCallback):
    def __init__(self, validation_data, num_samples = 10):
        super().__init__()
        self.num_samples = num_samples
        self.validation_data = validation_data
    
    def transform_test(self):
        test_labels = []
        test_data = []
        
        for data, label in self.validation_data.unbatch().as_numpy_iterator():
            test_data.append(data)
            test_labels.append(label)
            
        test_labels = np.array(test_labels)
        test_data = np.array(test_data)
            
        return test_data, test_labels
    
    def get_random_samples(self):
        test_data, test_labels = self.transform_test()
        
        rand_num = np.random.randint(0, len(test_data) - self.num_samples)
        
        sample_data = []
        sample_labels = []
        
        for i in range(len(test_data)):
            if i > rand_num and i <= self.num_samples + rand_num:
                sample_data.append(np.expand_dims(test_data[i], axis = 0))
                sample_labels.append(np.expand_dims(test_labels[i], axis = 0))
        
        sample_data = np.array(sample_data)
        sample_labels = np.array(sample_labels)
        
        return sample_data, sample_labels
            
    def name_labels(self, label):
        label = np.argmax(label)
        match label:
            case 0:
                return 'Angry'
            case 1:
                return 'Happy'
            case 2:
                return 'Sad'
        
    def make_epoch_preds(self):
        predicted_labels = []
        y_true = []
        probas = []
        
        for images, labels in self.validation_data.as_numpy_iterator():
            preds = self.model.predict(images)
            predicted_labels.extend(preds)
            y_true.extend(labels)
            probas.extend(preds)

        probas = np.array([1 - i for i in probas])
        predicted_labels = np.array(predicted_labels)
        y_true = np.array(y_true)

        return predicted_labels, y_true, probas
    
    def image_sampling_on_epoch(self):
        sample_data, sample_labels = self.get_random_samples()
                
        predictions = []

        for sample in sample_data:
            prediction = (self.model.predict(sample))
            predictions.append(prediction)
            
        predictions = np.array(predictions)

        for i in range(self.num_samples):
            image = sample_data[i]
            true_label = self.name_labels(sample_labels[i])
            pred_label = self.name_labels(predictions[i])
            caption = f'True: {true_label}, Pred: {pred_label}'
            wandb.log({f'validation_sample_{i}': [wandb.Image(image, caption = caption)]})
        
    def on_epoch_end(self, epoch, logs):
        self.image_sampling_on_epoch()
        
        pred_labels = ['Angry', 'Happy', 'Sad']
        predicted_y, y_true, probas = self.make_epoch_preds()
        
        predicted_y = np.argmax(predicted_y, axis = 1)
        y_true = np.argmax(y_true, axis = 1)
                
        wandb.log({'Confusion_Matrix' : wandb.plot.confusion_matrix(y_true = y_true, preds = predicted_y, class_names = pred_labels)})
        
        super().on_epoch_end(epoch, logs)

def get_dataset(configs: ModelConfigs):
    ds = create_dataset(configs)
    ratios = SplitRatios(0.7, 0.15, 0.15)
    train_ds, val_ds, test_ds = ds_shuffle_split_prefetch(ds, ratios)
    
    return train_ds, val_ds, test_ds

def tune_model_params(config = None):
    proj, ent = wandb_login_info()
    with wandb.init(project = proj, entity = ent, config = config):
        config = wandb.config

        model = make_sequential_wandb(config)
        train_ds, val_ds, test_ds = get_dataset(configs)
        
        wdbc = ImgWandbCb(validation_data = test_ds)
        loss = CategoricalCrossentropy()
        metrics = [CategoricalAccuracy(name = 'accuracy'), TopKCategoricalAccuracy(k = 2, name = 'top_k_acc'), Precision(name = 'precision'),
                Recall(name = 'recall')]

        model.compile(optimizer = Adam(learning_rate = config.learning_rate), loss = loss, metrics = metrics)
        model.fit(train_ds, validation_data = val_ds, epochs = config.epochs, verbose = 1, callbacks = [wdbc])

def make_wb_sweep_config(configs: ModelConfigs):
    
    sweep_config = {'name' : configs['name'],
                    'method' : configs['method'],
                    'metric' : configs['metric'][0],
                    'parameters' : {'image_size' : {'values' : [configs['image_size']]},
                                    'learning_rate' : {'values' : configs['learning_rate']},
                                    'epochs' : {'values' : configs['epochs']},
                                    'batch_size': {'values' : configs['batch_size']},
                                    'conv2d_01_filters' : {'values' : configs['conv2d_01_filters']},
                                    'conv2d_01_kernel' : {'values' : configs['conv2d_01_kernel']},
                                    'conv2d_01_strides' : {'values' : configs['conv2d_01_strides']},
                                    'conv2d_02_filters' : {'values' : configs['conv2d_02_filters']},
                                    'conv2d_02_kernel' : {'values' : configs['conv2d_02_kernel']},
                                    'conv2d_02_strides' : {'values' : configs['conv2d_02_strides']},
                                    'conv2d_03_filters' : {'values' : configs['conv2d_03_filters']},
                                    'conv2d_03_kernel' : {'values' : configs['conv2d_03_kernel']},
                                    'conv2d_03_strides' : {'values' : configs['conv2d_03_strides']},
                                    'dense_01': {'values' : configs['dense_01']},
                                    'dense_02': {'values' : configs['dense_02']},
                                    'dense_03' : {'values' : configs['dense_03']}}}
         
    return sweep_config
         
def config_values():
    name = 'Emotions_Detection'
    method = 'bayes'
    metric = {'name' : 'precision',
              'goal' : 'maximize'},
    image_size = (100, 100)
    learning_rate = [0.01, 0.001]
    epochs = [5, 10, 20]
    batch_size = [32]
    conv2d_01_filters = [12, 18, 24]
    conv2d_01_kernel = [4, 8, 12]
    conv2d_01_strides = [1, 2]
    conv2d_02_filters = [6, 12]
    conv2d_02_kernel = [3, 6]
    conv2d_02_strides = [1]
    conv2d_03_filters = [4, 8]
    conv2d_03_kernel = [2]
    conv2d_03_strides = [1]
    dense_01 = [40, 60, 100]
    dense_02 = [20, 40]
    dense_03 = [5, 10]
            
    configs = ModelConfigs (name = name, method = method, metrics = metric, image_size = image_size, learning_rate = learning_rate,
                            epochs = epochs, batch_size = batch_size, conv2d_01_filters = conv2d_01_filters, conv2d_01_kernel = conv2d_01_kernel,
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
    sweep_config = make_wb_sweep_config(configs)
    sweep_id = wandb.sweep(sweep_config, project = 'Emotions_Detection')
    wandb.agent(sweep_id, function = tune_model_params, count = 6)

    # model = load_model('Emotions_Detection/keras_models/test_01.keras')


