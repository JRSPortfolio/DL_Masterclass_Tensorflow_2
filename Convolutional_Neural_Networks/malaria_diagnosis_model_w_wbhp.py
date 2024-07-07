'''
Convolutional Neural Network model for malaria diagnosis based on cell images
'''


import numpy as np
from tensorflow.keras.optimizers import Adam  #type: ignore
from tensorflow.keras.losses import BinaryCrossentropy #type: ignore
from tensorflow.keras.metrics import BinaryAccuracy, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC #type: ignore
from model_maker import make_sequential_config_wandb
from data_retrive_transform import SplitRatios, get_dataset, ds_shuffle_split
import wandb
from wandb.integration.keras import WandbCallback

class ImgWandbCb(WandbCallback):
    def __init__(self, validation_data, num_samples = 10):
        super().__init__()
        self.num_samples = num_samples
        self.validation_data = validation_data
        
    def get_random_samples(self):
        rand_num = np.random.randint(0, len(self.validation_data) - self.num_samples)
        val_sample = self.validation_data.skip(rand_num).take(self.num_samples)
        return val_sample
    
    def name_labels(self, label):
        if label == 0:
            return 'Prasitized'
        else:
            return 'Uninfected'
        
    def make_epoch_preds(self):
        predicted_labels = []
        y_true = []
        probas = []
        
        for images, labels in self.validation_data.as_numpy_iterator():
            preds = self.model.predict(images)
            predicted_labels.extend((preds > 0.5).astype('int32'))
            y_true.extend(labels)
            probas.extend(preds)
            
        probas = np.column_stack([probas, [1 - i for i in probas]])
        predicted_labels = np.array(predicted_labels).flatten()
        y_true = np.array(y_true)
        
        return predicted_labels, y_true, probas
    
    def image_sampling_on_epoch(self):
        samples = self.get_random_samples()
        
        predictions = []
        predicted_labels = []
        pred_images = []
        
        for image, label in samples.as_numpy_iterator():
            prediction = (self.model.predict(image) > 0.5).astype('int32')
            predictions.append(prediction)
            predicted_labels.append(label)
            pred_images.append(image)
            
            
        predictions = np.array(predictions)
        predicted_labels = np.array(predicted_labels)
        pred_images = np.array(pred_images)
        
        for i in range(self.num_samples):
            image = pred_images[i]
            true_label = self.name_labels(predicted_labels[i])
            pred_label = self.name_labels(predictions[i])
            caption = f'True: {true_label}, Pred: {pred_label}'
            wandb.log({f'validation_sample_{i}': [wandb.Image(image, caption = caption)]})
        
    def on_epoch_end(self, epoch, logs):
        self.image_sampling_on_epoch()
        
        pred_labels = ['Parasitized', 'Uninfected']
        predicted_y, y_true, probas = self.make_epoch_preds()
                
        wandb.log({'Confusion_Matrix' : wandb.plot.confusion_matrix(y_true = y_true, preds =  predicted_y, class_names = pred_labels),
                   'ROC_Curve' : wandb.plot.roc_curve(y_true = y_true, y_probas = probas, labels = pred_labels)})
        
        super().on_epoch_end(epoch, logs)
            
def get_splits_from_dataset():
    dataset = get_dataset()
    ratios = SplitRatios(0.6, 0.2, 0.2)
    train_ds, val_ds, test_ds = ds_shuffle_split(dataset, ratios, 1)
    return train_ds, val_ds, test_ds

def sweep_models(config = None):
    with wandb.init(project = 'Malaria_Detection_Sweep', entity = 'blorgus', config = config):
        config = wandb.config
    
        model = make_sequential_config_wandb(config)
        
        train, validation, test = get_splits_from_dataset()
                
        wdbc = ImgWandbCb(validation_data = test)
        
        metrics = [TruePositives(name = 'tp'), FalsePositives(name = 'fp'), TrueNegatives(name = 'tn'), FalseNegatives(name = 'fn'),
                BinaryAccuracy(name = 'accuracy'), Precision(name = 'precision'), Recall(name = 'recall'), AUC(name = 'auc')]
        
        model.compile(optimizer = Adam(learning_rate = config.learning_rate), loss = BinaryCrossentropy(), metrics = metrics)
        model.fit(train, validation_data = validation, epochs = config.epochs, verbose = 1, callbacks = [wdbc])

def make_wb_sweep(learning_rate : list, epochs: list, batch_size: list, conv2d_01_filters : list,
                  conv2d_01_kernel : list, conv2d__01_strides: list, conv2d_02_filters : list,
                  conv2d_02_kernel : list, conv2d_02_strides: list, conv2d_03_filters : list,
                  conv2d_03_kernel : list, conv2d_03_strides: list, dense_01 : list, dense_02 : list):
    sweep_config = {'name' : 'Malaria_Detection_Sweep',
                    'method' : 'bayes',
                    'metric' : {'name' : 'precision',
                                'goal' : 'maximize'},
                    'parameters' : {'learning_rate' : {'values' : learning_rate},
                                    'epochs' : {'values' : epochs},
                                    'batch_size': {'values' : batch_size},
                                    'conv2d_01_filters' : {'values' : conv2d_01_filters},
                                    'conv2d_01_kernel' : {'values' : conv2d_01_kernel},
                                    'conv2d_01_strides' : {'values' : conv2d__01_strides},
                                    'conv2d_02_filters' : {'values' : conv2d_02_filters},
                                    'conv2d_02_kernel' : {'values' : conv2d_02_kernel},
                                    'conv2d_02_strides' : {'values' : conv2d_02_strides},
                                    'conv2d_03_filters' : {'values' : conv2d_03_filters},
                                    'conv2d_03_kernel' : {'values' : conv2d_03_kernel},
                                    'conv2d_03_strides' : {'values' : conv2d_03_strides},
                                    'dense_01': {'values' : dense_01},
                                    'dense_02': {'values' : dense_02}}}
    
    return sweep_config

##
## main
##

if __name__ == '__main__':
    
    learning_rate = [0.001, 0.01, 0.1]
    epochs = [4, 6, 10]
    batch_size = [32, 64]
    conv2d_01_filters = [6, 12, 24, 36]
    conv2d_01_kernel = [2, 4, 6]
    conv2d__01_strides = [1]
    conv2d_02_filters = [6, 12, 24, 36]
    conv2d_02_kernel = [2, 4, 6]
    conv2d_02_strides = [1]
    conv2d_03_filters = [6, 12, 24, 36]
    conv2d_03_kernel = [2, 4, 6]
    conv2d_03_strides = [1]
    dense_01 = [128, 64, 32, 16]
    dense_02 = [128, 64, 32, 16]
    
    
    sweep_config = make_wb_sweep(learning_rate, epochs, batch_size, conv2d_01_filters, conv2d_01_kernel, conv2d__01_strides,
                                 conv2d_02_filters, conv2d_02_kernel, conv2d_02_strides, conv2d_03_filters, conv2d_03_kernel,
                                 conv2d_03_strides, dense_01, dense_02)
    
    sweep_id = wandb.sweep(sweep_config, project = 'Malaria_Detection_Sweep')
    wandb.agent(sweep_id, function = sweep_models, count = 5)


