'''
Convolutional Neural Network model for malaria diagnosis based on cell images
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.optimizers import Adam  #type: ignore
from tensorflow.keras.losses import BinaryCrossentropy #type: ignore
from tensorflow.keras.metrics import BinaryAccuracy, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC #type: ignore
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard #type: ignore
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from model_maker import ModelArgs, make_sequential
from data_retrive_transform import SplitRatios, get_dataset, ds_shuffle_split, transform_test
from datetime import datetime
import tensorboard
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
            return 'Infected'
        else:
            return 'Uninfected'
    
    def on_epoch_end(self, epoch, logs = None):
        samples = self.get_random_samples()
        print()
        print(len(samples))
        print(type(samples))
        for batch in samples:
            images, labels = batch
            predictions = self.model.predict(images[:self.num_samples])
            predicted_labels = tf.argmax(predictions, axis=1)
        
        for i in range(self.num_samples):
            image = images[i]
            true_label = self.name_labels(labels[i])
            pred_label = self.name_labels(predicted_labels[i])
            caption = f"True: {true_label}, Pred: {pred_label}"
            print(caption)
            wandb.log({f"validation_sample_{i}": [wandb.Image(image, caption = caption)]})
            
def get_splits_from_dataset():
    dataset = get_dataset()
    ratios = SplitRatios(0.6, 0.2, 0.2)
    train_ds, val_ds, test_ds = ds_shuffle_split(dataset, ratios, 32)
    return train_ds, val_ds, test_ds


##
## Model creation with module
##
EPOCHS = 1

# def scheduler(epoch, lr):
#     if epoch <= 3:
#         return lr
#     else:
#         return lr * tf.math.exp(-0.1)

def scheduler(epoch, lr):
    print(epoch)
    if epoch <= 2:
        lr = 0.05
        return lr
    elif epoch <= 4:
        lr = 0.01
        return lr
    else:
        lr = 0.002
        return lr
    
def checkpoint_callback():
    filepath = 'Convolutional_Neural_Networks/model_checkpoints'
    callback = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 0, save_best_only = False,
                               save_weights_only = False, mode = 'auto', save_freq = 'epoch')
    return callback

def create_model(train, validation, model_args: ModelArgs):
    model = make_sequential(model_args)

    print(model.summary())
    
    learning_rate_scheduler = LearningRateScheduler(scheduler, verbose = 1)
    # checkpoint_cb = checkpoint_callback()
    # tb_logdir = f"Convolutional_Neural_Networks/tensorboard_logs/{datetime.now().strftime('%d-%m-%Y_%H:%M')}"
    # tb_callback = TensorBoard(log_dir = tb_logdir, histogram_freq = 1, write_graph = True, update_freq = 'epoch', profile_batch = '10,30')
    
    wandb.init(project = 'Malaria_Detection', entity = 'blorgus')
    wdbc = ImgWandbCb(validation_data = validation)
    
    metrics = [TruePositives(name = 'tp'), FalsePositives(name = 'fp'), TrueNegatives(name = 'tn'), FalseNegatives(name = 'fn'),
               BinaryAccuracy(name = 'accuracy'), Precision(name = 'precision'), Recall(name = 'recall'), AUC(name = 'auc')]
    model.compile(optimizer = Adam(learning_rate = 0), loss = BinaryCrossentropy(), metrics = metrics)
    model.fit(train, validation_data = validation, epochs = EPOCHS, verbose = 1, callbacks = [learning_rate_scheduler, wdbc])
    
    return model

##
##Model visualization and evaluation
##
def evaluate_model(model, test):        
    metrics =  pd.DataFrame(model.history.history)

    fig01, axes01 = plt.subplots(1, 3, figsize = (48, 24))
    plt.subplots_adjust(left = 0.04, right = 0.99, top = 0.99, bottom = 0.08)
    fig02, axes02 = plt.subplots(1, 1, figsize = (48, 24))
    plt.subplots_adjust(left = 0.04, right = 0.99, top = 0.99, bottom = 0.08)

    sns.lineplot(metrics[['accuracy', 'auc', 'val_accuracy', 'val_auc']], ax = axes01[0])
    axes01[0].set_xticks(range(EPOCHS))
    axes01[0].set_yticks(np.arange(0.85, 1.002, 0.001))
    axes01[0].grid(True)
    
    sns.lineplot(metrics[['precision', 'recall', 'val_precision', 'val_recall']], ax = axes01[1])
    axes01[1].set_xticks(range(EPOCHS))
    axes01[1].set_yticks(np.arange(0.85, 1.002, 0.001))
    axes01[1].grid(True)
    
    sns.lineplot(metrics[['loss', 'val_loss']], ax = axes01[2])
    axes01[2].set_xticks(range(EPOCHS))
    axes01[2].set_yticks(np.arange(0, 0.25, 0.001))
    axes01[2].grid(True)
    
    print(f'Model eval: {model.evaluate(test)}')

    ##
    ## Print test prediction reports
    ##
    test_data, test_labels = transform_test(test)

    preds = model.predict(test_data)
    pred_classes = (preds > 0.1).astype('int32')

    print(f'Classification Report:\n{classification_report(test_labels, pred_classes)}')
    print(f'Confusion Matrix:\n{confusion_matrix(test_labels, pred_classes)}')

    ##
    ## Plot ROC curve
    ##
    fp, tp, ths = roc_curve(test_labels, preds)
    sns.lineplot(x = fp, y = tp, ax = axes02)
    for k in range(0, len(ths), 25):
        plt.text(fp[k], tp[k], ths[k])
    plt.grid(True)

    plt.show()
##
## Model saved to file
##
def save_model(model, name: str):
    model.save(f'Convolutional_Neural_Networks/keras_models/{name}.keras')


##
## main
##

if __name__ == '__main__':
    train_ds, val_ds, test_ds = get_splits_from_dataset()
    
    model_args = ModelArgs(shape = (180, 180, 3),
                           filters = [6, 12, 36],
                           kernel_size = [2, 4, 6],
                           strides = [1, 1, 1],
                           padding = ['valid', 'valid', 'valid'],
                           img_activation = ['relu', 'relu', 'relu'],
                           units = [100, 20, 1],
                           d_activation = ['relu', 'relu', 'sigmoid'])

    model = create_model(train_ds, val_ds, model_args)
    # evaluate_model(model, test_ds)
        
    # save_model(model, 'malaria_diagnosis_da_01')


