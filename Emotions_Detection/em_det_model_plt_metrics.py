import tensorflow as tf
from data_preparation import SplitRatios, create_dataset, ds_shuffle_split
from set_model import ModelConfigs, make_sequential_model
from tensorflow.keras import Sequential #type:ignore
from tensorflow.keras.optimizers import Adam  #type: ignore
from tensorflow.keras.losses import CategoricalCrossentropy #type: ignore
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, TopKCategoricalAccuracy #type: ignore
from tensorflow.keras.models import load_model #type: ignore
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_dataset(configs: ModelConfigs):
    ds = create_dataset(configs)
    ratios = SplitRatios(0.8, 0.1, 0.1)
    train_ds, val_ds, test_ds = ds_shuffle_split(ds, ratios)
    
    return train_ds, val_ds, test_ds

def make_model(train_ds, val_ds, configs: ModelConfigs):
    metrics = [CategoricalAccuracy(name = 'accuracy'), TopKCategoricalAccuracy(k = 2, name = 'top_k_acc'), Precision(name = 'precision'),
               Recall(name = 'recall')]
    
    model = make_sequential_model(configs)
    
    loss = CategoricalCrossentropy()
    
    model.compile(optimizer = Adam(learning_rate = configs['learning_rate']), loss = loss, metrics = metrics)
    model.fit(train_ds, validation_data = val_ds, epochs = configs['epochs'], verbose = 1)
    
    return model

def set_conf_matrix(model, test_ds):
    test_data, test_labels = transform_test(test_ds)
    
    preds = []
    
    i = 0
    for image in test_data:
        image = np.array(image).reshape(1, 150, 150, 3)
        pred = model(image, training = False).numpy()[0]
        preds.append(pred)
    
    preds_labels = np.argmax(np.array(preds), axis = 1)
    test_labels = np.argmax(test_labels, axis = 1)
    
    cm = confusion_matrix(test_labels, preds_labels)
    print(cm)
        
    plt.figure(figsize = (36, 24))
    sns.heatmap(cm, annot = True)
    plt.title('Confusion Matrix')
    plt.ylabel('Real')
    plt.xlabel('Predicted')
     
def show_model_metrics(model, test_ds, configs: ModelConfigs):
    metrics =  pd.DataFrame(model.history.history)

    fig01, axes01 = plt.subplots(1, 3, figsize = (48, 24))
    plt.subplots_adjust(left = 0.04, right = 0.99, top = 0.99, bottom = 0.08)

    sns.lineplot(metrics[['accuracy', 'val_accuracy', 'top_k_acc', 'val_top_k_acc']], ax = axes01[0])
    axes01[0].set_xticks(range(configs['epochs']))
    axes01[0].set_yticks(np.arange(0, 1.05, 0.05))
    axes01[0].grid(True)
    
    sns.lineplot(metrics[['precision', 'recall', 'val_precision', 'val_recall']], ax = axes01[1])
    axes01[1].set_xticks(range(configs['epochs']))
    axes01[1].set_yticks(np.arange(0, 1.05, 0.05))
    axes01[1].grid(True)
    
    sns.lineplot(metrics[['loss', 'val_loss']], ax = axes01[2])
    axes01[2].set_xticks(range(configs['epochs']))
    axes01[2].set_yticks(np.arange(0, 1.05, 0.05))
    axes01[2].grid(True)
    
    print(f'Model eval: {model.evaluate(test_ds)}')
    
    set_conf_matrix(model, test_ds)
     
# def test_random_image(model):
#     folders = ['angry', 'happy', 'sad']
#     r_folder = np.random.randint(0, 3)
#     image_folder = f'Emotions_Detection/dataset/{folders[r_folder]}'
#     r_image = np.random.randint(0, len(os.listdir(image_folder)))
#     image = cv2.imread(f'{image_folder}/{r_image}.jpg')
#     im = tf.constant(image, dtype = tf.float32)
#     im = tf.expand_dims(im, axis = 0)
#     print(folders[tf.argmax(model(im), axis = -1).numpy()[0]])
    
#     print(image_folder)
#     print(r_image)
#     plt.imshow(tf.constant(image, dtype = tf.float32))
#     plt.show()
    
def config_values():
    # image_size = (200, 200)
    # learning_rate = [0.01]
    # epochs = [5]
    # batch_size = 16
    # conv2d_01_filters = [24]
    # conv2d_01_kernel = [6]
    # conv2d__01_strides = [1]
    # conv2d_02_filters = [16]
    # conv2d_02_kernel = [4]
    # conv2d_02_strides = [1]
    # conv2d_03_filters = [8]
    # conv2d_03_kernel = [2]
    # conv2d_03_strides = [1]
    # dense_01 = [128]
    # dense_02 = [32]
    
    # image_size = (350, 350)
    # learning_rate = 0.005
    # epochs = 300
    # batch_size = 16
    # conv2d_01_filters = 128
    # conv2d_01_kernel = 8
    # conv2d__01_strides = 2
    # conv2d_02_filters = 64
    # conv2d_02_kernel = 4
    # conv2d_02_strides = 1
    # conv2d_03_filters = 16
    # conv2d_03_kernel = 2
    # conv2d_03_strides = 1
    # dense_01 = 128
    # dense_02 = 64
    # dense_03 = 32
    
    image_size = (150, 150)
    learning_rate = 0.005
    epochs = 100
    batch_size = 16
    conv2d_01_filters = 80
    conv2d_01_kernel = 6
    conv2d__01_strides = 2
    conv2d_02_filters = 50
    conv2d_02_kernel = 4
    conv2d_02_strides = 1
    conv2d_03_filters = 16
    conv2d_03_kernel = 2
    conv2d_03_strides = 1
    dense_01 = 128
    dense_02 = 32
    dense_03 = 6
    
    configs = ModelConfigs (image_size = image_size, learning_rate = learning_rate, epochs = epochs,
                            batch_size = batch_size, conv2d_01_filters = conv2d_01_filters, conv2d_01_kernel = conv2d_01_kernel,
                            conv2d_01_strides = conv2d__01_strides, conv2d_02_filters = conv2d_02_filters, conv2d_02_kernel = conv2d_02_kernel,
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
  
def test_model(model, test_ds):
    labels = ['angry', 'happy', 'sad']
    test_data, test_labels = transform_test(test_ds)
    
    random_list = np.random.randint(0, len(test_data), 24).tolist()
    fig, axes = plt.subplots(4, 6, figsize = (36, 18))
    plt.subplots_adjust(left = 0.04, right = 0.96, top = 0.95, bottom = 0.1, wspace = 0.6, hspace = 0.6)

    ax_x = int(0)
    ax_y = int(0)
    for i in random_list:
        image = np.array(test_data[i]).reshape(1, 150, 150, 3)
        pred = model(image, training = False)
        axes[ax_x, ax_y].imshow(test_data[i] / 255)
        real_label_val = np.argmax(test_labels[i]).astype(int)
        pred_label_val = np.argmax(np.round(np.array(pred)).astype(int))
        
        axes[ax_x, ax_y].set_title(f"Real: {labels[real_label_val]} | Pred: {labels[pred_label_val]}")
        ax_y += 1
        if ax_y == 6:
            ax_y = 0
            ax_x += 1
                            
if __name__ == '__main__':
    configs = config_values()
    train_ds, val_ds, test_ds = get_dataset(configs)
    model = make_model(train_ds, val_ds,configs)
    show_model_metrics(model, test_ds, configs)
    # save_model(model, 'test_01')

    # model = load_model('Emotions_Detection/keras_models/test_01.keras')
    test_model(model, test_ds)
    plt.show()

