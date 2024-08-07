from tensorflow.keras.applications.efficientnet import EfficientNetB4 #type: ignore
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, TopKCategoricalAccuracy #type: ignore
from tensorflow.keras.losses import CategoricalCrossentropy #type: ignore
from tensorflow.keras.optimizers import Adam  #type: ignore
import matplotlib.pyplot as plt
from set_model import ModelConfigs, make_sequential_transfer_learning
from em_det_model_plt_metrics import get_dataset, show_model_metrics, test_model

def config_values():   
    image_size = (200, 200)
    learning_rate = 0.001
    epochs = 30
    batch_size = 32
    dense_01 = 512
    dense_02 = 128
    
    configs = ModelConfigs (image_size = image_size, learning_rate = learning_rate, epochs = epochs,
                            batch_size = batch_size, dense_01 = dense_01, dense_02 = dense_02)

    return configs

def set_backbone(configs: ModelConfigs):
    shape = (configs['image_size'][0], configs['image_size'][1], 3)
    backbone = EfficientNetB4(include_top = False, weights = 'imagenet', input_shape = shape)
    backbone.trainable = False
    
    return backbone

def make_model(train_ds, val_ds, configs: ModelConfigs):
    metrics = [CategoricalAccuracy(name = 'accuracy'), TopKCategoricalAccuracy(k = 2, name = 'top_k_acc'), Precision(name = 'precision'),
               Recall(name = 'recall')]
    
    backbone = set_backbone(configs)
    model = make_sequential_transfer_learning(configs, backbone)
    
    loss = CategoricalCrossentropy()
    
    model.compile(optimizer = Adam(learning_rate = configs['learning_rate']), loss = loss, metrics = metrics)
    model.fit(train_ds, validation_data = val_ds, epochs = configs['epochs'], verbose = 1)
    
    return model


if __name__ == '__main__':
    configs = config_values()
    train_ds, val_ds, test_ds = get_dataset(configs)
    model = make_model(train_ds, val_ds,configs)
    show_model_metrics(model, test_ds, configs)

    test_model(model, test_ds)
    plt.show()