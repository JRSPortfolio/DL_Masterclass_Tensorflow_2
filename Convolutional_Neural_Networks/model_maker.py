from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, InputLayer #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from tensorboard.plugins.hparams import api as hp

class ModelArgs(dict):
    def __init__(self, shape: tuple, filters: list, kernel_size: list, strides: list, padding: list, img_activation: list, units: list, d_activation: list):
        super(ModelArgs, self).__init__()
        self['shape'] = shape
        self['img_args'] = {}
        self['feat_args'] = {}
        for i in range(len(filters)):
            layer = f'layer_{i}'
            self['img_args'][layer] = [filters[i], kernel_size[i], strides[i], padding[i], img_activation[i]]
            
        for i in range(len(units)):
            layer = f'dense_{i}'
            self['feat_args'][layer] = [units[i], d_activation[i]]
            
            
def make_sequential(model_args: ModelArgs):
    model = Sequential()
    model.add(InputLayer(input_shape = model_args['shape'])) #input_shape changed to shape on tensorflow 2.16, but that version has unresolved issues with GPU recognition
    
    for key in model_args['img_args'].keys():
        model.add(Conv2D(filters = model_args['img_args'][key][0], kernel_size = model_args['img_args'][key][1],
                         strides = model_args['img_args'][key][2], padding = model_args['img_args'][key][3], activation = model_args['img_args'][key][4]))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size = 2, strides = 2))
        
    model.add(Flatten())
    
    for key in model_args['feat_args'].keys():
        model.add(Dense(units = model_args['feat_args'][key][0], activation = model_args['feat_args'][key][1]))
        if model_args['feat_args'][key][0] != 1:
            model.add(BatchNormalization())
            
    return model
    

def make_sequential_config_wandb(config):        
    model = Sequential([Conv2D(filters = config.conv2d_01_filters, kernel_size = config.conv2d_01_kernel,
                               strides = config.conv2d_01_strides, padding = 'valid', activation = 'relu',
                               input_shape = (90, 90, 3)),
                        BatchNormalization(),
                        MaxPool2D(pool_size = 2, strides = 2),
                        Conv2D(filters = config.conv2d_02_filters, kernel_size = config.conv2d_02_kernel,
                               strides = config.conv2d_02_strides, padding = 'valid', activation = 'relu'),
                        BatchNormalization(),
                        MaxPool2D(pool_size = 2, strides = 2),
                        Conv2D(filters = config.conv2d_03_filters, kernel_size = config.conv2d_03_kernel,
                               strides = config.conv2d_03_strides, padding = 'valid', activation = 'relu'),
                        BatchNormalization(),
                        MaxPool2D(pool_size = 2, strides = 2),
                        Flatten(),
                        Dense(units = config.dense_01, activation = 'relu'),
                        BatchNormalization(),
                        Dense(units = config.dense_02, activation = 'relu'),
                        BatchNormalization(),
                        Dense(units = 1, activation = 'sigmoid')])
    
    return model