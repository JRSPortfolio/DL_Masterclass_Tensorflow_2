from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, InputLayer #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Resizing, Rescaling #type: ignore

class ModelConfigs(dict):
    def __init__(self, name = None, method = None, metrics = None, image_size = None, learning_rate = None,
                 epochs = None, batch_size = None, conv2d_01_filters = None, conv2d_01_kernel = None,
                 conv2d_01_strides = None, conv2d_02_filters = None, conv2d_02_kernel = None,
                 conv2d_02_strides = None, conv2d_03_filters = None, conv2d_03_kernel = None,
                 conv2d_03_strides = None, dense_01 = None, dense_02 = None, dense_03 = None):
        """
        Configuration class for model arguments
        
        name: String with the name of the model/model in project
        method: String naming the method to be used in wandb sweep (grid, bayes, random)
        metrics: Dictionary to set the goal in wandb sweep (eg:  {'metric' : {'name' : 'accuracy',
                                                                               'goal' : 'maximize'}},)
        image_size: Tuple of two elements setting the image size
        
        Model Settings to be assign in lists:
        learning_rate = None
        epochs = None
        batch_size = None
        conv2d_01_filters = None
        conv2d_01_kernel = None,
        conv2d_01_strides = None
        conv2d_02_filters = None
        conv2d_02_kernel = None
        conv2d_02_strides = None
        conv2d_03_filters = None
        conv2d_03_kernel = None
        conv2d_03_strides = None
        dense_01 = None
        dense_02 = None
        dense_03 = None
        """
        
        super(ModelConfigs, self).__init__()
        self['name'] =  name
        self['method'] = method
        self['metric'] = metrics
        self['learning_rate'] = learning_rate,
        self['epochs'] = epochs
        self['batch_size'] = batch_size
        self['conv2d_01_filters'] = conv2d_01_filters
        self['conv2d_01_kernel'] = conv2d_01_kernel
        self['conv2d_01_strides'] = conv2d_01_strides
        self['conv2d_02_filters'] = conv2d_02_filters
        self['conv2d_02_kernel'] = conv2d_02_kernel
        self['conv2d_02_strides'] = conv2d_02_strides
        self['conv2d_03_filters'] = conv2d_03_filters
        self['conv2d_03_kernel'] = conv2d_03_kernel
        self['conv2d_03_strides'] = conv2d_03_strides
        self['dense_01'] = dense_01
        self['dense_02'] = dense_02
        self['dense_03'] = dense_03
        self['image_size'] = image_size
        

def make_sequential_model(config: ModelConfigs):
    resize_rescale = Sequential([Resizing(config['image_size'][0], config['image_size'][1]), Rescaling(1/255)])
    
    model = Sequential([InputLayer(input_shape = (None, None, 3)),
                        resize_rescale,
                        Conv2D(filters = config['conv2d_01_filters'], kernel_size = config['conv2d_01_kernel'],
                               strides = config['conv2d_01_strides'], padding = 'valid', activation = 'relu'),
                        BatchNormalization(),
                        MaxPool2D(pool_size = 2, strides = 2),
                        Conv2D(filters = config['conv2d_02_filters'], kernel_size = config['conv2d_02_kernel'],
                               strides = config['conv2d_02_strides'], padding = 'valid', activation = 'relu'),
                        BatchNormalization(),
                        MaxPool2D(pool_size = 2, strides = 2),
                        Conv2D(filters = config['conv2d_03_filters'], kernel_size = config['conv2d_03_kernel'],
                               strides = config['conv2d_03_strides'], padding = 'valid', activation = 'relu'),
                        BatchNormalization(),
                        MaxPool2D(pool_size = 2, strides = 2),
                        Flatten(),
                        Dense(units = config['dense_01'], activation = 'relu'),
                        BatchNormalization(),
                        Dense(units = config['dense_02'], activation = 'relu'),
                        BatchNormalization(),
                        Dense(units = config['dense_03'], activation = 'relu'),
                        BatchNormalization(),
                        Dense(units = 3, activation = 'softmax')])
    
    return model