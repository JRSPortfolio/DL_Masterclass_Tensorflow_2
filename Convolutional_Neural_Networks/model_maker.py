from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, InputLayer
from keras.models import Sequential


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
            layer = f'layer_{i}'
            self['feat_args'][layer] = [units[i], d_activation[i]]
            
# class FeatModArgs(dict):
#     def __init__(self, units: list, activation: list):
#         super(FeatModArgs, self).__init__()
#         for i in range(len(units)):
#             layer = f'layer_{i}'
#             self[layer] = [units[i], activation[i]]
            
# class FeaturesModel(Model):
#     def __init__(self, shape: tuple, img_args: ImgExtrArgs, feat_args: FeatModArgs, *args, **kwargs):
#         super(FeaturesModel, self).__init__(*args, **kwargs)
#         self.input_layer = InputLayer(shape = shape)
#         self.img_args = img_args
#         self.feat_args = feat_args
#         self.call_list = []

#         for key in img_args.keys():
#             self.call_list.append(Conv2D(filters = img_args[key][0], kernel_size = img_args[key][1], strides = img_args[key][2], padding = img_args[key][3], activation = img_args[key][4]))
#             self.call_list.append(BatchNormalization())
#             self.call_list.append(MaxPool2D(pool_size = 2, strides = 2))
        
#         self.call_list.append(Flatten())
        
#         for key in feat_args.keys():
#             self.call_list.append(Dense(units = feat_args[key][0], activation = feat_args[key][1]))
#             if feat_args[key][0] != 1:
#                 self.call_list.append(BatchNormalization())
    
#     def call(self, x):
#         for layer in self.call_list:
#             x = layer(x)   
#         return x
    


def make_sequential(model_args: ModelArgs):
    model = Sequential()
    model.add(InputLayer(shape = model_args['shape']))
    
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
    

