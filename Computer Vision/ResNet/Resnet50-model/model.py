"""Installing dependencies"""

import numpy as np
#import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
import pandas as pd
from keras.preprocessing import image

#Loading dataset
def load_dataset(path,target_size):
    """
    Argument:
    path - path of training/test datasets
    target_size - required size of image dataset
    Returns:
    dataset for training or test in X,Y pair
    """
    data = pd.read_csv(path)
    data = np.array(data)
    X = []
    for i in range(data.shape[0]):
        img_name = data[i,0]
        img_path = 'Downloads/Stage3/images/' + str(img_name)
        X.append(image.img_to_array(image.load_img(img_path, target_size = target_size)))
    X = np.array(X)
    if path == 'Downloads/Stage3/training_set.csv' or path == 'Downloads/Stage3/train2.csv':
        Y = data[:,1:]
    else:
        Y = None
    return X,Y
 
#Training datset
X3,Y3 = load_dataset('Downloads/Stage3/training_set.csv',target_size = (128,128))

#Identity block of resnet model
def identity_block(X,f,filters,stage,block):
    """
    X - input tensor of shape (m,nH_prev,nW_prev,nC_prev)
    f - filter shape for middle convolutional layer
    filters - list of no of filters to be used
    stage - integer representing the stage of the block
    block - integer representing the block
    """
    """
    Returns
    X - output tensor of shape (nH,nW,nC)
    """
    
    X_shortcut = X
    
    f1,f2,f3 = filters 
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    #First component of main path
    X = Conv2D(filters = f1, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    #Second component of main path
    X = Conv2D(filters = f2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    #Third component of main path
    X = Conv2D(filters = f3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    
    #Shortcut Path
    X = Add()([X_shortcut,X])
    X = Activation('relu')(X)
    
    return X
    
#Convolutional Block of resnet model
def convolutional_block(X,f,filters,stage,block,s):
    """
    X - input tensor of shape (m,nH_prev, nW_prev, nc_prev)
    f - shape of filter of middle convolutional layer 
    filters - list of no of filters to be used
    stage - integer representing the stage of the block
    block - integer representing the block
    s - stride to be used in layers
    """
    """
    Returns
    X - output tensor of shape(m,nH,nW,nC)
    """
    
    X_shortcut = X
    f1,f2,f3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    #First component of main path
    X = Conv2D(filters = f1, kernel_size = (1,1), strides = (s,s), name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    #Second component of main path
    X = Conv2D(filters = f2, kernel_size = (f,f),padding = 'same',strides = (1,1), name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    
    #Third component of main path
    X = Conv2D(filters = f3, kernel_size = (1,1), strides = (1,1), name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    
    #Shortcut Path
    X_shortcut = Conv2D(filters = f3, kernel_size = (1,1), strides = (s,s), name = conv_name_base + '1')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    
    #Add shortcut
    
    X = Add()([X_shortcut,X])
    X = Activation('relu')(X)
    
    return X
    
#Resnet model
def ResNet50(input_shape = (128,128,3) ,classes = 4):
    """
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL ->
    
    Arguments:
    input_shape -- shape of the images of the dataset
    
    Returns:
    model -- a Model() instance in Keras
    """
    X_input = Input(input_shape)
    
    X = ZeroPadding2D((3,3))(X_input)
    
    #Stage 1
    
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    
    #Stage 2
    
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    
    #Stage 3
    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block = 'a',s =2)
    X = identity_block(X, 3, [128,128,512], stage = 3, block = 'b')
    X = identity_block(X, 3, [128,128,512], stage = 3, block = 'c')
    X = identity_block(X, 3, [128,128,512], stage = 3, block = 'd')
    
    #Stage 4
    X = convolutional_block(X, f = 3, filters = [256,256,1024], stage = 4, block = 'a',s =2)
    X = identity_block(X, 3, [256,256,1024], stage = 4, block = 'b')
    X = identity_block(X, 3, [256,256,1024], stage = 4, block = 'c')
    X = identity_block(X, 3, [256,256,1024], stage = 4, block = 'd')
    X = identity_block(X, 3, [256,256,1024], stage = 4, block = 'e')
    X = identity_block(X, 3, [256,256,1024], stage = 4, block = 'f')
    
    #Stage 5
    X = convolutional_block(X, f = 3, filters = [512,512,2048], stage = 5, block = 'a',s =2)
    X = identity_block(X, 3, [512,512,2048], stage = 5, block = 'b')
    X = identity_block(X, 3, [512,512,2048], stage = 5, block = 'c')
    
    #AvgPool
    X = AveragePooling2D((2,2), name = 'avg_pool')(X)
    
    #output layer
    X = Flatten()(X)
    X = Dense(classes, activation='relu', name='fc' + str(classes))(X)
    
    #model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    return model
    
#Building model
model = Model(inputs = X_input, outputs = X, name='ResNet50')

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
model.fit(X3, Y3, epochs = 20, batch_size = 64)

#Loading test dataset
X_test,Y_test = load_dataset('Downloads/Stage3/test_set.csv',target_size = (128,128))

#Predictions on test data
Y_pred = model.predict(X_test)
