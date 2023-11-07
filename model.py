import tensorflow as tf
from tensorflow.keras.metrics import TopKCategoricalAccuracy, CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (GlobalAveragePooling2D, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, 
                                     Dropout, RandomRotation,RandomZoom , RandomWidth , RandomHeight , RandomBrightness,
                                     RandomContrast, Rescaling, Resizing, Reshape)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam , SGD 
from tensorflow.keras.callbacks import  ReduceLROnPlateau 
from tensorflow.keras.regularizers  import L2

def get_model():
    model = tf.keras.Sequential([
        InputLayer(input_shape = (32, 32, 3)),

        Conv2D(filters = 32 , kernel_size = 5, kernel_initializer="he_uniform" , padding='same',
            activation = 'relu'),
        BatchNormalization(),
        MaxPool2D (pool_size = 2, strides=2),

        Conv2D(filters = 64 , kernel_size = (3,3), kernel_initializer="he_uniform" , padding='same',activation = 'relu'),
        BatchNormalization(),
        MaxPool2D (pool_size = 2, strides=2),

        Conv2D(filters = 128 , kernel_size = (3,3), kernel_initializer="he_uniform" , padding='same',activation = 'relu'),
        BatchNormalization(),
        MaxPool2D (pool_size = 2, strides=2),
        Dropout(rate = 0.2 ),
        
        
        Conv2D(filters =  256, kernel_size = (3,3), kernel_initializer="he_uniform" , padding='same',activation = 'relu'),
        BatchNormalization(),
        MaxPool2D (pool_size = 2, strides=2),
        Dropout(rate = 0.2 ),

        Flatten(),
        
        
        Dense( 200, activation = "relu",   kernel_initializer="he_uniform"),
        BatchNormalization(),
        Dropout(rate = 0.1),
        
        Dense( 100, activation = "relu", kernel_initializer="he_uniform"),
        BatchNormalization(),

        Dense(10, activation = "softmax"),

    ])

    return model
  
#callbacks
def callback():
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1024,
    patience=3,
    verbose=0,)
    return reduce_lr_callback
#metrics
def metrics():
    metrics = [CategoricalAccuracy(name = "accuracy"), TopKCategoricalAccuracy(k=2, name = "top_k_accuracy")]
    return metrics

