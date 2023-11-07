import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import cv2
import os
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

from data_loader import train_val_dataset
from model import get_model ,callback , metrics


train_dataset , val_dataset = train_val_dataset()

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1024,
    patience=3,
    verbose=0,)

model = get_model()

metrics = [CategoricalAccuracy(name = "accuracy"), TopKCategoricalAccuracy(k=2, name = "top_k_accuracy")]

model.compile(optimizer=Adam(learning_rate = 0.001),
              loss='categorical_crossentropy',  
              metrics=metrics)

history = model.fit(train_dataset,
          epochs=30,
          validation_data=val_dataset,
         callbacks=[reduce_lr_callback])

model.save("model.hdf5")