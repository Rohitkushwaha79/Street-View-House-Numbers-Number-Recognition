import pandas as pd
import numpy as np
import mat73
import os
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf


train_mat_path = r'C:\Users\Harry\Desktop\number_model\SVHN\train_digitStruct.mat'
test_mat_path = r'C:\Users\Harry\Desktop\number_model\SVHN\test_digitStruct.mat'
train_img_path = r"C:\Users\Harry\Desktop\number_model\SVHN\train\train\\"
test_img_path = r"C:\Users\Harry\Desktop\number_model\SVHN\test\test\\"


# bbox generation
def bbox_generator(mat_path):
    dict = mat73.loadmat(mat_path)
    bbox_data = dict['digitStruct']['bbox']
    return bbox_data


# Dataframe
def to_DataFrame(data):
    height = []
    label = []
    left = []
    top = []
    width = []

    for item in data:
        height.append(item['height'])
        label.append(item['label'])
        left.append(item['left'])
        top.append(item['top'])
        width.append(item['width'])

    Dataframe = pd.DataFrame({'height': height, 'label': label, 'left': left, 'top': top, 'width': width})
    
    return Dataframe

def Dataframe(mat_path):
    bbox_data  = bbox_generator(mat_path)
    
    dataframe = to_DataFrame(bbox_data)
    return dataframe


# converting all non list values to list
def convert_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


# preprocessing train dataframe
def preprocess_train_val_dataframe(mat_path , train_img_path):
    dataframe = Dataframe(mat_path)
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].apply(convert_to_list)

    len_train = len(os.listdir(train_img_path))

    train_img_name=[i for i in range(1,len_train)]

    lis_train_img_path = [train_img_path + str(i) + ".png" for i in train_img_name]
    dataframe["file_path"] = lis_train_img_path

    train_df , val_df = train_test_split(dataframe , test_size=0.2)

    return train_df , val_df 

# preprocessing test dataframe
def preprocess_test_dataframe(mat_path , test_img_path):
    test_df = Dataframe(mat_path)
    for column in test_df.columns:
        test_df[column] = test_df[column].apply(convert_to_list)

    len_test = len(os.listdir(test_img_path))
    test_img_name=[i for i in range(1,len_test-1)]
    lis_test_img_path = [test_img_path + str(i) +".png" for i in test_img_name]
    test_df["file_path"] = lis_test_img_path

    return test_df

def preprocess_and_extract_roi(row):
    image = cv2.imread(row['file_path'])
    if image is None:
        return None, None

    top_list = row['top']
    left_list = row['left']
    height_list = row['height']
    width_list = row['width']
    label_list = row['label']

    preprocessed_images = []
    labels = []

    for i in range(len(top_list)):
        top = abs(top_list[i])
        left = abs(left_list[i])
        height = abs(height_list[i])
        width = abs(width_list[i])
        if label_list[i]%10==0:
            label = 0
        else:
            label = label_list[i]
        
        

        object_roi = image[int(top):int(top + height), int(left):int(left + width)]
        resized_object = cv2.resize(object_roi, (32, 32))

        preprocessed_images.append(resized_object)
        labels.append(label)

    return preprocessed_images, tf.keras.utils.to_categorical(labels)

def preprocess_and_extract_roi_batch(dataframe):
    preprocessed_images = []
    labels = []

    for index, row in dataframe.iterrows():
        images, lbls = preprocess_and_extract_roi(row)
        if images is not None and lbls is not None:
            preprocessed_images.extend(images)
            labels.extend(lbls)

    return preprocessed_images, labels

def pad_labels(labels, max_length):
    for i in range(len(labels)):
        label = labels[i]
        if len(label) < max_length:
            padding_length = max_length - len(label)
            labels[i] = np.concatenate((label, np.zeros(padding_length)))
    return labels

# creating train and val dataset form dataframes
def to_train_val_dataset(train_mat_path , train_mg_path):

    train_df , val_df  = preprocess_train_val_dataframe(train_mat_path , train_mg_path)

    train_preprocessed_images, train_labels = preprocess_and_extract_roi_batch(train_df)
    train_preprocessed_images = np.array(train_preprocessed_images)
    train_labels = np.array(train_labels , dtype="object")

    val_preprocessed_images, val_labels = preprocess_and_extract_roi_batch(val_df)
    val_preprocessed_images = np.array(val_preprocessed_images)
    val_labels = np.array(val_labels , dtype="object")


    train_labels = pad_labels(train_labels , 10)
    val_labels = pad_labels(val_labels , 10)

    train_labels = [tf.convert_to_tensor(label , dtype = tf.float32) for label in train_labels]
    val_labels = [tf.convert_to_tensor(label , dtype = tf.float32) for label in val_labels]

    train_images = tf.convert_to_tensor(train_preprocessed_images, dtype=tf.float32)
    val_images = tf.convert_to_tensor(val_preprocessed_images, dtype=tf.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

    return train_dataset , val_dataset 

# creating test dataset form dataframe
def to_test_dataset(test_mat_path , test_img_path):

    test_df  = preprocess_test_dataframe(test_mat_path , test_img_path)

    test_preprocessed_images, test_labels = preprocess_and_extract_roi_batch(test_df)
    test_preprocessed_images = np.array(test_preprocessed_images)
    test_labels = np.array(test_labels , dtype="object")

    test_labels = pad_labels(test_labels , 10)

    test_labels = [tf.convert_to_tensor(label , dtype = tf.float32) for label in test_labels]

    test_images = tf.convert_to_tensor(test_preprocessed_images, dtype=tf.float32)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))


    return test_dataset


agumenter_layer = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(factor=(-0.03, 0.03)),
    tf.keras.layers.RandomZoom(height_factor=(0.2, 0.2), width_factor=(0.2, 0.2)),
])
def process_data(image , labels):
    return agumenter_layer(image , training = True) , labels

# preprocessing train and val dataset 
def train_val_dataset(train_mat_path= train_mat_path , train_mg_path= train_img_path):  
    batch_size = 32
    train_dataset , val_dataset  = to_train_val_dataset(train_mat_path , train_mg_path)
    train_dataset = (train_dataset.batch(batch_size)
                 .shuffle(buffer_size = 1024, reshuffle_each_iteration = True)
                 .map(process_data)
                 .prefetch(tf.data.AUTOTUNE))
    val_dataset = val_dataset.shuffle(buffer_size = 1024, reshuffle_each_iteration = True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset , val_dataset

# preprocessing test dataset
def test_dataset(test_mat_patht= test_mat_path , test_img_path = test_img_path):
    batch_size = 32
    test_dataset = to_test_dataset(test_mat_path , test_img_path)
    test_dataset = test_dataset.batch(batch_size).shuffle(buffer_size = 1024, reshuffle_each_iteration = True).prefetch(tf.data.AUTOTUNE)
    return test_dataset
