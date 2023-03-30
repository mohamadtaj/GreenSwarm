import os
import cv2
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist, imdb, reuters
from sklearn.model_selection import train_test_split
import numpy as np
from utils import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

"""Prepare the numpy arrays from the images for each dataset"""
def get_data_xray(data_dir):
    img_size = 150
    data = []
    data_labels = []
    labels = ['NORMAL', 'PNEUMONIA']
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append(np.array(resized_arr))
                data_labels.append(class_num)
            except Exception as e:
                print(e)
    return np.array(data), np.array(data_labels)
    
def get_data_oct(data_dir):
    img_size = 90
    data = []
    data_labels = []
    labels = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append(np.array(resized_arr))
                data_labels.append(class_num)
            except Exception as e:
                print(e)
    return np.array(data), np.array(data_labels)

def get_data_intel(data_dir):
    img_size = 50
    data = []
    data_labels = []
    labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append(np.array(resized_arr))
                data_labels.append(class_num)
            except Exception as e:
                print(e)
    return np.array(data), np.array(data_labels)
    
def get_data_malaria(data_dir):    
    img_size = 150
    data = []
    data_labels = []
    labels = ['Uninfected', 'Parasitized']
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append(np.array(resized_arr))
                data_labels.append(class_num)
            except Exception as e:
                print(e)
    return np.array(data), np.array(data_labels)
    
    
"""load datasets with the minimal required preprocessig steps"""   
def load_dataset(data):

    if (data =="cifar" or data =="cifar10" or data =='CIFAR' or data =='CIFAR10'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        x_train = scale(x_train)
        x_test  = scale(x_test)
        
    elif (data == 'xray' or data == 'x'):

        train_path = './datasets/xray/train'
        val_path = './datasets/xray/val'
        test_path = './datasets/xray/test'      
        
        x_train, y_train = get_data_xray(train_path)
        x_val, y_val = get_data_xray(val_path)
        x_test, y_test = get_data_xray(test_path)
        
        x = np.concatenate((x_train,x_val,x_test), axis=0)         
        y = np.concatenate((y_train,y_val,y_test), axis=0)  

        y = y.flatten()
        x = scale(x)
        
        # split the data into train and test 75% / 25%. The test data is randomly prepared to be balanced.          
        test_size = x.shape[0] * 0.25
        
        np.random.seed(7)
        class_zero_idx = np.random.choice(np.where(y == 0)[0], size = int(test_size//2), replace=False)
        np.random.seed(9)
        class_one_idx = np.random.choice(np.where(y == 1)[0], size = int(test_size//2), replace=False)

        test_idx = np.concatenate((class_zero_idx, class_one_idx), axis=0)
        
        x_test = x[test_idx].copy()
        y_test = y[test_idx].copy()
            
        x_train = np.delete(x, test_idx, axis=0)
        y_train = np.delete(y, test_idx, axis=0)

        m = x_train.shape[0]
        n = x_test.shape[0]
        np.random.seed(0)
        permutation_train = np.random.permutation(m)
        np.random.seed(1)
        permutation_test = np.random.permutation(n) 
        
        x_train = x_train[permutation_train,:]
        y_train = y_train[permutation_train] 
        x_test = x_test[permutation_test,:]
        y_test = y_test[permutation_test]
        
    
    elif (data =='fashion'):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        shape_train = tuple(list(x_train.shape) + [1])
        shape_test = tuple(list(x_test.shape) + [1])
        x_train = x_train.reshape(shape_train)
        x_test = x_test.reshape(shape_test)
        x_train = scale(x_train)
        x_test  = scale(x_test)        

    elif (data == 'malaria'):

        path = './datasets/malaria'
        
        x, y = get_data_malaria(path)
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, shuffle=True)
        
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        x_train = scale(x_train)
        x_test  = scale(x_test)          

    elif (data == 'retinal'):

        train_path = './datasets/retinal/OCT2017/train'
        val_path = './datasets/retinal/OCT2017/val'
        test_path = './datasets/retinal/OCT2017/test'
        
        
        x_train, y_train = get_data_oct(train_path)
        x_val, y_val = get_data_oct(val_path)
        x_test, y_test = get_data_oct(test_path)

        shape_train = tuple(list(x_train.shape) + [1])
        shape_test = tuple(list(x_test.shape) + [1])
        x_train = x_train.reshape(shape_train)
        x_test = x_test.reshape(shape_test)

       
        y_train = y_train.flatten()
        y_test = y_test.flatten()        
        x_train = scale(x_train)
        x_test  = scale(x_test)
        
        train_size = len(x_train)
        test_size = len(x_test)
        
        x_train, y_train, x_test, y_test = reduce_size(x_train, y_train, x_test, y_test, train_size//5, test_size) # reduce the size of the datasetfor simplicity
        

    elif (data == 'intel'):

        train_path = './datasets/intel/seg_train'
        val_path = './datasets/intel/seg_test'
                
        x_train, y_train = get_data_intel(train_path)
        x_test, y_test = get_data_intel(val_path)
        
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        x_train = scale(x_train)
        x_test  = scale(x_test)
        
        
    elif (data =='imdb'):
        (x_train, y_train), (x_test, y_test) = imdb.load_data(
                                            num_words=10000,
                                            skip_top=20,
                                            maxlen=500,
                                            seed=0)       
        
        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=500)
        x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=500)
        
    elif (data =='reuters'):
        (x_train, y_train), (x_test, y_test) = reuters.load_data(
                                            num_words=10000,
                                            skip_top=20,
                                            maxlen=500,
                                            test_split=0.2,
                                            seed=0)       
        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=500) 
        x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=500)        
      
    return x_train, y_train, x_test, y_test