#!/usr/bin/env python
# coding: utf-8

import cv2
import math
import os
import shutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)'
pd.set_option('display.max_rows', None)
from IPython.display import display
import PIL
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import font_manager, rc
rc('font',family="AppleGothic")
plt.rcParams["font.family"]="AppleGothic" #plt 한글꺠짐
plt.rcParams["font.family"]="Arial" #외국어꺠짐
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 출력 설정
plt.rc('figure', figsize=(10,8))

sns.set(font="AppleGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid') #sns 한글깨짐
#그래프 세팅
font = {'family': 'serif',
        'color':  'white',
        'weight': 'normal',
        'size': 16,
        }

#마이너스 폰트
plt.rc('axes', unicode_minus=False) # 마이너스 폰트 설정

#시각화?
import platform
platform.system()

# 운영체제별 한글 폰트 설정
if platform.system() == 'Darwin': # Mac 환경 폰트 설정
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': # Windows 환경 폰트 설정
    plt.rc('font', family='Malgun Gothic')

#scikit-learn
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from skimage.io import imread

#tensorflow
import tensorflow as tf
# import tensorflow.compat.v2 as tf # 애플실리콘
# tf.enable_v2_behavior() #되는지 확인

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

#keras
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.utils import to_categorical


#dataset
import splitfolders
from sklearn.model_selection import train_test_split


def gpu_check():
    print(device_lib.list_local_devices())
    print('tf',tf.__version__)
    print('keras',keras.__version__)
    print('set_global_determinism(seed=1337) 이거 꼭 해라')
    print('set_global_determinism(seed=1337) 이거 꼭 해라')
    print('set_global_determinism(seed=1337) 이거 꼭 해라')
    


def get_label_dict(train_generator ):
# Get label to class_id mapping
    labels = (train_generator.class_indices)
    label_dict = dict((v,k) for k,v in labels.items())
    return  label_dict   

def get_labels( generator ):
    generator.reset()
    labels = []
    for i in range(len(generator)):
        labels.extend(np.array(generator[i][1]) )
    return np.argmax(labels, axis =1)

# def get_pred_labels(model, test_generator):
#     test_generator.reset()
#     pred_vec=model.predict_generator(test_generator,
#                                      steps=test_generator.n, #test_generator.batch_size
#                                      verbose=1)
#  
#     return np.argmax( pred_vec, axis = 1), np.max(pred_vec, axis = 1)

def get_pred_labels(model, test_generator):
    test_generator.reset()
    pred_vec=model.predict_generator(test_generator,
                                     steps=test_generator.n, #test_generator.batch_size
                                     verbose=1)
    predicted_classes = np.argmax(pred_vec, axis=1)
    pred_labels = predicted_classes.tolist()
    return pred_labels


def test_file_name(test_generator):
    test_file_name = []

    for file in test_generator.filenames:
        test_file_name.append(file)
        
    return test_file_name



def plot_history( H, NUM_EPOCHS ):
    plt.style.use("ggplot")
    fig = plt.figure()
    fig.set_size_inches(15, 5)
    
    fig.add_subplot(1, 3, 1)
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Validation Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")

    
    fig.add_subplot(1, 3, 2)
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["accuracy"], label="train_accuracy")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    
    fig.add_subplot(1, 3, 3)
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_accuracy"], label="val_accuracy")
    plt.title("Validation Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")


    plt.show()

SEED = 1337

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED']= str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS']= '1'
    os.environ['TF_CUDNN_DETERMINISTIC']= '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(seed=SEED)



class AttLayer(keras.layers.Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__(**kwargs)


def prepare_image_for_prediction( img):
       
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    # The below function inserts an additional dimension at the axis position provided
    img = np.expand_dims(img, axis=0)
    # perform pre-processing that was done when resnet model was trained.
    return preprocess_input(img)

def get_display_string(pred_class, label_dict):
    txt = ""
    for c, confidence in pred_class:
        print(c)
        print(confidence)
        txt += label_dict[c]
        txt += '['+ str(confidence) +']'
    return txt

def predict (model, real_path):
    img_list = get_img(real_path)
    try:
        for num, i in enumerate (img_list):
            resized_frame = cv2.resize(i, (IMG_SIZE,IMG_SIZE))
            frame_for_pred = prepare_image_for_prediction( resized_frame )
            pred_vec = model.predict(frame_for_pred)
            pred_class =[]
            confidence = np.round(pred_vec.max(),2)
            pc = pred_vec.argmax()
            pred_class.append( (pc, confidence) )
            txt = get_display_string(pred_class, label_dict)
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            show(i)
            print(txt)
    except TypeError:
        print('error')
        pass
    else:
        pass