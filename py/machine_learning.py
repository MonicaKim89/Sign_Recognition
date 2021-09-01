import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import pickle
import random

#datset
from sklearn.model_selection import train_test_split

##dataset preprocessing
from sklearn.preprocessing import StandardScaler

#multiclass classifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

#svm
from sklearn.svm import SVC

#validation
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

#evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


#model
def sav_model_loading (model_path):
    model_file = open(model_path, 'rb')
    model = pickle.load(model_file)
    model_file.close()
    print(model)
    return model





# dataset
def data_for_ml (categories, data_path):
    data = []

    

    for category in categories:
        file_path = os.path.join(data_path, category)
        print(file_path)
        label = categories.index(category)

        for img in os.listdir(file_path):
            imgpath = os.path.join(file_path, img)

            try: 
                pet_img = cv2.imread(imgpath, 0)
                pet_img = cv2.resize(pet_img, (200,200))

                image = np.array(pet_img).flatten()

                data.append([image, label])

            except Exception as e:
                pass

    print('data수: ', len(data))
    return data

def feature_label_maker(pickle_name, data):
    pick_in = open(pickle_name, 'wb')
    pickle.dump(data, pick_in)
    pick_in.close()

    pick_in = open(pickle_name, 'rb')
    data = pickle.load(pick_in)
    pick_in.close()

    features = []
    labels = []

    for feature, label in data:
        features.append(feature)
        labels.append(label)
    
    print(len(features))
    print(len(labels))
    
    return features, labels

#plots
def plot_precision_recall_vs_threshold(precision, recalls, thresholds):
    plt.plot(thresholds, precision[:-1], 'b--', label ='precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label = 'recall')
    plt.legend()