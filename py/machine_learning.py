import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import pickle
import random

#model
from sklearn.model_selection import train_test_split

#svm
from sklearn.svm import SVC

#validation
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV

#evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


#model
def sav_model_loading (model_path):
    model_file = open(model_path, 'rb')
    model = pickle.load(model_file)
    model_file.close()
    print(model)
    return model





# dataset

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
        
    return features, labels