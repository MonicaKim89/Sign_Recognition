import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import pickle
import random
import joblib #scikit learn 모델저장

# scikit learn
import sklearn
from sklearn.externals import joblib

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


# model
def sav_model_loading (model_path):
    model_file = open(model_path, 'rb')
    model = pickle.load(model_file)
    model_file.close()
    print(model)
    return model





# dataset
def data_for_ml (categories, data_path, num):
    data = []

    

    for category in categories:
        file_path = os.path.join(data_path, category)
        print(file_path)
        label = categories.index(category)

        for img in os.listdir(file_path):
            imgpath = os.path.join(file_path, img)

            try: 
                pet_img = cv2.imread(imgpath, 0)
                pet_img = cv2.resize(pet_img, (num, num))

                images = np.array(pet_img).flatten()

                data.append([images, label])

            except Exception as e:
                pass

    print('data수: ', len(data))
    return data

def feature_label_maker(data):
    features=[]
    labels=[]
    for feature, label in data:
        features.append(feature)
        labels.append(label)
        
    features = np.array(features)
    labels = np.array(labels)

    print('features: ', len(features))
    print('features ex: ', features[0])
    print('feature shape: ', feature.shape)
    print('-----------------------------')
    print('labels: ', len(labels))
    print('labels ex: ', labels[0])
    print('labels shape: ', feature.shape)

    return features, labels


#plots
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label ='precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label = 'recall')
    plt.legend()
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    

## 재현율에 대한 정밀도곡선을 그려서 정밀도/재현율 trade-off확인해보기
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # 대각 점선
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)   


#evaluation

print("Precision Score: ", precision_score(y_train_fragile, y_train_pred))
print("Recall Score: ", recall_score(y_train_fragile, y_train_pred))
print("F1-Score: ", f1_score(y_train_fragile, y_train_pred))