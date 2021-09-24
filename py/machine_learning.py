#warnings
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import pickle
import random
import joblib #scikit learn 모델저장
from collections import Counter


# scikit learn
import sklearn
# from sklearn.externals import joblib

#datset
from sklearn.model_selection import train_test_split

##dataset preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer

#multiclass classifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

#svm
from sklearn.svm import SVC


#ensemble
from sklearn.ensemble import RandomForestClassifier

#validation
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

#evaluation
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score
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
    img_path=[]
    

    for category in categories:
        file_path = os.path.join(data_path, category)
        print(file_path)
        label = categories.index(category)

        for img in os.listdir(file_path):
            imgpath = os.path.join(file_path, img)
            img_path.append(imgpath)

            try: 
                pet_img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                pet_img = cv2.cvtColor(pet_img, cv2.COLOR_BGR2RGB)
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
def get_clf_eval(y_test , pred):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    # ROC-AUC 추가 
    roc_auc = roc_auc_score(y_test, pred)
    print('오차 행렬')
    print(confusion)
    # ROC-AUC print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
    
    return confusion


def get_eval_by_threshold(y_test , pred_proba_c1, thresholds):
    # thresholds list객체내의 값을 차례로 iteration하면서 Evaluation 수행.
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1) 
        custom_predict = binarizer.transform(pred_proba_c1)
        print('임곗값:',custom_threshold)
        get_clf_eval(y_test , custom_predict)


def confusion_plot(confusion_array, x):
    plt.figure(figsize = (x,x))
    ax= plt.subplot() 
    sns.heatmap(confusion_array, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion matrix')
