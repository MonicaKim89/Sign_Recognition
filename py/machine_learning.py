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