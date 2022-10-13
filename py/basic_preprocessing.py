#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import glob
import math
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
# from IPython.display import Image

#cv
import cv2
import math
from PIL import Image
import math
from scipy import ndimage
import argparse
import imutils
import xml.etree.ElementTree as ET


#시각화
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.image as mpimg
from matplotlib import font_manager, rc
rc('font',family="AppleGothic")
plt.rcParams["font.family"]="AppleGothic" #plt 한글꺠짐
plt.rcParams["font.family"]="Arial" #외국어꺠짐
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 출력 설정
plt.rc('figure', figsize=(10,8))

sns.set(font="AppleGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid') #sns 한글깨짐


# In[3]:

def get_file_list(path):
    file_list = os.listdir(path)
    file_list.sort()
    
    list_file = []
    for i in file_list:
        list_file.append(path+i)
    return list_file

def direct_show(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize = (10,8))
    #xticks/yticks - 눈금표
    plt.xticks([])
    plt.yticks([])
    #코랩에서 안돌아감 주의
    plt.imshow(img, cmap= 'gray')
    plt.show()


# In[6]:


def INPUT_IMG(path):
    i = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    return img


# In[2]:


def show(img):
    #사이즈
    plt.figure(figsize = (10,8))
    #xticks/yticks - 눈금표
    plt.xticks([])
    plt.yticks([])
    #코랩에서 안돌아감 주의
    plt.imshow(img, cmap= 'gray')
    plt.show()


# In[3]:dfs


#이미지 수 확인하기
def count_img(path):
    os.chdir(path)
    files = os.listdir(path)
    for num, i in enumerate(files):
        if i[-1] =='g':
            num +=1
    print('이미지 수', num)


###이미지파일명
def img_names(path):
    os.chdir(path)
    files = os.listdir(path)
    names = []
    for num, i in enumerate(files):
        names.append(i)    
    return names
    

#이미지 불러오기
def get_img(path):
    data_path = os.path.join(path, '*g')
    files= glob.glob(data_path)
    img_list=[]
    for f1 in files:
        img = cv2.imread(f1, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)
#     print('이미지수',len(img_list))
#     print('show(get_img(list_file[1])[0]) 식으로 이미지 불러와서 img로 저장')
    
    return img_list


# In[5]:


def img_trim(img):
    img_ = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #워터마크 지우기
    blur = cv2.GaussianBlur(img, ksize=(7,7), sigmaX=100)
    ret, thresh1 = cv2.threshold(blur, img.mean(), 255, cv2.THRESH_BINARY)

    #엣지찾기
    edged = cv2.Canny(blur, 10, 250)

    #closed edge
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    #finding contour
    cont, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total=0

    #drawing contour
    # cont_img = cv2.drawContours(img, cont, -1, (0, 255, 0),3)

    #contours info
    contours_xy = np.array(cont)
    contours_xy.shape

    # x의 min과 max 찾기
    x_min, x_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
            x_min = min(value)
            x_max = max(value)

    # y의 min, max
    y_min, y_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
            y_min = min(value)
            y_max = max(value)

    # image trim
    x = x_min
    y = y_min
    w = (x_max-x_min)
    h = (y_max-y_min)

    img_trim = img_[y:y+h, x:x+w]

    return img_trim



# In[6]:

#이미지 로테이션
def img_rotation(img, num):
    try:
        if num==90:
             img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # 시계방향으로 90도 회전
        elif num==180:
            img = cv2.rotate(img, cv2.ROTATE_180) # 180도 회전
        elif num==270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) # 반시계방향으로 90도 회전 
                                                         # = 시계방향으로 270도 회전
    except:
        print('에러')
        
    return img


#이미지 저장
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

#show_comparing
def show_img_compar(img_1, img_2 ):
    f, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(img_1, cmap='gray')
    ax[1].imshow(img_2, cmap='gray')
    ax[0].axis('off') #hide the axis
    ax[1].axis('off')
    f.tight_layout()
    plt.show()