import pyzbar.pyzbar as pyzbar
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("C:\\Users\\yukir\\Documents\\Monicas_workspace\\Sign detection\\QR_barcode\\barcode.jpg")
if (img.shape) >=(800,1000):
    img =cv2.resize(img, (500,500))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dot=[]
decoded = pyzbar.decode(gray)
for i in decoded:
    try: 
        print(i.data)
        cv2.rectangle(img,(i.rect[0], i.rect[1]), (i.rect[0] + i.rect[2], i.rect[1]+i.rect[3]),(0,0,255),3)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        k=i.polygon
        print(len(k))
        for n in range(k):
            dot.append(k[n])
            pts = np.array(dot, np.int32)
            cv2.polylines(img, [pts], True, (255,0,0),3)
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()