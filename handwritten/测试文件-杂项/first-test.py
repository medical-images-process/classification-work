# coding=utf-8
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import scipy
import cv2
from fractions import Fraction

digits = load_digits()#载入数据
x_data = digits.data #数据
y_data = digits.target #标签

# 标准化
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data)
#分割数据1/4为测试数据，3/4为训练数据

mlp = MLPClassifier(hidden_layer_sizes=(100,50) ,max_iter=500)
mlp.fit(x_train,y_train )

predictions = mlp.predict(x_test)
print('first:\n',predictions)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test,predictions))

def image2Digit(image):
    # 调整为8*8大小
    im_resized = scipy.misc.imresize(image, (8,8))
    # RGB（三维）转为灰度图（一维）
    im_gray = cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY)
    # 调整为0-16之间（digits训练数据的特征规格）像素值——16/255
    im_hex = Fraction(16,255) * im_gray
    # 将图片数据反相（digits训练数据的特征规格——黑底白字）
    im_reverse = 16 - im_hex
    return im_reverse.astype(np.int)
image = scipy.misc.imread("5.jpg")
im_reverse = image2Digit(image)
print(im_reverse)
reshaped = im_reverse.reshape(1,64)
print('reshape:\n',reshaped)
predictions = mlp.predict(reshaped).reshape(2)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test,predictions))
