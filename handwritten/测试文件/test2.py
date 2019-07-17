# coding=utf-8
from PIL import Image
import argparse

ascii_char = '01'


def select_ascii_char(r, g, b):
    gray = int((19595 * r + 38469 * g + 7472 * b) >> 16)
    unit = 256.0 / len(ascii_char)
    return ascii_char[int(gray / unit)]


def preimg(img_name, width=100, height=100):
    img = Image.open(img_name)
    print(img.size)
    img = img.resize((width, height), Image.NEAREST)
    print(img.size)

    img.convert('L')
    return img


def img2char(img):
    res = ''
    width, height = img.size
    for h in range(height):
        for w in range(width):
            res += select_ascii_char(*img.getpixel((w, h))[:3])
        res += '\n'
    return res


def save_to_file(pic_str, filename):
    outfile = open(filename, 'a')
    outfile.write(pic_str)
    outfile.close


if __name__ == '__main__':
    img_name = '0.jpg'
    width = 112
    height = 112
    file = 'test.txt'
    img = preimg(img_name, width, height)
    pic_str = img2char(img)
    save_to_file(pic_str, file)

# -*- coding: utf-8 -*-
from numpy import *
import operator
from os import listdir
import time


def classify(inX, dataSet, labels, k):  # 4个输入参数分别为：用于分类的输入向量inX，输入的训练样本集dataSet，标签向量labels，选择最近邻居的数目k
    dataSetSize = dataSet.shape[0]  # 把行数求出来
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile是将inx数组重复n次,把相对距离的x y 求出来
    sqDiffMat = diffMat ** 2  # 表示乘方，diffMar^2
    sqDistance = sqDiffMat.sum(axis=1)  # axis = 1代表行向量相加， 0就是普通的求和
    distance = sqDistance ** 0.5  # 距离
    sortedDisIndicies = distance.argsort()  # np排序，代表的是返回从小到大排序的下标
    classCount = {}
    for i in range(k):  # 选择距离最近的k个点
        voteIlabel = labels[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # get返回指定键的值，否则返回默认值
    sortedClasscount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # 排序
    return sortedClasscount[0][0]


def img2vector(filename):  # 将图像文本向量化，将32*32矩阵转换成1*1024
    returnVect = []
    fr = open(filename)
    for i in range(32):  # 先读32行
        lineStr = fr.readline()
        for j in range(32):
            returnVect.append(int(lineStr[j]))
    return returnVect


def classnumCut(fileName):  # 解析出分类文件的数字
    fileStr = fileName.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    return classNumStr


def trainingDataSet():  # 构建训练集向量，对应分类的标签向量
    hwlabels = []
    trainingFileList = listdir('trainingDigits')  # 获取目录内容
    m = len(trainingFileList)  # 获取长度
    trainingMat = zeros((m, 1024))  # m维向量集的初始化
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 获取第i个训练文件
        hwlabels.append(classnumCut(fileNameStr))  # 将数字答案存起来
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)  # 将对应目录添加到矩阵里
    return hwlabels, trainingMat


def handwritingTest():
    hwLabels, trainingMat = trainingDataSet()  # 构建训练集
    testFileList = listdir('testDigits')  # 获取测试集
    errorCount = 0.0  # 统计错误率
    mTest = len(testFileList)
    t1 = time.time()
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = classnumCut(fileNameStr)
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        # 调用knn算法
        classfileResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print
        "the classfile came back with: %d, the real answer is: %d" % (classfileResult, classNumStr)
        if (classfileResult != classNumStr):  # 统计错误数据
            errorCount += 1.0
    print
    "\nthe total number of tests is: %d" % mTest  # 输出测试总样本数
    print
    "the total number of errors is: %d" % errorCount  # 输出测试错误样本数
    print
    "the total error rate is: %f" % (errorCount / float(mTest))  # 输出错误率
    t2 = time.time()
    print
    "Cost time: %.2fmin, %.4fs." % ((t2 - t1) // 60, (t2 - t1) % 60)  # 测试耗时


if __name__ == "__main__":
    handwritingTest()