# encoding: utf-8
from numpy import *
from os import listdir
import operator
import sys


# 添加内容：
# 图片转为文字图像：
# 图片存到testDigits下面

import os
from PIL import Image
from os import listdir
import shutil


def picture2code(filename1, filename2):
    image_file = Image.open(filename1)
    # 缩放成32*32
    image_file = image_file.resize((32, 32))
    # 转成黑白图像
    image_file = image_file.convert('1')

    width, height = image_file.size

    f1 = open(filename1, 'r')
    f2 = open(filename2, 'w')
    for i in range(height):
        for j in range(width):
            # 获取每个像素值
            pixel = int(image_file.getpixel((j, i)) / 255)
            # 黑白图像中0代表黑色，1代表白色
            # 我希望有内容的部分表示为1，所以将0和1互换
            if (pixel == 0):
                pixel = 1
            elif (pixel == 1):
                pixel = 0
            f2.write(str(pixel))
            if (j == width - 1):
                # 换行
                f2.write('\n')
    f1.close()
    f2.close()


path_picture = 'pictures'
path_txt = 'txt'
# 文件夹下所有文件
pictureList = listdir(path_picture)
picm = len(pictureList)
for i in range(picm):
    pictureNameStr = pictureList[i]
    # 图像路径的完整表示
    picturelocation = os.path.join(path_picture, pictureNameStr)
    # 获取文件前缀，即文件名
    pictureStr = pictureNameStr.split('.')[0]
    # 生成的文本路径的完整表示
    txtlocation = os.path.join(path_txt, '%s_%d.txt' %( pictureStr , i ))
    picture2code(picturelocation, txtlocation)
    # 提取数据向量



def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

def img2vector(filename):
    # 创建向量
    returnVect = zeros((1,1024))

    # 打开数据文件，读取每行内容
    fr = open(filename)

    for i in range(32):
        # 读取每一行
        lineStr = fr.readline()

        # 将每行前32字符转成int存入向量
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])

    return returnVect

def classify0(inX, dataSet, labels, k):
    # 获取样本数据数量
    dataSetSize = dataSet.shape[0]

    # 矩阵运算，计算测试数据与每个样本数据对应数据项的差值
    diffMat = tile(inX, (dataSetSize,1)) - dataSet

    # sqDistances 上一步骤结果平方和
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)

    # 取平方根，得到距离向量
    distances = sqDistances**0.5

    # 按照距离从低到高排序
    sortedDistIndicies = distances.argsort()
    classCount={}

    # 依次取出最近的样本数据
    for i in range(k):
        # 记录该样本数据所属的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    # 对类别出现的频次进行排序，从高到低
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    # 返回出现频次最高的类别
    return sortedClassCount[0][0]

def handwritingClassTest():
    # 样本数据的类标签列表
    hwLabels = []

    # 样本数据文件列表
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)

    # # 初始化样本数据矩阵（M*1024）
    trainingMat = zeros((m,1024))

    # # 依次读取所有样本数据到数据矩阵
    for i in range(m):
        # 提取文件名中的数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)

        # 将样本数据存入矩阵
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)

    # 循环读取测试数据
    testFileList = listdir('digits/testDigits')

    # 初始化错误率
    errorCount = 0.0
    mTest = len(testFileList)


    # 循环测试每个测试数据文件
    for i in range(mTest):
        # 提取文件名中的数字
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        # 提取数据向量
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        # 对数据文件进行分类
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        # 打印KNN算法分类结果和真实的分类
        print(u'模型判断结果为: %d,真实数据为: %d,测试数据文件名为：%s' % (classifierResult, classNumStr, fileNameStr))

        # 判断KNN算法结果是否准确
        if classifierResult != classNumStr: errorCount += 1.0

    # 打印错误率
    print(u'\n错误次数: %d' % errorCount)
    print(u'\n错误率： %f' % (errorCount/float(mTest)))


#测试自己输入的文件内容
def handwritingClass2():
    # 样本数据的类标签列表
    hwLabels = []

    # 样本数据文件列表
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)

    # # 初始化样本数据矩阵（M*1024）
    trainingMat = zeros((m, 1024))

    # # 依次读取所有样本数据到数据矩阵
    for i in range(m):
        # 提取文件名中的数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)

        # 将样本数据存入矩阵
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)

    # 循环读取测试数据
    testFileList = listdir('txt')

    # 初始化错误率
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        vectorUnderTest = img2vector('txt/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print(u'模型判断结果为: %d,测试数据文件名为：%s' % (classifierResult, fileNameStr))


print('\nWitch one do you want?\n'
      '1.test your own number.\n'
      '2.test machine number'
    )
choice = input('>')
#input输入为文本
if choice == "1":
    handwritingClass2()
elif choice == "2":
    handwritingClassTest()
else:
    print('Please input the right number\n',input)
