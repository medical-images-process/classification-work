# coding=utf-8

from sklearn import datasets, svm, metrics
import numpy as np
import matplotlib.pyplot as plt

# 加载sklearn.datasets自带的digits数据集
digits = datasets.load_digits()
# 采用SVM分类器
classifier = svm.SVC(gamma=0.001, C=100)

samples = len(digits.data)
X, y = digits.data, digits.target
half = samples // 2
expected = []
predictX = []
images = []

# 取前一半数据作为trainsets，后一半数据为testsets,打印训练结果
classifier.fit(X[:half], y[:half])
print
"Classification report for classifier %s:\n%s\n" % (
classifier, metrics.classification_report(y[half:], classifier.predict(X[half:])))

# 在后一半数据中随意选取四组数据验证，并且以图像形式绘出

for i in range(0, 4):
    index = np.random.random_integers(half, samples)
    predictX.append(X[index])
    expected.append(y[index])
    images.append(digits.images[index])
predicted = list(classifier.predict(predictX))

for index, image in enumerate(images):
    plt.subplot(4, 1, index + 1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Expected:%s,Predicted:%s' % (expected[index], predicted[index]))
plt.show()