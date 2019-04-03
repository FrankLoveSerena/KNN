#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# __author__ = 'Frank'
# 导入包
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 加载数据
digits = load_digits()
data = digits.data
# 数据探索
print(data.shape)
# 查看第一幅图像
print(digits.images[0])
# 第一幅图像代表的数字
print(digits.target[0])
# 将第一幅图像显示出来
plt.gray()
plt.imshow(digits.images[0])
plt.show()
# 分割数据，将25%的数据做测试集，其余做训练集
x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size = 0.25, random_state = 33)
# 将特征集规范化
ss = StandardScaler()
ss_train = ss.fit_transform(x_train)
ss_test = ss.transform(x_test)
# 创建KNN模型
knn = KNeighborsClassifier()
knn.fit(ss_train, y_train)
y_predict = knn.predict(ss_test)
acc_score = accuracy_score(y_test, y_predict)
print(round(acc_score, 4))

# 下边我们建立SVM，naive bayes和decision tree模型，以作比较
# 创建SVM模型
from sklearn.svm import SVC

svm = SVC(gamma = 'auto')
svm.fit(ss_train, y_train)
svm_predict = svm.predict(ss_test)
print("SVM准确率为：", accuracy_score(y_test, svm_predict))
# 创建naive bayes模型
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
# 由于多项式朴素贝叶斯的特征值不能为负数，所以我们利用min_max方法对元特征集规范化
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()
mm_train = mm.fit_transform(x_train)
mm_test = mm.transform(x_test)
# 拟合模型并测试
nb.fit(mm_train, y_train)
nb_predict = nb.predict(mm_test)
print("naive bayes准确率为：", accuracy_score(y_test, nb_predict))
# 创建decision tree模型
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(mm_train, y_train)
dt_predict = dt.predict(mm_test)
print("decision tree准确率为：", accuracy_score(y_test, dt_predict))
