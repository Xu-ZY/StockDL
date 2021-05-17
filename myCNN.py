# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : xzy
# @FILE     : myCNN.py
# @Time     : 2021/5/15 22:18
# @Software : PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.optimizers import Adam


# 获取数据文件命令参数
# dataFilePath = sys.argv[1]

# 加载数据
# raw_dataset = pd.read_csv(dataFilePath)
raw_dataset = pd.read_csv('ITC.csv')

# 查看数据集
print(raw_dataset.head())
print(raw_dataset.corr())
# 我想预测收盘价

# 特征选择只考虑收盘价close
close = raw_dataset["Close"].to_numpy()
date = raw_dataset["Date"].to_numpy()
print("Close : ", close)

# 数据可视化
x_dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in date.tolist()]
y_close = close.tolist()
plt.plot(x_dates, y_close)
plt.xlabel("dates")
plt.ylabel("close")
# plt.show()

# 特征选择使用定义的窗口大小创建特征和目标
feature = []
target = []
window_size = 7
for i in range(window_size+1, len(y_close)):
    target.append(y_close[i])
    datarow = []
    for j in range(i-window_size, i):
        datarow.append(y_close[j])
    feature.append(datarow)

print("features : ", feature[0:5])
print("targets : ", target[0:5])

# 数据处理，所有数据按照8:2分训练集测试集
train_perct = 80
dividor = int((len(y_close)*train_perct)/100)
x_train = np.array(feature[:dividor])
y_train = np.array(target[:dividor])
x_test = np.array(feature[dividor:])
y_test = np.array(target[dividor:])

# 用reshape()改成适合io和CNN的形状输入
x_train = x_train.reshape((-1, 7, 1))
x_test = x_test.reshape((-1, 7, 1))
print("X_train shape : ", x_train.shape)
print("X_test shape : ", x_test.shape)


# 建立 CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3,
                 activation='relu', input_shape=(7, 1)))
# model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=2))#防止过拟合只有输入矩阵的1/2
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))
opt = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=opt, metrics=['mse'])
#print(model.summary())


# tranning
batch_size = 64
epochs = 5
verbose = True
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(x_test, y_test), verbose=verbose)


# 利用试验数据对模型进行检验和评价
ground_truth = y_test
predicted = model.predict(x_test).tolist()
size_of_groundTruth = ground_truth.size
x_axis_data = []
for i in range(size_of_groundTruth):
    x_axis_data.append(x_dates[dividor+i])

plt.plot(x_axis_data, predicted, '--',
         label="Predicion", color="red")
plt.plot(x_axis_data, ground_truth.tolist(),
         label="Ground Truth", color="yellow")
plt.xlabel("dates")
plt.ylabel("close")
plt.legend()
plt.show()

def errorCalculation(groundTruth, predicted):
    square_errors = 0
    size = 0
    for i in range(groundTruth.size):
        error = groundTruth[i]-predicted[i]
        square_error = error * error
        square_errors += square_error
        size = size + 1
    return square_errors / size


print("MSE of Prediction with GroundTruth(y_test) : ",
      errorCalculation(ground_truth, predicted))
