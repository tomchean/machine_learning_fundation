# -*- coding: utf-8 -*-
import csv
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

data = pd.read_csv('./train.csv', encoding = 'big5')

data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

feature = np.empty([18, 12 * 480])
for fea in range(4320):
    index = int(fea / 18)
    feature[fea % 18, index * 24 : (index + 1) * 24] = raw_data[fea]

raw_data = np.copy(feature)
mean_x = np.empty([18])
std_x = np.empty([18])

for i in range(18):
    mean_x[i] = np.mean(feature[i]) #18 * 5 
    std_x[i] = np.std(feature[i]) #18 * 5 
    for j in range(12 * 480):
        if (std_x[i] != 0):
            feature[i][j] = (feature[i][j] - mean_x[i]) / std_x[i]
        else :
            feature[i][j] = 0

smooth_feature = np.empty([18, 12 * 480])
for i in range(18):
    for j in range(12 * 480):
        if j % 480 == 0:
            smooth_feature[i][j] = feature[i][j]
            continue
        avg = (feature[i][j-1] + feature[i][j])/2
        smooth_feature[i][j] = avg

feature = smooth_feature

np.save('mean.npy', mean_x)
np.save('std.npy', std_x)

x = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            index = month * 480 + day * 24 + hour
            x[month * 471 + day * 24 + hour, :] = feature[:,index : index + 9].reshape(-1)
            y[month * 471 + day * 24 + hour, 0] = raw_data[9, index + 9]


x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)

dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x_train_set = np.concatenate((np.ones([math.floor(len(x) * 0.8), 1]), x_train_set), axis = 1).astype(float)
x_validation = np.concatenate((np.ones([math.floor(len(x) * 0.2)+1, 1]), x_validation), axis = 1).astype(float)
learning_rate = 100
iter_time = 100000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2))/471/12/0.8)#rmse
    gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    if(t % 1000 ==0):
        val_loss = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/471/12/0.2)#rmse
        print(str(t) + ":" + str(loss), str(val_loss))

np.save('weight.npy', w)
guess = np.dot(x_validation, w)
ans = y_validation

with open('check.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(len(ans)):
        row = [guess[i], ans[i]]
        csv_writer.writerow(row)


"""# **Testing**

載入 test data，並且以相似於訓練資料預先處理和特徵萃取的方式處理，使 test data 形成 240 個維度為 18 * 9 + 1 的資料。
"""

testdata = pd.read_csv('./test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        index = int(j/9)
        if std_x[index] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[index]) / std_x[index]
        else:
            test_x[i][j] = 0

smooth_test = np.empty([240, 18*9], dtype = float)
for i in range(len(test_x)):
    for j in range(18):
        for k in range(9):
            if k % 9 == 0:
                smooth_test[i][j*9 + k] = test_x[i][j*9 + k]
                continue
            avg = (test_x[i][j*9+ k-1] + test_x[i][j*9 + k])/2
            smooth_test[i][j*9 + k] = avg

test_x = smooth_test

test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

"""# **Save Prediction to CSV File**"""
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        if ans_y[i][0] < 0:
            ans_y[i][0] = 0
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)

