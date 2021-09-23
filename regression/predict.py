# -*- coding: utf-8 -*-
import csv
import sys
import pandas as pd
import numpy as np
import math

testing_data = sys.argv[1]
output_file = sys.argv[2]

mean_x = np.load('./mean.npy')
std_x = np.load('./std.npy')

testdata = pd.read_csv(testing_data, header = None, encoding = 'big5')
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
with open(output_file, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        if ans_y[i][0] < 0:
            ans_y[i][0] = 0
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)

