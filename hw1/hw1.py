import sys
import pandas as pd
import numpy as np
import math
import csv
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv', encoding='big5')

# 获取24小时的数据，通过切片除去前面的日期、测站、测项
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24:(day + 1) * 24] = raw_data[18 * (20 * month + day):18 * (20 * month + day + 1), :]
    month_data[month] = sample

# 每10个小时形成一个数据，共有12*471（480-9）个数据，每个数据有9*18个特征
x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour:day * 24 + hour + 9].reshape(1,
                                                                                                                    -1)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]
# print(x)
# print(y)


# 正则化
mean_x = np.mean(x, axis=0) # 计算平均数
std_x = np.std(x, axis=0) # 计算标准差
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# 划分训练集和数据集,测试不同模型时才用
# x_train_set = x[:math.floor(len(x)*0.8), :]
# y_train_set = y[:math.floor(len(y)*0.8), :]
# x_validation = x[math.floor(len(x)*0.8):, :]
# y_validation = y[math.floor(len(y)*0.8):, :]


# 训练——梯度下降
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12*471, 1]), x), axis=1).astype((float))
learning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 1e-10
loss_list = []
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) /471/12)
    loss_list.append(loss)
    if(t%100 == 0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x,w) - y)
    adagrad += gradient**2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)
# print(w)

testdata = pd.read_csv('test.csv', header=None, encoding='big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)

plt.figure(figsize = (13, 7))
plt.plot(range(iter_time), loss_list)
plt.show()