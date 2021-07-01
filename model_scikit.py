# coding=utf-8
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from data_post_process import data_post_process
from util import cnames


def mlp_fun(train_x, train_y):
    mlpr = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100),  # 4层隐含层
                        activation='tanh',
                        solver='adam',
                        alpha=0.0001,  # L2惩罚参数
                        max_iter=600,
                        random_state=123, )
    mlpr.fit(train_x, train_y)  # mlp回归拟合
    y_pred = mlpr.predict(train_x)

    return y_pred


def plot_curve(y_res, dims, color):
    plt.title('regression lab curve')
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(dims)]
    plt.plot(x, y_res, color=color)
    plt.savefig("lab_curve.png")
    plt.show()


def calculate_diff(y_pred, Y):
    '''
    :param y_res: 81, 1
    :param Y: 1, 81
    :return: n,1

    '''
    y_pred = np.array(y_pred)
    mse = metrics.mean_squared_error(y_pred, Y)
    # print("mean squared error: {}".format(mse))
    return mse


def skmlp(X, Y):
    colors = list(cnames.keys())
    scale = StandardScaler(with_mean=True, with_std=True)
    dataXS = scale.fit_transform(X)
    dims = Y.shape[1]
    y_res = []
    y_train = []
    plt.title('regression lab curve')
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(dims)]
    for i in range(dims):
        Y_data = Y[:, i]
        train_x, test_x, train_y, test_y = train_test_split(dataXS, Y_data, test_size=0.3, random_state=2)
        y_pred = mlp_fun(train_x, train_y)
        #     y_res.append(np.mean(y_pred))
        # plot_curve(y_res, dims)
        y_res.append(y_pred)
        y_train.append(train_y)
    sample_num = len(y_res[0])
    mse = []
    for i in range(sample_num):
        single_y = [a[i] for a in y_res]
        y_gt = [a[i] for a in y_train]
        # 计算 y_res 和 Y 的误差
        mse.append(calculate_diff(single_y, Y[i, :]))
        plt.plot(x, single_y, color='cornflowerblue')
        plt.plot(x, y_gt, color='pink')
    plt.savefig("lab_curve.png")
    plt.show()
    print(
        "sample numbers: {}, max mse: {}, min mse: {}, mean_mse {}".format(len(mse), max(mse), min(mse), np.mean(mse)))
    # print(collections.Counter(mse))
    print("mse 众数：{}".format(max(mse, key=mse.count)))

    # # 对测试集上进行预测
    # pre_y = mlpr.predict(test_x)
    # print("mean absolute error:", metrics.mean_absolute_error(test_y,pre_y))
    # print("mean squared error:", metrics.mean_squared_error(test_y,pre_y))
    # # 输出在测试集上的R^2
    # print("在训练集上的R^2:", mlpr.score(train_x, train_y))
    # print("在测试集上的R^2:", mlpr.score(test_x, test_y))


# 1. 怎么计算异常点的误差? 标准反射率曲线怎么定义? done.
# 2. 输入当中有稍微离群一些的点嘛？  离群点的误差和平均误差的差异咋样呀？ done, 有离群点.
# 3. 怎么搭起模型, 实现改变膜厚,然后反射率变化,曲线拟合越来越好?

if __name__ == "__main__":
    file1 = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#膜色文件与EVT文件对应表.xlsx'
    file2 = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\膜色数据.xlsx'
    evt_cc_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC'
    data_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\org_refine_thickness_lab_curve.json'
    if not os.path.exists(data_js):
        data_post_process(file1, file2, evt_cc_dir, data_js).run()
        print("data process done!")
    else:
        print("data has already processed! start mlp！！!")

    with open(data_js, encoding="utf-8") as reader:
        thickness_lab_curve = json.load(reader)
    Y = []
    for thickness, lab_curve in thickness_lab_curve.items():
        Y.append(thickness_lab_curve[thickness])
    X = list(thickness_lab_curve.keys())
    X = [i.split(',')[:-1] for i in X]
    X = [[float(i) for i in a] for a in X]
    X = [[i[0], i[1], i[2] * 1.1, i[3], i[4] * 1.1, i[5] * 1.1, i[6]] for i in X]
    Y = [[float(i) for i in a] for a in Y]
    X = np.array(X)
    Y = np.array(Y)
    skmlp(X, Y)
