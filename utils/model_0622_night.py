# coding=utf-8
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from data_load import DataLoader
from data_post_process import data_post_process
from mlp_torch import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
from util import cnames
from torch.autograd import Variable

colors = list(cnames.keys())
from sklearn.neural_network import MLPRegressor


def mlp_fun(test_x, test_y, train_x, train_y):
    # mlp regression
    mlpr = MLPRegressor(hidden_layer_sizes=(1,),  # 4层隐含层
                        activation='tanh',
                        solver='adam',
                        alpha=0.0001,  # L2惩罚参数
                        max_iter=1,
                        random_state=123, )
    mlpr.fit(train_x, train_y)
    pre_y = mlpr.predict(test_x)
    print("mean absolute error:", metrics.mean_absolute_error(test_y, pre_y))


def compute_loss(t, y):
    return nn.MSELoss()(y, t)


def show_y_pred(y, gt_y=None, epo=None, best=None):
    sample_num, dims = y.shape
    plt.title('epoch {} lab_curve'.format(epo))
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(dims)]
    for i in range(sample_num):
        single_y = y[i, :]
        single_gt_y = gt_y[i, :]
        if i == 0:
            plt.plot(x, single_gt_y, color='cornflowerblue', label='origin')
            plt.plot(x, single_y, color='moccasin', label='mlp regression')
        else:
            plt.plot(x, single_gt_y, color='cornflowerblue')
            plt.plot(x, single_y, color='moccasin')
    plt.plot(x, best, color='red', label='target')
    plt.legend()
    plt.savefig("lab_curve1.png")
    plt.show()


def plot_loss(loss):
    x = [i for i in range(len(loss))]
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.plot(x, loss)
    plt.show()


def generate_data(file1, file2, evt_cc_dir, data_js):
    # load json data
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
    # X = [[i[0],i[1],i[2],i[3],i[4]*2,i[5],i[6]] for i in X]
    Y = [[float(i) for i in a] for a in Y]
    X = np.array(X)
    Y = np.array(Y)
    # print(X.shape, Y.shape)
    return X, Y


def compare_res(best):
    best_ = np.array(best)
    y1 = np.load(r'./train.npy')
    y2 = np.load(r'./fine_tune.npy')
    mse1 = []
    mse2 = []
    sample_num, dims = y1.shape
    plt.title('compare lab_curve')
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(dims)]
    for i in range(sample_num):
        a = y1[i, :]
        b = y2[i, :]
        mse1.append(metrics.mean_squared_error(a, best_))
        mse2.append(metrics.mean_squared_error(b, best_))
        plt.plot(x, a, color='cornflowerblue')
        plt.plot(x, b, color='hotpink')
        if i == 0:
            plt.plot(x, a, color='cornflowerblue', label='base')
            plt.plot(x, b, color='lightpink', label='fine-tune')
        else:
            plt.plot(x, a, color='cornflowerblue')
            plt.plot(x, b, color='lightpink')
    plt.plot(x, best, color='red', label='target')
    plt.legend()
    plt.savefig("compare_lab_curve.png")
    plt.show()
    print("base mse: {}, fine_tune mse: {}".format(np.mean(mse1), np.mean(mse2)))
    print(np.mean(mse1) > np.mean(mse2))


def run(model, optimizer, train_dataloader, epochs, best, is_train=True):
    if is_train:
        loss_list = []
        for epoch in range(epochs):
            # print('-' * 10, 'epoch: {}'.format(epoch + 1), '-' * 10)
            for ii, (data, label) in enumerate(train_dataloader):
                input = Variable(data, requires_grad=False)
                target = Variable(label)
                optimizer.zero_grad()
                score = model(input)
                loss = compute_loss(score, target)
                loss_list.append(loss)
                # print('-' * 10, 'loss: {}'.format(loss), '-' * 10)
                loss.backward()
                optimizer.step()
                if epoch == epochs - 1:
                    model.eval()
                    preds = model(data)
                    y_pred = preds.detach().numpy()
                    np.save(r'./train.npy', y_pred)
                    show_y_pred(y_pred, gt_y=label, epo=epoch + 1, best=best)
        plot_loss(loss_list)
        torch.save(model.state_dict(), "./mlp.pth")
    else:
        model.load_state_dict(torch.load("./mlp.pth"))
        for index, p in enumerate(model.parameters()):
            p.requires_grad = False
        loss_list = []
        for epoch in range(epochs):
            for ii, (data, label) in enumerate(train_dataloader):
                input = Variable(data, requires_grad=True)
                target = best * data.shape[0]
                target = np.array(target)
                target = np.reshape(target, (data.shape[0], -1))
                target = Variable(torch.from_numpy(target).float())
                optimizer.zero_grad()
                score = model(input)
                loss = compute_loss(score, target)
                # print('-' * 10, 'loss: {}'.format(loss), '-' * 10)
                loss.backward()
                optimizer.step()
                loss_list.append(loss)
                print("epoch {}, input: {}".format(epoch, scale.inverse_transform(input.detach().numpy())[11]))
        print(loss_list.index(min(loss_list)), '-===')
        # model.eval()
        # if epoch == epochs-1:
        #     preds = model(data)
        #     y_pred = preds.detach().numpy()
        #     np.save(r'./fine_tune.npy', y_pred)
        #     show_y_pred(y_pred, gt_y=label, epo=epoch + 1, best=best)
        #     # print("epoch {}, input: {}".format(epoch, input))
        #     X = scale.inverse_transform(input.detach().numpy())
        #     print(X)
        for ii, (data, label) in enumerate(train_dataloader):
            print(scale.inverse_transform(data)[11])


if __name__ == "__main__":
    # train or fine-tune
    flag = 0
    best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
            1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
            0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
            0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
            3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]
    file1 = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#膜色文件与EVT文件对应表.xlsx'
    file2 = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\膜色数据.xlsx'
    evt_cc_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC'
    data_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\thickness_lab_curve.json'
    X, Y = generate_data(file1, file2, evt_cc_dir, data_js)
    hiden_dim = 50
    epochs_train = 100
    epochs_finetune = 33
    input_dim = X.shape[-1]
    output_dim = Y.shape[-1]
    batch_size = X.shape[0]
    # # 数据规整化
    scale = StandardScaler(with_mean=True, with_std=True)
    X = scale.fit_transform(X)  # 注意后面观察膜厚的变化,需要用到它的逆操作: X = scale.inverse_transform(X)
    # train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=2)
    train_x, train_y = X, Y  # all in train
    train_dataloader = DataLoader((train_x, train_y), batch_size=batch_size, batch_first=False, device=device)
    # val_dataloader = DataLoader((test_x, test_y), batch_first=False, device=device)
    model = MLP(input_dim, hiden_dim, output_dim).to(device)
    # print(model)
    optimizer = optimizers.Adam(model.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999), amsgrad=True)
    if flag == 1:
        run(model, optimizer, train_dataloader, epochs_train, best)
    elif flag == 0:
        run(model, optimizer, train_dataloader, epochs_finetune, best, is_train=False)
        compare_res(best)
    else:
        compare_res(best)  # 仅一个可视化功能,展现手动double某一层膜厚,lab_curve曲线的变化.
