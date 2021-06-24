# coding=utf-8
# cornflowerblue moccasin

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from sklearn.model_selection import train_test_split
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
from sklearn import metrics


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


def show_y_pred(y, colors, epoch, train=True):
    sample_num, dims = y.shape  # 92, 81
    if train:
        plt.title('train epoch {}'.format(epoch))
    else:
        plt.title('eval epoch {}'.format(epoch))
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(dims)]
    for i in range(sample_num):
        single_y = y[i, :]
        single_gt_y = y[i, :]
        plt.plot(x, single_y, color=colors[i])
    plt.savefig("torch_lab_curve.png")
    plt.show()


def train_step(model, optimizer, x, y, epoch):
    input = Variable(x)
    preds = model(input)
    model.train()
    # 关闭mlp的隐含层参数更新
    for index, p in enumerate(model.parameters()):
        p.requires_grad = False
    loss = compute_loss(y, preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch == 199:
        y_pred = preds.detach().numpy()
        show_y_pred(y_pred, colors, epoch + 1, train=True)
    return loss


def eval_step(model, x, colors, epoch):
    model.eval()
    preds = model(x)
    y_pred = preds.detach().numpy()
    show_y_pred(y_pred, colors, epoch + 1, train=False)


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
    # X = [[i[0],i[1],i[2]*1.1,i[3],i[4]*1.2,i[5]*1.2,i[6]] for i in X]
    Y = [[float(i) for i in a] for a in Y]
    X = np.array(X)
    Y = np.array(Y)
    # print(X.shape, Y.shape)
    return X, Y


def plot_y(y):
    sample_num, dims = y.shape  # 92, 81
    plt.title('original lab_curve')
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(dims)]
    for i in range(sample_num):
        single_y = y[i, :]
        plt.plot(x, single_y, color=colors[i])
    plt.savefig("gt_lab_curve.png")
    plt.show()


def run(model, train_dataloader, val_dataloader, epochs, is_train=True):
    if is_train:
        optimizer = optimizers.Adam(model.parameters(),
                                    lr=0.001,
                                    betas=(0.9, 0.999), amsgrad=True)
        loss_list = []
        for (x, y) in train_dataloader:
            plot_y(y)
            break
        for epoch in range(epochs):
            print('-' * 10, 'epoch: {}'.format(epoch + 1), '-' * 10)
            train_loss = 0.
            for (x, y) in train_dataloader:  # y：(92, 81)
                loss = train_step(model, optimizer, x, y, epoch)
                train_loss += loss.item()
            train_loss /= len(train_dataloader)
            print('loss: {:.3f}'.format(train_loss))
            loss_list.append(train_loss)
            if epoch and (epoch + 1) % 50 == 0:  # test data
                for (x, _) in val_dataloader:
                    eval_step(model, x, colors, epoch)
        plot_loss(loss_list)
        torch.save(model.state_dict(), "./mlp.pth")
    else:  # train=False
        model.load_state_dict(torch.load("./mlp.pth"))
        # params = list(model.named_parameters())
        # print(params[0])
        for (x, y) in val_dataloader:
            model.eval()
            preds = model(x)
            loss = compute_loss(y, preds)
            print(loss)


# def gan_sample(model, train_dataloader, val_dataloader, epochs):
#     model.load_state_dict(torch.load("./mlp.pth"))
#
#     for (x, y) in train_dataloader:
#         model.train()
#         preds = model(x)
#         loss = compute_loss(y, preds)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if epoch == 199:
#             y_pred = preds.detach().numpy()
#             show_y_pred(y_pred, colors, epoch+1, train=True)
#         return loss


if __name__ == "__main__":
    # train or eval or gan
    flag = 1  # 只修改input值,不更新模型隐含层的参数了..

    file1 = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#膜色文件与EVT文件对应表.xlsx'
    file2 = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\膜色数据.xlsx'
    evt_cc_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC'
    data_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\thickness_lab_curve.json'
    X, Y = generate_data(file1, file2, evt_cc_dir, data_js)
    hiden_dim = 100
    epochs = 200
    input_dim = X.shape[-1]
    output_dim = Y.shape[-1]
    batch_size = X.shape[0]
    # 数据规整化
    scale = StandardScaler(with_mean=True, with_std=True)
    dataXS = scale.fit_transform(X)
    train_x, test_x, train_y, test_y = train_test_split(dataXS, Y, test_size=0.3, random_state=2)
    train_dataloader = DataLoader((train_x, train_y), batch_size=batch_size, batch_first=False, device=device)
    val_dataloader = DataLoader((test_x, test_y), batch_first=False, device=device)
    model = MLP(input_dim, hiden_dim, output_dim).to(device)
    print(model)

    if flag == 1:
        run(model, train_dataloader, val_dataloader, epochs)
    elif flag == 0:
        run(model, train_dataloader, val_dataloader, epochs, is_train=False)
    # elif flag == 3:
    #     gan_sample(model, train_dataloader, val_dataloader, epochs)

    # else:
    #     # sklearn test
    #     diff = [16.9562, 11.1065,  6.1142, 29.6493, 98.2522, 86.2919, 40.4601]
    #     print(X[0], '===')
    #     for i in range(X.shape[0]):
    #         X[i] = np.array(diff)
    #     print(X[0], '===')
    #     scale = StandardScaler(with_mean=True, with_std=True)
    #     dataXS = scale.fit_transform(X)
    #     train_x, test_x, train_y, test_y = train_test_split(dataXS, Y, test_size=0.3, random_state=2)
    #     mlp_fun(test_x, test_y, train_x, train_y)
