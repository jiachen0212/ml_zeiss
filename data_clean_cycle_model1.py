# coding=utf-8

'''
deta 模型验证.
寻找时序上连续的一些样本,做10维 deta_thickness 到 16维 sensor std 的映射.

2021.07.23 chen_jia 

'''

import json
import os
import numpy as np
import torch
import torch.optim as optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_load import DataLoader
from mlp_torch import MLP
from util import weighted_mse
import matplotlib.pyplot as plt
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn

def compute_loss(t, y):
    return nn.MSELoss()(y, t)


def show_y_pred(y, gt_y=None, epo=None, best=None, flag='eval'):
    sample_num, dims = y.shape
    plt.title('{} epoch {} lab_curve'.format(flag, epo + 1))
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [i for i in range(dims)]
    for i in range(sample_num):
        single_y = y[i, :]
        single_gt_y = gt_y[i, :]
        plt.plot(x, single_gt_y, color='pink')
        plt.plot(x, single_y, color='black')
    plt.legend()
    # plt.savefig("lab_curve.png")
    plt.show()


def plot_loss(loss):
    x = [i for i in range(len(loss))]
    plt.title('modify-thickness loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, loss)
    # plt.savefig("pretrain_loss.png")
    plt.show()


def generate_data(number33_thick10sensor8step_lab_js):

    if not os.path.exists(number33_thick10sensor8step_lab_js):
        print('lose number_thick10_sensor_std.json!')
    else:
        with open(number33_thick10sensor8step_lab_js, encoding="utf-8") as reader:
            f_lab = json.load(reader)
    X, Y = [], []
    part_clean = open(r'./part_clean_number.txt', 'r').readlines()[0]
    part_clean_number = part_clean.split(',')[:-1]
    # y_inds = [1,3,6,7,11,12,13,14]
    y_inds = open(r'./std_index.txt').readlines()[0].split(',')[:-1]

    for number33, thicksensor_lab in f_lab.items():
        if number33 in part_clean_number:
            x = thicksensor_lab[0].split(',')[:-1]
            y = [x[int(i)] for i in y_inds]
            assert len(x) == 26
            X.append(x[:10])
            Y.append(y)
    X = [[float(i) for i in x] for x in X]
    Y = [[float(i) for i in a] for a in Y]

    return X, Y


def compare_res(best):
    y1 = np.load(r'./step1_y.npy')
    y2 = np.load(r'./step2_y.npy')
    print(y1.shape, y2.shape)
    mse1 = []
    mse2 = []
    sample_num, dims = y1.shape
    plt.title('compare lab_curve')
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [i for i in range(16)]
    for i in range(sample_num):
        a = y1[i, :]
        b = y2[i, :]
        mse1.append(weighted_mse(a))
        mse2.append(weighted_mse(b))
        plt.plot(x, a, color='cornflowerblue')
        plt.plot(x, b, color='hotpink')
        if i == 0:
            plt.plot(x, a, color='cornflowerblue', label='base')
            plt.plot(x, b, color='lightpink', label='modified')
        else:
            plt.plot(x, a, color='cornflowerblue')
            plt.plot(x, b, color='lightpink')
    plt.plot(x, best, color='red', label='target')
    plt.legend()
    # plt.savefig("compare_lab_curve.png")
    plt.show()
    print("base mse: {}, fine_tune mse: {}".format(np.mean(mse1), np.mean(mse2)))


def run_train(X, Y, hiden_dim, output_dim, epochs, i, j):

    batch_size = X.shape[0]
    input_dim = X.shape[-1]

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=3)
    print("train size: {}".format(train_x.shape[0]))
    print("validation size: {}".format(test_x.shape[0]))
    train_dataloader = DataLoader((train_x, train_y), batch_size=batch_size, batch_first=False, device=device)
    val_dataloader = DataLoader((test_x, test_y), batch_size=batch_size, batch_first=False, device=device)
    model = MLP(input_dim, hiden_dim, output_dim).to(device)
    print(model)
    optimizer = optimizers.Adam(model.parameters(),
                                      lr=0.01,
                                      betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)

    loss_list = []
    for epoch in range(epochs):
        train_loss = 0
        for ii, (data, label) in enumerate(train_dataloader):
            input = Variable(data, requires_grad=False)
            target = Variable(label)
            optimizer.zero_grad()
            score = model(input)
            loss = compute_loss(score, target)
            print('-' * 10, 'epoch {} loss: {}'.format(epoch, loss), '-' * 10)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        loss_list.append(train_loss)
        if (epoch + 1) % 2000 == 0:
            model.eval()
            for ii, (input, gt) in enumerate(val_dataloader):
                model.eval()
                pred = model(input)
                y_pred = pred.detach().numpy()
                show_y_pred(y_pred, gt, epo=epoch, flag='validation')
        if epoch == epochs - 1:
            model.eval()
            for ii, (input, org) in enumerate(train_dataloader):
                model.eval()
                pred = model(input)
                y = pred.detach().numpy()
                show_y_pred(y, org, epo=epoch, flag='train')
                np.save(r'./step1_y.npy', y)
    plot_loss(loss_list)
    torch.save(model.state_dict(), "./modle1_{}_{}.pth".format(i, j))

def run_test(scale, mlp_pth, X, Y, hiden_dim, output_dim, epochs):
    print("data size: {}".format(X.shape[0]))
    input_dim = X.shape[1]
    model = MLP(input_dim, hiden_dim, output_dim).to(device)
    print(model)
    model.load_state_dict(torch.load(mlp_pth))
    all_data = DataLoader((X, Y), batch_size=X.shape[0], batch_first=False, device=device)

    for index, p in enumerate(model.parameters()):
        p.requires_grad = False

    loss_list = []
    for epoch in range(epochs):
        train_loss = 0
        for ii, (data, label) in enumerate(all_data):
            if epoch == 0:
                x_data = scale.inverse_transform(data.detach().numpy())
                np.save(r'./start_x.npy', x_data)
                np.save(r'./start_lab.npy', label)
            data = Variable(data, requires_grad=True)
            optimizer = optimizers.Adam({data},
                                        lr=1,
                                        betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
            # optimizer = torch.optim.SGD(params=[data], lr=100, momentum=0.8, dampening=0.5, weight_decay=0.01,
            #                             nesterov=False)
            optimizer.zero_grad()
            score = model(data)
            loss = compute_loss(score, label)
            y_pred = score.detach().numpy()
            loss.backward()
            # print(data.grad[0])
            optimizer.step()
            train_loss += loss.item()
            if (epoch) % 200 == 0:
                show_y_pred(y_pred, label, epo=epoch, flag='validation')
            if epoch == epochs - 1:
                model.eval()
                preds = model(data)
                x_data = scale.inverse_transform(data.detach().numpy())
                np.save(r'./modified_x.npy', x_data)
                np.save(r'./modified_lab.npy', label)
        train_loss /= len(all_data)
        loss_list.append(train_loss)

    plot_loss(loss_list)
    print(loss_list.index(min(loss_list)))
    print(max(loss_list), min(loss_list))


if __name__ == "__main__":

    root_dir = r'D:\work\project\卡尔蔡司AR镀膜\第三批'
    sub_dir = r'0705'
    number33_thick10sensor8step_lab_js = os.path.join(root_dir, sub_dir, 'thick14hc3sensor64_lab.json')

    hiden_dim = 200
    epochs_train = 1000
    epochs_test = 100
    X, Y = generate_data(number33_thick10sensor8step_lab_js)
    X = np.array(X)
    Y = np.array(Y)
    scale = StandardScaler(with_mean=True, with_std=True)
    X = scale.fit_transform(X)

    i, j = 0, 0
    flag = 0

    if flag == 1:
        print("data_size: {}, {}".format(X.shape, Y.shape))
        output_dim = Y.shape[1]

        run_train(X, Y, hiden_dim, output_dim, epochs_train, i, j)

    elif flag == 0:
        output_dim = Y.shape[1]

        # 这里可以细粒度到每一个样本的修改值, 后续可优化
        mod = [0.001, -0.0016, -0.0037, -0.0012, 0.0013]
        mod = [1+a for a in mod]

        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i][j] *= mod[j]

        mlp_list = ['./modle1_0_0.pth']
        for mlp_pth in mlp_list:
            run_test(scale, mlp_pth, X, Y, hiden_dim, output_dim, epochs_test)




