# coding=utf-8

'''
根据loss拆分 data

'''

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optimizers
import torch.nn as nn
from torch.autograd import Variable
from utils.my_mse_loss import split_data_loss
from data_load import DataLoader
from mlp_torch import MLP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_loss(t, y, n):
    loss, inds = split_data_loss()(t, y, n)
    return loss, inds


def compute_loss1(t, y):
    return nn.MSELoss()(y, t)


def show_y_pred(y, gt_y):
    sample_num, dims = y.shape
    plt.title('model1')
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [i for i in range(dims)]
    for i in range(sample_num):
        single_y = y[i, :]
        single_gt_y = gt_y[i, :]
        plt.plot(x, single_gt_y, color='pink')
        plt.plot(x, single_y, color='cornflowerblue')
    plt.legend()
    # plt.savefig("lab_curve.png")
    plt.show()


def plot_loss(loss, lr):
    x = [i for i in range(len(loss))]
    plt.title(lr)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, loss)
    plt.show()


def generate_data(number33_thick10sensor8step_lab_js, flag, cyc_ind):
    if not os.path.exists(number33_thick10sensor8step_lab_js):
        print('lose number_thick10_sensor_std.json!')
    else:
        with open(number33_thick10sensor8step_lab_js, encoding="utf-8") as reader:
            f_lab = json.load(reader)
    X, Y = [], []
    std14_number = dict()
    if flag == 0:
        for number33, thicksensor_lab in f_lab.items():
            if number33 != "33321021303":
                all = thicksensor_lab[0].split(',')[:-1]
                x = all[:10]
                y = all[10:]
                y.pop(0)
                y.pop(7)
                assert len(x) == 10
                assert len(y) == 14
                X.append([float(a) for a in x])
                Y.append([float(a) for a in y])
                std14_number[''.join(str(round(float(r),4))+ ',' for r in y)] = number33
    return X, Y, std14_number


def run_train(X, Y, hiden_dim, output_dim, epochs, mean, std, lr, flag, cyc_ind):
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    print(X.shape, '===')
    batch_size = X.shape[0]
    for i in range(batch_size):
        X[i] = [(X[i][j] - mean[j]) / std[j] for j in range(10)]
    input_dim = X.shape[-1]

    train_dataloader = DataLoader((X, Y), batch_size=batch_size, batch_first=False, device=device)
    model = MLP(input_dim, hiden_dim, output_dim).to(device)
    if flag == 1:
        model.load_state_dict(torch.load(r'./model1.pth'))
    optimizer = optimizers.Adam(model.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-4)

    for epoch in range(epochs):
        for ii, (data, label) in enumerate(train_dataloader):
            input = Variable(data, requires_grad=False)
            target = Variable(label)
            optimizer.zero_grad()
            score = model(input)
            loss, inds = compute_loss(score, target, n=87)
            loss.backward()
            optimizer.step()
        if epoch == epochs - 1:
            part_x, part_y = [], []
            for ind in inds:
                part_x.append(data[ind].detach().numpy())
                part_y.append(label[ind].detach().numpy())
            part_y = np.array(part_y)
            part_x = np.array(part_x)
            np.save('X_{}.npy'.format(cyc_ind), part_x)
            np.save('Y_{}.npy'.format(cyc_ind), part_y)
            remain_ind = [i for i in range(label.shape[0]) if i not in inds]
            remain_x = []
            remain_y = []
            for ind in remain_ind:
                remain_y.append(label[ind].detach().numpy())
                remain_x.append(data[ind].detach().numpy())
            np.save('X.npy', remain_x)
            np.save('Y.npy', remain_y)
            if epoch == epochs - 1:
                print(loss)

    torch.save(model.state_dict(), "./model1_68.pth")


def part_train(hiden_dim, output_dim, epochs, mean, std, cyc_ind):
        X = np.load('X_{}.npy'.format(cyc_ind))
        Y = np.load('Y_{}.npy'.format(cyc_ind))
        print(X.shape, 'loss part')
        batch_size = X.shape[0]
        for i in range(batch_size):
            X[i] = [(X[i][j] - mean[j]) / std[j] for j in range(10)]
        input_dim = X.shape[-1]

        train_dataloader = DataLoader((X, Y), batch_size=batch_size, batch_first=False, device=device)
        model = MLP(input_dim, hiden_dim, output_dim).to(device)
        # model.load_state_dict(torch.load('./model1.pth'))
        optimizer = optimizers.Adam(model.parameters(),
                                    lr=0.001,
                                    betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)

        for epoch in range(epochs):
            for ii, (data, label) in enumerate(train_dataloader):
                input = Variable(data, requires_grad=False)
                target = Variable(label)
                optimizer.zero_grad()
                score = model(input)
                loss = compute_loss1(score, target)
                loss.backward()
                optimizer.step()
                if epoch % 200 == 0:
                    print(loss.item())
                if epoch == epochs-1:
                    model.eval()
                    pred = model(data).detach().numpy()
                    show_y_pred(pred, label)
        torch.save(model.state_dict(), "./model1_{}.pth".format(cyc_ind+19))


def Model1(flag=None, cyc_ind=None):
    root_dir = r'D:\work\project\卡尔蔡司AR镀膜\第三批'
    sub_dir = r'0705'
    number33_thick10sensor8step_lab_js = os.path.join(root_dir, sub_dir, 'thick14hc3sensor64_lab.json')

    hiden_dim = 200
    epochs_train = 2000

    mean_ = [float(i) for i in open(r'./thickness_mean.txt', 'r').readlines()[0].split(',')[:-1]]
    std_ = [float(i) for i in open(r'./thickness_std.txt', 'r').readlines()[0].split(',')[:-1]]
    assert len(mean_) == 10

    if flag == 0:
        # lr = open(r'./model1_flag1_best_lr.txt', 'r').readlines()[0].split(',')[0]
        X, Y, std_num = generate_data(number33_thick10sensor8step_lab_js, flag, cyc_ind)
        # X = np.array(X)
        # Y = np.array(Y)
        # output_dim = Y.shape[1]
        # run_train(X, Y, hiden_dim, output_dim, epochs_train, mean_, std_, float(lr), flag, 2)
        lqy = np.load('./Y_2.npy')
        f = open('./lq_numbers.txt', 'w')
        std_nums = list(std_num.keys())
        lq_numbers = []
        for y in lqy:
            tmp = ''.join(str(round(o, 4))+',' for o in y)
            if tmp in std_nums:
                # print(std_nums.index(tmp))
                lq_numbers.append(std_num[tmp])
                f.write(std_num[tmp]+',')

    elif flag == 3:
        part_train(hiden_dim, 14, 2000, mean_, std_, cyc_ind)
        print("got model1_{}.pth".format(cyc_ind+19))


def split():
    c = 0
    inds = [i for i in range(12)]
    for ind in inds:
        x = np.load("X_{}.npy".format(ind))
        c += x.shape[0]
    print(c)


if __name__ == "__main__":
    loss = Model1(flag=0)
    # split()

    # inds = [i for i in range(3)]
    # for ind in inds:
    #     Model1(flag=3, cyc_ind=ind)








