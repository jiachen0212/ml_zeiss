# coding=utf-8
import torch
import os
import json
from read_csvs import get8step_sensor_feature
import xlrd
from sklearn.metrics import mean_squared_error
from data_load import DataLoader
from mlp_torch import MLP
import torch.nn as nn
import numpy as np
import torch.optim as optimizers
from torch.autograd import Variable
import matplotlib.pyplot as plt
from util import calculate_Lab
from util import weighted_mse


def compute_loss(t, y):
    return nn.MSELoss()(y, t)


def compare_res(best, number):
    y1 = np.load(r'./step1_y.npy')[0].tolist()
    y2 = np.load(r'./step2_y.npy')[0].tolist()
    plt.title(number)
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(81)]
    print("原曲线lab: ", calculate_Lab(y1))
    print("调整后曲线lab: ", calculate_Lab(y2))
    plt.plot(x, y1, color='cornflowerblue', label='real curve')
    plt.plot(x, y2, color='lightpink', label='modified curve')
    plt.plot(x, best, color='red', label='best')
    plt.legend()
    plt.show()


def get_thickness(number, ind, mean_sensor, std_sensor, mean_thick, std_thick, single_data, flag=None):
    best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
            1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
            0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
            0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
            3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]

    # step1. std -> lab
    model2_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\2'
    model1_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\1'
    thick10std14 = single_data[0].split(',')[:-1]
    lab = single_data[1]
    std14 = thick10std14[10:]
    std14.pop(0)
    std14.pop(7)
    std14 = [float(a) for a in std14]
    std14 = [(std14[i] - mean_sensor[i]) / std_sensor[i] for i in range(len(std_sensor))]
    X = np.array([std14])
    Y = np.array([lab])
    # 保存原始真实lab曲线
    np.save(r'./step1_y.npy', Y)
    mse_0 = mean_squared_error(lab, best)
    print("初始与best的mse: {}".format(mse_0))

    # 导入model2
    model2 = os.path.join(model2_dir, 'model2_fine_tune{}.pth'.format(ind))
    model = MLP(14, 80, 81).to('cpu')
    model.load_state_dict(torch.load(model2))

    # 开始微调
    for index, p in enumerate(model.parameters()):
        p.requires_grad = False

    best_epoch_model1_epoch_lr = './{}.txt'.format(number)
    if flag == 0:
        if os.path.exists(best_epoch_model1_epoch_lr):
            os.remove(best_epoch_model1_epoch_lr)
        tmp_ = open(best_epoch_model1_epoch_lr, 'a')

    if flag == 0:
        losses = []
        epochs = 3500
        for epoch in range(epochs):
            if epoch > 0:
                X = np.load('./modify.npy')
            all_data = DataLoader((X, Y), batch_size=1, batch_first=False, device='cpu')
            for ii, (data, _) in enumerate(all_data):
                data = Variable(data, requires_grad=True)
                target = [best]
                target = np.array(target)
                target = np.reshape(target, (data.shape[0], -1))
                target = Variable(torch.from_numpy(target).float())
                optimizer = optimizers.Adam({data},
                                            lr=0.001,
                                            betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
                optimizer.zero_grad()
                score = model(data)
                loss = compute_loss(score, target)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                if epoch == epochs-2:
                    score = model(data)
                    pred = score.detach().numpy()
                    np.save(r'./step2_y.npy', pred)
                    # print("和best曲线的loss: {}".format(compute_loss(score, target).detach().numpy()))
                np.save(r'./modify.npy', data.detach().numpy())
                if epoch == epochs-1:
                    tmp = data.detach().numpy()[0]
                    modified_std = [tmp[i] * std_sensor[i] + mean_sensor[i] for i in range(len(mean_sensor))]
                    np.save(r'./modified_std.npy', modified_std)
        min_loss = min(losses)
        index = losses.index(min_loss)
        print('固定model2参数, 微调std值优化lab_curve, mse为: ', index, min_loss)
        tmp_.write(str(index+2)+',')
    else:
        tmp_ = open(best_epoch_model1_epoch_lr, 'r')
        epoch_ycy = int(tmp_.readlines()[0].split(',')[0])
        print(epoch_ycy)
        for epoch in range(epoch_ycy):
            if epoch > 0:
                X = np.load('./modify.npy')
            all_data = DataLoader((X, Y), batch_size=1, batch_first=False, device='cpu')
            for ii, (data, _) in enumerate(all_data):
                data = Variable(data, requires_grad=True)
                target = [best]
                target = np.array(target)
                target = np.reshape(target, (data.shape[0], -1))
                target = Variable(torch.from_numpy(target).float())
                optimizer = optimizers.Adam({data},
                                            lr=0.001,
                                            betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
                optimizer.zero_grad()
                score = model(data)
                loss = compute_loss(score, target)
                loss.backward()
                optimizer.step()
                if epoch == epoch_ycy-2:
                    score = model(data)
                    pred = score.detach().numpy()
                    np.save(r'./step2_y.npy', pred)
                np.save(r'./modify.npy', data.detach().numpy())
                if epoch == epoch_ycy-1:
                    print("lab loss: {}".format(loss.item()))
                    tmp = data.detach().numpy()[0]
                    modified_std = [tmp[i] * std_sensor[i] + mean_sensor[i] for i in range(len(mean_sensor))]
                    np.save(r'./modified_std.npy', modified_std)

        compare_res(best, number)

    # step2. std -> thickness
    thick10std14 = single_data[0].split(',')[:-1]
    thick10 = [float(a) for a in thick10std14[:10]]
    base_thickness = thick10
    thick10 = [(thick10[j]-mean_thick[j])/std_thick[j] for j in range(len(std_thick))]
    std = [float(a) for a in thick10std14[10:]]
    std.pop(0)
    std.pop(7)
    assert len(std) == 14
    X, Y = np.array([thick10]), np.array([std])
    model = MLP(10, 200, 14).to('cpu')
    model1 = os.path.join(model1_dir, 'model1_fine_tune{}.pth'.format(ind))
    model.load_state_dict(torch.load(model1))
    # 开始微调
    for index, p in enumerate(model.parameters()):
        p.requires_grad = False
    # 复制model2中的std期望值到model1的输出端
    Y = np.reshape(np.load(r'./modified_std.npy'), (1, -1))

    if flag == 0:
        # lrs = [0.0005, 0.0008, 0.001, 0.002, 0.003]
        lrs = [0.0005]
        epochs = 6000
        losses = []
        min_loss_index = []
        for lr in lrs:
            ll = []
            for epoch in range(epochs):
                if epoch > 0:
                    X = np.load('./modify1.npy')
                all_data = DataLoader((X, Y), batch_size=1, batch_first=False, device='cpu')
                for ii, (data, gt) in enumerate(all_data):
                    data = Variable(data, requires_grad=True)
                    target = Variable(gt)
                    optimizer = optimizers.Adam({data},
                                                lr=lr,
                                                betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
                    optimizer.zero_grad()
                    score = model(data)
                    loss = compute_loss(score, target)
                    ll.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    np.save(r'./modify1.npy', data.detach().numpy())
                    if epoch == epochs-1:
                        model.eval()
                        tmp = data.detach().numpy()[0]
                        modified_thick = [tmp[i] * std_thick[i] + mean_thick[i] for i in range(len(mean_thick))]
                        np.save(r'./modified_thickness.npy', modified_thick)
            cur_lr_min_loss = min(ll)
            cur_lr_index = ll.index(cur_lr_min_loss)
            losses.append(cur_lr_min_loss)
            min_loss_index.append(cur_lr_index)
        last_min_loss = min(losses)
        best_index = losses.index(last_min_loss)
        best_lr = lrs[best_index]
        best_epoch = min_loss_index[best_index]
        print("最佳lr：{}, 最佳epoch: {}".format(best_lr, best_epoch))
        tmp_.write(str(best_lr)+','+str(best_epoch+2))
    else:
        tmp_ = open(best_epoch_model1_epoch_lr, 'r')
        cy = tmp_.readlines()[0].split(',')
        epoch_ycy = int(cy[2])
        lr_ycy = float(cy[1])
        print(lr_ycy, epoch_ycy)
        for epoch in range(epoch_ycy):
            if epoch > 0:
                X = np.load('./modify1.npy')
            all_data = DataLoader((X, Y), batch_size=1, batch_first=False, device='cpu')
            for ii, (data, gt) in enumerate(all_data):
                data = Variable(data, requires_grad=True)
                target = Variable(gt)
                optimizer = optimizers.Adam({data},
                                            lr=lr_ycy,
                                            betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
                optimizer.zero_grad()
                score = model(data)
                loss = compute_loss(score, target)
                loss.backward()
                optimizer.step()
                np.save(r'./modify1.npy', data.detach().numpy())
                if epoch == epoch_ycy - 1:
                    model.eval()
                    print("std loss: {}".format(loss.item()))
                    # print(model(data).detach().numpy()[0], 'last')
                    tmp = data.detach().numpy()[0]
                    modified_thick = [tmp[i] * std_thick[i] + mean_thick[i] for i in range(len(mean_thick))]
                    np.save(r'./modified_thickness.npy', modified_thick)

        # print("原始thickness: {}".format(base_thickness))
        # round_model_thickness = [round(a, 2) for a in modified_thick]
        # print("model thickness: {}".format(round_model_thickness))
        # print("修改diff: {}".formatS([round_model_thickness[i]-base_thickness[i] for i in range(10)]))
    return modified_thick