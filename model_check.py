# coding=utf-8
# 1. 生成json测试数据
import torch
import os
import json
from read_csvs import get8step_sensor_feature
import xlrd
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


def plot_loss(loss, name=None):
    x = [i for i in range(len(loss))]
    plt.title(name)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, loss)
    plt.show()


def prepare_data(best):
    '''
    r'./sensor_csv.txt'这个文件里要准备好本次测试数据的: .csv name

    '''
    base_data_dir = r'D:\work\project\卡尔蔡司AR镀膜\0731data\7.31\7.31\机台文件'
    number_json = r'D:\work\project\卡尔蔡司AR镀膜\0731data\7.31\7.31\number33_lab.json'
    csv_dict_js = r'D:\work\project\卡尔蔡司AR镀膜\0731data\7.31\7.31\csv_dict.json'
    number_evts = r'D:\work\project\卡尔蔡司AR镀膜\0731data\7.31\7.31\number33_evtpair.json'
    final_js = r'D:\work\project\卡尔蔡司AR镀膜\0731data\7.31\7.31\finally.json'

    f = r'D:\work\project\卡尔蔡司AR镀膜\0731data\7.31\7.31\7.31.xlsx'
    wb = xlrd.open_workbook(f)
    data = wb.sheet_by_name('双面数据')
    title = data.row_values(1)
    start, end = title.index(380.0), title.index(780.0)
    labs = []
    lines = [2, 9, 16, 23, 30]
    x = [380+5*i for i in range(81)]
    colors = ['cornflowerblue', 'lightpink', 'hotpink', 'green', 'yellow']
    for ind, line in enumerate(lines):
        lab = data.row_values(line)[start:end+1]
        plt.plot(x, lab, color=colors[ind], label='{} curve'.format(ind))
        plt.legend()
        print(1 - weighted_mse(lab))
        print(calculate_Lab(lab))
        labs.append(lab)
    plt.plot(x, best, color='red', label='best')
    plt.show()


    number_thickness = dict()
    # 背面+正面
    thickness_list = [[24.95,12.4,37.07,107.65,97.1,24.46,11.7,36.34,102.2,93.6],
    [24.9,12.38,36.99,106.8,96.6,24.46,11.7,36.34,101.8,93.6],
    [24.9,12.38,36.99,106.4,96.1,24.46,11.7,36.34,101.8,93.7],
    [24.9,12.38,36.99,105.8,95.8,24.46,11.7,36.34,101.5,93.95],
    [24.82,12.34,36.88,105,95.1,24.46,11.7,36.34,101.2,93.95]]
    thickness_list = [[str(i) for i in a] for a in thickness_list]
    numbers = ['33388888888', '33388888866', '33388888899', '33388888800', '33388888811']
    for ind, number in enumerate(numbers):
        number_thickness[number] = [thickness_list[ind], labs[ind]]
    data = json.dumps(number_thickness)
    with open(number_json, 'w') as js_file:
        js_file.write(data)

    # generate data
    get8step_sensor_feature(7, number_evts, base_data_dir, csv_dict_js, number_json, final_js, ['ACT_O1_QCMS_THICKNESS_CH1'])


def compare_res(best):
    y1 = np.load(r'./step1_y.npy')[0].tolist()
    y2 = np.load(r'./step2_y.npy')[0].tolist()
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(81)]
    print("原曲线lab: ", calculate_Lab(y1))
    print("调整后曲线lab: ", calculate_Lab(y2))
    plt.plot(x, y1, color='black', label='real curve')
    plt.plot(x, y2, color='cornflowerblue', label='modified curve')
    plt.plot(x, best, color='red', label='best')
    plt.legend()
    # plt.savefig(r'./0802.png')
    plt.show()



def test_model1_and_model2(test_data_js, mean_thick, std_thick, mean_sensor, std_sensor, flag=None):
    '''
    model1,正向验证. 膜厚->lab曲线
    flag == 1, 2
    '''
    best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
            1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
            0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
            0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
            3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]

    f = json.load(open(test_data_js, 'r'))
    X1, Y1 = [], []
    X2, Y2 = [], []
    number_names = []
    for k, v in f.items():
        thick10std16 = v[0].split(',')[:-1]
        assert len(thick10std16) == 26
        lab = v[1]
        thick = [float(a) for a in thick10std16[:10]]
        std = [float(a) for a in thick10std16[10:]]
        std.pop(0)
        std.pop(7)
        # 未归一化的std值, 作为model1的输出
        Y1.append(std)
        std = [(std[i]-mean_sensor[i])/std_sensor[i] for i in range(len(std_sensor))]
        thick = [(thick[i]-mean_thick[i])/std_thick[i] for i in range(len(std_thick))]
        # 归一化后的thickness作为model1的输入
        X1.append(thick)
        # 归一化后的std值作为model2的输入
        X2.append(std)
        number_names.append(k)
        # lab曲线为model2的输出
        Y2.append(lab)

    if flag == 1:
        model1_pth_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model1'
        pths = os.listdir(model1_pth_dir)
        pths.pop(7)

        model = MLP(10, 200, 14).to('cpu')
        model1_output_std = []
        # 针对每一条测试样本, 获取model1的最佳选择
        for ind, x1 in enumerate(X1):
            losses = []
            all_data = DataLoader((np.array([x1]), np.array([Y1[ind]])), batch_size=1, batch_first=False, device='cpu')
            for m1 in pths:
                model1 = os.path.join(model1_pth_dir, m1)
                model.load_state_dict(torch.load(model1))
                for jj, (a, lab) in enumerate(all_data):
                    model.eval()
                    pred = model(a)
                    loss = compute_loss(pred, lab)
                    losses.append(loss.item())
            index = losses.index(min(losses))
            print("样本 {} 的最佳model1 index: {}, loss值: {}".format(ind, pths[index], losses[index]))

            pth = pths[index]
            model1 = os.path.join(model1_pth_dir, pth)
            model.load_state_dict(torch.load(model1))

            # fine-tune最佳model1.pth
            optimizer = optimizers.Adam(model.parameters(),
                                        lr=0.001,
                                        betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
            loss_list = []
            for epoch in range(100):
                for ii, (data, label) in enumerate(all_data):
                    input = Variable(data, requires_grad=False)
                    target = Variable(label)
                    optimizer.zero_grad()
                    score = model(input)
                    loss = compute_loss(score, target)
                    loss.backward()
                    optimizer.step()
                    loss_list.append(loss.item())
            plot_loss(loss_list)
            if number_names[ind] == "33321042704":
                torch.save(model.state_dict(), "./model1_fine_tune_R4.pth")

            # fine好之后,对样本eval()
            for jj, (a, gt) in enumerate(all_data):
                model.eval()
                pred = model(a)
                print("fine-tune后, 样本{} 的 {} 的 loss: {}".format(ind, pths[index], compute_loss(pred, gt).item()))
                pred_ = pred.detach().numpy()[0]
                model1_output_std.append(pred_)
        model1_output_std = np.array(model1_output_std)
        np.save(r'./model1_std_value.npy', model1_output_std)


    if flag == 2:
        # std -> lab
        # 首先根据真实std值,得到每条测试数据的最佳model2选择
        model2_pth_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model2'
        pths = os.listdir(model2_pth_dir)
        pths.pop(7)
        model = MLP(14, 80, 81).to('cpu')

        for ind, x2 in enumerate(X2):
            losses = []
            all_data = DataLoader((np.array([x2]), np.array([Y2[ind]])), batch_size=1, batch_first=False, device='cpu')
            for m2 in pths:
                model2 = os.path.join(model2_pth_dir, m2)
                model.load_state_dict(torch.load(model2))
                for jj, (a, lab) in enumerate(all_data):
                    model.eval()
                    pred = model(a)
                    loss = compute_loss(pred, lab)
                    losses.append(loss.item())
            index = losses.index(min(losses))
            print("样本 {} 的最佳model2 index: {}, loss值: {}".format(ind, pths[index], losses[index]))
            aa = [380 + 5 * i for i in range(81)]

            # 最佳model2.pth fine-tune
            pth = pths[index]
            model2 = os.path.join(model2_pth_dir, pth)
            model.load_state_dict(torch.load(model2))
            optimizer = optimizers.Adam(model.parameters(),
                                        lr=0.001,
                                        betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
            loss_list = []
            for epoch in range(100):
                for ii, (data, label) in enumerate(all_data):
                    input = Variable(data, requires_grad=False)
                    target = Variable(label)
                    optimizer.zero_grad()
                    score = model(input)
                    loss = compute_loss(score, target)
                    loss.backward()
                    optimizer.step()
                    loss_list.append(loss.item())
            plot_loss(loss_list)
            if number_names[ind] == "33321042704":
                torch.save(model.state_dict(), "./model2_fine_tune_R4.pth")
            for jj, (a, gt) in enumerate(all_data):
                model.eval()
                pred = model(a)
                print("fine-tune后, 样本{} 的 {}的 loss: {}".format(ind, pths[index], compute_loss(pred, gt).item()))

            # fine-tune结束, 开始读取model1中的std输出值,得到联合model1,2的正向拟合lab结果.
            model1_output_std = np.load(r'./model1_std_value.npy')
            x2 = model1_output_std[ind]
            x2 = [(x2[j] - mean_sensor[j]) / std_sensor[j] for j in range(len(std_sensor))]
            single_data = DataLoader((np.array([x2]), np.array([Y2[ind]])), batch_size=1, batch_first=False, device='cpu')
            for jj, (a, lab) in enumerate(single_data):
                model.eval()
                pred = model(a).detach().numpy()[0]
                plt.plot(aa, pred, color='red', label='model1+2 lab')
                plt.plot(aa, lab.detach().numpy()[0], color='black', label='real lab')
                plt.plot(aa, best, color='cornflowerblue', label='best')
                plt.legend()
                print("真实曲线lab值: ", calculate_Lab(lab.detach().numpy()[0]))
                print("模型拟合曲线lab值: ", calculate_Lab(pred))
                plt.title(number_names[ind])
                plt.show()


# 反向验证
def modify_std(final_js, mean_sensor, std_sensor, best):
    # std -> lab
    f = json.load(open(final_js, 'r'))
    data = f['33321042704']
    thick10std14 = data[0].split(',')[:-1]
    # 真实lab曲线
    lab = data[1]
    std14 = thick10std14[10:]
    std14.pop(0)
    std14.pop(7)
    std14 = [float(a) for a in std14]
    std14 = [(std14[i] - mean_sensor[i]) / std_sensor[i] for i in range(len(std_sensor))]
    X = np.array([std14])
    Y = np.array([lab])
    # 保存原始真实lab曲线
    np.save(r'./step1_y.npy', Y)
    print("真实曲线mse: {}".format(weighted_mse(lab)))

    # 寻找最佳的model2.pth
    all_data = DataLoader((X, Y), batch_size=1, batch_first=False, device='cpu')
    model = MLP(14, 80, 81).to('cpu')
    # model2_pth_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model2'
    # pths = os.listdir(model2_pth_dir)
    # losses = []
    # for m2 in pths:
    #     model2 = os.path.join(model2_pth_dir, m2)
    #     model.load_state_dict(torch.load(model2))
    #     for jj, (a, lab) in enumerate(all_data):
    #         model.eval()
    #         pred = model(a)
    #         loss = compute_loss(pred, lab)
    #         losses.append(loss.item())
    # index = losses.index(min(losses))
    # print("不ok样本的最佳model2 index: {}, loss值: {}".format(pths[index], losses[index]))
    #
    # # 最佳model2.pth fine-tune
    # pth = pths[index]
    # model2 = os.path.join(model2_pth_dir, pth)
    model2 = r"./model2_fine_tune_R4.pth"
    model.load_state_dict(torch.load(model2))
    # 实验发现, model2不做微调比较好~
    # optimizer = optimizers.Adam(model.parameters(),
    #                             lr=0.001,
    #                             betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
    # loss_list = []
    # for epoch in range(100):
    #     for ii, (data, label) in enumerate(all_data):
    #         input = Variable(data, requires_grad=False)
    #         target = Variable(label)
    #         optimizer.zero_grad()
    #         score = model(input)
    #         loss = compute_loss(score, target)
    #         loss.backward()
    #         optimizer.step()
    #         loss_list.append(loss.item())
    #         if epoch == 99:
    #             print("fine-tune 后 loss: {}".format(loss.item()))
    # plot_loss(loss_list)
    # # fine-tune model2结束, 开始微调std值
    for index, p in enumerate(model.parameters()):
        p.requires_grad = False
    losses = []
    epochs = 3000

    # R3：
    # l = [5.17, 2.87, 1.04, 0.95, 0.42, 0.26, 0.26, 0.56, 1.06, 1.28, 1.32, 1.4, 1.65, 2.03, 2.14, 2.0, 1.68, 1.45, 1.5, 1.67, 1.8, 1.69, 1.4, 1.07, 0.85, 0.93, 1.1, 1.18, 1.1, 0.86, 0.61, 0.39, 0.34, 0.42, 0.52, 0.53, 0.49, 0.57, 0.36, 0.25, 0.13, 0.1, 0.09, 0.14, 0.19, 0.22, 0.2, 0.15, 0.14, 0.1, 0.03, 0.07, 0.22, 0.33, 0.6, 0.67, 0.7, 0.75, 0.68, 0.63, 0.72, 0.79, 0.98, 1.23, 1.55, 1.84, 2.23, 2.62, 2.96, 3.22, 3.4, 3.57, 3.68, 3.79, 4.08, 4.08, 4.24, 4.39, 4.77, 5.17, 5.6]
    # R1
    # l = [4.97, 3.12, 1.81, 0.91, 0.55, 0.48, 0.48, 0.55, 0.87, 1.18, 1.39, 1.61, 1.7, 1.8, 1.89, 2.01, 2.02, 1.9, 1.73, 1.61, 1.52, 1.49, 1.48, 1.41, 1.25, 1.02, 0.9, 0.83, 0.88, 0.88, 0.84, 0.75, 0.62, 0.47, 0.35, 0.3, 0.3, 0.44, 0.4, 0.39, 0.32, 0.26, 0.17, 0.09, 0.04, 0.08, 0.08, 0.14, 0.16, 0.16, 0.18, 0.14, 0.13, 0.17, 0.2, 0.33, 0.64, 0.81, 0.9, 0.98, 1.09, 1.15, 1.25, 1.37, 1.49, 1.58, 1.79, 2.12, 2.41, 2.82, 3.19, 3.55, 3.92, 4.25, 4.59, 4.69, 4.89, 5.09, 5.35, 5.49, 5.71]
    # R2
    # l = [5.83, 3.76, 1.95, 1.07, 0.78, 0.33, 0.25, 0.44, 0.93, 1.04, 0.85, 1.19, 1.74, 2.1, 2.06, 1.77, 1.56, 1.61, 1.82, 1.9, 1.8, 1.42, 1.09, 0.97, 1.05, 1.22, 1.29, 1.12, 0.82, 0.53, 0.38, 0.43, 0.53, 0.63, 0.67, 0.6, 0.46, 0.29, 0.15, 0.12, 0.12, 0.19, 0.31, 0.23, 0.15, 0.13, 0.09, 0.08, 0.08, 0.19, 0.21, 0.34, 0.36, 0.43, 0.24, 0.13, 0.17, 0.21, 0.33, 0.6, 0.94, 1.22, 1.54, 1.88, 2.17, 2.22, 2.32, 2.37, 2.37, 2.39, 2.4, 2.45, 2.62, 2.96, 3.4, 3.98, 4.53, 4.93, 5.57, 6.17, 6.71]
    for epoch in range(epochs):
        if epoch > 0:
            X = np.load('./modify.npy')
        all_data = DataLoader((X, Y), batch_size=1, batch_first=False, device='cpu')
        for ii, (data, _) in enumerate(all_data):
            data = Variable(data, requires_grad=True)
            target = [best]
            # target = [l]
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
    plot_loss(losses, 'wei-tiao model2')
    print(losses.index(min(losses)), min(losses))
    compare_res(best)
    # compare_res(l)


def modify_thickness(final_js, mean_thick, std_thick):
    # thickness -> std
    f = json.load(open(final_js, 'r'))
    data = f['33321042704']
    thick10std14 = data[0].split(',')[:-1]
    thick10 = [float(a) for a in thick10std14[:10]]
    thick10 = [(thick10[j]-mean_thick[j])/std_thick[j] for j in range(len(std_thick))]
    std = [float(a) for a in thick10std14[10:]]
    std.pop(0)
    std.pop(7)
    assert len(std) == 14
    X, Y = np.array([thick10]), np.array([std])
    all_data = DataLoader((X, Y), batch_size=1, batch_first=False, device='cpu')
    model = MLP(10, 200, 14).to('cpu')

    # # 寻找最佳model1
    # model1_pth_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model1'
    # pths = os.listdir(model1_pth_dir)
    # losses = []
    # for m1 in pths:
    #     model1 = os.path.join(model1_pth_dir, m1)
    #     model.load_state_dict(torch.load(model1))
    #     for jj, (a, lab) in enumerate(all_data):
    #         model.eval()
    #         pred = model(a)
    #         loss = compute_loss(pred, lab)
    #         losses.append(loss.item())
    # index = losses.index(min(losses))
    # print("不ok样本的最佳model1 index: {}, loss值: {}".format(pths[index], losses[index]))
    #
    # # 先fine-tune
    # pth = pths[index]
    # model1 = os.path.join(model1_pth_dir, pth)
    model1 = r"./model1_fine_tune_R4.pth"
    model.load_state_dict(torch.load(model1))

    # 实验来看, model1也不要微调...
    # optimizer = optimizers.Adam(model.parameters(),
    #                             lr=0.001,
    #                             betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
    # loss_list = []
    # for epoch in range(100):
    #     for ii, (data, label) in enumerate(all_data):
    #         input = Variable(data, requires_grad=False)
    #         target = Variable(label)
    #         optimizer.zero_grad()
    #         score = model(input)
    #         loss = compute_loss(score, target)
    #         loss.backward()
    #         optimizer.step()
    #         loss_list.append(loss.item())
    #     if epoch == 99:
    #         print("fine-tune 后的loss: {}".format(loss.item()))
    # plot_loss(loss_list)

    # 基于fine-tune的model1, 开始微调thickness
    for index, p in enumerate(model.parameters()):
        p.requires_grad = False
    # 复制model2中的std期望值到model1的输出端
    Y = np.reshape(np.load(r'./modified_std.npy'), (1, -1))
    # print("期望std输出值: {}".format(Y))
    epochs = 1639
    losses = []
    for epoch in range(epochs):
        if epoch > 0:
            X = np.load('./modify1.npy')
        all_data = DataLoader((X, Y), batch_size=1, batch_first=False, device='cpu')
        for ii, (data, gt) in enumerate(all_data):
            data = Variable(data, requires_grad=True)
            target = Variable(gt)
            optimizer = optimizers.Adam({data},
                                        lr=0.0005,
                                        betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
            optimizer.zero_grad()
            score = model(data)
            loss = compute_loss(score, target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            np.save(r'./modify1.npy', data.detach().numpy())
            if epoch == epochs-1:
                model.eval()
                print(model(data).detach().numpy()[0], 'last')
                tmp = data.detach().numpy()[0]
                modified_thick = [tmp[i] * std_thick[i] + mean_thick[i] for i in range(len(mean_thick))]
                np.save(r'./modified_thickness.npy', modified_thick)
    plot_loss(losses, 'wei-tiao model1')
    print(losses.index(min(losses)), min(losses))

    return modified_thick



def Model_Check(test_data_js, targ=None):
    '''
    反向验证也是算法部署上线的使用过程.

    '''
    best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
            1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
            0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
            0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
            3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]

    mean_thick = [float(i) for i in open(r'./thickness_mean.txt', 'r').readlines()[0].split(',')[:-1]]
    std_thick = [float(i) for i in open(r'./thickness_std.txt', 'r').readlines()[0].split(',')[:-1]]

    mean_sensor = [float(i) for i in open(r'./sensor_mean.txt', 'r').readlines()[0].split(',')[:-1]]
    std_sensor = [float(i) for i in open(r'./sensor_std.txt', 'r').readlines()[0].split(',')[:-1]]
    assert len(mean_sensor) == len(std_sensor)
    assert len(mean_thick) == len(std_thick)

    # 正向验证: 膜厚到lab曲线 model1+2联合. flag == 1, then flag == 2
    if targ == 1:
        test_model1_and_model2(test_data_js, mean_thick, std_thick, mean_sensor, std_sensor, flag=1)
        test_model1_and_model2(test_data_js, mean_thick, std_thick, mean_sensor, std_sensor, flag=2)

    # 反向验证: 一条不ok的样本, 先过model2, 再过model1, 看看膜厚调整值的变化
    elif targ == 2:
        modify_std(test_data_js, mean_sensor, std_sensor, best)
        thicks = modify_thickness(test_data_js, mean_thick, std_thick)
        print(thicks)



