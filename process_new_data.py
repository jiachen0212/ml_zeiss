# coding=utf-8
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optimizers
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from data_load import DataLoader
from data_post_process import data_post_process
from mlp_torch import MLP
from util import calculate_Lab
from util import cnames
from util import weighted_mse
from utils.my_mse_loss import my_mse_loss
from utils.my_mse_loss import my_mse_loss1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
colors = list(cnames.keys())


def compute_loss(t, y):
    return my_mse_loss()(y, t)


def compute_loss1(t, y):
    return my_mse_loss1()(y, t)
    # return nn.MSELoss()(y, t)


def show_y_pred(y, gt_y=None, epo=None, best=None, flag='eval'):
    sample_num, dims = y.shape
    plt.title('model2')
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(81)]
    for i in range(sample_num):
        single_y = y[i, :]
        single_gt_y = gt_y[i, :]
        if i == 0:
            plt.plot(x, single_gt_y, color='pink', label='real curve')
            plt.plot(x, single_y, color='black', label='model regress')
        else:
            plt.plot(x, single_gt_y, color='pink')
            plt.plot(x, single_y, color='black')
    plt.legend()
    plt.show()


def plot_loss(loss):
    x = [i for i in range(len(loss))]
    plt.title('modify-thickness loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, loss)
    # plt.savefig("pretrain_loss.png")
    plt.show()


def generate_data(file1, file2, base_data_dir, CC_dir, CX_dir, number33_thick10sensor8step_lab_js,
                  number33_thicklab, evt_pair, num_evt12, csv_dict_js, flag):
    tmp = number33_thick10sensor8step_lab_js
    if not os.path.exists(tmp):
        data_post_process(file1, file2, base_data_dir, CC_dir, CX_dir, number33_thick10sensor8step_lab_js,
                          number33_thicklab, evt_pair, num_evt12, csv_dict_js).run()
        print("data process done!")

    with open(tmp, encoding="utf-8") as reader:
        f_lab = json.load(reader)

    X, Y = [], []
    number_list = []

    if flag == 0 or flag == 2:
        # 微调阶段, 只使用一个清洗周期内的数据
        part_clean = open(r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_data_txts\part_clean_number8.txt', 'r').readlines()[0]
        part_clean_number = part_clean.split(',')[:-1]
        for number33, thicksensor_lab in f_lab.items():
            if number33 == "33121032303":
                x = thicksensor_lab[0].split(',')[:-1][10:]
                X.append([float(i) for i in x])
                y = thicksensor_lab[1]
                Y.append(y)
                number_list.append(number33)

    if flag == 1:
        all_thickness = []
        # 使用所有的数据训model2
        for number33, thicksensor_lab in f_lab.items():
            all = thicksensor_lab[0].split(',')[:-1]
            all_thickness.append([float(a) for a in all[:10]])
            x = all[10:]
            assert len(x) == 16
            X.append([float(i) for i in x])
            Y.append(thicksensor_lab[1])
        Y = [[float(i) for i in a] for a in Y]

        # 记录下所有样本的thickness mean, std统计信息
        mean_ = np.mean(all_thickness, axis=0)
        std_ = np.std(all_thickness, axis=0)
        # thickness mean, std落盘
        mean_f = open(r'./thickness_mean.txt', 'w')
        std_f = open(r'./thickness_std.txt', 'w')
        for value in mean_:
            mean_f.write(str(value) + ',')
        for value in std_:
            std_f.write(str(value) + ',')

    # 剔除 lab曲线反射率均值<1的样本
    for i in range(len(Y)):
        if np.mean(Y[i]) < 1:
            X.pop(i)
            if not flag:
                number_list.pop(i)
    Y = [a for a in Y if np.mean(a[1]) >= 1]

    # for x in X:
    #     x.pop(0)
    #     x.pop(7)
    # print(len(X), len(X[0]))
    #
    # mean_ = np.mean(X, axis=0)
    # std_ = np.std(X, axis=0)
    # print("X_mean, std 维度: {}".format(mean_.shape))
    # # sensor mean, std 落盘
    # mean_f = open(r'./sensor_mean.txt', 'w')
    # std_f = open(r'./sensor_std.txt', 'w')
    # for value in mean_:
    #     mean_f.write(str(value) + ',')
    # for value in std_:
    #     std_f.write(str(value) + ',')

    return X, Y, number_list


def compare_res(best, lr):
    y1 = np.load(r'./step1_y.npy')
    y2 = np.load(r'./step2_y.npy')
    mse1 = []
    mse2 = []
    sample_num, dims = y1.shape
    plt.title('compare lab_curve lr: {}'.format(lr))
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(81)]
    for i in range(sample_num):
        a = y1[i, :]
        b = y2[i, :]
        # print("base lab: L: {}, A: {}, B: {}".format(calculate_Lab(a)[0], calculate_Lab(a)[1], calculate_Lab(a)[2]))
        # print("modified lab: L: {}, A: {}, B: {}".format(calculate_Lab(b)[0], calculate_Lab(b)[1], calculate_Lab(b)[2]))
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


def run_train(X, Y, input_dim, hiden_dim, output_dim, epochs, mean_, std_):
    batch_size = X.shape[0]
    for i in range(batch_size):
        X[i] = [(X[i][j] - mean_[j]) / std_[j] for j in range(X.shape[1])]
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=3)
    print("train size: {}, validation size: {}".format(train_x.shape[0], test_x.shape[0]))
    train_dataloader = DataLoader((train_x, train_y), batch_size=batch_size, batch_first=False, device=device)
    val_dataloader = DataLoader((test_x, test_y), batch_size=batch_size, batch_first=False, device=device)
    model = MLP(input_dim, hiden_dim, output_dim).to(device)
    print(model)
    optimizer = optimizers.Adam(model.parameters(),
                                lr=0.001,
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
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        loss_list.append(train_loss)
        if (epoch + 1) % 200 == 0:
            model.eval()
            for ii, (input, gt) in enumerate(val_dataloader):
                model.eval()
                pred = model(input)
                y_pred = pred.detach().numpy()
                show_y_pred(y_pred, gt, epo=epoch, flag='validation')
            for ii, (input, gt) in enumerate(train_dataloader):
                model.eval()
                pred = model(input)
                y_pred = pred.detach().numpy()
                show_y_pred(y_pred, gt, epo=epoch, flag='train')
        if epoch == epochs - 1:
            model.eval()
            for ii, (input, org) in enumerate(train_dataloader):
                model.eval()
                pred = model(input)
                y = pred.detach().numpy()
                show_y_pred(y, org, epo=epoch, flag='train')
    plot_loss(loss_list)
    torch.save(model.state_dict(), "./model2.pth")


def run_test(best_lab, lr, mlp_pth, X, Y, input_dim, hiden_dim, output_dim, epochs, mean_, std_):
    model = MLP(input_dim, hiden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(mlp_pth))

    for index, p in enumerate(model.parameters()):
        p.requires_grad = False

    # test
    all_data = DataLoader((X, Y), batch_size=X.shape[0], batch_first=False, device=device)
    for ii, (data, lab) in enumerate(all_data):
        pred = model(data).detach().numpy()[0]
        loss = weighted_mse(pred)
        print(loss, 'cj')

    losses = []
    for epoch in range(epochs):
        if epoch > 0:
            X = np.load('./data_modify.npy')

        all_data = DataLoader((X, Y), batch_size=X.shape[0], batch_first=False, device=device)

        for ii, (data, lab) in enumerate(all_data):
            target = [best_lab]
            target = np.array(target)
            target = np.reshape(target, (data.shape[0], -1))
            target = Variable(torch.from_numpy(target).float())

            data = Variable(data, requires_grad=True)
            optimizer = optimizers.Adam({data},
                                        lr=lr,
                                        betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
            optimizer.zero_grad()
            score = model(data)
            pred = score.detach().numpy()
            loss = compute_loss1(score, target)
            # print('-' * 10, 'epoch {} loss: {}'.format(epoch, loss), '-' * 10)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch == 0:
                np.save(r'./step1_y.npy', pred)
            if epoch == epochs - 2:
                np.save(r'./step2_y.npy', pred)
            np.save(r'./data_modify.npy', data.detach().numpy())
            if epoch == epochs - 1:
                tmp = data.detach().numpy()[0]
                modified_std = [tmp[i] * std_[i] + mean_[i] for i in range(len(std_))]
                np.save(r'./modified_std', modified_std)
    min_index = losses.index(min(losses))
    print("loss 最小的loss：{}, 对应的epoch: {}".format(min(losses), min_index))
    # plot_loss(losses)
    return min(losses), min_index


if __name__ == "__main__":

    # 标准lab曲线
    best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
            1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
            0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
            0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
            3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]
    _ = calculate_Lab(best)

    root_dir = r'D:\work\project\卡尔蔡司AR镀膜\第四批'
    base_data_dir = os.path.join(root_dir, r'all')
    sub_dir = r'0811'
    js_save_path = os.path.join(root_dir, sub_dir)
    if not os.path.exists(js_save_path):
        os.mkdir(js_save_path)
    evt_cc_dir = os.path.join(root_dir, r'cc')
    CC_dir = os.path.join(root_dir, r'cc')
    CX_dir = os.path.join(root_dir, r'cx')
    file1 = os.path.join(root_dir, r'33.xlsx')
    file2 = os.path.join(root_dir, r'双面膜色.xlsx')
    # .json落盘文件
    number33_thick10sensor8step_lab_js = os.path.join(root_dir, sub_dir, 'thick14hc3sensor64_lab.json')
    number33_thicklab_js = os.path.join(root_dir, sub_dir, 'number33_thick10lab.json')

    # 其他中间落盘文件
    evt_pair = os.path.join(root_dir, sub_dir, '正背面_thickness_evtname.txt')
    num_evt12 = os.path.join(root_dir, sub_dir, 'num_evt12.json')
    csv_dict_js = os.path.join(root_dir, sub_dir, 'evtname_sensor_name_value.json')

    hiden_dim = 80
    epochs_train = 800
    epochs_test = 1000

    flag = 1

    X, Y, number_list = generate_data(file1, file2, base_data_dir, CC_dir, CX_dir,
                                      number33_thick10sensor8step_lab_js,
                                      number33_thicklab_js, evt_pair, num_evt12, csv_dict_js, flag)

    # 除去index 0,8: 不要正背面的step 1 std, 数值是0无回归意义
    print(X[0])
    print("除去index 0,8: 不要正背面的step 1 std, 数值是0无回归意义")
    x_inds = [i for i in range(16)]
    x_inds = [i for i in x_inds if i != 0 and i != 8]
    X = [[x[i] for i in x_inds] for x in X]
    print(X[0])
    X = np.array(X)
    Y = np.array(Y)
    print("X.shape: {}, Y.shape: {}".format(X.shape, Y.shape))

    #
    # # combine 第三四批的thickness std mean std值
    # dir1 = r'D:\work\project\卡尔蔡司AR镀膜\第三批'
    # dir2 = r'D:\work\project\卡尔蔡司AR镀膜\第四批'
    # thickness_mean = r'thickness_mean.txt'
    # thickness_std = r'thickness_std.txt'
    # sensor_mean = r'sensor_mean.txt'
    # sensor_std = r'sensor_std.txt'
    # thickness_mean1 = [float(a) for a in open(os.path.join(dir1, sensor_std), 'r').readlines()[0].split(',')[:-1]]
    # thickness_mean2 = [float(a) for a in open(os.path.join(dir2, sensor_std), 'r').readlines()[0].split(',')[:-1]]
    #
    # A = [a*587 for a in thickness_mean1]
    # B = [a*77 for a in thickness_mean2]
    # C = [(A[i]+B[i])/664 for i in range(14)]
    #
    # # mean_f = open(r'./sensor_mean.txt', 'w')
    # # std_f = open(r'./sensor_std.txt', 'w')
    # # for value in mean_:
    # #     mean_f.write(str(value) + ',')
    # # for value in std_:
    # #     std_f.write(str(value) + ',')
    #
    # mean_f = open(r'./sensor_std.txt', 'w')
    # for value in C:
    #     mean_f.write(str(value) + ',')

