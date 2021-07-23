# coding=utf-8
import json
import os
import random
import numpy as np
import torch
import torch.optim as optimizers
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from data_load import DataLoader
from data_post_process import data_post_process
from mlp_torch import MLP
from utils.my_mse_loss import my_mse_loss
from utils.my_mse_loss import my_mse_loss1
# 衡量lab曲线与标准曲线的误差/相似度
from util import weighted_mse
import matplotlib.pyplot as plt
from util import cnames
from util import calculate_Lab
from util import Select_feature
from torch.autograd import Variable
from sklearn.neural_network import MLPRegressor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
colors = list(cnames.keys())


def get_important_x():
    weights = [1 for i in range(81)]
    nms = [380 + i * 5 for i in range(81)]
    a = [380, 450, 680, 685, 690, 695, 700, 780]
    for n in a:
        weights[nms.index(n)] = 2
    return weights


def mlp_fun(test_x, test_y, train_x, train_y):
    # mlp regression
    mlpr = MLPRegressor(hidden_layer_sizes=(1,),
                        activation='tanh',
                        solver='adam',
                        alpha=0.0001,  # L2惩罚参数
                        max_iter=1,
                        random_state=123,)
    mlpr.fit(train_x, train_y)
    pre_y = mlpr.predict(test_x)
    print("mean absolute error:", metrics.mean_absolute_error(test_y, pre_y))


def compute_loss(t, y):
    return my_mse_loss()(y, t)

def compute_loss1(t, y):
    return my_mse_loss1()(y, t)
    # return nn.MSELoss()(y, t)

def show_y_pred(y, gt_y=None, epo=None, best=None, flag='eval'):
    best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
            1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
            0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
            0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
            3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]
    sample_num, dims = y.shape
    plt.title('{} epoch {} lab_curve'.format(flag, epo + 1))
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(dims)]
    for i in range(sample_num):
        single_y = y[i, :]
        single_gt_y = gt_y[i, :]
        plt.plot(x, single_gt_y, color='pink')
        plt.plot(x, single_y, color='black')
    if best:
        plt.plot(x, best, color='red', label='target')
    plt.legend()
    plt.savefig("lab_curve.png")
    plt.show()


def plot_loss(loss):
    x = [i for i in range(len(loss))]
    plt.title('modify-thickness loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, loss)
    plt.savefig("pretrain_loss.png")
    plt.show()


def remove2and7(number_thick_lab):
    thick10_lab = dict()
    X, Y = [], []
    for number33, flab in number_thick_lab.items():
        assert len(flab[0].split(',')) == 18
        Y.append(flab[1])
        X.append(flab[0])
        # 耗材和最后的空格
    X = [i.split(',')[:-4] for i in X]
    # 剔除不能调整的第二,七层
    for x in X:
        x.pop()
        x.pop(1)
        x.pop(5)
        x.pop(6)
    for i in range(len(X)):
        thick10_lab[''.join(i+',' for i in X[i])] = Y[i]
    data = json.dumps(thick10_lab)
    with open('./thick10lab.json', 'w') as js_file:
        js_file.write(data)
    return X, Y

def generate_data(data_part1, file1, file2, process_data, base_data_dir, CC_dir, CX_dir, thick14_hc3_sensor16_lab_js,
                  number33_thick10sensor8step_lab_js, feature135_lab_js, thick10_lab_js, flag=0):  # flag=0,默认选最新最多的特征

    X_list = [number33_thick10sensor8step_lab_js, thick10_lab_js]
    tmp = X_list[flag]
    if not os.path.exists(tmp):
        data_post_process(file1, file2, process_data, base_data_dir, CC_dir, CX_dir,
                          thick14_hc3_sensor16_lab_js, number33_thick10sensor8step_lab_js, thick10_lab_js).run()
        print("data process done!")
    else:
        print("data has already processed! start mlp！！!")
    with open(tmp, encoding="utf-8") as reader:
        f_lab = json.load(reader)

    X, Y = [], []
    part_clean = open(r'./part_clean_number.txt', 'r').readlines()[0]
    part_clean_number = part_clean.split(',')[:-1]
    for number33, thicksensor_lab in f_lab.items():
        if number33 in part_clean_number:
            x = thicksensor_lab[0].split(',')[:-1]
            assert len(x) == 26
            X.append(x)
            Y.append(thicksensor_lab[1])
    X = [[float(i) for i in x] for x in X]
    Y = [[float(i) for i in a] for a in Y]

    # 剔除坏掉的样本
    for i in range(len(Y)):
        if np.mean(Y[i]) < 1:
            X.pop(i)
    Y = [a for a in Y if np.mean(a) >= 1]

    return X, Y



def generate_data_test():
    X, Y = [], []
    f = json.load(open('./f16lab.json', 'r'))
    for f17, lab in f.items():
        print(f17.split(',')[:-1])
        print('\n')
        X.append([float(n) for n in f17.split(',')[:-1]])
        Y.append(lab)
    X = np.array(X)
    Y = np.array(Y)

    return X, Y


# def compare_res(best):
#     best_ = np.array(best)
#     y2 = np.load(r'./step2_y.npy')
#     y1 = np.load(r'./step1_y.npy')
#     mse2 = []
#     sample_num, dims = y2.shape
#     plt.title('compare lab_curve')
#     plt.xlabel("Wave-length")
#     plt.ylabel("Reflectance")
#
#     x = [380 + 5 * i for i in range(dims)]
#     for i in range(sample_num):
#         b = y2[i, :]
#         mse2.append(weighted_mse(b))
#     print("fine_tune mse: {}".format(np.mean(mse2)))


def compare_res(best):
    best_ = np.array(best)
    y1 = np.load(r'./step1_y.npy')
    y2 = np.load(r'./step2_y.npy')
    print(y1.shape, y2.shape)
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
    plt.savefig("compare_lab_curve.png")
    plt.show()
    print("base mse: {}, fine_tune mse: {}".format(np.mean(mse1), np.mean(mse2)))

def run_train(X, Y, hiden_dim, output_dim, epochs):
    print("data_size: {}, {}".format(X.shape, Y.shape))

    batch_size = X.shape[0]
    input_dim = X.shape[-1]

    # 数据规整化
    scale = StandardScaler(with_mean=True, with_std=True)
    X_ = scale.fit_transform(X)
    train_x, test_x, train_y, test_y = train_test_split(X_, Y, test_size=0.25, random_state=3)
    print("train size: {}".format(train_x.shape[0]))
    print("validation size: {}".format(test_x.shape[0]))
    train_dataloader = DataLoader((train_x, train_y), batch_size=batch_size, batch_first=False, device=device)
    val_dataloader = DataLoader((test_x, test_y), batch_size=batch_size, batch_first=False, device=device)
    model = MLP(input_dim, hiden_dim, output_dim).to(device)
    print(model)
    optimizer = optimizers.Adam(model.parameters(),
                                      lr=0.001,
                                      betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)  # L2正则

    loss_list = []
    for epoch in range(epochs):
        train_loss = 0
        # print('-' * 10, 'epoch: {}'.format(epoch + 1), '-' * 10)
        for ii, (data, label) in enumerate(train_dataloader):
            input = Variable(data, requires_grad=False)
            target = Variable(label)
            optimizer.zero_grad()
            score = model(input)
            loss = compute_loss(score, target)
            # print('-' * 10, 'epoch {} loss: {}'.format(epoch, loss), '-' * 10)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        loss_list.append(train_loss)
        if (epoch + 1) % 1000 == 0:
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
    torch.save(model.state_dict(), "./mlp.pth")

def run_test(mlp_pth, X, Y, hiden_dim, output_dim, epochs):
    print("data_size: {}, {}".format(X.shape, Y.shape))
    input_dim = X.shape[1]
    model = MLP(input_dim, hiden_dim, output_dim).to(device)
    print(model)
    model.load_state_dict(torch.load(mlp_pth))

    scale = StandardScaler(with_mean=True, with_std=True)
    X_ = scale.fit_transform(X)
    all_data = DataLoader((X_, Y), batch_size=X_.shape[0], batch_first=False, device=device)

    for index, p in enumerate(model.parameters()):
        p.requires_grad = False

    loss_list = []
    for epoch in range(epochs):
        train_loss = 0
        for ii, (data, label) in enumerate(all_data):
            if epoch == 0:
                model.eval()
                preds = model(data)
                y_pred = preds.detach().numpy()
                x_data = scale.inverse_transform(data.detach().numpy())
                np.save(r'./step1_y.npy', y_pred)
                np.save(r'./start_x.npy', x_data)
                np.save(r'./start_lab.npy', label)
            # 用标准曲线作为target,逼近膜厚去拟合最佳曲线
            target = best * data.shape[0]
            target = np.array(target)
            target = np.reshape(target, (data.shape[0], -1))
            target = Variable(torch.from_numpy(target).float())

            data = Variable(data, requires_grad=True)
            optimizer = optimizers.Adam({data},
                                        lr=0.1,
                                        betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
            optimizer.zero_grad()
            score = model(data)
            loss = compute_loss1(score, target)
            loss.backward()
            # print(data.grad[0])
            optimizer.step()
            train_loss += loss.item()
            if epoch == epochs - 1:
                model.eval()
                preds = model(data)
                y_pred = preds.detach().numpy()
                x_data = scale.inverse_transform(data.detach().numpy())
                np.save(r'./step2_y.npy', y_pred)
                np.save(r'./modified_x.npy', x_data)
                np.save(r'./modified_lab.npy', label)
        train_loss /= len(all_data)
        loss_list.append(train_loss)
        # print('-' * 10, 'loss: {}'.format(train_loss), '-' * 10)
    plot_loss(loss_list)
    print(loss_list.index(min(loss_list)))  # 返回fine-tune阶段min_loss出现的epoch
    print(max(loss_list), min(loss_list))

def data_info(X, Y):
    '''

    :param X:
    :param Y:
    :return: 检查一个膜厚设置输入, 是否对应多条lab曲线.
    '''
    print("X shape:", X.shape)
    print("Y shape", Y.shape)
    data_length = X.shape[0]
    all_same = [True for i in X[0]]
    print(all_same)

    for i in range(1, data_length):
        for index, val in enumerate(X[0]):
            if X[i][index] != val:
                all_same[index] = False
    print(all_same)

    for i in range(data_length):
        for j in range(i):
            Xi = X[i]
            Xj = X[j]
            identical = (Xi == Xj).all()
            if identical:
                print("For input X {} and {} are identical:{}".format(i, j, Xi))
                Yi = Y[i]
                Yj = Y[j]
                y_identical = (Yi == Yj).all()
                if y_identical:
                    print("\tY {} and {} are the same.".format(i, j))
                else:
                    y_delta = Yi - Yj
                    print("\tY {} and {} not the same.".format(i, j))
                    # print("\tY[{}]:{}".format(i,Yi))
                    # print("\tY[{}]:{}".format(j,Yj))
                    print("\tDelta: {}".format(sum(abs(y_delta))))


def concate_data(a, b, c):
    js1 = json.load(open(a, 'r'))
    js2 = json.load(open(b, 'r'))
    print("pre_data_size: {}, cur_data_size: {}".format(len(js1), len(js2)))
    all_js = dict()
    for k, v in js1.items():
        all_js[k] = v
    for k, v in js2.items():
        all_js[k] = v
    print(len(all_js))
    data = json.dumps(all_js)
    with open(c, 'w') as js_file:
        js_file.write(data)



def show_all_y(Y, best):
    n, m = Y.shape[:2]
    plt.title('all Y')
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(m)]
    count = 0
    for i in range(n):
        single_y = Y[i, :]
        if i == 0:
            if single_y[74] > 4.5:
                plt.plot(x, single_y, color='black', label='origin_bad_y')
                count += 1
            else:
                plt.plot(x, single_y, color='cornflowerblue', label='origin_y')
            plt.plot(x, best, color='red', label='best')
        else:
            if single_y[74] > 4.5:
                plt.plot(x, single_y, color='black')
                count += 1
            else:
                plt.plot(x, single_y, color='cornflowerblue', )
    # print(count)
    plt.legend()
    plt.show()



if __name__ == "__main__":

    x = [380 + i * 5 for i in range(81)]
    import_index = x.index(750)

    # 1train or 0modified_thickness
    flag = 1
    # get_important_x()

    # 标准lab曲线
    best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
            1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
            0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
            0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
            3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]
    _ = calculate_Lab(best)
    part_root_dir1 = r'D:\work\project\卡尔蔡司AR镀膜\第二批7.1\0701'
    root_dir = r'D:\work\project\卡尔蔡司AR镀膜\第三批'
    base_data_dir = os.path.join(root_dir, r'33机台文件')
    sub_dir = r'0705'
    js_save_path = os.path.join(root_dir, sub_dir)
    if not os.path.exists(js_save_path):
        os.mkdir(js_save_path)
    evt_cc_dir = os.path.join(root_dir, r'cc')
    CC_dir = os.path.join(root_dir, r'cc')
    CX_dir = os.path.join(root_dir, r'cx')
    file1 = os.path.join(root_dir, r'匹配关系2021.1~2021.6.xlsx')
    file2 = os.path.join(root_dir, r'33# DVS双面膜色2021.1~2021.6.xlsx')
    # 此文档用于关联周期信息,筛选相同膜厚设置值所对应的lab曲线
    process_data = os.path.join(root_dir, r'镀膜炉工艺记录2021.1~2021.6.xlsx')
    # .json都是数据处理中需要落盘的信息
    data_js = os.path.join(root_dir, sub_dir, 'thickness_lab_curve.json')
    refine_data_json = os.path.join(root_dir, sub_dir, 'refine_thickness_lab_curve.json')
    bad_js = os.path.join(root_dir, sub_dir, 'bad_thick_lab.json')
    oneone_evt_thickness = os.path.join(root_dir, sub_dir, 'oneone_evt_thickness.json')
    evt_33number = os.path.join(root_dir, sub_dir, 'evt_33number.json')
    # 加入3维耗材信息
    num33_hc_js = os.path.join(root_dir, sub_dir, '33_hc.json')
    number33_thick_js = os.path.join(root_dir, sub_dir, '33number_thickness.json')
    thick_hc_lab_js = os.path.join(root_dir, sub_dir, r'thick_hc_lab.json')
    # 加入16维sensor列时序特征
    thick14_hc3_sensor16_lab_js = os.path.join(root_dir, sub_dir, 'thick14hc3sensor16_lab.json')
    # 加入64维 8step sensor时序特征
    csv_dict_js = os.path.join(root_dir, sub_dir, 'evtname_sensor_name_value.json')
    number33_thick10sensor8step_lab_js = os.path.join(root_dir, sub_dir, 'thick14hc3sensor64_lab.json')
    # 再加入19列有意义数据的38维特征
    feature135_lab_js = os.path.join(root_dir, sub_dir, 'feature135_lab.json')

    # merge two_part_data_json
    data_part1 = os.path.join(part_root_dir1, 'all.json')
    full_135feature_js = os.path.join(root_dir, sub_dir, 'all.json')
    number33_thicklab_js = r'./number33_thick10lab.json'
    hiden_dim = 80
    epochs_train = 5000
    epochs_test = 1000
    X, Y = generate_data(data_part1, file1, file2, process_data, base_data_dir, CC_dir, CX_dir,
                         thick14_hc3_sensor16_lab_js, number33_thick10sensor8step_lab_js, feature135_lab_js,
                         number33_thicklab_js)
    X = [x[:10] for x in X]
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    output_dim = Y.shape[1]

    if flag == 1:
        run_train(X, Y, hiden_dim, output_dim, epochs_train)

    elif flag == 0:
        mlp_list = ['./mlp_6_1.pth']
        for mlp_pth in mlp_list:
            run_test(mlp_pth, X, Y, hiden_dim, output_dim, epochs_test)
            compare_res(best)

