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


def plot_loss(loss):
    x = [i for i in range(len(loss))]
    plt.title('fine_tune loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, loss)
    plt.show()


def prepare_data():
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
    for line in lines:
        lab = data.row_values(line)[start:end+1]
        print(lab)
        labs.append(lab)

    number_thickness = dict()
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
    plt.plot(x, y1, color='cornflowerblue', label='real curve')
    plt.plot(x, y2, color='lightpink', label='modified curve')
    plt.plot(x, best, color='red', label='target')
    plt.legend()
    plt.savefig(r'./0802.png')
    plt.show()



def test_model1_and_model2(mean_thick, std_thick, mean_sensor, std_sensor):
    '''
    联合测试 model1,2. 5条给到的测试数据可用于 fine-tune model1,2的 cycle子模型

    '''

    final_js = r'D:\work\project\卡尔蔡司AR镀膜\0731data\7.31\7.31\finally.json'
    f = json.load(open(final_js, 'r'))
    X1, Y1 = [], []
    X2, Y2 = [], []
    for k, v in f.items():
        thick10std16 = v[0].split(',')[:-1]
        assert len(thick10std16) == 26
        lab = v[1]
        thick = [float(a) for a in thick10std16[:10]]
        std = [float(a) for a in thick10std16[10:]]
        std.pop(0)
        std.pop(7)
        Y1.append(std)
        std = [(std[i]-mean_sensor[i])/std_sensor[i] for i in range(len(std_sensor))]
        thick = [(thick[i]-mean_thick[i])/std_thick[i] for i in range(len(std_thick))]
        X1.append(thick)
        X2.append(std)
        Y2.append(lab)

    '''
    # for model1
    model1_pth_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model1'
    pths = os.listdir(model1_pth_dir)
    model = MLP(10, 200, 14).to('cpu')

    # get model1的最佳选择
    # for ind, x1 in enumerate(X1):
    #     losses = []
    #     all_data = DataLoader((np.array([x1]), np.array([Y1[ind]])), batch_size=1, batch_first=False, device='cpu')
    #     for m1 in pths:
    #         model1 = os.path.join(model1_pth_dir, m1)
    #         model.load_state_dict(torch.load(model1))
    #         for jj, (a, lab) in enumerate(all_data):
    #             model.eval()
    #             pred = model(a)
    #             loss = compute_loss(pred, lab)
    #             losses.append(loss.item())
    #     ind = losses.index(min(losses))
    #     print(ind, pths[ind], losses[ind])

    pth = pths[8]
    # 5条数据拿过来fine-tune model1_16.pth
    all_data = DataLoader((np.array(X1), np.array(Y1)), batch_size=len(X1), batch_first=False, device='cpu')
    model1 = os.path.join(model1_pth_dir, pth)
    model.load_state_dict(torch.load(model1))

    # 2. 用此条数据 finetune下model1的状态
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
    torch.save(model.state_dict(), "./model1_fine_tune.pth")
    # fine好之后,对样本一一eval()
    std_prds = []
    for ind, x1 in enumerate(X1):
        single_data = DataLoader((np.array([x1]), np.array([Y1[ind]])), batch_size=1, batch_first=False, device='cpu')
        for jj, (a, gt) in enumerate(single_data):
            model.eval()
            pred = model(a)
            # print(compute_loss(pred, gt).item())
            pred_ = pred.detach().numpy()[0]
            std_prds.append(pred_)
    np.save(r'./model1_std_value.npy', np.array(std_prds))
    '''


    # for model2
    model2_pth_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model2'
    pths = os.listdir(model2_pth_dir)
    model = MLP(14, 80, 81).to('cpu')

    # get model2的最佳选择, index=13,
    # for ind, x2 in enumerate(X2):
    #     losses = []
    #     all_data = DataLoader((np.array([x2]), np.array([Y2[ind]])), batch_size=1, batch_first=False, device='cpu')
    #     for m2 in pths:
    #         model2 = os.path.join(model2_pth_dir, m2)
    #         model.load_state_dict(torch.load(model2))
    #         for jj, (a, lab) in enumerate(all_data):
    #             model.eval()
    #             pred = model(a)
    #             loss = compute_loss(pred, lab)
    #             losses.append(loss.item())
    #     ind = losses.index(min(losses))
    #     print(ind, pths[ind], losses[13])

    # pth = pths[13]
    # # 5条数据拿过来fine-tune model2_6.pth
    # all_data = DataLoader((np.array(X2), np.array(Y2)), batch_size=len(X2), batch_first=False, device='cpu')
    # model2 = os.path.join(model2_pth_dir, pth)
    # model.load_state_dict(torch.load(model2))
    #
    # # 2. 用5条测试数据 finetune下model1的状态
    # optimizer = optimizers.Adam(model.parameters(),
    #                             lr=0.001,
    #                             betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
    # loss_list = []
    # for epoch in range(100):
    #     for ii, (data, label) in enumerate(all_data):
    #         print(data.shape)
    #         input = Variable(data, requires_grad=False)
    #         target = Variable(label)
    #         optimizer.zero_grad()
    #         score = model(input)
    #         loss = compute_loss(score, target)
    #         loss.backward()
    #         optimizer.step()
    #         loss_list.append(loss.item())
    # plot_loss(loss_list)
    # torch.save(model.state_dict(), "./model2_fine_tune.pth")

    # fine好之后,对样本一一eval()
    model2 = os.path.join(r'./model2_fine_tune.pth')
    model.load_state_dict(torch.load(model2))
    aa = [380 + 5 * i for i in range(81)]
    X2_ = np.load(r'./model1_std_value.npy').tolist()
    X2 = []
    for x in X2_:
        x2 = x
        x2 = [(x2[j] - mean_sensor[j]) / std_sensor[j] for j in range(len(std_sensor))]
        X2.append(x2)
    for ind, x2 in enumerate(X2):
        single_data = DataLoader((np.array([x2]), np.array([Y2[ind]])), batch_size=1, batch_first=False, device='cpu')
        for jj, (a, lab) in enumerate(single_data):
            model.eval()
            pred = model(a).detach().numpy()[0]
            plt.plot(aa, pred, color='pink')
            plt.plot(aa, lab.detach().numpy()[0], color='cornflowerblue')
            print("真实曲线lab值: ", calculate_Lab(lab.detach().numpy()[0]))
            print("模型拟合曲线lab值: ", calculate_Lab(pred))
            plt.show()

# 真实曲线lab值:  (4.716380620702281, 1.0988634707032874, -15.056682777944086)
# 模型拟合曲线lab值:  (4.333570192372064, 1.2596831897940253, -14.111424794028649)



# 反向验证
def modify_std(mean_sensor, std_sensor):
    final_js = r'D:\work\project\卡尔蔡司AR镀膜\0731data\7.31\7.31\finally.json'
    f = json.load(open(final_js, 'r'))
    data = f['33388888888']
    thick10std14 = data[0].split(',')[:-1]
    lab = data[1]
    std14 = thick10std14[10:]
    std14.pop(0)
    std14.pop(7)
    std14 = [float(a) for a in std14]
    std14 = [(std14[i] - mean_sensor[i]) / std_sensor[i] for i in range(len(std_sensor))]
    # model2,1的最贴近model分别是: index13,index8
    all_data = DataLoader((np.array([std14]), np.array(lab)), batch_size=1, batch_first=False, device='cpu')
    model = MLP(14, 80, 81).to('cpu')
    model.load_state_dict(torch.load(r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model2\model2_6.pth'))
    # 先fine-tune再 run model
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
    # plot_loss(loss_list)
        if epoch == 99:
            torch.save(model.state_dict(), "./mm2.pth")
    model = MLP(14, 80, 81).to('cpu')
    # model.load_state_dict(torch.load("./mm2.pth"))
    # 不要fine-tune
    model.load_state_dict(torch.load("D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model2\model2_6.pth"))
    for index, p in enumerate(model.parameters()):
        p.requires_grad = False

    losses = []
    X = np.array([std14])
    Y = np.array([lab])
    np.save(r'./step1_y.npy', Y)
    epochs = 2000
    best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
            1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
            0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
            0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
            3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]
    l1 = [4.97, 3.12, 1.81, 0.91, 0.55, 0.48, 0.48, 0.55, 0.87, 1.18, 1.39, 1.61, 1.7, 1.8, 1.89, 2.01, 2.02, 1.9, 1.73, 1.61, 1.52, 1.49, 1.48, 1.41, 1.25, 1.02, 0.9, 0.83, 0.88, 0.88, 0.84, 0.75, 0.62, 0.47, 0.35, 0.3, 0.3, 0.44, 0.4, 0.39, 0.32, 0.26, 0.17, 0.09, 0.04, 0.08, 0.08, 0.14, 0.16, 0.16, 0.18, 0.14, 0.13, 0.17, 0.2, 0.33, 0.64, 0.81, 0.9, 0.98, 1.09, 1.15, 1.25, 1.37, 1.49, 1.58, 1.79, 2.12, 2.41, 2.82, 3.19, 3.55, 3.92, 4.25, 4.59, 4.69, 4.89, 5.09, 5.35, 5.49, 5.71]
    for epoch in range(epochs):
        if epoch > 0:
            X = np.load('./modify.npy')
        all_data = DataLoader((X, Y), batch_size=1, batch_first=False, device='cpu')
        for ii, (data, _) in enumerate(all_data):
            data = Variable(data, requires_grad=True)
            target = [l1]
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
            np.save(r'./modify.npy', data.detach().numpy())
            if epoch == epochs-1:
                tmp = data.detach().numpy()[0]
                modified_std = [tmp[i] * mean_sensor[i] + mean_sensor[i] for i in range(len(mean_sensor))]
                np.save(r'./modified_std.npy', modified_std)
    plot_loss(losses)
    print(losses.index(min(losses)))
    compare_res(best)


def modify_thickness(mean_thick, std_thick):
    final_js = r'D:\work\project\卡尔蔡司AR镀膜\0731data\7.31\7.31\finally.json'
    f = json.load(open(final_js, 'r'))
    data = f['33388888888']
    thick10std14 = data[0].split(',')[:-1]
    thick10 = [float(a) for a in thick10std14[:10]]
    thick10 = [(thick10[j]-mean_thick[j])/std_thick[j] for j in range(len(std_thick))]
    std = [float(a) for a in thick10std14[10:]]
    std.pop(0)
    std.pop(7)
    assert len(std) == 14
    all_data = DataLoader((np.array([thick10]), np.array([std])), batch_size=1, batch_first=False, device='cpu')
    model = MLP(10, 200, 14).to('cpu')
    model.load_state_dict(torch.load(r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model1\model1_16.pth'))
    # 先fine-tune 再 run model1
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
    # plot_loss(loss_list)
        if epoch == 99:
            torch.save(model.state_dict(), "./mm1.pth")
    model.load_state_dict(torch.load("./mm1.pth"))
    X = np.array([thick10])
    Y = np.load(r'./modified_std.npy')
    epochs = 20000
    losses = []
    for epoch in range(epochs):
        if epoch > 0:
            X = np.load('./modify1.npy')
        all_data = DataLoader((X, Y), batch_size=1, batch_first=False, device='cpu')
        for ii, (data, gt) in enumerate(all_data):
            data = Variable(data, requires_grad=True)
            target = Variable(gt)
            optimizer = optimizers.Adam({data},
                                        lr=0.001,
                                        betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
            optimizer.zero_grad()
            score = model(data)
            loss = compute_loss(score, target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            np.save(r'./modify1.npy', data.detach().numpy())
            if epoch == epochs-1:
                tmp = data.detach().numpy()[0]
                print(tmp)
                modified_std = [tmp[i] * std_thick[i] + mean_thick[i] for i in range(len(mean_thick))]
                np.save(r'./modified_thickness.npy', modified_std)
    plot_loss(losses)
    print(losses.index(min(losses)), min(losses))




if __name__ == "__main__":
    prepare_data()
    mean_thick = [float(i) for i in open(r'./thickness_mean.txt', 'r').readlines()[0].split(',')[:-1]]
    std_thick = [float(i) for i in open(r'./thickness_std.txt', 'r').readlines()[0].split(',')[:-1]]

    mean_sensor = [float(i) for i in open(r'./sensor_mean.txt', 'r').readlines()[0].split(',')[:-1]]
    std_sensor = [float(i) for i in open(r'./sensor_std.txt', 'r').readlines()[0].split(',')[:-1]]
    # 正向验证: 膜厚到lab曲线 model1+2联合.
    # test_model1_and_model2(mean_thick, std_thick, mean_sensor, std_sensor)

    # 反向验证: 一条不ok的样本, model1,2过一遍,得到一个膜厚推荐修改结果. 看看前后lab三个值的变化情况, 看看膜厚推荐修改
    # 和蔡司提供的修改是否大致符合.
    modify_std(mean_sensor, std_sensor)
    modify_thickness(mean_thick, std_thick)
    res = np.load(r'./modified_thickness.npy')
    print(res)


