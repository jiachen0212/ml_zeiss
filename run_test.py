# # coding=utf-8
# '''
# 在第三批数据里,不在清洗周期内的几条数据找出来（最好model 2的validation）。
# 找最相近的model1，得到std，然后拟合lab 曲线。 和数据真实的lab曲线对比, check model1-2的有效性.
# edited by chenjia 2021.07.31
#
# '''
# import torch
# import os
# import json
# from read_csvs import get8step_sensor_feature
#
#
#
# best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
#             1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
#             0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
#             0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
#             3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]
#
# base_data_dir = r'D:\work\project\卡尔蔡司AR镀膜\0731data\33# 21-7-31\机台数据'
# number_json = r'D:\work\project\卡尔蔡司AR镀膜\0731data\number33_lab.json'
# csv_dict_js = r'D:\work\project\卡尔蔡司AR镀膜\0731data\csv_dict.json'
# numbers = ['33888888888', '33888888889', '33888888890']
# number_evts = r'D:\work\project\卡尔蔡司AR镀膜\0731data\number33_evtpair.json'
# final_js = r'D:\work\project\卡尔蔡司AR镀膜\0731data\finally.json'
# number_thickness = dict()
# thickness_list = [[24.46,11.7,36.34,101.7,92.65,24.95,12.4,37.07,104.4,94.5], [24.46,11.7,36.34,103 ,93.2,24.95,12.4,37.07,106.5,95.3], [24.46,11.7,36.34,103,93.4,24.95,12.4,37.07,106.8,95.3]]
# thickness_list = [[str(i) for i in a] for a in thickness_list]
#
# for ind, number in enumerate(numbers):
#     number_thickness[number] = [thickness_list[ind], best]
# data = json.dumps(number_thickness)
# with open(number_json, 'w') as js_file:
#     js_file.write(data)
#
# # generate data
# get8step_sensor_feature(7, number_evts, base_data_dir, csv_dict_js, number_json, final_js, ['ACT_O1_QCMS_THICKNESS_CH1'])
#
#
# import json
# import os
# import torch.optim as optimizers
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from data_load import DataLoader
# from mlp_torch import MLP
# from mm_Model1 import Model1
# from mm_Model2 import Model2
# from thickness10_modified import modify
# from torch.autograd import Variable
#
#
# def top3_model1(pths, losses, n=3):
#     top_pths = []
#     print(print([round(a, 3) for a in losses]))
#     for i in range(n):
#         min1 = min(losses)
#         top_pths.append(pths[losses.index(min1)])
#         print(pths[losses.index(min1)])
#         pths.remove(pths[losses.index(min1)])
#         losses.remove(min1)
#         print(print([round(a, 3) for a in losses]), i)
#
#     return top_pths
#
#
# def compute_loss(t, y):
#     return nn.MSELoss()(y, t)
#
#
# def plot_loss(loss):
#     x = [i for i in range(len(loss))]
#     plt.title('fine_tune loss')
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.plot(x, loss)
#     plt.show()
#
#
# # 4. 针对每一待优化炉样本, 做lab,std,膜厚微调
# def modify_thickness(model1_pth_dir, model, all_data, flag, cycle=None):
#     if flag == 1:
#         loss1_ = []
#         for pth in pths:
#             model1_path = os.path.join(model1_pth_dir, pth)
#             model.load_state_dict(torch.load(model1_path))
#             for ii, (input, gt) in enumerate(all_data):
#                 model.eval()
#                 pred = model(input)
#                 loss = compute_loss(pred, gt)
#                 loss1_.append(loss.item())
#         res = loss1_.index(min(loss1_))
#         print(res)
#         return pths[res]
#
#
# if __name__ == "__main__":
#
#     # flag33 = 3   # test all
#     flag33 = 2   # check model2 是否准
#
#     all_json_ = json.load(open(final_js, 'r'))
#     numbers = list(all_json_.keys())
#
#     # 1. 拟合model2
#     # Model2(flag=1)
#     mean_ = [float(i) for i in open(r'./thickness_mean.txt', 'r').readlines()[0].split(',')[:-1]]
#     std_ = [float(i) for i in open(r'./thickness_std.txt', 'r').readlines()[0].split(',')[:-1]]
#
#     # 2. 找到最相近的model1状态
#     number = '33321020305'
#     feature26_lab = all_json_[number][0]
#     x = [float(i) for i in feature26_lab.split(',')[:-1][:10]]
#     # data normalization
#     x = [(x[i] - mean_[i]) / std_[i] for i in range(len(std_))]
#     std14 = feature26_lab.split(',')[:-1][10:]
#     std14.pop(0)
#     std14.pop(7)
#     std14 = [float(a) for a in std14]
#     Y = np.array([std14])
#     X = np.array([x])
#     all_data = DataLoader((X, Y), batch_size=X.shape[0], batch_first=False, device='cpu')
#
#     # 3. 准备model1 list
#     model1_pth_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model1'
#     pths = os.listdir(model1_pth_dir)
#     loss1_ = []
#     model = MLP(10, 200, 14).to('cpu')
#     model1_pth = modify_thickness(model1_pth_dir, model, all_data, flag=1)
#     model1 = os.path.join(model1_pth_dir, model1_pth)
#     model.load_state_dict(torch.load(model1))
#
#     # 2. 用此条数据 finetune下model1的状态
#     optimizer = optimizers.Adam(model.parameters(),
#                                 lr=0.001,
#                                 betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
#     loss_list = []
#     for epoch in range(100):
#         for ii, (data, label) in enumerate(all_data):
#             input = Variable(data, requires_grad=False)
#             target = Variable(label)
#             optimizer.zero_grad()
#             score = model(input)
#             loss = compute_loss(score, target)
#             loss.backward()
#             optimizer.step()
#             loss_list.append(loss.item())
#     plot_loss(loss_list)
#     torch.save(model.state_dict(), "./model1_fine_tune.pth")
#
#     # 3. 得到std, 准备放入model2拟合lab曲线
#     mean_1 = [float(i) for i in open(r'./sensor_mean.txt', 'r').readlines()[0].split(',')[:-1]]
#     std_1 = [float(i) for i in open(r'./sensor_std.txt', 'r').readlines()[0].split(',')[:-1]]
#     for ii, (input, gt) in enumerate(all_data):
#         model.eval()
#         pred = model(input).detach().numpy()[0]
#         pred = [(pred[k] - mean_1[k]) / std_1[k] for k in range(len(pred))]
#         std14 = np.array([pred])
#         Y = np.array([best])
#         all_data2 = DataLoader((std14, Y), batch_size=1, batch_first=False, device='cpu')
#         model2 = r'./model2.pth'
#         Model2 = MLP(14, 80, 81).to('cpu')
#         Model2.load_state_dict(torch.load(model2))
#         for jj, (a, _) in enumerate(all_data2):
#             Model2.eval()
#             pred = Model2(a).detach().numpy()[0]
#             # print(pred)
#             x = [380 + 5 * i for i in range(81)]
#             plt.title('CS')
#             plt.plot(x, pred, color='pink')
#             plt.plot(x, best, color='red')
#             plt.show()
#
#     X = []
#     Y = []
#     for number in numbers:
#         feature26_lab = all_json_[number][0]
#         x = [float(i) for i in feature26_lab.split(',')[:-1][:10]]
#         x = [(x[i] - mean_[i]) / std_[i] for i in range(len(std_))]
#         std14 = feature26_lab.split(',')[:-1][10:]
#         std14.pop(0)
#         std14.pop(7)
#         std14 = [float(a) for a in std14]
#         X.append(x)
#         Y.append(std14)
#     Y = np.array(Y)
#     X = np.array(X)
#     all_data = DataLoader((X, Y), batch_size=X.shape[0], batch_first=False, device='cpu')
#
#     if flag33 == 3:
#         print('=======')
#         print(model1_pth)
#         model1 = os.path.join(model1_pth_dir, model1_pth)
#         model.load_state_dict(torch.load(model1))
#         optimizer = optimizers.Adam(model.parameters(),
#                                     lr=0.001,
#                                     betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
#         loss_list = []
#         for epoch in range(100):
#             for ii, (data, label) in enumerate(all_data):
#                 input = Variable(data, requires_grad=False)
#                 target = Variable(label)
#                 optimizer.zero_grad()
#                 score = model(input)
#                 loss = compute_loss(score, target)
#                 loss.backward()
#                 optimizer.step()
#                 loss_list.append(loss.item())
#         plot_loss(loss_list)
#         mean_1 = [float(i) for i in open(r'./sensor_mean.txt', 'r').readlines()[0].split(',')[:-1]]
#         std_1 = [float(i) for i in open(r'./sensor_std.txt', 'r').readlines()[0].split(',')[:-1]]
#         names = ['CS', 'R1', 'R2']
#         for ii, (input, gt) in enumerate(all_data):
#             model.eval()
#             for index in range(input.shape[0]):
#                 pred = model(input).detach().numpy()[index]
#                 pred = [(pred[k] - mean_1[k]) / std_1[k] for k in range(len(pred))]
#                 std14 = np.array([pred])
#                 Y = np.array([best])
#                 all_data2 = DataLoader((std14, Y), batch_size=X.shape[0], batch_first=False, device='cpu')
#                 model2 = r'./model2.pth'
#                 Model2 = MLP(14, 80, 81).to('cpu')
#                 Model2.load_state_dict(torch.load(model2))
#                 for jj, (a, _) in enumerate(all_data2):
#                     Model2.eval()
#                     pred = Model2(a).detach().numpy()[0]
#                     x = [380 + 5 * i for i in range(81)]
#                     plt.title(names[index])
#                     plt.plot(x, pred, color='black')
#                     plt.plot(x, best, color='red')
#                     plt.show()
#
#     if flag33 == 2:
#         colors = ['black', 'firebrick', 'forestgreen']
#         print(mean_1)
#         print('\n')
#         print(std_1)
#         X = []
#         Y = []
#         for number in numbers:
#             thick10std16 = all_json_[number][0].split(',')[:-1]
#             std16 = [float(k) for k in thick10std16[10:]]
#             thickness10 = thick10std16[:10]
#             std16.pop(0)
#             std16.pop(7)
#             std16 = [(std16[i] - mean_1[i]) / std_1[i] for i in range(len(std16))]
#             print(std16)
#             X.append(std16)
#             Y.append(best)
#         Y = np.array(Y)
#         X = np.array(X)
#         all_data2 = DataLoader((X, Y), batch_size=X.shape[0], batch_first=False, device='cpu')
#         Model2.load_state_dict(torch.load(r'./model2.pth'))
#
#         model2_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model2'
#         model2s = os.listdir(model2_dir)
#         losses2 = []
#         for m2 in model2s:
#             model2_path = os.path.join(model2_dir, m2)
#             Model2 = MLP(14, 80, 81).to('cpu')
#             Model2.load_state_dict(torch.load(model2_path))
#             for jj, (a, _) in enumerate(all_data2):
#                 Model2.eval()
#                 pred = Model2(a).detach().numpy()
#                 loss = compute_loss(pred, target)
#
#         # for jj, (a, _) in enumerate(all_data2):
#         #     Model2.eval()
#         #     pred = Model2(a).detach().numpy()
#         #     for i in range(pred.shape[0]):
#         #         x = [380 + 5 * i for i in range(81)]
#         #         plt.plot(x, pred[i], color=colors[0])
#         #         plt.plot(x, best, color='red')
#         #         plt.show()
#
#




# data 0802
# coding=utf-8
'''
在第三批数据里,不在清洗周期内的几条数据找出来（最好model 2的validation）。
找最相近的model1，得到std，然后拟合lab 曲线。 和数据真实的lab曲线对比, check model1-2的有效性.
edited by chenjia 2021.07.31

'''
import torch
import os
import json
from read_csvs import get8step_sensor_feature



best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
            1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
            0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
            0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
            3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]

final_js = r'D:\work\project\卡尔蔡司AR镀膜\0731data\finally.json'


import json
import os
import torch.optim as optimizers
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_load import DataLoader
from mlp_torch import MLP
from mm_Model1 import Model1
from mm_Model2 import Model2
from thickness10_modified import modify
from torch.autograd import Variable


def top3_model1(pths, losses, n=3):
    top_pths = []
    print(print([round(a, 3) for a in losses]))
    for i in range(n):
        min1 = min(losses)
        top_pths.append(pths[losses.index(min1)])
        print(pths[losses.index(min1)])
        pths.remove(pths[losses.index(min1)])
        losses.remove(min1)
        print(print([round(a, 3) for a in losses]), i)

    return top_pths


def compute_loss(t, y):
    return nn.MSELoss()(y, t)


def plot_loss(loss):
    x = [i for i in range(len(loss))]
    plt.title('fine_tune loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, loss)
    plt.show()


# 4. 针对每一待优化炉样本, 做lab,std,膜厚微调
def modify_thickness(model1_pth_dir, model, all_data, flag, cycle=None):
    if flag == 1:
        loss1_ = []
        for pth in pths:
            model1_path = os.path.join(model1_pth_dir, pth)
            model.load_state_dict(torch.load(model1_path))
            for ii, (input, gt) in enumerate(all_data):
                model.eval()
                pred = model(input)
                loss = compute_loss(pred, gt)
                loss1_.append(loss.item())
        res = loss1_.index(min(loss1_))
        print(res)
        return pths[res]


if __name__ == "__main__":
    from util import calculate_Lab
    from sklearn.metrics import mean_squared_error as MSE

    final_js = r'D:\work\project\卡尔蔡司AR镀膜\0731data\finally.json'
    all_json_ = json.load(open(final_js, 'r'))
    feature26_lab = all_json_['33321020305'][0]

    mean_ = [float(i) for i in open(r'./thickness_mean.txt', 'r').readlines()[0].split(',')[:-1]]
    std_ = [float(i) for i in open(r'./thickness_std.txt', 'r').readlines()[0].split(',')[:-1]]

    thick10 = [float(i) for i in feature26_lab.split(',')[:-1][:10]]
    thick10 = [(thick10[i] - mean_[i]) / std_[i] for i in range(len(std_))]

    # std14 = feature26_lab.split(',')[:-1][10:]
    # std14.pop(0)
    # std14.pop(7)
    # std14 = [float(a) for a in std14]
    # X = np.array([thick10])
    # Y = np.array([std14])


    # for model2
    mean_ = [float(i) for i in open(r'./sensor_mean.txt', 'r').readlines()[0].split(',')[:-1]]
    std_ = [float(i) for i in open(r'./sensor_std.txt', 'r').readlines()[0].split(',')[:-1]]
    std14 = [9.747287,6.2930427,4.4760237,13.22046,34.151115,32.785973,13.786832,9.388135,6.2186465,4.4377556,13.423934,33.58079,32.44427,13.229015]
    std14 = [(std14[i]-mean_[i])/std_[i] for i in range(len(std_))]
    X = np.array([std14])
    Y = np.array([all_json_['33321020305'][1]])

    print(X)
    print(Y)

    all_data = DataLoader((X, Y), batch_size=X.shape[0], batch_first=False, device='cpu')
    print(calculate_Lab(all_json_['33321020305'][1]))

    # 3. 准备model1 list
    # model1_pth_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model1'
    # pths = [os.listdir(model1_pth_dir)[11]]
    # loss1_ = []
    # model = MLP(10, 200, 14).to('cpu')
    # losses = []
    # for m2 in pths:
    #     model1 = os.path.join(model1_pth_dir, m2)
    #     model.load_state_dict(torch.load(model1))
    #     for jj, (a, lab) in enumerate(all_data):
    #         model.eval()
    #         pred = model(a)
    #         y = pred.detach().numpy()[0]
    #         loss = compute_loss(pred, lab)
    #         losses.append(loss.item())
    # ind = losses.index(min(losses))
    # print(ind)
    # print(pths[ind], losses[ind])

    # fine tune model1
    # optimizer = optimizers.Adam(model.parameters(),
    #                             lr=0.001,
    #                             betas=(0.9, 0.999), amsgrad=True, weight_decay=1e-5)
    # model1 = os.path.join(model1_pth_dir, r'model1_4.pth')
    # model.load_state_dict(torch.load(model1))
    # loss_list = []
    # epochs = 60
    # for epoch in range(epochs):
    #     for jj, (a, lab) in enumerate(all_data):
    #         input = Variable(a, requires_grad=False)
    #         target = Variable(lab)
    #         optimizer.zero_grad()
    #         score = model(input)
    #         loss = compute_loss(score, target)
    #         loss.backward()
    #         optimizer.step()
    #         loss_list.append(loss.item())
    # # plot_loss(loss_list)
    #         if epoch == epochs - 1:
    #             model.eval()
    #             pred = model(a)
    #             y = pred.detach().numpy()[0]
    #             print(y)






    # 3. 准备model2 list
    model2_pth_dir = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model2'
    pths = os.listdir(model2_pth_dir)[1:]
    pths = [pths[13]]
    print(pths)
    loss1_ = []
    model = MLP(14, 80, 81).to('cpu')
    losses = []
    for m2 in pths:
        model2 = os.path.join(model2_pth_dir, m2)
        model.load_state_dict(torch.load(model2))
        for jj, (a, lab) in enumerate(all_data):
            model.eval()
            pred = model(a)
            loss = compute_loss(pred, lab)
            losses.append(loss.item())
            y = pred.detach().numpy()[0]
            print(MSE(calculate_Lab(all_json_['33321020305'][1]), y))
            print(calculate_Lab(y))
            x = [380+i*5 for i in range(len(y))]
            plt.title('compare lab')
            plt.plot(x, y, color='pink')
            plt.plot(x, lab[0], color='cornflowerblue')
            plt.show()

    ind = losses.index(min(losses))
    print(ind)
    print(pths[ind], losses[ind])