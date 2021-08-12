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


def plot_loss(loss, name):
    x = [i for i in range(len(loss))]
    plt.title(name)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, loss)
    plt.show()


def prepare_data(best):
    '''
    r'./sensor_csv.txt'这个文件里要准备好本次测试数据的: .csv name
    number33_evtpair.json 需要自己写.

    '''
    base_data_dir = r'D:\work\project\卡尔蔡司AR镀膜\0812\机台文件'
    number_json = r'D:\work\project\卡尔蔡司AR镀膜\0812\number33_lab.json'
    csv_dict_js = r'D:\work\project\卡尔蔡司AR镀膜\0812\csv_dict.json'
    number_evts = r'D:\work\project\卡尔蔡司AR镀膜\0812\number33_evtpair.json'
    final_js = r'D:\work\project\卡尔蔡司AR镀膜\0812\finally.json'

    f = r'D:\work\project\卡尔蔡司AR镀膜\0812\双面.xlsx'
    wb = xlrd.open_workbook(f)
    data = wb.sheet_by_name('Sheet1')
    title = data.row_values(3)
    print(title)
    start, end = title.index(380), title.index(780)
    labs = []
    # lines: 哪几行有81维lab曲线数据
    lines = [5, 6]
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
    # thickness_list = [[24.8, 12.2, 36.88, 103.6, 94.8, 24.46, 11.6,36.34,100.6,92.3],
    #                   [24.8, 12.23, 36.88, 103.76, 94.53, 24.46, 11.6, 36.34, 100.6, 92.86]]
    # thickness_list = [[26.04,12,38.53,108.95,99.26,25,10.5,35,96,90]]

    thickness_list = [[ ]]

    thickness_list = [[str(i) for i in a] for a in thickness_list]
    numbers = ['33338882233']
    for ind, number in enumerate(numbers):
        number_thickness[number] = [thickness_list[ind], labs[ind]]
    data = json.dumps(number_thickness)
    with open(number_json, 'w') as js_file:
        js_file.write(data)

    # generate data
    get8step_sensor_feature(7, number_evts, base_data_dir, csv_dict_js, number_json, final_js, ['ACT_O1_QCMS_THICKNESS_CH1'])



if __name__ == "__main__":
    '''
    ycy

    '''
    best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
            1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
            0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
            0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
            3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]

    prepare_data(best)





