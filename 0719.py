# coding=utf-8
import numpy as np
import json
import torch
from model1 import MLP
from sklearn.preprocessing import StandardScaler
from data_load import DataLoader
import torch.nn as nn

def fun1():
    all_data = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\第三批\0705\thick14hc3sensor64_lab.json', 'r'))
    f = json.load(open(r'./large_ng.json', 'r'))
    selected_f = open(r'./Select_feature.txt', 'r').readlines()[0]
    seleted = [int(i) for i in selected_f.split(',')[:-1]]
    large_ng = dict()
    sensor_thick = dict()
    # 只展示 single 一条数据
    largest = list(f.keys())[0]
    for num, v in all_data.items():
        if num == largest:
            large_ng[num] = v
    large_ng_lab = dict()
    f_seleted_33 = dict()
    for num, f_lab in large_ng.items():
        f = []
        f32 = f_lab[0].split(',')[10:-1]
        thick10 = f_lab[0].split(',')[:10]
        assert len(f32) == 16
        for ind in seleted:
            f.append(f32[ind])
        f_str = ''.join(i + ',' for i in f)
        thick10_str = ''.join(i+',' for i in thick10)
        large_ng_lab[f_str] = f_lab[1]
        f_seleted_33[f_str] = num
        sensor_thick[f_str] = thick10_str

    data = json.dumps(large_ng_lab)
    with open('./f16lab.json', 'w') as js_file:
        js_file.write(data)

    data = json.dumps(f_seleted_33)
    with open('./f1633.json', 'w') as js_file:
        js_file.write(data)

    data = json.dumps(sensor_thick)
    with open('./sensor_thick.json', 'w') as js_file:
        js_file.write(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def fun2(device):
    model1_list = [r'./thickhc2sensor1.pth', r'./thickhc2sensor2.pth', r'./thickhc2sensor3.pth', r'./thickhc2sensor4.pth', r'./thickhc2sensor5.pth', r'./thickhc2sensor6.pth', r'./thickhc2sensor7.pth', r'./thickhc2sensor8.pth']
    data = json.load(open('./f16lab.json', 'r'))
    thick_sensor = json.load(open('./sensor_thick.json', 'r'))
    X, Y = [], []
    for sensor, lab in data.items():
        X.append(thick_sensor[sensor].split(',')[:-1])
        sensor_all = [float(i) for i in sensor.split(',')[:-1]]
        sensor = [float(i) for i in [sensor_all[1], sensor_all[6], sensor_all[9], sensor_all[14]]]
        Y.append(sensor)
    X = np.array(X)
    Y = np.array(Y)
    scale = StandardScaler(with_mean=True, with_std=True)
    X_ = scale.fit_transform(X)
    all_data = DataLoader((X_, Y), batch_size=X_.shape[0], batch_first=False, device=device)
    mse = []
    for i, m in enumerate(model1_list):
        if i == 0:
            model = MLP(10, 20, 4).to(device)
        else:
            model = MLP(10, 40, 4).to(device)
        model.load_state_dict(torch.load(m))
        model.eval()
        for ii, (input, org) in enumerate(all_data):
            model.eval()
            pred = model(input)
            v = nn.MSELoss()(pred, org).detach().numpy()
            print(v)
            mse.append(v)
    print("最符合的sub_model1 index: {}".format(mse.index(min(mse))))

# fun2(device)

