# coding=utf-8
'''
en_decoder 思路, 压缩再恢复时序数据, 取压缩后的小维度特征.
但没用上lstm的思想..

'''
import os.path
import torch
import math
import json
import torch.nn as nn
import matplotlib.pyplot as plt
from data_load import DataLoader
from sklearn.model_selection import train_test_split as TTS
import numpy as np
import pandas as pd


def strip_a(a):
    a = a.strip("['] ")
    return a


class AutoEncoder(nn.Module):
    def __init__(self, sqe_len, hidden1, embed_len):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(

            # 两层hidden
            nn.Linear(sqe_len, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, embed_len),
            nn.Tanh(),

            # 只一层hidden然后直接输出
            # nn.Linear(sqe_len, embed_len),
            # nn.Tanh(),

            # 一层hidden+一层线性
            # nn.Linear(sqe_len, hidden1),
            # nn.Tanh(),
            # nn.Linear(hidden1, embed_len),
        )
        self.decoder = nn.Sequential(

            # 两层hidden
            nn.Linear(embed_len, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, sqe_len),
            # nn.Sigmoid()
            nn.Tanh(),

            # 只一层hidden然后直接输出
            # nn.Linear(embed_len, sqe_len),
            # nn.Tanh(),

            # 一层hidden+一层线性
            # nn.Linear(embed_len, hidden1),
            # nn.Linear(hidden1, sqe_len),
            # nn.Tanh(),

        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def get_step_index(thick_col):
    ll = len(thick_col)
    inds = []
    for i in range(ll-1):
        # 膜厚值到0突变, 则i为index节点
        if thick_col[i] > 0 and thick_col[i+1] == 0:
            inds.append(i)
    if len(inds) == 6:
        return inds
    if len(inds) > 6:
        return inds[:-1]


def sampling_for_each_sensor_col(sensor_data, col_name):
    each_step_nums = [10, 6, 8, 12, 16, 14, 10]
    thick_col = sensor_data['ACT_O1_QCMS_THICKNESS']
    sensor_col = [a for a in sensor_data[col_name]]
    step_inds = get_step_index(thick_col)
    assert len(step_inds) >= 6
    # 获取7个step段
    step_datas = []
    for i in range(6):
        if i == 0:
            step_datas.append(sensor_col[: step_inds[i] + 1])
        else:
            step_datas.append(sensor_col[step_inds[i-1]+1: step_inds[i]+1])
    step_datas.append(sensor_col[step_inds[-1] + 1:])

    assert len(step_datas) == 7
    sampled_sensor_col = []
    for i in range(len(step_datas)):
        tmp_step_data = step_datas[i]
        step = math.floor(len(tmp_step_data) / each_step_nums[i])
        step_data = tmp_step_data[::step][:each_step_nums[i]]
        sampled_sensor_col.extend(step_data)
    assert len(sampled_sensor_col) == np.sum(each_step_nums)

    return sampled_sensor_col


def generate_data(datas, sensor_col, ovens, mean_std_txt, flag=None):
    # if os.path.exists('{}_{}.npy'.format(sensor_col, flag)):
    #     return np.load('{}_{}.npy'.format(sensor_col, flag))
    # else:
    X = []
    for data in datas:
        ll = len(data)
        for i in range(ll):
            oven = str(data.iloc[i]['OvenNo'])
            if oven in ovens:
                try:
                    sensor_data = pd.read_csv(data.iloc[i]['path_sensor_data'])
                except:
                    continue
                sensor_col_data = sampling_for_each_sensor_col(sensor_data, sensor_col)
                X.append(sensor_col_data)

    mean_, std_ = [], []
    if flag == "train":
        ff = open(mean_std_txt, 'w')
        mean_ = np.mean(X, axis=0)
        std_ = np.std(X, axis=0)
        std_ = [a + 1e-6 for a in std_]
        for x_mean_ in mean_:
            ff.write(str(x_mean_) + ',')
        ff.write('\n')
        for x_std_ in std_:
            ff.write(str(x_std_) + ',')
        ff.write('\n')
    elif flag == "test":
        ff = open(mean_std_txt, 'r').readlines()
        mean_ = ff[0].split(',')[:-1]
        mean_ = [float(a) for a in mean_]
        std_ = ff[1].split(',')[:-1]
        std_ = [float(a) for a in std_]

    X = [[(x[k] - mean_[k]) / std_[k] for k in range(len(std_))] for x in X]
    X = np.array(X)
    X[np.isnan(X)] = 0
    np.save('{}_{}.npy'.format(sensor_col, flag), X)

    return X



def en_decoder(hidden1, embed_len, EPOCH, BATCH_SIZE, LR, X, test_x, sensor_col):
    X_train, X_test, y_train, y_test = TTS(X, X, test_size=0.3, random_state=66)
    train_loader = DataLoader((X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, batch_first=False, device='cpu')
    val_loader = DataLoader((X_test, y_test), batch_size=BATCH_SIZE, shuffle=True, batch_first=False, device='cpu')

    # 注意这里不做shuffle
    all_loader = DataLoader((X, X), batch_size=X.shape[0], shuffle=False, batch_first=False, device='cpu')
    all_loader_test = DataLoader((test_x, test_x), batch_size=X.shape[0], shuffle=False, batch_first=False, device='cpu')

    Coder = AutoEncoder(X.shape[1], hidden1, embed_len)
    optimizer = torch.optim.Adam(Coder.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    train_loss, val_loss = [], []
    for epoch in range(EPOCH):
        loss_value = 0
        for step, (x, y) in enumerate(train_loader):
            b_x = x.view(-1, x.shape[1])
            b_y = x.view(-1, x.shape[1])
            encoded, decoded = Coder(b_x)
            loss = loss_func(decoded, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value += loss.data
        loss_value /= len(train_loader)
        print('Epoch :', epoch, '|', 'train_loss:%.4f' % loss_value)
        train_loss.append(loss_value)
        if (epoch + 1) % 100 == 0:
            loss_val = 0
            for step, (x, y) in enumerate(val_loader):
                b_x = x.view(-1, x.shape[1])
                b_y = x.view(-1, x.shape[1])
                _, decoded_data = Coder(b_x)
                loss = loss_func(decoded_data, b_y)
                loss_val += loss.data
            loss_val /= len(val_loader)
            print('Epoch :', epoch, '|', 'val_loss:%.4f' % loss_val)
            val_loss.append(loss_val)
        if (epoch + 1) % 500 == 0 or (epoch == EPOCH-1):
            plt.plot([a for a in range(len(train_loss))], train_loss, label='train_loss')
            plt.legend()
            # plt.show()
            plt.plot([a for a in range(len(val_loss))], val_loss, label='val_loss')
            plt.legend()
            # plt.show()
        # 最后一个epoch, 把sensor_col的embedding后的向量输出
        if epoch == EPOCH - 1:
            for step, (x, _) in enumerate(all_loader):
                b_x = x.view(-1, x.shape[1])
                encoded_train, _ = Coder(b_x)
                torch.save(Coder, './{}.pkl'.format(sensor_col))
            for step, (x, _) in enumerate(all_loader_test):
                b_x = x.view(-1, x.shape[1])
                encoded_test, _ = Coder(b_x)

            return encoded_train, encoded_test


def slim_en_decoder(hidden1, embed_len, EPOCH, BATCH_SIZE, LR, X):
    X_train, X_test, y_train, y_test = TTS(X, X, test_size=0.3, random_state=33)
    train_loader = DataLoader((X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, batch_first=False, device='cpu')
    val_loader = DataLoader((X_test, y_test), batch_size=BATCH_SIZE, shuffle=True, batch_first=False, device='cpu')
    # 注意这里不做shuffle
    all_loader = DataLoader((X, X), batch_size=X.shape[0], shuffle=False, batch_first=False, device='cpu')

    Coder = AutoEncoder(X.shape[1], hidden1, embed_len)
    optimizer = torch.optim.Adam(Coder.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    train_loss, val_loss = [], []
    for epoch in range(EPOCH):
        loss_value = 0
        for step, (x, y) in enumerate(train_loader):
            b_x = x.view(-1, x.shape[1])
            b_y = x.view(-1, x.shape[1])
            encoded, decoded = Coder(b_x)
            loss = loss_func(decoded, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value += loss.data
        loss_value /= len(train_loader)
        print('Epoch :', epoch, '|', 'train_loss:%.4f' % loss_value)
        train_loss.append(loss_value)
        if (epoch + 1) % 100 == 0:
            loss_val = 0
            for step, (x, y) in enumerate(val_loader):
                b_x = x.view(-1, x.shape[1])
                b_y = x.view(-1, x.shape[1])
                _, decoded_data = Coder(b_x)
                loss = loss_func(decoded_data, b_y)
                loss_val += loss.data
            loss_val /= len(val_loader)
            print('Epoch :', epoch, '|', 'val_loss:%.4f' % loss_val)
            val_loss.append(loss_val)
        if epoch == EPOCH - 1:
            for step, (x, _) in enumerate(all_loader):
                b_x = x.view(-1, x.shape[1])
                encoded, _ = Coder(b_x)
                torch.save(Coder, './fc.pkl')
                return encoded
        if (epoch + 1) % 500 == 0 or (epoch == EPOCH-1):
            plt.plot([a for a in range(len(train_loss))], train_loss, label='train_loss')
            # plt.legend()
            plt.show()
            plt.plot([a for a in range(len(val_loss))], val_loss, label='val_loss')
            plt.legend()
            # plt.show()


if __name__ == '__main__':

    data1 = r'./0910_cx_data.csv'
    data2 = r'./0917_cx_data.csv'
    data3 = r'./1008_cx_data.csv'
    data4 = r'./1021_cx_data.csv'
    data5 = r'./1105_cx_data.csv'
    data6 = r'./1121_cx_data.csv'

    datas = [pd.read_csv(data1), pd.read_csv(data2), pd.read_csv(data3), pd.read_csv(data4), pd.read_csv(data5),
                pd.read_csv(data6)]

    sensor_cols = ["ACT_V1_IONIVAC_CH", "ACT_V1_PENNINGVAC_CH", "ACT_V1_THERMOVAC_CH", "ACT_V1_THERMOVAC_PREVLINE", "ACT_V1_THERMOVAC_HP", "ACT_V1_THERMOVAC_HP2", "ACT_V1_PRESSURE_CH", "AI_V1_POLYCOLD_TEMP", "ACTN_F1_FLOW1", "ACT_F1_FLOW1", "STAT_LT_CRYSTAL_CH1", "ACT_HEATER2_TEMPERATURE", "ACT_Q10_CURRENT_ANODE", "ACT_Q10_VOLTAGE_ANODE", "ACT_Q10_CURRENT_CATHODE", "ACT_Q10_VOLTAGE_CATHODE", "ACT_Q10_CURRENT_NEUTRAL", "ACT_Q10_ION_FLOW1", "ACT_Q10_ION_FLOW2", "STA_Q10_IONSOURCE_SHUTTER_IOP", "ACT_V1_MEISSNER_POLYCOLDTEMP"]
    # thickness_cols = ["ACT_O1_QCMS_THICKNESS", "ACT_O1_QCMS_RATE", "ACT_O1_QCMS_THICKNESS_CH1", "ACT_O1_QCMS_RATE_CH1"]
    # sensor_cols = thickness_cols

    oven_train = np.load('./data1_5_oven_train.npy')
    oven_test = np.load('./data6_oven_test.npy')

    seeds = [33]
    for seed in seeds:
        # oven_train = np.load(r'./train_oven_seed{}.npy'.format(seed))
        # oven_test = np.load(r'./test_oven_seed{}.npy'.format(seed))

        train_feature_js = './all_en_decoder_feature_train_seed{}.json'.format(seed)
        test_feature_js = './all_en_decoder_feature_test_seed{}.json'.format(seed)

        all_x_encoded_train = None
        all_x_encoded_test = None
        epochs = [1200]*len(sensor_cols)
        hidden1 = 16
        embed_len = 8
        BATCH_SIZE = 8
        LR = 0.0005
        for col_ind, sensor_col in enumerate(sensor_cols):
            mean_std_txt = './ende_{}_mean_std.txt'.format(sensor_col)
            X = generate_data(datas, sensor_col, oven_train, mean_std_txt, flag='train')
            x_test = generate_data(datas, sensor_col, oven_test, mean_std_txt, flag='test')
            # assert oven_train.shape[0] == X.shape[0]
            # assert oven_test.shape[0] == x_test.shape[0]
            EPOCH = epochs[col_ind]
            x_encoded_train, x_encoded_test = en_decoder(hidden1, embed_len, EPOCH, BATCH_SIZE, LR, X, x_test, sensor_col)
            if col_ind == 0:
                all_x_encoded_train = x_encoded_train
                all_x_encoded_test = x_encoded_test
            else:
                all_x_encoded_train = torch.cat((all_x_encoded_train, x_encoded_train), 1)
                all_x_encoded_test = torch.cat((all_x_encoded_test, x_encoded_test), 1)
        all_x_encoded_train = all_x_encoded_train.detach().numpy()
        all_x_encoded_test = all_x_encoded_test.detach().numpy()

        # all_train_oven_feature
        all_train_oven_feature = dict()
        for ind, train_o in enumerate(oven_train):
            try:
                all_train_oven_feature[train_o] = all_x_encoded_train[ind].tolist()
            except:
                continue
        # all_test_oven_feature
        all_test_oven_feature = dict()
        for ind, test_o in enumerate(oven_test):
            try:
                all_test_oven_feature[test_o] = all_x_encoded_test[ind].tolist()
            except:
                continue

        data = json.dumps(all_train_oven_feature)
        with open(train_feature_js, 'w') as js_file:
            js_file.write(data)

        data = json.dumps(all_test_oven_feature)
        with open(test_feature_js, 'w') as js_file:
            js_file.write(data)

    #2. [all_data_size, 8x8]: 64维压缩至8维  en_decoder
    # all_x_encoded = np.load(r'./168dims_sensor_feature.npy')
    # hidden1 = 10
    # embed_len = 8
    # EPOCH = 1500
    # BATCH_SIZE = 16
    # LR = 0.001
    # print("8x8 dims sensor_feature -> 8 dims sensor_feature..")
    # slimed_sensor_feature = slim_en_decoder(hidden1, embed_len, EPOCH, BATCH_SIZE, LR, all_x_encoded).detach().numpy()
    # np.save(r'./8dims_sensor_feature.npy', slimed_sensor_feature)

