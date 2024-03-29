# coding=utf-8
'''
en_decoder 思路, 压缩再恢复时序数据, 取压缩后的小维度特征.
但没用上lstm的思想..

'''
import os.path
import torch
import torchvision
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from data_load import DataLoader
from sklearn.model_selection import train_test_split as TTS
import time
import numpy as np
import pandas as pd

torch.manual_seed(1)


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


def generate_data(datas, sen_col_name):
    if os.path.exists(sen_col_name+'.npy'):
        X = np.load(sen_col_name+'.npy')
    else:
        X = []
        OvenNos = []
        max_lens = 0
        for data in datas:
            ll = len(data)
            for i in range(ll-1):
                sensor_data_path = data.iloc[i]['path_sensor_data']
                try:
                    sensor_data = pd.read_csv(sensor_data_path[:-3] + 'CSV')[sen_col_name]
                    sensor_data = pd.read_csv(sensor_data_path[:-3] + 'csv')[sen_col_name]
                    max_lens = max_lens if max_lens > len(sensor_data) else len(sensor_data)
                except:
                    continue

        for data in datas:
            ll = len(data)
            for i in range(ll-1):
                sensor_data_path = data.iloc[i]['path_sensor_data']
                OvenNo = data.iloc[i]['OvenNo']
                try:
                    sensor_data = pd.read_csv(sensor_data_path[:-3] + 'CSV')[sen_col_name]
                    sensor_data = pd.read_csv(sensor_data_path[:-3] + 'csv')[sen_col_name]
                    sensor_data = [a for a in sensor_data]
                    OvenNos.append(OvenNo)
                except:
                    continue
                sensor_data_max_len = sensor_data + [0]*(max_lens-len(sensor_data))
                X.append(sensor_data_max_len)
        # X normalize
        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        assert len(x_mean) == max_lens
        X = [[(x[k] - x_mean[k]) / x_std[k] for k in range(len(x_std))] for x in X]
        X = np.array(X)
        # 处理部分nan数据, 置0
        X[np.isnan(X)] = 0
        np.save(sen_col_name+'.npy', X)
        np.save(sen_col_name+'_ovens.npy', np.array(OvenNos))

    return X



def en_decoder(hidden1, embed_len, EPOCH, BATCH_SIZE, LR, X, sensor_col):
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
        if (epoch + 1) % 500 == 0 or (epoch == EPOCH-1):
            plt.plot([a for a in range(len(train_loss))], train_loss, label='train_loss')
            plt.legend()
            plt.show()
            plt.plot([a for a in range(len(val_loss))], val_loss, label='val_loss')
            plt.legend()
            plt.show()
        # 最后一个epoch, 把sensor_col的embedding后的向量输出
        if epoch == EPOCH - 1:
            for step, (x, _) in enumerate(all_loader):
                b_x = x.view(-1, x.shape[1])
                encoded, _ = Coder(b_x)
                torch.save(Coder, './{}.pkl'.format(sensor_col))
                return encoded


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
            plt.legend()
            plt.show()
            plt.plot([a for a in range(len(val_loss))], val_loss, label='val_loss')
            plt.legend()
            plt.show()


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


def sampling_for_each_sensor_col(j):
    # 获取到了每个step段数据, [::x]去采样
    sample_step = [50, 10, 15, 12, 35, 12, 40]
    # 每个step段内, 采样20个点, 拼接成20x8=160长度的数据列
    # 1.可以根据处理好的data.csv中的 'Step_start_timestamp'获取每个step的起始时间,
    # 2.也可以根据 'ACT_O1_QCMS_THICKNESS'列膜厚值到0的突变时间, 来获取每个step的起始时间.. [使用2.]
    data1 = pd.read_csv(r'./0910_cx_data.csv')
    sensor_path = data1.iloc[j]['path_sensor_data']
    sensor_data = pd.read_csv(sensor_path)
    thick_col = sensor_data['ACT_O1_QCMS_THICKNESS']
    sensor_col = [a for a in sensor_data['STAT_LT_CRYSTAL_CH1']]
    step_inds = get_step_index(thick_col)
    assert len(step_inds) == 6
    # 获取7个step段
    step_datas = []
    for i in range(len(step_inds)):
        if i == 0:
            step_datas.append(sensor_col[: step_inds[i] + 1])
        else:
            step_datas.append(sensor_col[step_inds[i-1]+1: step_inds[i]+1])
    step_datas.append(sensor_col[step_inds[-1] + 1: ])

    assert len(step_datas) == 7
    sampled_sensor_col = []
    for i in range(len(step_datas)):
        sampled_sensor_col.extend(step_datas[i][::sample_step[i]])

    return sampled_sensor_col




if __name__ == '__main__':

    data1 = r'./0910_cx_data.csv'
    data2 = r'./0917_cx_data.csv'
    data3 = r'./1008_cx_data.csv'
    data4 = r'./1021_cx_data.csv'
    x_npy_name = './all_sensor_data.npy'
    datas = [pd.read_csv(data1), pd.read_csv(data2), pd.read_csv(data3), pd.read_csv(data4)]

    sensor_cols = ["ACT_V1_IONIVAC_CH", "ACT_V1_PENNINGVAC_CH", "ACT_V1_THERMOVAC_CH", "ACT_V1_THERMOVAC_PREVLINE",
                   "ACT_V1_THERMOVAC_HP", "ACT_V1_THERMOVAC_HP2", "ACT_V1_PRESSURE_CH", "AI_V1_POLYCOLD_TEMP",
                   "ACTN_F1_FLOW1", "ACT_F1_FLOW1", "STAT_LT_CRYSTAL_CH1", "ACT_HEATER2_TEMPERATURE", "ACT_Q10_CURRENT_ANODE",
                   "ACT_Q10_VOLTAGE_ANODE", "ACT_Q10_CURRENT_CATHODE", "ACT_Q10_VOLTAGE_CATHODE", "ACT_Q10_CURRENT_NEUTRAL",
                   "ACT_Q10_ION_FLOW1", "ACT_Q10_ION_FLOW2", "STA_Q10_IONSOURCE_SHUTTER_IOP", "ACT_V1_MEISSNER_POLYCOLDTEMP"]

    # all_x_encoded = None
    # epochs = [1200, 100] + [1500]*19
    # # epochs = [1200, 100, 1500, 1500, 1500, 100, 1500, 1500, 1500, ]
    # for col_ind, sensor_col in enumerate(sensor_cols):
    #     # 1. 提取各个sensor_col的 8dims_feature 并 concate  21x8=168dims
    #     X = generate_data(datas, sensor_col)
    #     ovens = np.load(sensor_col + '_ovens.npy')
    #     hidden1 = 16
    #     embed_len = 8
    #     EPOCH = epochs[col_ind]
    #     BATCH_SIZE = 8
    #     LR = 0.0005
    #     print("En_Decoder: {}, data_size: {}, Ovens_size: {}".format(sensor_col, X.shape, ovens.shape))
    #     X = np.load(sensor_col + '.npy')
    #     x_encoded = en_decoder(hidden1, embed_len, EPOCH, BATCH_SIZE, LR,
    #                            X, sensor_col)
    #     if col_ind == 0:
    #         all_x_encoded = x_encoded
    #     else:
    #         all_x_encoded = torch.cat((all_x_encoded, x_encoded), 1)
    # all_x_encoded = all_x_encoded.detach().numpy()
    # np.save(r'./168dims_sensor_feature.npy', all_x_encoded)

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

    sampling_for_each_sensor_col(33)

