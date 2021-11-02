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


class AutoEncoder(nn.Module):
    def __init__(self, sqe_len, hidden1, embed_len):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(sqe_len, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, embed_len),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_len, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, sqe_len),
            nn.Sigmoid()
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
        max_lens = 0
        for data in datas:
            ll = len(data)
            for i in range(ll-1):
                sensor_file_name1, sensor_file_name2 = data.iloc[i]['FileID'][3:], data.iloc[i + 1]['FileID'][3:]
                if int(sensor_file_name2) - int(sensor_file_name1) <= 3:
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
                sensor_file_name1, sensor_file_name2 = data.iloc[i]['FileID'][3:], data.iloc[i + 1]['FileID'][3:]
                # 连续炉号筛选
                if int(sensor_file_name2) - int(sensor_file_name1) <= 3:
                    sensor_data_path = data.iloc[i]['path_sensor_data']
                    try:
                        sensor_data = pd.read_csv(sensor_data_path[:-3] + 'CSV')[sen_col_name]
                        sensor_data = pd.read_csv(sensor_data_path[:-3] + 'csv')[sen_col_name]
                        sensor_data = [a for a in sensor_data]
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

    return X


def en_decoder_test(trainable_sensor_col, hidden1, embed_len, EPOCH, BATCH_SIZE, LR, X_train, X_test, y_train, y_test, sensor_col):
    train_loader = DataLoader((X_train, y_train), batch_size=BATCH_SIZE, batch_first=False, device='cpu')
    val_loader = DataLoader((X_test, y_test), batch_size=BATCH_SIZE, batch_first=False, device='cpu')

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
            # if epoch == EPOCH - 1:
            #     print(encoded)
        loss_value /= len(train_loader)
        print('Epoch :', epoch, '|', 'train_loss:%.4f' % loss_value)
        if np.isnan(loss_value):
            return None
        train_loss.append(loss_value)
        if (epoch + 1) % 100 == 0:
            # test val loss
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
        # if (epoch + 1) % 200 == 0:
        #     plt.plot([a for a in range(len(train_loss))], train_loss, label='train_loss')
        #     plt.legend()
        #     plt.show()
        #     plt.plot([a for a in range(len(val_loss))], val_loss, label='val_loss')
        #     plt.legend()
        #     plt.show()
    trainable_sensor_col.write(sensor_col+',')

    return Coder


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
        if (epoch + 1) % 1000 == 0 or (epoch == EPOCH-1):
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


def exists_sensor_cols(datas, sensor_cols, slim_sensor_cols_txt):
    if os.path.exists(slim_sensor_cols_txt):
        slimed_sensor_cols = open(slim_sensor_cols_txt, 'r').readlines()[0].split(',')[:-1]
    else:
        each_sensor_counts = [0]*len(sensor_cols)
        for data in datas:
            ll = len(data)
            for i in range(ll):
                sensor_data_path = data.iloc[i]['path_sensor_data']
                for col_ind, sensor_col in enumerate(sensor_cols):
                    try:
                        sensor_data = pd.read_csv(sensor_data_path[:-3] + 'CSV')[sensor_col]
                        sensor_data = pd.read_csv(sensor_data_path[:-3] + 'csv')[sensor_col]
                        each_sensor_counts[col_ind] = each_sensor_counts[col_ind] + 1
                    except:
                        continue
        slimed_sensor_cols = []
        data_lens = max(each_sensor_counts)
        for col_ind, sensor_counts in enumerate(each_sensor_counts):
            if sensor_counts == data_lens:
                slimed_sensor_cols.append(sensor_cols[col_ind])
        # 落盘
        ff = open(slim_sensor_cols_txt, 'w')
        for col in slimed_sensor_cols:
            ff.write(col+',')

    return slimed_sensor_cols





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
    # 部分sensor.csv没有记录以上全部的21列重要意义数据列, 精简一下
    slim_sensor_cols_txt = r'./slim_sensor_cols.txt'
    slim_sensor_cols = exists_sensor_cols(datas, sensor_cols, slim_sensor_cols_txt)
    trainable_sensor_txt = './trainable_sensor_col.txt'

    if not os.path.exists(trainable_sensor_txt):
        trainable_sensor_col = open(trainable_sensor_txt, 'w')
        for sensor_col in slim_sensor_cols:
            X = generate_data(datas, sensor_col)
            hidden1 = 20
            embed_len = 8
            EPOCH = 50
            BATCH_SIZE = 16
            LR = 0.001
            X_train, X_test, y_train, y_test = TTS(X, X, test_size=0.3, random_state=33)
            en_decoder_model = en_decoder_test(trainable_sensor_col, hidden1, embed_len, EPOCH, BATCH_SIZE, LR, X_train, X_test, y_train, y_test, sensor_col)
    else:
        # 1. 提取各个sensor_col的 8dims_feature 并 concate  21x8=168dims
        sensor_cols = open(trainable_sensor_txt, 'r').readlines()[0].split(',')[:-1]
        assert len(sensor_cols) == len(slim_sensor_cols)
        hidden1 = 16
        embed_len = 8
        EPOCH = 1300
        BATCH_SIZE = 16
        LR = 0.001
        all_x_encoded = None
        for col_ind, sensor_col in enumerate(sensor_cols):
            print("En_Decoder: {}".format(sensor_col))
            X = np.load(sensor_col + '.npy')
            x_encoded = en_decoder(hidden1, embed_len, EPOCH, BATCH_SIZE, LR,
                                   X, sensor_col)
            if col_ind == 0:
                all_x_encoded = x_encoded
            else:
                all_x_encoded = torch.cat((all_x_encoded, x_encoded), 1)
        all_x_encoded = all_x_encoded.detach().numpy()
        np.save(r'./64dims_sensor_feature.npy', all_x_encoded)

        #2. [all_data_size, 8x8]: 64维压缩至8维  en_decoder
        # all_x_encoded = np.load(r'./64dims_sensor_feature.npy')
        # hidden1 = 10
        # embed_len = 8
        # EPOCH = 50
        # BATCH_SIZE = 16
        # LR = 0.001
        # print("8x8 dims sensor_feature -> 8 dims sensor_feature..")
        # slimed_sensor_feature = slim_en_decoder(hidden1, embed_len, EPOCH, BATCH_SIZE, LR, all_x_encoded).detach().numpy()
        # np.save(r'./8dims_sensor_feature.npy', slimed_sensor_feature)




