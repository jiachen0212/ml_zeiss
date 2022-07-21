# coding=utf-8
'''
first run
'''
import pandas as pd
import os
import json
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split as TTS
import math
from keras import backend as K
from keras.models import load_model


def build_model(embedd_ed_dim, sqe_len):
    # define model
    model = Sequential()
    # batch_size, sqe_len, 1
    model.add(LSTM(embedd_ed_dim, activation='relu', input_shape=(sqe_len, 1)))
    model.add(RepeatVector(sqe_len))
    model.add(LSTM(embedd_ed_dim, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    # 打印网络信息
    # model.summary()

    return model


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


def sampling_for_each_sensor_col(data1, j, col_name):
    # 每个step段的长度约: [690, 90, 130, 100, 390, 160, 340], 每段采样的个数: [10, 6, 8, 12, 16, 14, 10]
    each_step_nums = [10, 6, 8, 12, 16, 14, 10]
    # 1.可以根据处理好的data.csv中的 'Step_start_timestamp'获取每个step的起始时间
    # 2.也可以根据 'ACT_O1_QCMS_THICKNESS'列膜厚值到0的突变时间, 来获取每个step的起始时间.. [使用2.]
    sensor_path = data1.iloc[j]['path_sensor_data']
    try:
        sensor_data = pd.read_csv(sensor_path)
    except:
        return []
    thick_col = sensor_data['ACT_O1_QCMS_THICKNESS']
    sensor_col = [a for a in sensor_data[col_name]]
    # plt.plot([i for i in range(len(sensor_col))], sensor_col)
    # plt.show()
    step_inds = get_step_index(thick_col)
    assert len(step_inds) >= 6

    # 获取7个step段
    step_datas = []
    for i in range(6):
        if i == 0:
            step_datas.append(sensor_col[: step_inds[i] + 1])
        else:
            step_datas.append(sensor_col[step_inds[i-1]+1: step_inds[i]+1])
    step_datas.append(sensor_col[step_inds[-1] + 1: ])

    assert len(step_datas) == 7
    sampled_sensor_col = []
    for i in range(len(step_datas)):
        tmp_step_data = step_datas[i]
        step = math.floor(len(tmp_step_data) / each_step_nums[i])
        step_data = tmp_step_data[::step][:each_step_nums[i]]
        sampled_sensor_col.extend(step_data)
    assert len(sampled_sensor_col) == np.sum(each_step_nums)

    return sampled_sensor_col


def bad_Ovens(datas):
    bads = []
    for data in datas:
        ll = len(data)
        for i in range(ll-1):
            OvenNo = data.iloc[i]['OvenNo']
            path = data.iloc[i]['path_sensor_data']
            try:
                _ = pd.read_csv(path)
                bads.append(OvenNo)
            except:
                continue
    return bads


import collections
def slim_ovens_x(ovens, X):
    distribute = collections.Counter(ovens)
    slime_ovens = list(distribute.keys())
    indexs = [ovens.index(a) for a in slime_ovens]

    return [X[ind] for ind in indexs], slime_ovens


def generate_data(col_name, datas, std_mean_txt, ovens, flag=None):
    X = []
    for data in datas:
        ll = len(data)
        for i in range(ll):
            oven = str(data.iloc[i]['OvenNo'])
            if oven in ovens:
                _data = sampling_for_each_sensor_col(data, i, col_name)
                if len(_data) > 0:
                    X.append(_data)
    mean_, std_ = [], []
    if flag == "train":
        ff = open(std_mean_txt, 'w')
        mean_ = np.mean(X, axis=0)
        std_ = np.std(X, axis=0)
        std_ = [a+1e-6 for a in std_]
        for x_mean_ in mean_:
            ff.write(str(x_mean_) + ',')
        ff.write('\n')
        for x_std_ in std_:
            ff.write(str(x_std_) + ',')
        ff.write('\n')
    elif flag == 'test':
        ff = open(std_mean_txt, 'r').readlines()
        mean_ = ff[0].split(',')[:-1]
        mean_ = [float(a) for a in mean_]
        std_ = ff[1].split(',')[:-1]
        std_ = [float(a) for a in std_]
    X = [[(x[k] - mean_[k]) / std_[k] for k in range(len(std_))] for x in X]
    X = np.array(X)

    return X


def lr_schedule(epoch):
    """
    根据epoch来调整学习率
    :param epoch:
    :return:
    """
    learning_rate = 0.001
    if epoch > 100:
        learning_rate *= 0.5
    elif epoch > 200:
        learning_rate *= 0.1
    elif epoch > 300:
        learning_rate = 0.05

    return learning_rate


def lstm_col_feature_list(sensor_cols, datas, embedd_ed_dim, ok_ovens, seed, oven_train, oven_test):
    sqe_len = 76

    for col_name in sensor_cols:
        std_mean_txt = './{}_mean_std.txt'.format(col_name)
        X_train = generate_data(col_name, datas, std_mean_txt, oven_train, flag='train')
        X_test = generate_data(col_name, datas, std_mean_txt, oven_test, flag='test')
        # assert X_test.shape[0] == len(oven_test)
        # assert X_train.shape[0] == len(oven_train)
        # X_train, X_test, _, _ = TTS(X, X, test_size=0.25, random_state=seed)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], -1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], -1))

        # build_model
        epochs = 300
        model = build_model(embedd_ed_dim, sqe_len)

        # 创建学习率调整回调
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        early_stop = EarlyStopping(monitor='val_loss', patience=20)
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, X_train,
                  epochs=epochs,
                  # validation_split=0.3,
                  validation_split=0.001,
                  validation_freq=50,
                  callbacks=[lr_callback, early_stop])

        # save model
        # model.save('tf_model_{}.h5'.format(col_name))

        # model predicate
        get_3rd_layer_output = K.function([model.layers[0].input],
                                          [model.layers[0].output])
        train_feature = get_3rd_layer_output([X_train])[0]
        test_feature = get_3rd_layer_output([X_test])[0]

        np.save(r'./lstm_train_{}.npy'.format(col_name), train_feature)
        np.save(r'./lstm_test_{}.npy'.format(col_name), test_feature)



def lstm_feature(datas, sensor_cols, seed):
    ok_ovens = None
    oven_train, oven_test = [], []
    for data in datas[:-1]:
        ll = len(data)
        for i in range(ll):
            oven = str(data.iloc[i]['OvenNo'])
            oven_train.append(oven)
    np.save('./train.npy', np.array(oven_train))
    ll = len(datas[-1])
    for i in range(ll):
        oven = str(datas[-1].iloc[i]['OvenNo'])
        oven_test.append(oven)
    np.save('./test.npy', np.array(oven_test))


    # 1. train model and get tf_features
    embedd_ed_dim = 18
    lstm_col_feature_list(sensor_cols, datas, embedd_ed_dim, ok_ovens, seed, oven_train, oven_test)

    # 2. concate lstm_features
    all_train_features = None
    all_test_features = None
    for col_ind, col_name in enumerate(sensor_cols):
        train_f = np.load(r'./lstm_train_{}.npy'.format(col_name))
        test_f = np.load(r'./lstm_test_{}.npy'.format(col_name))

        if col_ind == 0:
            all_train_features = train_f
            all_test_features = test_f
        else:
            all_train_features = np.concatenate((all_train_features, train_f), axis=1)
            all_test_features = np.concatenate((all_test_features, test_f), axis=1)

    # all_train_oven_feature
    all_train_oven_feature = dict()
    for ind, train_o in enumerate(oven_train):
        try:
            all_train_oven_feature[train_o] = all_train_features[ind].tolist()
        except:
            continue

    # all_test_oven_feature
    all_test_oven_feature = dict()
    for ind, test_o in enumerate(oven_test):
        try:
            all_test_oven_feature[test_o] = all_test_features[ind].tolist()
        except:
            continue

    data = json.dumps(all_train_oven_feature)
    with open(r'./lstm_train_features_seed{}.json'.format(seed), 'w') as js_file:
        js_file.write(data)

    data = json.dumps(all_test_oven_feature)
    with open(r'./lstm_test_features_seed{}.json'.format(seed), 'w') as js_file:
        js_file.write(data)

    return all_train_oven_feature, all_test_oven_feature


if __name__ == '__main__':

    train_csv = pd.read_csv(r'./till_0516_train.csv', encoding='utf-8')
    test_csv = pd.read_csv(r'./till_0516_test.csv', encoding='utf-8')

    all_data = [train_csv, test_csv]

    sensor_cols = ["ACT_V1_IONIVAC_CH", "ACT_V1_PENNINGVAC_CH", "ACT_V1_THERMOVAC_CH", "ACT_V1_THERMOVAC_PREVLINE", "ACT_V1_THERMOVAC_HP", "ACT_V1_THERMOVAC_HP2", "ACT_V1_PRESSURE_CH", "AI_V1_POLYCOLD_TEMP", "ACTN_F1_FLOW1", "ACT_F1_FLOW1", "STAT_LT_CRYSTAL_CH1", "ACT_HEATER2_TEMPERATURE", "ACT_Q10_CURRENT_ANODE", "ACT_Q10_VOLTAGE_ANODE", "ACT_Q10_CURRENT_CATHODE", "ACT_Q10_VOLTAGE_CATHODE", "ACT_Q10_CURRENT_NEUTRAL", "ACT_Q10_ION_FLOW1", "ACT_Q10_ION_FLOW2", "STA_Q10_IONSOURCE_SHUTTER_IOP", "ACT_V1_MEISSNER_POLYCOLDTEMP"]
    # thickness_cols = ["ACT_O1_QCMS_THICKNESS", "ACT_O1_QCMS_RATE", "ACT_O1_QCMS_THICKNESS_CH1", "ACT_O1_QCMS_RATE_CH1"]
    # sensor_cols += thickness_cols

    # lstm的模型就固定用seed=66的训测数据做 lstm的模型就这么定, 后面来了试数据再直接run模型得到lstm特征
    # seeds = [i + 33 for i in range(5)]
    seeds = [33]
    for seed in seeds:
        all_train_oven_feature, all_test_oven_feature = lstm_feature(all_data, sensor_cols, seed)