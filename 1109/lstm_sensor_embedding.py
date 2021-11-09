# coding=utf-8
import pandas as pd
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split as TTS
import math
from keras import backend as K

'''
# 这里创建了一个多输出的神经网络，完成了一个多输出回归的任务，然后使用decoder1来重构，decoder2来预测，据说能够提高精度.
# lstm autoencoder reconstruct and predict sequence
from numpy import array
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

# define input sequence
seq_in = array([i for i in range(40)])
# reshape input into [samples, timesteps, features]
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))
# prepare output sequence
seq_out = seq_in[:, 1:, :]
n_out = n_in - 1
# define encoder
visible = Input(shape=(n_in,1))
encoder = LSTM(100, activation='relu')(visible)
# define reconstruct decoder
decoder1 = RepeatVector(n_in)(encoder)
decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(1))(decoder1)
# define predict decoder
decoder2 = RepeatVector(n_out)(encoder)
decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
decoder2 = TimeDistributed(Dense(1))(decoder2)
# tie it together
model = Model(inputs=visible, outputs=[decoder1, decoder2])
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(seq_in, [seq_in,seq_out], epochs=300, verbose=0)
# demonstrate prediction
yhat = model.predict(seq_in, verbose=0)
print(yhat)

'''

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
                sensor_data = pd.read_csv(path)
                bads.append(OvenNo)
            except:
                continue
    return bads

def generate_data(col_name, datas, std_mean_txt):
    # generate or load .npy data
    if os.path.exists(r'./tf_X_{}.npy'.format(col_name)):
        return np.load(r'./tf_X_{}.npy'.format(col_name))
    else:
        OvenNos = []
        ff = open(std_mean_txt, 'w')
        X = []
        bads = bad_Ovens(datas)
        for data in datas:
            ll = len(data)
            for i in range(ll):
                oven = data.iloc[i]['OvenNo']
                if data.iloc[i]['OvenNo'] in bads:
                    test_data = sampling_for_each_sensor_col(data, i, col_name)
                    if len(test_data) != 0:
                        X.append(test_data)
                        OvenNos.append(oven)
        # data normalization
        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        x_std = [a+1e-6 for a in x_std]
        for x_mean_ in x_mean:
            ff.write(str(x_mean_) + ',')
        ff.write('\n')
        for x_std_ in x_std:
            ff.write(str(x_std_) + ',')
        ff.write('\n')

        X = [[(x[k] - x_mean[k]) / x_std[k] for k in range(len(x_std))] for x in X]
        X = np.array(X)
        print("{} normalized data has NaN: {}".format(col_name, np.isnan(X).any()))

        np.save(r'./tf_X_{}.npy'.format(col_name), X)

        # all Ovennos
        np.save(r"./Ovens.npy", np.array(OvenNos))
        return X


# random test
# import random
# test_data = [97+random.random() for i in range(77)]
# class MyCallback(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []
#         return
#
#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
#         return


def test_eval(model, col_name, X_test, y_test, std_mean_txt, yiliu_sensor_col):
    # test data eval
    test_res = model.predict(X_test)  # batch 76, 1
    test_pred = np.squeeze(test_res).tolist()
    test_real = np.squeeze(y_test).tolist()
    normalized_mse = mean_squared_error(test_pred, test_real)

    # de_normalization
    ff = open(std_mean_txt, 'r').readlines()
    mean_ = [float(a) for a in ff[0].split(',')[:-1]]
    std_ = [float(b) for b in ff[1].split(',')[:-1]]
    test_real = [[a[i] * std_[i] + mean_[i] for i in range(len(std_))] for a in test_real]
    test_pred = [[a[i] * std_[i] + mean_[i] for i in range(len(std_))] for a in test_pred]
    denormalized_mse = mean_squared_error(test_pred, test_real)

    print("{} normalized_mse: {}, de_normalized_mse: {}".format(col_name, normalized_mse, denormalized_mse))
    # print("real test: {}".format(test_real[0]))
    # print("pred test: {}".format(test_pred[0]))

    if denormalized_mse > 1:
        info = "{} mse: {}".format(col_name, denormalized_mse)
        yiliu_sensor_col.write(info + '\n')



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


def train_tf_lstm_to_get_col_feature_list(sensor_cols, datas, embedd_ed_dim):
    # 一些col的采样后的76个序列数据的mean std 特别大, 比较异常.
    yiliu_sensor_col = open(r'./yichang_col.txt', 'w')

    for col_name in sensor_cols:
        std_mean_txt = './{}_mean_std.txt'.format(col_name)
        X = generate_data(col_name, datas, std_mean_txt)
        data_size, sqe_len = X.shape[0], X.shape[1]
        X_ = X.reshape((data_size, sqe_len, -1))
        print(X_.shape)

        # 搭建 model
        epochs = 500
        model = build_model(embedd_ed_dim, sqe_len)
        X_train, X_test, y_train, y_test = TTS(X_, X_, test_size=0.2, random_state=66)

        # cb = MyCallback()
        # model.fit(X_train, y_train, epochs=epochs, batch_size=16,
        #           validation_data=(X_test, y_test), shuffle=True,
        #           callbacks=[cb], verbose=0)

        # 创建学习率调整回调
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        early_stop = EarlyStopping(monitor='val_loss', patience=20)
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_, X_,
                  epochs=epochs,
                  validation_split=0.3,
                  validation_freq=50,
                  callbacks=[lr_callback, early_stop])

        # only test
        test_eval(model, col_name, X_test, y_test, std_mean_txt, yiliu_sensor_col)

        # model predicate
        get_3rd_layer_output = K.function([model.layers[0].input],
                                          [model.layers[0].output])
        layer_output = get_3rd_layer_output([X])[0]
        print("{} tf_8dims_feature: {}".format(col_name, layer_output[6]))
        np.save(r'./tf_feature_{}.npy'.format(col_name), layer_output)


if __name__ == '__main__':

    data1 = r'./0910_cx_data.csv'
    data2 = r'./0917_cx_data.csv'
    data3 = r'./1008_cx_data.csv'
    data4 = r'./1021_cx_data.csv'
    data5 = r'./1105_cx_data.csv'
    datas = [pd.read_csv(data1), pd.read_csv(data2), pd.read_csv(data3), pd.read_csv(data4), pd.read_csv(data5)]

    sensor_cols = ["ACT_V1_THERMOVAC_PREVLINE", "ACT_V1_THERMOVAC_CH", "ACT_V1_IONIVAC_CH", "ACT_V1_PENNINGVAC_CH",
                   "ACT_V1_THERMOVAC_HP", "ACT_V1_THERMOVAC_HP2", "ACT_V1_PRESSURE_CH", "AI_V1_POLYCOLD_TEMP",
                   "ACTN_F1_FLOW1", "ACT_F1_FLOW1", "STAT_LT_CRYSTAL_CH1", "ACT_HEATER2_TEMPERATURE","ACT_Q10_CURRENT_ANODE",
                   "ACT_Q10_VOLTAGE_ANODE", "ACT_Q10_CURRENT_CATHODE", "ACT_Q10_VOLTAGE_CATHODE", "ACT_Q10_CURRENT_NEUTRAL",
                   "ACT_Q10_ION_FLOW1", "ACT_Q10_ION_FLOW2", "ACT_O1_QCMS_RATE", "STA_Q10_IONSOURCE_SHUTTER_IOP", "ACT_V1_MEISSNER_POLYCOLDTEMP"]

    #1. train model and get tf_features
    embedd_ed_dim = 18
    train_tf_lstm_to_get_col_feature_list(sensor_cols, datas, embedd_ed_dim)

    #2. concate tf_features
    all_tf_features = None
    for col_ind, col_name in enumerate(sensor_cols):
        cur_tf_feature = np.load(r'./tf_feature_{}.npy'.format(col_name))
        if col_ind == 0:
            all_tf_features = cur_tf_feature
        else:
            all_tf_features = np.concatenate((all_tf_features, cur_tf_feature), axis=1)
    np.save(r'./all_tf_sensor_features.npy', all_tf_features)