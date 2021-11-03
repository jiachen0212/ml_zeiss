# coding=utf-8
import pandas as pd
from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
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

def build_model(embedd_ed_dim, epochs):
    # define model
    model = Sequential()
    model.add(LSTM(embedd_ed_dim, activation='relu', input_shape=(n_in, 1)))
    model.add(RepeatVector(n_in))
    model.add(LSTM(embedd_ed_dim, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    # 打印网络信息
    # model.summary()

    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(sequence, sequence, epochs=epochs, verbose=0)

    # 获取lstm 的 encoder 结果
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[0].output])
    layer_output = get_3rd_layer_output([np.array([X[0]])])[0][0]  # len==8的list

    return model, layer_output


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


def generate_data(col_name, datas, std_mean_txt):
    ff = open(std_mean_txt, 'w')
    X = []
    for data in datas:
        ll = len(data)
        for i in range(ll):
            test_data = sampling_for_each_sensor_col(data, i, col_name)
            if len(test_data) != 0:
                X.append(test_data)
    # data normalization
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    for x_mean_ in x_mean:
        ff.write(str(x_mean_) + ',')
    ff.write('\n')
    for x_std_ in x_std:
        ff.write(str(x_std_) + ',')
    ff.write('\n')

    X = [[(x[k] - x_mean[k]) / x_std[k] for k in range(len(x_std))] for x in X]
    X = np.array(X)
    np.save(r'./tf_X.npy', X)

    return X


# random test
# import random
# test_data = [97+random.random() for i in range(77)]
# 训出nan结果, 跟序列长度有关系.. 长度>35貌似就不行了...

if __name__ == '__main__':

    data1 = r'./0910_cx_data.csv'
    data2 = r'./0917_cx_data.csv'
    data3 = r'./1008_cx_data.csv'
    data4 = r'./1021_cx_data.csv'
    datas = [pd.read_csv(data1), pd.read_csv(data2), pd.read_csv(data3), pd.read_csv(data4)]
    col_name = 'STAT_LT_CRYSTAL_CH1'
    std_mean_txt = './{}_mean_std.txt'.format(col_name)

    X = generate_data(col_name, datas, std_mean_txt)
    print(X.shape)

    # define input sequence
    sequence = array(X[0])
    # reshape input into [samples, timesteps, features]
    n_in = len(sequence)
    sequence = sequence.reshape((1, n_in, 1))

    # 搭建 model
    embedd_ed_dim = 8
    epochs = 300
    model, embedd_ed_res = build_model(embedd_ed_dim, epochs)

    # demonstrate recreation
    yhat = model.predict(sequence, verbose=0)

