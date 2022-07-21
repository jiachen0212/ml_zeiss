# coding=utf-8
'''
return dict: oven: tsf_feature
second run

'''
import os
import numpy as np
import pandas as pd
import tsfresh as tsf
import json
import math


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
    step_len = len(step_datas)
    sampled_sensor_col = []
    for i in range(step_len):
        tmp_step_data = step_datas[i]
        step = math.floor(len(tmp_step_data) / each_step_nums[i])
        if step < 1:
            step_data = tmp_step_data
        else:
            step_data = tmp_step_data[::step][:each_step_nums[i]]
        sampled_sensor_col.extend(step_data)
    # assert len(sampled_sensor_col) == np.sum(each_step_nums)
    # if len(sampled_sensor_col) < np.sum(each_step_nums):
    #     print(len(sampled_sensor_col))

    return sampled_sensor_col


def tsf_all_in(single_col_data):
    x = single_col_data
    ts = pd.Series(x)

    #1. 平方和: 绝对能量值
    # ae1 = tsf.feature_extraction.feature_calculators.abs_energy(ts)

    #2. 描述时序数据相邻观测值之间的绝对波动情况
    ae2 = tsf.feature_extraction.feature_calculators.absolute_sum_of_changes(ts)

    #3. 统计时序数据各界差分间的聚合特征，依据不同的聚合函数，刻画时序数据的不同特征
    param = [{'f_agg': 'mean', 'maxlag': 2}]
    ae3 = tsf.feature_extraction.feature_calculators.agg_autocorrelation(ts, param)

    # 5. 时序数据的周期不可预测波动性
    ae5 = tsf.feature_extraction.feature_calculators.approximate_entropy(ts, 10, 0.1)

    #6. 自回归系数
    param = [{'coeff': 0, 'k': 10}]
    ae6 = tsf.feature_extraction.feature_calculators.ar_coefficient(ts, param)

    #7. 时序平稳性
    param = [{'attr': 'pvalue'}]
    ae7 = tsf.feature_extraction.feature_calculators.augmented_dickey_fuller(ts, param)

    #8.
    ae8 = tsf.feature_extraction.feature_calculators.autocorrelation(ts, 2)

    #9. 非线性度
    ae9 = tsf.feature_extraction.feature_calculators.c3(ts, 2)

    #10. 时序数据复杂度
    ae10 = tsf.feature_extraction.feature_calculators.cid_ce(ts, True)

    #11.
    ae11 = tsf.feature_extraction.feature_calculators.first_location_of_maximum(ts)

    #12.
    ae12 = tsf.feature_extraction.feature_calculators.first_location_of_minimum(ts)

    #13.
    ae13 = tsf.feature_extraction.feature_calculators.kurtosis(ts)

    #14.
    ae14 = tsf.feature_extraction.feature_calculators.large_standard_deviation(ts, 0.2)
    ae14 = 1.0 if ae14 == True else 0.0

    #15.
    ae15 = tsf.feature_extraction.feature_calculators.last_location_of_maximum(ts)

    #16.
    ae16 = tsf.feature_extraction.feature_calculators.last_location_of_minimum(ts)

    #17.
    ae17 = tsf.feature_extraction.feature_calculators.longest_strike_above_mean(ts)

    #18.
    ae18 = tsf.feature_extraction.feature_calculators.longest_strike_below_mean(ts)

    return [ae2, ae3[0][1], ae5, ae6[0][1], ae7[0][1], ae8, ae9, ae10, ae11, ae12, ae13, ae14, ae15, ae16, ae17, ae18]


def get_tsf_features(data, sensor_cols):
    all_sensor_tsf_features = []
    for col_name in sensor_cols:
        single_col = sampling_for_each_sensor_col(data, col_name)
        tsf_features = tsf_all_in(single_col)
        all_sensor_tsf_features.extend([float(a) for a in tsf_features])
    assert len(all_sensor_tsf_features) == 16*len(sensor_cols)

    return all_sensor_tsf_features


def get_tsf_feature(datas, sensor_cols, tsf_feature_jsname):

    tsf_features = dict()
    for data in datas:
        ll = len(data)
        for i in range(ll):
            path = data.iloc[i]['path_sensor_data']
            oven = str(data.iloc[i]['OvenNo'])
            try:
                sensor_data = pd.read_csv(path)
            except:
                continue
            single_sensor_tdf_features = get_tsf_features(sensor_data, sensor_cols)
            tsf_features[oven] = single_sensor_tdf_features

    data = json.dumps(tsf_features)
    with open(tsf_feature_jsname, 'w') as js_file:
        js_file.write(data)

    return tsf_features


def split_train_test_feature():
    oven_train = np.load('./data1_5_oven_train.npy')
    oven_test = np.load('./data6_oven_test.npy')
    data1_6_feature = json.load(open(r'./data1_6_tsf_features_all.json', 'r'))
    data1_5_train_feature = dict()
    data6_test_feature = dict()
    for oven, feature in data1_6_feature.items():
        if oven in oven_train:
            data1_5_train_feature[oven] = feature
        elif oven in oven_test:
            data6_test_feature[oven] = feature
    data = json.dumps(data1_5_train_feature)
    with open(r'./data1_5_train_tsf_feature.json', 'w') as js_file:
        js_file.write(data)
    data = json.dumps(data6_test_feature)
    with open(r'./data6_test_tsf_feature.json', 'w') as js_file:
        js_file.write(data)


def generate_tsf_feature(train_csv, test_csv, tsf_train_feature, tsf_test_feature):

    sensor_cols = ["ACT_V1_IONIVAC_CH", "ACT_V1_PENNINGVAC_CH", "ACT_V1_THERMOVAC_CH", "ACT_V1_THERMOVAC_PREVLINE", "ACT_V1_THERMOVAC_HP", "ACT_V1_THERMOVAC_HP2", "ACT_V1_PRESSURE_CH", "AI_V1_POLYCOLD_TEMP", "ACTN_F1_FLOW1", "ACT_F1_FLOW1", "STAT_LT_CRYSTAL_CH1", "ACT_HEATER2_TEMPERATURE", "ACT_Q10_CURRENT_ANODE", "ACT_Q10_VOLTAGE_ANODE", "ACT_Q10_CURRENT_CATHODE", "ACT_Q10_VOLTAGE_CATHODE", "ACT_Q10_CURRENT_NEUTRAL", "ACT_Q10_ION_FLOW1", "ACT_Q10_ION_FLOW2", "STA_Q10_IONSOURCE_SHUTTER_IOP", "ACT_V1_MEISSNER_POLYCOLDTEMP"]
    thickness_cols = ["ACT_O1_QCMS_THICKNESS", "ACT_O1_QCMS_RATE", "ACT_O1_QCMS_THICKNESS_CH1", "ACT_O1_QCMS_RATE_CH1"]
    # sensor_cols = thickness_cols
    _ = get_tsf_feature([train_csv], sensor_cols, tsf_train_feature)
    _ = get_tsf_feature([test_csv], sensor_cols, tsf_test_feature)
    
    print('sensor.csv tsf feature generate done. ~')

    # split_train_test_feature()


if __name__ == '__main__':

    all_train_data, online_test_data = pd.read_csv('./till_0721_train.csv', encoding='utf-8'), pd.read_csv('./till_0721_test.csv', encoding='utf-8')
    tsf_train_feature, tsf_test_feature = './tsf_train_feature.json', './tsf_test_feature.json'
    generate_tsf_feature(all_train_data, online_test_data, tsf_train_feature, tsf_test_feature)







