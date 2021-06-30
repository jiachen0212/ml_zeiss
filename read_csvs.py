# coding=utf-8
import json
import os
import numpy as np
import pandas as pd
import tsfresh as tsf


def get_evt_sensor_dict(sensor_lines, process_lines):
    '''
    step_time = dict()   # process_name:step_name

    sensor.csv info:
    running_times = []
    sensor_values = []

    sensor_index_dict = dict()
    '''
    step_time = dict()
    for index, line in enumerate(process_lines):
        split_line = line.strip()
        if not split_line:
            continue
        spline = split_line[:-1].split(",")
        if index >= 5:
            if len(spline) > 2:
                # 这是个坑点, 01:04:37 统一写成 1:04:37
                if 'Start Step No.' in split_line:
                    if len(spline[0].split(':')[0]) > 1 and spline[0][0] == '0':
                        # process_time = spline[0].lstrip('0')
                        process_time = spline[0][1:]
                    else:
                        process_time = spline[0]
                    process_name = spline[2]
                    step_time[process_name] = process_time
        # 部分evt中有step no9,则不记录evt的最末行时间,不运行以下if.
        if len(step_time) == 8:
            if 'Total Process Time' in split_line:
                if len(spline[0].split(':')[0]) > 1 and spline[0][0] == '0':
                    process_time = spline[0][1:]
                else:
                    process_time = spline[0]
                process_name = spline[2]
                step_time[process_name] = process_time
    assert len(step_time) == 9

    running_times = []
    sensor_values = []
    sensor_names = []
    for index, line in enumerate(sensor_lines):
        split_line = line.strip()
        if not split_line:
            continue
        spline = split_line[:-1].split(',')
        if index == 0:
            # 所有sensor参数名
            sensor_names = spline[5:]
        else:
            # 同上step_time, 时间统一格式处理
            if len(spline[1].split(':')[0]) > 1 and spline[1][0] == '0':
                tmp = spline[1][1:]
            else:
                tmp = spline[1]
            running_times.append(tmp)
            sensor_values.append(spline[5:])
    return step_time, running_times, sensor_values, sensor_names


def get_time_sensor(evt_dict, running_times, sensor_values, evtname):
    '''
    evt_dict: time: process
    sensor.csv: running_times, sensor_values

    '''
    time_sensor = dict()
    evt_process_time = []
    for step_name, time in evt_dict.items():
        evt_process_time.append(time)
    lens = len(evt_process_time)   # lens == 9  step1~8 + end_line
    assert lens == 9
    for i in range(lens - 1):
        start, end = evt_process_time[i], evt_process_time[i + 1]
        sensor_start = running_times.index(start)
        try:
            sensor_end = running_times.index(end)
        except:
            # 最后evt中的时间点,sensor.csv中可能找不到,数据就直接读取到sensor.csv的最后一行
            sensor_end = len(running_times)

        # check time
        # print(dir_name, start, end)
        # print(running_times[sensor_start-1], running_times[sensor_start], \
        # running_times[sensor_end], running_times[sensor_end+1])
        cur_sensor_value = sensor_values[sensor_start:sensor_end]
        time_sensor[start] = cur_sensor_value
    # check
    # tmp = list(time_sensor.keys())[2]
    # print(dir_name, tmp, time_sensor[tmp][0])
    return time_sensor


def process_sensor(evt_dict, time_sensor):
    '''

    :param evt_dict:
    :param time_sensor:
    :return:
    time is the tmp_param
    '''
    evt_times = []
    for step_name, time in evt_dict.items():
        evt_times.append(time)
    # 根据evt的time,对应获取sensor_value
    process_sensor_dict = dict()
    for step_name, time in evt_dict.items():
        try:
            sensor_value_list = time_sensor[time]
        except:
            # print(dir_name, time)   # evt文件末尾的几行,无信息量
            continue
        process_sensor_dict[step_name] = sensor_value_list

    return process_sensor_dict


def get_info(path, miss_info):
    '''
    :param path: dir path
    :return: all_dir's process_sensor_value_dict. a list

    '''
    miss_dir = open(miss_info, 'w')
    all_process_sensor = dict()
    all_sensor_index_dict = dict()
    dir_names = os.listdir(path)
    for dir_name in dir_names:
        sensor_csv = os.path.join(path, dir_name, "{}.csv".format(dir_name))
        process_csv = os.path.join(path, dir_name, "evt{}.csv".format(dir_name))
        try:
            sensor_lines = open(sensor_csv, "r")
            process_lines = open(process_csv, "r")
        except:
            # print("{} loss evt or csv file".format(dir_name))
            # 落盘csv损失的文件夹名称至txt
            miss_dir.write(dir_name + '\n')
            continue
        # step1.
        # 为每一对csv文件,存储evt, sensor dict{}
        evt_dict, running_times, sensor_values, sensor_names = get_evt_sensor_dict(sensor_lines, process_lines,
                                                                                   dir_name)

        # step2.
        # 时间和sensor_value关联:
        # evt中最后一行的时间在sensor中可能找不到,它是evt的end信息
        time_sensor = get_time_sensor(evt_dict, running_times, sensor_values, dir_name)

        # step3.
        # process_name 和 sensor_value联系起来
        process_sensor_dict = process_sensor(evt_dict, time_sensor)
        all_process_sensor[dir_name] = process_sensor_dict
        all_sensor_index_dict[dir_name] = sensor_names
    miss_dir.close()

    return all_process_sensor, all_sensor_index_dict


def get_info_0630(path):
    '''
    :param path: dir path
    :return: all_dir's process_sensor_value_dict. a list

    '''
    files = os.listdir(path)
    evtfiles = [i for i in files if "EVT" in i]
    all_process_sensor = dict()
    all_sensor_index_dict = dict()
    for evtname in evtfiles:
        sensor_csv = os.path.join(path, evtname[3:])
        process_csv = os.path.join(path, evtname)
        sensor_lines = open(sensor_csv, "r")
        process_lines = open(process_csv, "r")

        # step1.
        # 为每一对csv文件,存储evt, sensor dict{}
        evt_dict, running_times, sensor_values, sensor_names = get_evt_sensor_dict(sensor_lines, process_lines)

        # step2.
        # 时间和sensor_value关联:
        # evt中最后一行的时间在sensor中可能找不到,它是evt的end信息
        time_sensor = get_time_sensor(evt_dict, running_times, sensor_values, evtname)

        # step3.
        # process_name 和 sensor_value联系起来
        process_sensor_dict = process_sensor(evt_dict, time_sensor)
        all_process_sensor[evtname] = process_sensor_dict
        all_sensor_index_dict[evtname] = sensor_names

    return all_process_sensor, all_sensor_index_dict


def process_sensor_param_dict(all_process_sensor, all_sensor_name, csv_dict_js, oneone_evt_thick_js):
    '''
    :param all_process_sensor: dir_name: each_dict
    each_dict: process_name: sensor_value_list
    sensor_index_dict: index, sensor_name
    :param oneone_evt_thick_js: 用于精简csv_dict_js, 剔除匹配不到lab值的evt_file
    :return:

    '''
    oneone_evt_lab = json.load(open(oneone_evt_thick_js, 'r'))
    evts = list(oneone_evt_lab.keys())
    evts = [i+'.CSV' for i in evts]
    csv_dict = dict()
    # 蔡司提供的,重要sensor数据列
    f = open(r'./info_sensor.txt', 'r')
    sen_list = (f.readlines()[0]).split(',')[:-1]
    for dir_name, process_sensor in all_process_sensor.items():
        if dir_name in evts:
            sensor_name_list = all_sensor_name[dir_name]
            dict_ = dict()
            for process_name, sensor_value_list in process_sensor.items():
                tmp_dict = dict()
                for index, each_sensor in enumerate(sensor_name_list):
                    if each_sensor in sen_list:
                        # 获取每一sensor参数的,所有列数据值. each_sensor_name: [value]
                        tmp_dict[each_sensor] = [row[index] for row in sensor_value_list]
                dict_[process_name] = tmp_dict
            csv_dict[dir_name] = dict_
            assert len(dict_) == 8
    # {dir_name: {process_name: {each_sensor: [value_list]}}}
    # 落盘
    js = json.dumps(csv_dict)
    with open(csv_dict_js, 'w') as js_:
        js_.write(js)
    return csv_dict


def times_feature(data):
    f_sensor = []
    ts = pd.Series(data)
    ae1 = tsf.feature_extraction.feature_calculators.ar_coefficient(ts, [{'coeff': 0, 'k': 10}])
    f_sensor.append(ae1[0][1])
    # # 时序数据的平稳性
    ae2 = tsf.feature_extraction.feature_calculators.augmented_dickey_fuller(ts, [{'attr': 'pvalue'}])
    f_sensor.append(ae2[0][1])
    f_sensor.append(np.mean(data))
    f_sensor.append(np.std(data, ddof=1))
    return f_sensor


def evt_128senftur(csv_dict_js, sub_sen_list):
    print('read csv_dict_js ~')
    evt_128sensor = dict()
    evt_sen_name_value = json.load(open(csv_dict_js, 'r'))
    for evtname, v in evt_sen_name_value.items():
        evt_8steps_sensor_feature = []  # len=128  8*4*4
        for process, sensor_dict in v.items():
            for sen_n, sen_v in sensor_dict.items():
                if sen_n in sub_sen_list:
                    each_step_sensor_feature = times_feature([float(i) for i in sen_v])  # len=4
                    evt_8steps_sensor_feature.extend(each_step_sensor_feature)
        evt_128sensor[evtname] = ''.join(str(i)+',' for i in evt_8steps_sensor_feature)

    return evt_128sensor


def get8step_sensor_feature(base_path, csv_dict_js, thick14_hc3_sensor144_lab_js, thick14_hc3_sensor16_lab_js, oneone_evt_thick_js, thick7_lab_js, sub_sen_list):
    print("add 8*4*4 part snesor feature!!! ")
    if not os.path.exists(csv_dict_js):
        all_process_sensor, all_sensor_name = get_info_0630(base_path)
        _ = process_sensor_param_dict(all_process_sensor, all_sensor_name, csv_dict_js, oneone_evt_thick_js)
    else:
        evt_128sensor_dict = evt_128senftur(csv_dict_js, sub_sen_list)
        evt_7thick = json.load(open(oneone_evt_thick_js, 'r'))
        thick7_lab = json.load(open(thick7_lab_js, 'r'))
        thick14hc3sensor16sensor128_lab = dict()
        # key value 转换
        thick_hc_sen16_lab = json.load(open(thick14_hc3_sensor16_lab_js, 'r'))
        lab_thick_hc_sen16_lab = dict()
        for k, v in thick_hc_sen16_lab.items():
            lab_thick_hc_sen16_lab[''.join(str(i) for i in v)] = k  # lab：膜厚耗材sensor16
        for evt, thick7 in evt_7thick.items():
            lab = thick7_lab[thick7]
            old_thick_hc_sensor = lab_thick_hc_sen16_lab[''.join(str(i) for i in lab)]   # 14+3+16维
            new_thick_hc_sensor = old_thick_hc_sensor + evt_128sensor_dict[evt+'.CSV']
            thick14hc3sensor16sensor128_lab[new_thick_hc_sensor] = lab
        # 落盘
        js = json.dumps(thick14hc3sensor16sensor128_lab)
        with open(thick14_hc3_sensor144_lab_js, 'w') as js_:
            js_.write(js)


if __name__ == "__main__":

    base_path = r"D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC"
    csv_dict_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\evtname_sensor_name_value.json'
    oneone_evt_thick_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\oneone_evt_thickness.json'
    sub_sen_list = ['ACT_O1_QCMS_THICKNESS', 'ACT_O1_QCMS_RATE', 'ACT_O1_QCMS_THICKNESS_CH1', 'ACT_O1_QCMS_RATE_CH1']
    thick14_hc3_sensor16_lab_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\thick14hc3sensor16_lab.json'
    thick7_lab_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\org_refine_thickness_lab_curve.json'
    thick14_hc3_sensor144_lab_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\thick14hc3sensor144_lab.json'

    get8step_sensor_feature(base_path, csv_dict_js, thick14_hc3_sensor144_lab_js, thick14_hc3_sensor16_lab_js, oneone_evt_thick_js,
                            thick7_lab_js)



