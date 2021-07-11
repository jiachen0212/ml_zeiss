# coding=utf-8
import json
import os
import copy
import numpy as np


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
    # assert len(step_time) == 9
    if len(step_time) == 9:
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
    else:
        return None, None, None, None


def get_time_sensor(evt_dict, running_times, sensor_values, evtname):
    '''
    evt_dict: time: process
    sensor.csv: running_times, sensor_values

    '''
    time_sensor = dict()
    evt_process_time = []
    for step_name, time in evt_dict.items():
        evt_process_time.append(time)
    lens = len(evt_process_time)  # lens == 9  step1~8 + end_line
    assert lens == 9
    for i in range(lens - 1):
        start, end = evt_process_time[i], evt_process_time[i + 1]
        try:
            sensor_start = running_times.index(start)
        except:
            return time_sensor  # start都找不到,数据记录肯定有问题,这个样本数据就不要了.
        try:
            sensor_end = running_times.index(end)
        except:
            if i == lens-2:
                # 最后evt中的时间点,sensor.csv中可能找不到,数据就直接读取到sensor.csv的最后一行
                sensor_end = len(running_times)
            else:
                return time_sensor
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


def get_info_0630(path, exists_sensors):
    '''
    :param path: dir path
    :return: all_dir's process_sensor_value_dict. a list

    '''
    all_process_sensor = dict()
    all_sensor_index_dict = dict()
    evtfiles = ['evt'+i for i in exists_sensors]
    for evtname in evtfiles:
        sensor_csv = os.path.join(path, evtname[3:])
        process_csv = os.path.join(path, evtname)
        try:
            sensor_lines = open(sensor_csv, "r")
            process_lines = open(process_csv, "r")
        except:
            continue
        # step1.
        # 为每一对csv文件,存储evt, sensor dict{}
        evt_dict, running_times, sensor_values, sensor_names = get_evt_sensor_dict(sensor_lines, process_lines)

        if sensor_values:
            # step2.
            # 时间和sensor_value关联:
            # evt中最后一行的时间在sensor中可能找不到,它是evt的end信息
            time_sensor = get_time_sensor(evt_dict, running_times, sensor_values, evtname)

            # step3.
            # process_name 和 sensor_value联系起来
            if time_sensor:
                process_sensor_dict = process_sensor(evt_dict, time_sensor)
                all_process_sensor[evtname] = process_sensor_dict
                all_sensor_index_dict[evtname] = sensor_names
        else:
            continue
    print("all_process_sensor len: {}, all_sensor_index_dict len: {}".format(len(all_process_sensor), len(all_sensor_index_dict)))
    return all_process_sensor, all_sensor_index_dict


def process_sensor_param_dict(all_process_sensor, all_sensor_name, csv_dict_js, sen_list, exists_sensors):
    '''
    :param all_process_sensor: dir_name: each_dict
    each_dict: process_name: sensor_value_list
    sensor_index_dict: index, sensor_name
    :return:

    '''
    csv_dict = dict()
    evts = ['evt'+i for i in exists_sensors]
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
            # pre code
            # csv_dict[dir_name] = dict_
            # assert len(dict_) == 8

            # 0711
            if len(dict_) == 8:
                csv_dict[dir_name] = dict_
    print("evt_8steps sensor len: {}".format(len(csv_dict)))

    # {dir_name: {process_name: {each_sensor: [value_list]}}}
    js = json.dumps(csv_dict)
    with open(csv_dict_js, 'w') as js_:
        js_.write(js)



def times_feature(data):
    f_sensor = []
    # ts = pd.Series(data)
    # ae1 = tsf.feature_extraction.feature_calculators.ar_coefficient(ts, [{'coeff': 0, 'k': 10}])
    # f_sensor.append(ae1[0][1])
    # # # 时序数据的平稳性
    # ae2 = tsf.feature_extraction.feature_calculators.augmented_dickey_fuller(ts, [{'attr': 'pvalue'}])
    # f_sensor.append(ae2[0][1])
    f_sensor.append(np.mean(data))
    f_sensor.append(np.std(data, ddof=1))
    return f_sensor


def num33_64sensor(csv_dict_js, sub_sen_list, num_evt12_data):
    numb33_64sensor = dict()
    evt_sen_name_value = json.load(open(csv_dict_js, 'r'))
    for numb33, evtcccx in num_evt12_data.items():
        # evtxxxx.csv
        evtcc = evtcccx[0]
        evtcx = evtcccx[1]
        evt_8steps_sensor_feature1 = []
        evt_8steps_sensor_feature2 = []
        if evtcc in evt_sen_name_value and evtcx in evt_sen_name_value:
            for process, sensor_dict in evt_sen_name_value[evtcc].items():
                for sen_n, sen_v in sensor_dict.items():
                    if sen_n in sub_sen_list:
                        each_step_sensor_feature = times_feature([float(i) for i in sen_v])
                        evt_8steps_sensor_feature1.extend(each_step_sensor_feature)
            for process, sensor_dict in evt_sen_name_value[evtcx].items():
                for sen_n, sen_v in sensor_dict.items():
                    if sen_n in sub_sen_list:
                        each_step_sensor_feature = times_feature([float(i) for i in sen_v])
                        evt_8steps_sensor_feature2.extend(each_step_sensor_feature)
            evt_8steps_sensor_feature = [(evt_8steps_sensor_feature1[i] + evt_8steps_sensor_feature2[i])/2 for i in range(len(evt_8steps_sensor_feature1))]
        # 仅存在正面不存在背面sensor_csv
        elif not evtcc in evt_sen_name_value and evtcx in evt_sen_name_value:
            for process, sensor_dict in evt_sen_name_value[evtcx].items():
                for sen_n, sen_v in sensor_dict.items():
                    if sen_n in sub_sen_list:
                        each_step_sensor_feature = times_feature([float(i) for i in sen_v])
                        evt_8steps_sensor_feature2.extend(each_step_sensor_feature)
            evt_8steps_sensor_feature = evt_8steps_sensor_feature2
        # 仅存在背面不存在正面sensor_csv
        elif evtcc in evt_sen_name_value and not evtcx in evt_sen_name_value:
            for process, sensor_dict in evt_sen_name_value[evtcc].items():
                for sen_n, sen_v in sensor_dict.items():
                    if sen_n in sub_sen_list:
                        each_step_sensor_feature = times_feature([float(i) for i in sen_v])
                        evt_8steps_sensor_feature1.extend(each_step_sensor_feature)
            evt_8steps_sensor_feature = evt_8steps_sensor_feature1
        else:
            continue
        assert len(evt_8steps_sensor_feature) == 64
        numb33_64sensor[numb33] = ''.join(str(i) + ',' for i in evt_8steps_sensor_feature)

    return numb33_64sensor


def get8step_sensor_feature(num_evt12, base_data_dir, csv_dict_js, thick14_hc3_sensor16_lab_js, thick14_hc3_sensor80_lab_js, sen_list):
    '''
    关联 evt和 sensor_value,获取工艺的8个step起始时间,并对应计算各个阶段内,thickness,rate的时序特征. 8*4*2 = 64维

    '''
    print("add 8*4*2  8steps sensor feature, mean and std ~")
    num_evt12_data = json.load(open(num_evt12, 'r'))
    exists_sensor = open(r'./sensor_csv.txt', 'r').readlines()[0]
    exists_sensors = exists_sensor.split(',')[:-1]
    number33_thick14hc3sensor16_lab = json.load(open(thick14_hc3_sensor16_lab_js, 'r'))
    thick14hc3sensor80_lab = copy.deepcopy(number33_thick14hc3sensor16_lab)
    all_process_sensor, all_sensor_name = get_info_0630(base_data_dir, exists_sensors)
    process_sensor_param_dict(all_process_sensor, all_sensor_name, csv_dict_js, sen_list, exists_sensors)
    numb33_64sensor_dict = num33_64sensor(csv_dict_js, sen_list, num_evt12_data)

    for num, thickhcsensor in number33_thick14hc3sensor16_lab.items():
        try:
            sensor64 = numb33_64sensor_dict[num]
        except:
            continue
        thick14hc3sensor80_lab[num][0] = number33_thick14hc3sensor16_lab[num][0] + sensor64
    assert len(thick14hc3sensor80_lab[num][0].split(',')) == 98
    js = json.dumps(thick14hc3sensor80_lab)
    with open(thick14_hc3_sensor80_lab_js, 'w') as js_:
        js_.write(js)
    print("thick14hc3sensor80 len: {}".format(len(thick14hc3sensor80_lab)))


if __name__ == "__main__":
    root_dir = r'D:\work\project\卡尔蔡司AR镀膜\第二批7.1'
    refine_data_json = os.path.join(root_dir, r'0701', 'refine_thickness_lab_curve.json')
    bad_js = os.path.join(root_dir, r'0701', 'bad_thick_lab.json')
    oneone_evt_thickness = os.path.join(root_dir, r'0701', 'oneone_evt_thickness.json')
    evt_33number = os.path.join(root_dir, r'0701', 'evt_33number.json')

    base_path = r"D:\work\project\卡尔蔡司AR镀膜\第二批7.1\机台文件\1.6&1.67_DVS_CC"

    oneone_evt_thick_js = os.path.join(root_dir, r'0701', 'oneone_evt_thickness.json')
    sub_sen_list = ['ACT_O1_QCMS_THICKNESS', 'ACT_O1_QCMS_RATE', 'ACT_O1_QCMS_THICKNESS_CH1', 'ACT_O1_QCMS_RATE_CH1']
    thick14_hc3_sensor16_lab_js = os.path.join(root_dir, r'0701', 'thick14hc3sensor16_lab.json')
    thick7_lab_js = os.path.join(root_dir, r'0701', 'refine_thickness_lab_curve.json')
