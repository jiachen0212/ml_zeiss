# coding=utf-8
import json
import os
import copy
import numpy as np


def get_evt_sensor_dict(sensor_lines, process_lines, sen_list):
    '''
    step_time = dict()   # start step No. of : time

    sensor.csv info:
    running_times = []
    sen_list: sensor_values = []

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
        # 部分evt中有step no.9,则不需要记录evt的最末行时间,不运行以下if.
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
        for index, line in enumerate(sensor_lines):
            split_line = line.strip()
            if not split_line:
                continue
            spline = split_line[:-1].split(',')
            if index == 0:
                all_sensor_names = spline
                ind = all_sensor_names.index(sen_list[0])
            else:
                # 同上step_time, 时间统一格式处理
                if len(spline[1].split(':')[0]) > 1 and spline[1][0] == '0':
                    tmp = spline[1][1:]
                else:
                    tmp = spline[1]
                running_times.append(tmp)
                # 这里只有两列只考虑两列sensor数据列
                sensor_values.append([spline[ind]])
        return step_time, running_times, sensor_values
    else:
        return None, None, None


def get_time_sensor(step_time_dict, running_times, sensor_values, evtname):
    '''
    evt_dict: time: process
    sensor.csv: running_times, sensor_values

    '''
    time_sensor = dict()
    evt_process_time = []
    for step_name, time in step_time_dict.items():
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
        # print(evtname, start, end)
        # print(running_times[sensor_start-1], running_times[sensor_start], running_times[sensor_end], running_times[sensor_end+1])

        cur_sensor_value = sensor_values[sensor_start:sensor_end]
        time_sensor[start] = cur_sensor_value

    # check
    # tmp = list(time_sensor.keys())[4]
    # print(evtname, tmp, time_sensor[tmp])

    return time_sensor


def process_sensor(step_time_dict, time_sensor, n_thickness):
    '''

    :param step_time_dict:
    :param time_sensor:
    :return:
    time is the tmp_param
    '''
    evt_times = []
    for step_name, time in step_time_dict.items():
        evt_times.append(time)
    # 根据evt文件中的steptime,对应获取sensor_value
    step_sensor_dict = dict()
    for step_name, time in step_time_dict.items():
        try:
            sensor_value_list = time_sensor[time]
        except:
            # print(dir_name, time)   # evt文件末尾的几行,无信息量
            continue
        step_sensor_dict[step_name] = sensor_value_list
    assert len(step_sensor_dict) == n_thickness + 1

    return step_sensor_dict


def get_info_0630(n_thickness, sen_list, path, exists_sensors):
    '''
    :param path: dir path
    :return: all_dir's process_sensor_value_dict. a list

    '''
    all_step_sensor = dict()
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
        # 得到step no.x of 9: time dict, sensor.csv文件的 time list sensor_value list
        step_time_dict, running_times, sensor_values = get_evt_sensor_dict(sensor_lines, process_lines, sen_list)

        if sensor_values:
            # step2.
            # 通过时间, 获取8段step的 sensor_value list:
            # evt中最后一个时间在sensor.csv中可能找不到,就直接读取到最后一行即可
            time_sensor = get_time_sensor(step_time_dict, running_times, sensor_values, evtname)

            # step3.
            # step no.x of 9 和 sensor_value联系起来
            if len(time_sensor) == n_thickness + 1:
                step_sensor_dict = process_sensor(step_time_dict, time_sensor, n_thickness)
                all_step_sensor[evtname] = step_sensor_dict
        else:
            continue
    print("all_process_sensor len: {}".format(len(all_step_sensor)))
    return all_step_sensor


def process_sensor_param_dict(all_process_sensor, csv_dict_js, sen_list, exists_sensors, n_thickness):
    '''
    :param all_process_sensor: evt_name: each_dict
    each_dict: process_name: sensor_value_list
    sensor_index_dict: index, sensor_name
    :return:

    '''
    csv_dict = dict()
    evts = ['evt'+i for i in exists_sensors]
    for evt_name, process_sensor in all_process_sensor.items():
        if evt_name in evts:
            step_names = list(process_sensor.keys())
            dict_ = dict()
            for step_name in step_names:
                tmp_dict = dict()
                for i, sensor_name in enumerate(sen_list):
                    tmp_dict[sensor_name] = [row[i] for row in process_sensor[step_name]]
                dict_[step_name] = tmp_dict
            if len(dict_) == n_thickness + 1:
                csv_dict[evt_name] = dict_
    print("evt_8steps sensor len: {}".format(len(csv_dict)))
    # print(csv_dict['evt21020930.csv']['Start Step No. 4 of 9'])

    # {dir_name: {process_name: {each_sensor: [value_list]}}}
    js = json.dumps(csv_dict)
    with open(csv_dict_js, 'w') as js_:
        js_.write(js)



def times_feature(data, process, evtcc, sen_n):
    f_sensor = []
    # ts = pd.Series(data)
    # ae1 = tsf.feature_extraction.feature_calculators.ar_coefficient(ts, [{'coeff': 0, 'k': 10}])
    # f_sensor.append(ae1[0][1])
    # # # 时序数据的平稳性
    # ae2 = tsf.feature_extraction.feature_calculators.augmented_dickey_fuller(ts, [{'attr': 'pvalue'}])
    # f_sensor.append(ae2[0][1])
    assert len(data) != 0
    # f_sensor.append(np.mean(data))
    f_sensor.append(np.std(data, ddof=1))
    # print(len(data), process, evtcc, data, sen_n)
    # f_sensor.append(max(data))
    # if len(data) == 0:
    #     print(process, evtcc, sen_n)
    #     f_sensor.append(0)
    # else:
    #     f_sensor.append(max(data))
    return f_sensor


def num33_64sensor(csv_dict_js, sub_sen_list, num_evt12_data):
    numb33_64sensor = dict()
    evt_sen_name_value = json.load(open(csv_dict_js, 'r'))
    for numb33, evtcccx in num_evt12_data.items():
        evtcc = evtcccx[0]
        evtcx = evtcccx[1]
        evt_8steps_sensor_feature1 = []
        evt_8steps_sensor_feature2 = []
        if evtcc in evt_sen_name_value and evtcx in evt_sen_name_value:
            for process, sensor_dict in evt_sen_name_value[evtcc].items():
                for sen_n, sen_v in sensor_dict.items():
                    if sen_n in sub_sen_list:
                        each_step_sensor_feature = times_feature([float(i) for i in sen_v], process, evtcc, sen_n)
                        evt_8steps_sensor_feature1.extend(each_step_sensor_feature)
            for process, sensor_dict in evt_sen_name_value[evtcx].items():
                for sen_n, sen_v in sensor_dict.items():
                    if sen_n in sub_sen_list:
                        each_step_sensor_feature = times_feature([float(i) for i in sen_v], process, evtcc, sen_n)
                        evt_8steps_sensor_feature2.extend(each_step_sensor_feature)
            # evt_8steps_sensor_feature = [(evt_8steps_sensor_feature1[i] + evt_8steps_sensor_feature2[i])/2 for i in range(len(evt_8steps_sensor_feature1))]   # 取mean
            evt_8steps_sensor_feature = evt_8steps_sensor_feature1 + evt_8steps_sensor_feature2   # concate
        # 仅存在正面不存在背面sensor_csv
        elif not evtcc in evt_sen_name_value and evtcx in evt_sen_name_value:
            for process, sensor_dict in evt_sen_name_value[evtcx].items():
                for sen_n, sen_v in sensor_dict.items():
                    if sen_n in sub_sen_list:
                        each_step_sensor_feature = times_feature([float(i) for i in sen_v], process, evtcc, sen_n)
                        evt_8steps_sensor_feature2.extend(each_step_sensor_feature)
            evt_8steps_sensor_feature = evt_8steps_sensor_feature2 + evt_8steps_sensor_feature2
        # 仅存在背面不存在正面sensor_csv
        elif evtcc in evt_sen_name_value and not evtcx in evt_sen_name_value:
            for process, sensor_dict in evt_sen_name_value[evtcc].items():
                for sen_n, sen_v in sensor_dict.items():
                    if sen_n in sub_sen_list:
                        each_step_sensor_feature = times_feature([float(i) for i in sen_v], process, evtcc, sen_n)
                        evt_8steps_sensor_feature1.extend(each_step_sensor_feature)
            evt_8steps_sensor_feature = evt_8steps_sensor_feature1 + evt_8steps_sensor_feature1
        else:
            numb33_64sensor[numb33] = ''
        assert len(evt_8steps_sensor_feature) == 16
        numb33_64sensor[numb33] = ''.join(str(i) + ',' for i in evt_8steps_sensor_feature)

    return numb33_64sensor


def get8step_sensor_feature(n_thickness, num_evt12, base_data_dir, csv_dict_js, thick10_lab_js, thick10_sensor8step_lab_js, sen_list):
    '''
    关联 evt和 sensor_value,获取工艺的8个step起始时间,并对应计算各个阶段内,thickness,rate的时序特征. 8*4*2 = 64维

    '''
    num_evt12_data = json.load(open(num_evt12, 'r'))
    exists_sensor = open(r'./sensor_csv.txt', 'r').readlines()[0]
    exists_sensors = exists_sensor.split(',')[:-1]
    number33_thick10_lab = json.load(open(thick10_lab_js, 'r'))
    thick10sensor32_lab = copy.deepcopy(number33_thick10_lab)
    all_step_sensor = get_info_0630(n_thickness, sen_list, base_data_dir, exists_sensors)
    process_sensor_param_dict(all_step_sensor, csv_dict_js, sen_list, exists_sensors, n_thickness)
    numb33_64sensor_dict = num33_64sensor(csv_dict_js, sen_list, num_evt12_data)

    for num, thickhcsensor in number33_thick10_lab.items():
        sensor64 = numb33_64sensor_dict[num]
        if sensor64:
            new_f = ''.join(i+',' for i in number33_thick10_lab[num][0]) + sensor64
            assert len(new_f.split(',')) == 27
            thick10sensor32_lab[num][0] = new_f
        else:
            del thick10sensor32_lab[num]

    js = json.dumps(thick10sensor32_lab)
    with open(thick10_sensor8step_lab_js, 'w') as js_:
        js_.write(js)
    print("thick10sensor8step len: {}".format(len(thick10sensor32_lab)))


if __name__ == "__main__":
    root_dir = r'D:\work\project\卡尔蔡司AR镀膜\第二批7.1'
    refine_data_json = os.path.join(root_dir, r'0701', 'refine_thickness_lab_curve.json')
    bad_js = os.path.join(root_dir, r'0701', 'bad_thick_lab.json')
    oneone_evt_thickness = os.path.join(root_dir, r'0701', 'oneone_evt_thickness.json')
    evt_33number = os.path.join(root_dir, r'0701', 'evt_33number.json')

    base_path = r"D:\work\project\卡尔蔡司AR镀膜\第二批7.1\机台文件\1.6&1.67_DVS_CC"

    oneone_evt_thick_js = os.path.join(root_dir, r'0701', 'oneone_evt_thickness.json')
    sub_sen_list = ['ACT_O1_QCMS_THICKNESS', 'ACT_O1_QCMS_RATE', 'ACT_O1_QCMS_THICKNESS_CH1', 'ACT_O1_QCMS_RATE_CH1']
    thick10_lab_js = os.path.join(root_dir, r'0701', 'thick14hc3sensor16_lab.json')
    thick7_lab_js = os.path.join(root_dir, r'0701', 'refine_thickness_lab_curve.json')
