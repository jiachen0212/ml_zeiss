# coing=utf-8
import json
import os
import shutil

import numpy as np
import pandas as pd
import tsfresh as tsf

'''
peak峰值个数,峰值
周期性,趋势性
均值方差 min max
7步工艺分段, 每一段内膜厚值变化与各个sensor变化可关联起来.

...

'''


def same_machine_recip(dir_):
    files = os.listdir(dir_)
    files = [i for i in files if "evt" in i]
    PLC_version = []
    MachineName = []
    RecipeName = []
    RecipeName_dict = dict()
    for file in files:
        f = open(os.path.join(dir_, file), 'r')
        for index, line in enumerate(f):
            # if index == 1:
            #     PLC_version.append(line.split(',')[-1])
            # elif index == 2:
            #     MachineName.append(line.split(',')[-1])
            if index == 3:
                recipename = line[:-1].split(',')[-1]
                if recipename:
                    RecipeName.append(recipename)
                    if recipename and recipename not in RecipeName_dict:
                        RecipeName_dict[recipename] = []
                        RecipeName_dict[recipename].append(file)
                    elif recipename:
                        RecipeName_dict[recipename].append(file)
    count = 0
    for k, v in RecipeName_dict.items():
        # print(len(v))
        count += len(v)
        path = os.path.join(dir_, k)
        # isExists = os.path.exists(path)
        os.makedirs(path)
        for cvs_name in v:
            filename0 = os.path.join(dir_, cvs_name)
            filename1 = os.path.join(path, cvs_name)
            filename0_ = os.path.join(dir_, cvs_name[3:])
            filename1_ = os.path.join(path, cvs_name[3:])
            try:
                shutil.copy(filename0, filename1)
            except:
                print(filename0)
                pass
            try:
                shutil.copy(filename0_, filename1_)
            except:
                print(filename0_)
                pass
    print(count)


def get_sub_title(dir_):
    '''
    当前文件夹下并非所有文件都share sensor title
    会不会存在一些情况,不同的csv,虽然一些列都是var==0,但是value并不相同,导致最终膜厚不同?
    有些列略微不同,也可能是sensor记录错误导致的. [硬件统计误差]

    '''
    files = os.listdir(dir_)
    files = [i for i in files if "EVT" not in i]
    all = []
    for file in files:
        full_path = os.path.join(dir_, file)
        f = open(full_path, 'r')
        for index, line in enumerate(f):
            if index == 0:
                sensor_list = line[:-1].split(',')[5:]
                all.extend(sensor_list)
    all_sensor = list(set(all))
    f = open(os.path.join(r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs', 'sub_title.txt'), 'w')
    for sensor in all_sensor:
        f.write(sensor + ',')


def get_thickness(csv_dir, files):
    thickness = open(r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\thickness.txt', 'r')
    line = thickness.readlines()[0]
    thickness_list = line.split(',')
    file_thickness = dict()

    for file in files:
        full_path = os.path.join(csv_dir, file)
        if "EVT" not in file:
            tmp = []
            # process = pd.read_csv(full_path, skiprows=5, error_bad_lines=False)
            sensor = pd.read_csv(full_path, error_bad_lines=False)
            for thickness_ in thickness_list:
                thickness_data = sensor[thickness_]
                col = [float(i) for i in thickness_data]  # 获取到了thickness列数据
                if np.var(col) > 1e-20:
                    tmp.append(thickness_)
            file_thickness[file] = tmp

    all = []
    for key, value in file_thickness.items():
        all.extend(value)
    print(len(list(set(all))))  # len=3
    # get:
    # thickness_list = ['ACT_O1_QCMS_NOMTHICKNESS', 'ACT_O1_QCMS_THICKNESS', 'ACT_O1_QCMS_THICKNESS_CH1']


def evt_csv_par(csv_dir):
    files = os.listdir(csv_dir)
    print("before recopy: {}".format(len(files)))
    evt = [i for i in files if "EVT" in i]
    csv = [i for i in files if "EVT" not in i]
    print("evt files: {}, csv files: {}".format(len(evt), len(csv)))
    data = []
    for f1 in evt:
        for f2 in csv:
            if f1[3:] == f2:
                data.append(f1)
    for f in evt:
        if f not in data:
            os.recopy(r"D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件\\" + f)
    for f in csv:
        if "EVT" + f not in data:
            os.recopy(r"D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件\\" + f)
    files = os.listdir(csv_dir)
    print("after recopy: {}".format(len(files)))


def get_file_sensor(csv_dir, sensores, js_name):
    files = os.listdir(csv_dir)
    useful_col = dict()
    line = sensores.readlines()[0]
    process_list = line[:-1].split(',')[5:]
    for file in files:
        full_path = os.path.join(csv_dir, file)
        if "EVT" not in file:
            tmp = dict()
            sensor = pd.read_csv(full_path, error_bad_lines=False)
            for process_ in process_list:
                try:
                    # 有些方差为0的列已经删掉了, 所以部分title_name读不到数据.
                    thickness_data = sensor[process_]
                except:
                    pass
                col = [float(i) for i in thickness_data]  # 获取到了csv的列数据
                # 做一个min-max归一化
                # mmin, mmax = min(col), max(col)
                # col = [(i-mmin)/(mmax-mmin) for i in col]
                if np.var(col) > 1e-12:
                    tmp[process_] = col
            useful_col[file] = tmp

    json_sensor = json.dumps(useful_col)

    with open(js_name, 'w') as js_file:
        js_file.write(json_sensor)
    print("get {}".format(js_name))


def get_evt_thickness(csv_dir, miss_, evt_thickness_val, evt_process_time):
    evt_thickness = dict()
    evt_process = dict()
    files = os.listdir(csv_dir)
    for file in files:
        full_path = os.path.join(csv_dir, file)
        if "EVT" in file:
            tmp = []
            process_step_time = []
            # process = pd.read_csv(full_path, skiprows=5, error_bad_lines=False)
            process = open(full_path, 'r')
            for index, line in enumerate(process):
                if 'Thickness' in line:
                    tmp.append(line.split(',')[4])
                if 'Start Step No.' in line:
                    process_step_time.append(line.split(',')[0])
            if len(tmp) == 0:
                miss_.write(file + ',')
                # miss_.write(file + '\n')
            else:
                evt_thickness[file] = tmp
                evt_process[file] = process_step_time

    json_sensor = json.dumps(evt_thickness)
    with open(evt_thickness_val, 'w') as js_file:
        js_file.write(json_sensor)
    js = json.dumps(evt_process)
    with open(evt_process_time, 'w') as js_:
        js_.write(js)
    print("get {} and {}".format(evt_thickness_val, evt_process_time))


def select_csv(useful_csv, evt_thickness_val, csv_dir, file_sensor_dict):
    useful_csv_tmp = open(useful_csv, "w", encoding="utf-8")
    reader = open(file_sensor_dict, encoding="utf-8")
    js_file = json.load(reader)
    with open(evt_thickness_val, encoding="utf-8") as reader:
        evt_thickness = json.load(reader)
        for file, sensor_list in evt_thickness.items():
            if len(sensor_list) == 7:
                useful_csv_tmp.write(file + ',')
    useful_csv_tmp.close()

    csvs = open(useful_csv, "r", encoding="utf-8")
    csvs_ = csvs.readlines()[0]
    csv_list = csvs_[:-1].split(',')  # 559
    csv_list = [i[3:] for i in csv_list] + csv_list
    files = os.listdir(csv_dir)
    files = [i for i in files if "EVT" not in i]
    for file in files:
        if file not in csv_list:
            full_path1 = os.path.join(csv_dir, file)
            full_path2 = os.path.join(csv_dir, "EVT" + file)
            os.recopy(full_path1)
            os.recopy(full_path2)
            del js_file[file]  # 删除膜层不足7的csv键值
    json_sensor = json.dumps(js_file)
    with open(file_sensor_dict, 'w') as js_file:
        js_file.write(json_sensor)
    print("update {}".format(file_sensor_dict))
    print("get {}".format(useful_csv))


def cut_cols(root_dir, titles):
    subdirs = os.listdir(root_dir)
    subdirs = [i for i in subdirs if '.txt' not in i]
    line = titles.readlines()[0]
    process_list = line[:-1].split(',')[5:]
    for dir_ in subdirs:
        files = os.listdir(os.path.join(root_dir, dir_))
        for file in files:
            full_path = os.path.join(root_dir, dir_, file)
            if "EVT" not in file:
                sensor = pd.read_csv(full_path, error_bad_lines=False)
                for process_ in process_list:
                    thickness_data = sensor[process_]
                    col = [float(i) for i in thickness_data]  # 获取到了csv的列数据
                    if np.var(col) < 1e-12:
                        sensor.drop([process_], axis=1, inplace=True)  # 删除方差<1e-12的列
                        print(file, process_)
                sensor.to_csv(full_path, index=False)


def get_thick_sensor(evt_thickness, thickness_sensor_file, file_sensor_dict):
    '''
    关联起  thickness, sensor_value, lab_curve
    :param evt_thickness:
    :param thickness_sensor_file:
    :param file_sensor_dict:
    :return:
    '''
    thickness_sensor = dict()
    # 这个js是数据清洗后的, 已此为基准去找evt-thickness
    evt_thick_dict = json.load(open(evt_thickness, 'r'))  # EVT21050425
    # evts = list(evt_thick_dict.keys())

    # evtname: sensor_list dict
    evtname_sensor = json.load(open(file_sensor_dict, 'r'))  # 21042718.csv
    # 根据：evt_dict(evt:33)、number33_lab(33:lab)、file_sensor_dict(evt:sensor_value_list)
    for evt, thick in evt_thick_dict.items():
        thickness_sensor[thick] = evtname_sensor[evt[3:] + '.csv']
    #     print(thick, '===', len(evtname_sensor[evt[3:]+'.csv']))
    # print(len(thickness_sensor))
    # 落盘json
    data = json.dumps(thickness_sensor)
    with open(thickness_sensor_file, 'w') as js_file:
        js_file.write(data)


def TSD(val):
    '''

    时序分解函数
    :return: St,Tt,Et
    '''
    St, Tt, Et = 0, 0, 0

    return St, Tt, Et


def feature_project(file_sensor_dict, ):
    with open(file_sensor_dict, "r", encoding="utf-8") as reader:
        js = json.load(reader)
        for key, value in js.items():
            '''
            do something
            这里就对这些方差不是0的列数据, 做一些特征提取, featuretools、时序分解都可以
            like： https://github.com/blue-yonder/tsfresh

            '''
            data_dict = value
            for sensor, val in data_dict.items():
                # 时序分解算法,处理每一列sensor采样值
                s, t, e = TSD(val)

    return


def rate_thickness_check(csv_dir):
    f = open(r'./thickness_int_float_2rate.txt', 'w')
    files = os.listdir(csv_dir)
    files = [a for a in files if "EVT" in a]
    for file in files:
        full_path = os.path.join(csv_dir, file)
        process = open(full_path, 'r')
        thick_int = []  # 存储膜厚设置值
        thick_float = []  # 存储膜厚实测值
        rate = []  # 速率实测值
        for index, line in enumerate(process):
            if 'Thickness' in line:
                thick_int.append(float(line.split(',')[4]))
                thick_float.append(float(line.split(',')[6]))
            if 'Rate' in line:
                rate.append(float(line.split(',')[6]))
        # print(thick_int)
        # print(thick_float)
        # print(rate)
        diff = [abs(thick_float[i] - thick_int[i]) for i in range(7)]
        rate = [i * 2 for i in rate]
        # print(diff)
        for i in range(7):
            if diff[i] > rate[i]:
                print()
                f.write('{},{},{}\n'.format(file, str(diff[i]), str(rate[i])))


def bad_sample_clean(thickness_js, evt_thick):
    # dict key-value 转换
    f = open(r'./bad_sample.txt', 'w')
    thick_evt = dict()
    with open(evt_thick, encoding="utf-8") as reader:
        tmp = json.load(reader)
        for evt, thick in tmp.items():
            thick_evt[thick] = evt

    with open(thickness_js, encoding="utf-8") as reader:
        thick_lab = json.load(reader)
        for thick, lab in thick_lab.items():
            part1 = lab[4:24]  # 400~500
            peak1 = max([float(i) for i in part1])
            part2 = lab[6:14]
            low_peak = min([float(i) for i in part2])
            if peak1 > 3:
                # print(thick_evt[thick])
                f.write(thick_evt[thick] + '\n')
            if low_peak < 0.14:
                f.write(thick_evt[thick] + '\n')


def sensor_csv_feature(sen_list, data):
    # 一个data/evt样本输入进来,输出特征维度: 4*2+2*2=12维
    f_sensor = []
    for sen in sen_list:
        col = [i for i in data[sen]]
        ts = pd.Series(col)
        # 时序数据的的周期性、不可预测性和波动性
        ae1 = tsf.feature_extraction.feature_calculators.ar_coefficient(ts, [{'coeff': 0, 'k': 10}])
        f_sensor.append(ae1[0][1])
        # # 时序数据的平稳性
        ae2 = tsf.feature_extraction.feature_calculators.augmented_dickey_fuller(ts, [{'attr': 'pvalue'}])
        f_sensor.append(ae2[0][1])
        # rate的 mean, std,
        if 'RATE' in sen:
            mean = np.mean([i for i in data[sen]][2:])
            std = np.var([i for i in data[sen]][2:])
            f_sensor.append(mean)
            f_sensor.append(std)
    # scale = StandardScaler(with_mean=True, with_std=True)
    # f_sensor = np.reshape(np.array(f_sensor), (-1, 12))
    f_sensor = [round(i * 10, 3) for i in f_sensor]
    return ''.join(str(i) + ',' for i in f_sensor)


def add_sensor_feature(data_dir, evt_7thick_js, thick7_lab_js, thick_hc_lab_js, sen_list,
                       thick14_hc3_sensor16_lab_js):
    evt_7thick = json.load(open(evt_7thick_js, 'r'))
    thick7_lab = json.load(open(thick7_lab_js, 'r'))
    thick14hc3sensor12_lab = dict()
    lab_thick_hc = dict()
    thick_hc_lab = json.load(open(thick_hc_lab_js, 'r'))
    for thick_hc, lab in thick_hc_lab.items():
        lab_thick_hc[''.join(str(i) for i in lab)] = thick_hc
    for evtname, thick7 in evt_7thick.items():
        # 等待补充的17维feature
        thick14_hc3 = lab_thick_hc[''.join(str(i) for i in thick7_lab[thick7])]
        # 读取sensor数据csv文件
        full_file_path = os.path.join(data_dir, evtname[3:] + '.csv')
        data = pd.read_csv(full_file_path, error_bad_lines=False)
        sensor12 = sensor_csv_feature(sen_list, data)
        thick14_hc3_sensor12 = thick14_hc3 + sensor12
        thick14hc3sensor12_lab[thick14_hc3_sensor12] = thick7_lab[thick7]
    data = json.dumps(thick14hc3sensor12_lab)
    with open(thick14_hc3_sensor16_lab_js, 'w') as js_file:
        js_file.write(data)


def sensor_feature(sen_list, data, title):
    # data 是excel数据文件
    # 一个data/evt样本输入进来,输出特征维度: 4*2+2*2=12维
    f_sensor = []
    for sen in sen_list:
        index = title.index(sen)
        col = data.col_values(index)[2:]
        ts = pd.Series(col)  # 数据x假设已经获取
        # 时序数据的的周期性、不可预测性和波动性
        ae1 = tsf.feature_extraction.feature_calculators.ar_coefficient(ts, [{'coeff': 0, 'k': 10}])
        f_sensor.append(ae1[0][1])
        # # 时序数据的平稳性
        ae2 = tsf.feature_extraction.feature_calculators.augmented_dickey_fuller(ts, [{'attr': 'pvalue'}])
        f_sensor.append(ae2[0][1])
        # rate的 mean, std,
        if 'RATE' in sen:
            mean = np.mean([data.col_values(index)[2:]])
            std = np.var([data.col_values(index)[2:]])
            f_sensor.append(mean)
            f_sensor.append(std)
    return f_sensor


def backup_usful_sensor_feature():
    tmp = open(r'./info_sensor_nothick.txt', 'w')
    f = os.path.join(csv_dir, r'21051026.csv')
    sensor_csv = pd.read_csv(f, error_bad_lines=False)
    ok_sen_list = ['ACT_V1_IONIVAC_CH', 'ACT_V1_THERMOVAC_CH', 'ACT_V1_THERMOVAC_PREVLINE', 'ACT_V1_THERMOVAC_HP',
                   'ACT_V1_PRESSURE_CH', 'AI_V1_POLYCOLD_TEMP', 'ACTN_F1_FLOW1', 'ACT_F1_FLOW1',
                   'ACT_O1_QCMS_THICKNESS', 'ACT_O1_QCMS_RATE', 'ACT_O1_QCMS_THICKNESS_CH1', 'ACT_O1_QCMS_RATE_CH1',
                   'STAT_LT_CRYSTAL_CH1', 'ACT_HEATER2_TEMPERATURE', 'ACT_Q10_CURRENT_ANODE', 'ACT_Q10_VOLTAGE_ANODE',
                   'ACT_Q10_CURRENT_CATHODE', 'ACT_Q10_VOLTAGE_CATHODE', 'ACT_Q10_CURRENT_NEUTRAL', 'ACT_Q10_ION_FLOW1',
                   'ACT_Q10_ION_FLOW2', 'STA_Q10_IONSOURCE_SHUTTER_IOP', 'ACT_V1_MEISSNER_POLYCOLDTEMP']
    ok_sen_list = [i for i in ok_sen_list if i not in sen_list]
    for a in ok_sen_list:
        tmp.write(a + ',')
    f = []
    for sen_n in ok_sen_list:
        try:
            col = sensor_csv[sen_n]
        except:
            continue
        # ok_sen_list.append(sen_n)
        col_data = [i for i in col]
        x = [i for i in range(len(col_data))]
        # plt.title(sen_n)
        # plt.plot(x, col_data)
        # plt.savefig(os.path.join(r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\重要snesor列数据曲线', sen_n))
        # plt.show()
        ts = pd.Series(col)
        ae1 = tsf.feature_extraction.feature_calculators.ar_coefficient(ts, [{'coeff': 0, 'k': 10}])
        f.append(ae1[0][1])
        # ae2 = tsf.feature_extraction.feature_calculators.augmented_dickey_fuller(ts, [{'attr': 'pvalue'}])
        ae3 = tsf.feature_extraction.feature_calculators.binned_entropy(ts, 10)  # 信息熵,可以考虑加入
        f.append(ae3)


def usful_sensor_feature(sensor_csv):
    sensor_csv = pd.read_csv(sensor_csv, error_bad_lines=False)
    tmp = open(r'./info_sensor_nothick.txt', 'r')
    ok_sen_list = tmp.readlines()[0].split(',')[:-1]
    f = []
    for sen_n in ok_sen_list:
        col = sensor_csv[sen_n]
        col_data = [i for i in col]
        ts = pd.Series(col_data)
        ae1 = tsf.feature_extraction.feature_calculators.ar_coefficient(ts, [{'coeff': 0, 'k': 10}])
        f.append(ae1[0][1])
        # ae2 = tsf.feature_extraction.feature_calculators.augmented_dickey_fuller(ts, [{'attr': 'pvalue'}])
        ae3 = tsf.feature_extraction.feature_calculators.binned_entropy(ts, 10)  # 信息熵,可以考虑加入
        f.append(ae3)
    return ''.join(str(i) + ',' for i in f)


def all_usful_sensor_except_thickness(csv_dir, org_refine_thick_lab, oneone_evt_thick, thick14_hc3_sensor64_lab_js,
                                      feature135_lab_js):
    '''
    整合出来已经处理的thickness、rate这四列之外的,有意义数据列feature
    :return: 19*2=38维特征

    '''
    thick7_lab = json.load(open(org_refine_thick_lab, 'r'))
    evt_7thick = json.load(open(oneone_evt_thick, 'r'))
    feature135_lab = dict()
    # key value 转换
    feature97_lab = json.load(open(thick14_hc3_sensor64_lab_js, 'r'))
    lab_feature97 = dict()
    for k, v in feature97_lab.items():
        lab_feature97[''.join(str(i) for i in v)] = k  # lab:feature97
    for evt, thick7 in evt_7thick.items():
        feature38_sensor = usful_sensor_feature(os.path.join(csv_dir, evt[3:] + '.csv'))
        lab = thick7_lab[thick7]
        old_thick_hc_sensor = lab_feature97[''.join(str(i) for i in lab)]
        new_thick_hc_sensor = old_thick_hc_sensor + feature38_sensor
        feature135_lab[new_thick_hc_sensor] = lab

    # 落盘
    js = json.dumps(feature135_lab)
    with open(feature135_lab_js, 'w') as js_:
        js_.write(js)
    print("got {}!!".format(feature135_lab_js))


if __name__ == "__main__":
    # csv_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件'
    csv_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC'
    thickness_list = ['ACT_O1_QCMS_NOMTHICKNESS', 'ACT_O1_QCMS_THICKNESS',
                      'ACT_O1_QCMS_THICKNESS_CH1']  # 'ACT_O1_QCMS_THICKNESS_CH1'这个为膜厚代表列ba

    # step0.
    # evt_csv_par(csv_dir)   # evt和csv文件需要一一对应,首先进行文件一一对应清洗

    # step1.  工艺机器名,区分开
    # same_machine_recip(csv_dir)

    # step2. get file_sensor_dict
    titles = open(r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\title.txt', 'r')
    file_sensor_dict = r"D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\file_sensor.json"
    if not os.path.exists(file_sensor_dict):
        get_file_sensor(csv_dir, titles, file_sensor_dict)

    # step3. fet thickness_sensor_file
    evt_thickness = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\oneone_evt_thickness.json'
    # 需要生成的thinkness_sensor.json
    thickness_sensor_file = r"D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\thickness_sensor.json"
    if not os.path.exists(thickness_sensor_file):
        get_thick_sensor(evt_thickness, thickness_sensor_file, file_sensor_dict)

    # refine thick_lab
    refine_thick_lab = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\org_refine_thickness_lab_curve.json'
    oneone_evt_thick = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\oneone_evt_thickness.json'
    feature135_lab_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\feature135_lab.json'

    '''
    ar_coefficient  自回归系数: 衡量时序数据的的周期性、不可预测性和波动性
    augmented_dickey_fuller(x, param) 扩展迪基-福勒检验（ADF检验）: 测试一个自回归模型是否存在单位根，衡量时序数据的平稳性
    # change_quantiles(x, ql, qh, isabs, f_agg) 给定区间的时序数据描述统计 : 先用ql和qh两个分位数在x中确定出一个区间，然后在这个区间里计算时序数据的均值、绝对值、连续变化值。（浮点数）
    
    '''

    evt_7thick = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\oneone_evt_thickness.json'
    thick7_lab = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\org_refine_thickness_lab_curve.json'
    thick_hc_lab_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\thick_hc_lab.json'
    data_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC'
    thick14_hc3_sensor64_lab_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\thick14hc3sensor64_lab.json'
    sen_list = ['ACT_O1_QCMS_THICKNESS', 'ACT_O1_QCMS_RATE', 'ACT_O1_QCMS_THICKNESS_CH1', 'ACT_O1_QCMS_RATE_CH1']
    # add_sensor_feature(data_dir, evt_7thick, thick7_lab, thick_hc_lab_js, sen_list, thick14_hc3_sensor16_lab_js)
    # all_usful_sensor_except_thickness(csv_dir, refine_thick_lab, oneone_evt_thick, thick14_hc3_sensor64_lab_js,
    #                                   feature135_lab_js)
    dir_ = r'D:\work\project\卡尔蔡司AR镀膜\第二批7.1\机台文件'
    # same_machine_recip(dir_)

    # y1 = np.load(r'./1.npy')
    # y2 = np.load(r'./2.npy')
    # m, n = y1.shape
    # print(m, n)
    # count = 0
    # for i in range(m):
    #     for j in range(m):
    #         if y1[i].all() == y2[j].all():
    #             count += 1
    # print(count)

    # check 耗材js data
    # hc_js1 = r'D:\work\project\卡尔蔡司AR镀膜\第三批\0705\thick_hc_lab.json'
    # hc_js2 = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\tmp.json'
    # hc1 = json.load(open(hc_js1, 'r'))
    # hc2 = json.load(open(hc_js2, 'r'))
    # k1s = []
    # v1s = []
    # k2s = []
    # v2s = []
    # for k, v in hc1.items():
    #     k1s.append(k)
    #     v1s.append(v)
    # for k, v in hc2.items():
    #     k2s.append(k)
    #     v2s.append(v)
    # f = open(r'data_info_0707.txt', 'w')
    # for i, k2 in enumerate(k2s):
    #     try:
    #         ind = k1s.index(k2)
    #         if v1s[ind] != v2s[i]:
    #             print(v1s[ind])
    #             print(v2s[i])
    #             f.write("thickness14_hc3: " + k2 + '\n')
    #             s1 = ''.join(str(i)+',' for i in v1s[ind])
    #             s2 = ''.join(str(i) + ',' for i in v2s[i])
    #             f.write("lab1: " + s1+'\n')
    #             f.write("lab2: " + s2+'\n')
    #             f.write('\n')
    #     except:
    #         continue



    # 0709 计算结果相似度
    import numpy as np
    import matplotlib.pyplot as plt


    def xsd(a, best):
        a_ = np.array(a)
        b_ = np.array(best)
        print(np.corrcoef([a_, b_])[0][1])
        weights = [1 for i in range(81)]
        nms = [380 + i * 5 for i in range(81)]
        t = [380, 400, 405, 410, 435, 440, 445, 635, 640, 645, 780]
        for n in t:
            weights[nms.index(n)] = 2

        res = [(a[i] - best[i]) ** 2 * weights[i] for i in range(81)]
        print("mse: {}".format(np.mean(res)))
        return np.corrcoef([a_, b_])[0][1], np.mean(res)


    # plt.xlabel("Wave-length")
    # plt.ylabel("Reflectance")
    #
    # plt.plot(nms, best, color='pink', label='best curve')
    # plt.plot(nms, a, color='black', label='sample_curve')
    # plt.legend()
    # plt.show()


    # a = [11.27, 7.28, 5.08, 3.53, 2.08, 1.12, 0.77, 0.73, 0.64, 0.74, 1.18, 1.4, 1.6, 1.65, 1.73, 1.92, 2.02, 1.99, 1.84,
    #      1.68, 1.61, 1.64, 1.65, 1.54, 1.3, 1.09, 0.96, 0.97, 0.98, 0.94, 0.86, 0.72, 0.59, 0.57, 0.51, 0.49, 0.49, 0.54,
    #      0.51, 0.45, 0.37, 0.3, 0.27, 0.23, 0.2, 0.3, 0.26, 0.24, 0.22, 0.14, 0.17, 0.2, 0.23, 0.36, 0.31, 0.35, 0.39, 0.52,
    #      0.52, 0.52, 0.56, 0.67, 0.78, 0.87, 1.06, 1.25, 1.5, 1.65, 1.85, 2.1, 2.29, 2.49, 2.7, 2.92, 3.14, 3.33, 3.62,
    #      3.75, 4.1, 4.31, 4.67]

    best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
            1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
            0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
            0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
            3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]


    # y1 = np.load(r'./train.npy')
    # y2 = np.load(r'./modified_x.npy')
    # mse1 = []
    # mse2 = []
    # a1 = []
    # b1 = []
    # sample_num, dims = y1.shape
    # for i in range(sample_num):
    #     a = y1[i, :]
    #     b = y2[i, :]
    #     corr, mse = xsd(a, best)
    #     a1.append(corr)
    #     mse1.append(mse)
    # print(np.mean(a1), np.mean(mse))   # b: 0.9581901163212438 0.08417979853069367   # a: 0.9555943864195805 0.30164862646093643
    # 0.9577425168730154 0.1332909929220622

    import heapq
    def get_max_index(nums):
        temp = []
        Inf = 0
        for i in range(3):
            temp.append(nums.index(max(nums)))
            nums[nums.index(max(nums))] = Inf
        temp.sort()
        return temp

    import collections

    all_ = []
    y1 = np.load(r'./1.npy')
    print(y1.shape)
    y2 = np.load(r'./2.npy')
    for i in range(y1.shape[0]):
        a = []
        base = y1[i, :].tolist()
        modif = y2[i, :].tolist()
        for j in range(y1.shape[1]):
            diff = base[j] - modif[j]
            diff /= base[j]
            diff = abs(diff)
            a.append(diff)
        res = get_max_index(a)
        # print(res)
        all_.extend(res)
    out = collections.Counter(all_)
    print(out)

#
# def lower_filename(base_dir, tmp_dir):
#     '''
#
#     :param base_dir: D:\work\project\卡尔蔡司AR镀膜\第三批\33机台文件
#     :param tmp_dir: 临时文件夹, 后续会被删除可随意命名
#     :return:
#     '''
#     # base_dir = r'D:\work\project\卡尔蔡司AR镀膜\第三批\33机台文件'
#     files = os.listdir(base_dir)
#     for f in files:
#         if '.7z' in f or '.zip' in f:
#             os.recopy(os.path.join(base_dir, f))
#     fs = os.listdir(base_dir)
#     if not os.path.exists(tmp_dir):
#         os.mkdir(tmp_dir)
#     for f in fs:
#         path1 = os.path.join(base_dir, f)
#         path2 = os.path.join(tmp_dir, f.lower())
#         shutil.copy(path1, path2)
#     os.rmdir(base_dir)
#     os.rename(tmp_dir, base_dir)

# def split_CX_CC(base_data_dir):
#     pre_dir = base_data_dir[:-1 - len(base_data_dir.split('\\')[-1])]
#     cc_dir = os.path.join(pre_dir, r'cc')
#     cx_dir = os.path.join(pre_dir, r'cx')
#     if not os.path.exists(cc_dir):
#         os.mkdir(cc_dir)
#     if not os.path.exists(cx_dir):
#         os.mkdir(cx_dir)
#     all_file = os.listdir(base_data_dir)
#     evt_files = [i for i in all_file if "evt" in i]
#     cc = []
#     cx = []
#     for evt_f in evt_files:
#         path = os.path.join(base_data_dir, evt_f)
#         print(path)
#         with open(path, 'r') as f:
#             for index, line in enumerate(f):
#                 if 'Recipe Name :' in line:
#                     cccx = line.split(',')[1]
#                     if 'CC' in cccx:
#                         cc.append(evt_f)
#                     elif 'CX' in cccx:
#                         cx.append(evt_f)
#     for cc_ in cc:
#         path_sensor1 = os.path.join(base_data_dir, cc_[3:])
#         path = os.path.join(base_data_dir, cc_)
#         path_evt2 = os.path.join(cc_dir, cc_)
#         shutil.copy(path, path_evt2)
#         if os.path.exists(path_sensor1):
#             path_sensor2 = os.path.join(cc_dir, cc_[3:])
#             shutil.copy(path_sensor1, path_sensor2)
#     for cx_ in cx:
#         path_sensor1 = os.path.join(base_data_dir, cx_[3:])
#         path = os.path.join(base_data_dir, cx_)
#         path_evt2 = os.path.join(cx_dir, cx_)
#         shutil.copy(path, path_evt2)
#         if os.path.exists(path_sensor1):
#             path_sensor2 = os.path.join(cx_dir, cx_[3:])
#             shutil.copy(path_sensor1, path_sensor2)

    from util import calculate_Lab
    def lab_okng(v, v_range):
        V = float(v)
        if V >= v_range[0] and V <= v_range[1]:
            return 1
        return 0

    import xlrd
    num33_lab = dict()
    msqx = r'D:\work\project\卡尔蔡司AR镀膜\第三批\33# DVS双面膜色2021.1~2021.6.xlsx'
    wb = xlrd.open_workbook(msqx)
    data = wb.sheet_by_name(r'Sheet1')
    rows = data.nrows
    need_title = [r'炉序号', r'L值',  r'A值', r'B值']
    lab_curve_index = [r'380', r'780']
    number_index = dict()
    number_labokng = dict()
    numb33_lab = dict()
    numokng = dict()
    a = 0
    for count in range(10):
        try:
            title = data.row_values(count)
            num33_ind, Lind, Aind, Bind = title.index(need_title[0]), title.index(need_title[1]), title.index(need_title[2]), title.index(need_title[3])
            labstart, labend = title.index(lab_curve_index[0]), title.index(lab_curve_index[1])
            a += 1
            break
        except:
            continue
    for i in range(a, rows):
        num33 = data.cell(i, num33_ind).value
        number_index[num33] = number_index.get(num33, 0) + 1
        if number_index[num33] == 4:
            L = data.cell(i, Lind).value
            A = data.cell(i, Aind).value
            B = data.cell(i, Bind).value
            label = lab_okng(L, [3.3, 6.8]) and lab_okng(A, [-2, 2]) and lab_okng(B, [-18, -15])
            number_labokng[num33] = label
            tmp = data.row_values(i)[labstart: labend + 1]
            tmp = [i for i in tmp if i != '']
            if len(tmp) == 81:
                numb33_lab[num33] = tmp
                l, a, b = calculate_Lab(tmp)
                res = lab_okng(l, [3.3, 6.8]) and lab_okng(a, [-2, 2]) and lab_okng(b, [-18, -15])
                numokng[num33] = res
    a = 0
    for num, lab in numokng.items():
        if lab != number_labokng[num]:
            print(lab, number_labokng[num])
            a += 1
    print(a)


