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
    files = [i for i in files if "EVT" in i]
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
            shutil.move(filename0, filename1)
            shutil.move(filename0_, filename1_)
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
    print("before remove: {}".format(len(files)))
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
            os.remove(r"D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件\\" + f)
    for f in csv:
        if "EVT" + f not in data:
            os.remove(r"D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件\\" + f)
    files = os.listdir(csv_dir)
    print("after remove: {}".format(len(files)))


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
            os.remove(full_path1)
            os.remove(full_path2)
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
    evtname_sensor = json.load(open(file_sensor_dict, 'r'))  # 21042718.CSV
    # 根据：evt_dict(evt:33)、number33_lab(33:lab)、file_sensor_dict(evt:sensor_value_list)
    for evt, thick in evt_thick_dict.items():
        thickness_sensor[thick] = evtname_sensor[evt[3:] + '.CSV']
    #     print(thick, '===', len(evtname_sensor[evt[3:]+'.CSV']))
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


if __name__ == "__main__":
    # csv_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件'
    csv_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC_all'
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
    refine_thick_lab = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\org_refine_thickness_lab_curve.json'
    oneone_evt_thick = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\oneone_evt_thickness.json'


    # check 膜厚设置、实测、rate值
    # rate_thickness_check(csv_dir)

    # 异常样本剔除
    # bad_sample_clean(refine_thick_lab, oneone_evt_thick)

    # step4.
    # thickness-sensor
    # thickness-lab
    # thick_lab_file = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\refine_thickness_lab_curve.json'
    # thick_lab = open(thick_lab_file, 'r')
    # thick_lab_dict = json.load(thick_lab)
    # thick_sensor_dict = json.load(open(thickness_sensor_file, 'r'))
    # assert len(thick_sensor_dict) == len(thick_lab_dict)
    # for thick, sensor in thick_sensor_dict.items():
    #     # sensor is a dict: sensor[sensor_name]=sensor_value
    #     for sensor_name, sensor_value in sensor.items():
    #         # tffresh(sensor_value)

    # js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\thick_hc_lab.json'
    # thick_hc_lab = json.load(open(js, 'r'))
    # for k, lab in thick_hc_lab.items():
    #     print(k)
    #     lab = [float(i) for i in lab]
    #     x = [380+5*i for i in range(len(lab))]
    #     plt.plot(x, lab, color='cornflowerblue', label='origin')
    #     plt.show()

    # file = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\机台文件说明.xlsx'
    # wb = xlrd.open_workbook(file)
    # data = wb.sheet_by_name('数据点')
    # title = data.row_values(1)
    # rows = data.nrows
    # sensor_part_dict = dict()
    # f = open(r'info_sensor.txt', 'w')
    # for i in range(1, rows):
    # sensor_part_dict[r'分段真空计'] = data.row_values(1)[:6]
    # sensor_part_dict[r'炉膛压力'] = data.row_values(1)[6]
    # sensor_part_dict[r'深冷机制冷温度'] = data.row_values(1)[7]
    # sensor_part_dict[r'炉膛氧流量'] = data.row_values(1)[8:10]
    # sensor_part_dict[r'膜厚及速率曲线'] = data.row_values(1)[10:14]
    # sensor_part_dict[r'晶振频率'] = data.row_values(1)[14]
    # sensor_part_dict[r'机台温度曲线'] = data.row_values(1)[15]
    # sensor_part_dict[r'离子源相关曲线'] = data.row_values(1)[16:24]
    # sensor_part_dict[r'深冷机曲线'] = data.row_values(1)[24]
    # data = json.dumps(sensor_part_dict)
    # with open(r'./info_sensor_dict.json', 'w', encoding="utf-8") as js_file:
    #     js_file.write(data)
    # for i in data.row_values(1):
    #     f.write(i+',')

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


    '''
    ar_coefficient  自回归系数: 衡量时序数据的的周期性、不可预测性和波动性
    augmented_dickey_fuller(x, param) 扩展迪基-福勒检验（ADF检验）: 测试一个自回归模型是否存在单位根，衡量时序数据的平稳性
    # change_quantiles(x, ql, qh, isabs, f_agg) 给定区间的时序数据描述统计 : 先用ql和qh两个分位数在x中确定出一个区间，然后在这个区间里计算时序数据的均值、绝对值、连续变化值。（浮点数）
    
    '''

    evt_7thick = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\oneone_evt_thickness.json'
    thick7_lab = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\org_refine_thickness_lab_curve.json'
    thick_hc_lab_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\thick_hc_lab.json'
    data_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC'
    thick14_hc3_sensor12_lab_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\thick14hc3sensor12_lab.json'
    # f = open(r'info_sensor.txt', 'r')
    # sen_list = (f.readlines()[0]).split(',')[:-1]
    sen_list = ['ACT_O1_QCMS_THICKNESS', 'ACT_O1_QCMS_RATE', 'ACT_O1_QCMS_THICKNESS_CH1', 'ACT_O1_QCMS_RATE_CH1']


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
                           thick14_hc3_sensor12_lab_js):
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
            full_file_path = os.path.join(data_dir, evtname[3:] + '.CSV')
            data = pd.read_csv(full_file_path, error_bad_lines=False)
            sensor12 = sensor_csv_feature(sen_list, data)
            thick14_hc3_sensor12 = thick14_hc3 + sensor12
            thick14hc3sensor12_lab[thick14_hc3_sensor12] = thick7_lab[thick7]
        data = json.dumps(thick14hc3sensor12_lab)
        with open(thick14_hc3_sensor12_lab_js, 'w') as js_file:
            js_file.write(data)


    add_sensor_feature(data_dir, evt_7thick, thick7_lab, thick_hc_lab_js, sen_list, thick14_hc3_sensor12_lab_js)
