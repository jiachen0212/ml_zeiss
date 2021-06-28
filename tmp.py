'''
# thickness列统计
f = open(r"D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\title.txt", 'r')
line = f.readlines()[0]
title = line[:-1].split(',')
thickness = []
for name in title:
    if "THICKNESS" in name:
        thickness.append(name)
data = pd.read_csv(r"D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件\21042719.CSV")
all = []
for thick in thickness:
    col = [i for i in data[thick]]
    print(np.var(col), thick)
    if np.var(col) > 1e-12:
        all.append(thick)
print(all)

import xlrd
# 对excel进行修改读写
from xlutils.copy import copy

book1 = xlrd.open_workbook(file)
book2 = copy(book1)
sheet = book2.get_sheet(0)  # 获取第几个sheet页
sheet.write(0, 0, 'time')
sheet.write(0, 1, 'id')
for i in range(474):
    # time
    sheet.write(i + 1, 0, i + 1)
    sheet.write(474 + 1 + i, 0, i + 1)
    sheet.write(474 * 2 + 1 + i, 0, i + 1)
    sheet.write(474 * 3 + 1 + i, 0, i + 1)

    # id
    sheet.write(i + 1, 1, 1)
    sheet.write(474 + 1 + i, 1, 2)
    sheet.write(474 * 2 + 1 + i, 1, 3)
    sheet.write(474 * 3 + 1 + i, 1, 4)

for i in range(2, 197):
    sheet.write(0, i, 'x')

book2.save(r"D:\work\project\卡尔蔡司AR镀膜\文档s\cj1.xls")  # 保存修改后excel

# show每一列sensor数据图
data = pd.read_csv(r"D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件\21042719.CSV")
js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\file_sensor.json'
with open(js, encoding="utf-8") as reader:
    js_file = json.load(reader)
    tmp_dict = js_file["21042719.CSV"]
    keys = list(tmp_dict.keys())
    for sensor_name in keys:
        if "THICKNESS" in sensor_name:
            col = [i for i in data[sensor_name]]
            x = [i for i in range(len(col))]
            plt.title(sensor_name)
            plt.plot(x, col)
            plt.show()

# test1. 时序分解
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

elecequip = read_csv(r"cj.csv")
result = seasonal_decompose(np.array(elecequip), model='additive', freq=80)
result.plot()
plt.show()

# 剔除全为0的列数据  some bug
feat_names = []
for i, value in enumerate(X):
    if i == 0:
        feat_names = [i for i in value]
print(len(feat_names))
# for feat in feat_names:
#     col_data = [i for i in X[feat]]
#     if set(col_data) == {0}:
#         X.drop([feat], axis=1)
# print(X)


# 0624 check_data.py

# step3.
miss_ = open(r"D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\miss_thickness.txt", 'w')
evt_thickness_val = r"D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\evt_thickness_val.json"
evt_process_time = r"D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\evt_process_time.json"
# get_evt_thickness(csv_dir, miss_, evt_thickness_val, evt_process_time)

# step4.
useful_csv = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\seven_levels_csv.txt'
# select_csv(useful_csv, evt_thickness_val, csv_dir, file_sensor_dict)

# step5. 构建特征工程获取特征
# feature_project(file_sensor_dict, )

# 0616new
# step6.
# cut_cols(csv_dir, titles)
# step7.
dir_ = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC'
get_sub_title(dir_)

js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\file_sensor.json'
all = []
with open(js, encoding="utf-8") as reader:
    js_file = json.load(reader)
    print(len(js_file))
    # for job_name, sensor_dict in js_file.items():
    #     for process, value in sensor_dict.items():
    #         print(process)
    #         print(value)

def tffresh():
    def fun(f):
        return f

        # tsfresh
        # https://github.com/blue-yonder/tsfresh
        # https://tsfresh.readthedocs.io/en/latest/text/quick_start.html

    from tsfresh import extract_features

    file = r"D:\work\project\卡尔蔡司AR镀膜\文档s\cj.csv"
    f = pd.read_csv(file, encoding="utf-8")
    # 写一个剔除无效数据列的函数: 方差为0, 空数据列
    f = fun(f)
    extracted_features = extract_features(f, column_id="id", column_sort="time")
'''


import json
f = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\refine_thickness_lab_curve.json'
with open(f, encoding="utf-8") as reader:
    dict_ = json.load(reader)
    for k, v in dict_.items():
        print(k)





evt33 = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#膜色文件与EVT文件对应表.xlsx'
wb = xlrd.open_workbook(evt33)
data = wb.sheet_by_name('Sheet1')
number33_evts = dict()
rows = data.nrows
for i in range(1, rows):
    number33 = data.cell(i, 2).value
    if number33 not in number33_evts:
        number33_evts[number33] = []
    number33_evts[number33].append(data.cell(i, 5).value)

CC_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC'
CX_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CX'
dirs = [CC_dir, CX_dir]
cxs = os.listdir(CX_dir)
ccs = os.listdir(CC_dir)
f = open(r'正背面_thickness_evtname.txt', 'w')

evt_cc = []
for num33, evt_list in number33_evts.items():
    if len(evt_list) == 2:
        if (evt_list[1] + '.CSV' in cxs and evt_list[0] + '.CSV' in ccs) or (
                evt_list[0] + '.CSV' in cxs and evt_list[1] + '.CSV' in ccs):
            evts = []
            for dir_ in dirs:
                for i in range(2):
                    if os.path.exists(os.path.join(dir_, evt_list[i] + '.CSV')):
                        evts.append(os.path.join(dir_, evt_list[i] + '.CSV'))
            thickness1 = []
            thickness2 = []
            # print(evts)
            evt1, evt2 = evts[0], evts[1]
            with open(evt1, 'r') as file:
                for line in file:
                    if "Thickness" in line:
                        thickness1.append(line.split(',')[4])
            with open(evt2, 'r') as file:
                for line in file:
                    if "Thickness" in line:
                        thickness2.append(line.split(',')[4])
            # if thickness1 != thickness2:
            f.write(num33 + ',')
            f.write(evt1.split('\\')[-1] + ',' + evt2.split('\\')[-1] + ',')
            f.write(''.join(str(i) + ' ' for i in thickness1) + ',')
            f.write(''.join(str(i) + ' ' for i in thickness2) + '\n')
            evt_cc.append(evt1.split('\\')[-1])

