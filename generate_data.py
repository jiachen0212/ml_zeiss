# coding=utf-8
'''
按照xixi的方案, 生成实验数据

## 特征工程

train / val .csv

#### 序号

* OvenNo

#### 当前实验特征

* thinkness_front_1,3,4,5,6:  正面五层可调节膜厚

* thinkness_back1,3,4,5,6:  反面五层可调节膜厚
* 构造特征: 10d可调节膜厚相比于上一炉的变化值 d0,d1,.....,d9

#### 机器状态特征

* cycle: 所处清洗周期(categories)

* last_curve_385,390,...,780：上一炉的81d Lab曲线

* L_value: 上一炉的L值
* A_value: 上一炉的A值
* B_value: 上一炉的B值
* start_time: 上一炉从start到step1的时间差(该特征数值会随炉数逐渐增加，反映机器状态)
* material: 耗材特征

#### 目标特征

* curve_385,390,...,780: 当前实验的81d Lab曲线
* sensor数据中的特征(需要特征工程)



## 模型

输入：feature_list = [当前实验特征, 机器状态特征]

输出：81d Lab曲线 curve_385,390,...,780

辅助输出或中间输出: sensor特征

该模型构建了最后一次试验后，膜厚调整值到Lab曲线的映射关系；模型可为树模型也可为nn；

通过最优化算法调整输入膜厚调节值即膜厚推荐值d0,d1,...d9，可最小化模型输出与标准Lab曲线的差异；最优化算法可为暴力搜索，贝叶斯优化抑或是梯度反传；



## 推理流程

模型 Model， 当前实验特征可由d0到d9表示

估计Lab = Model (d0,d1,...,d9, other_feature)

目标函数 = 估计Lab - 标准Lab

在可调整的约束范围内优化d0到d9，使得目标函数最小化

得到膜厚调整推荐值d0，d1， ....， d9


'''

from torch import nn
import json
from data_post_process import data_process
import xlrd
import os
from util import calculate_Lab


pair_file = r'D:\work\project\卡尔蔡司AR镀膜\第三批\匹配关系2021.1~2021.6.xlsx'
lab_file = r'D:\work\project\卡尔蔡司AR镀膜\第三批\33# DVS双面膜色2021.1~2021.6.xlsx'
all_data = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\第三批\0705\thick14hc3sensor64_lab.json', 'r'))
file = r'D:\work\project\卡尔蔡司AR镀膜\第三批\镀膜炉工艺记录2021.1~2021.6.xlsx'
evtcccx_dir = r'D:\work\project\卡尔蔡司AR镀膜\第三批\33机台文件'

# 按时间排序的, 有lab曲线的所有numbers
numbers = []
for k, v in all_data.items():
    numbers.append(k)


# def get_materials():
    # materials = dict()
    # wb = xlrd.open_workbook(file)
    # info_title = ['炉序号', '离子枪灯丝', '电子枪灯丝', '挡板']
    # data = wb.sheet_by_name('Sheet1')
    # title = data.row_values(2)
    # ind, ind1, ind2, ind3 = title.index(info_title[0]), title.index(info_title[1]), title.index(info_title[2]), title.index(info_title[3])
    # rows = data.nrows
    #
    # for i in range(3, rows):
    #     number_cur = data.cell(i, ind).value
    #     if number_cur in numbers and number_cur not in materials:
    #         lzq, dzq, db = data.cell(i, ind1).value, data.cell(i, ind2).value, data.cell(i, ind3).value
    #         materials[number_cur] = [lzq, dzq, db]
    #
    # print(len(materials))
    # 耗材特征一加上来, 数据量少了很多, 所以就暂时还是不考虑吧..



def get_cycle_index(pair_file, all_numbers):

    # 得到各个清洗周期的时间点
    r = open(r'D:\work\project\卡尔蔡司AR镀膜\xixi\cycle.txt', 'r').readlines()
    times = []
    for line in r:
        month, day = line.split('/')[:2]
        value = 60 * int(month) + int(day)
        times.append(value)

    number_cycle_index = dict()
    tmp_number = dict()
    wb = xlrd.open_workbook(pair_file)
    # info_title = ['OvenNo', 'FileID']
    data = wb.sheet_by_name('Sheet1')
    # title = data.row_values(0)
    rows = data.nrows
    for i in range(1, rows):
        cur_num = data.cell(i, 2).value
        if cur_num in all_numbers and cur_num not in tmp_number:
            tmp_number[cur_num] = 1
            evt_name = data.cell(i, 5).value[6:-2]
            evt_name_value = int(evt_name[0]) * 60 + int(evt_name[1:])
            for ind in range(len(times)-1):
                if times[ind] <= evt_name_value < times[ind+1]:
                    number_cycle_index[cur_num] = ind

    data = json.dumps(number_cycle_index)
    with open(r'./number_cycle_index.json', 'w') as js_file:
        js_file.write(data)



# 组合训练数据的所有信息
# number: [thickness10, lab81, lab3, start_cost_time, cycle_index]
single_cycle_data = dict()
machine_start_cost_time = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\第三批\0705\number_start_cost_time.json', 'r'))
number_cycle_index = json.load(open(r'./number_cycle_index.json', 'r'))

for num in numbers:
    thick10 = [float(a) for a in all_data[num][0].split(',')[:10]]
    lab_value = calculate_Lab(all_data[num][1])
    try:
        cost_time = machine_start_cost_time[num]
        cycle_index = number_cycle_index[num]
    except:
        continue
    # 10+3+2
    single_cycle_data[num] = thick10 + lab_value + cost_time + [cycle_index]
    # print(single_cycle_data[num])
data = json.dumps(single_cycle_data)
with open(r'./all_data_deta_thickness_lab_cost_time_cycle_index.json', 'w') as js_file:
    js_file.write(data)


# step4.
# train_data json
# [deta_thickness, pre_lab, cur_machine_start_cost_time]
train_data = dict()
lens = len(single_cycle_data)
for i in range(1, lens):
    try:
        pre_thick = single_cycle_data[numbers[i-1]][:10]
        cur_thick = single_cycle_data[numbers[i]][:10]
        deta_thick = [cur_thick[a] - pre_thick[a] for a in range(len(cur_thick))]
        pre_lab_value = single_cycle_data[numbers[i-1]][10:13]
    except:
        continue
    # 10+3+2+1 81
    train_data[numbers[i]] = deta_thick + pre_lab_value + single_cycle_data[numbers[i]][13:] + all_data[numbers[i]][1]
    # print(deta_thick + pre_lab_value + single_cycle_data[numbers[i]][13:])
print("train data size: {}".format(len(train_data)))
data = json.dumps(train_data)
with open(r'./all_data_train.json', 'w') as js_file:
    js_file.write(data)



