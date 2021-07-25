# coding=utf-8
'''
    edited by chen-jia 2021.0722

'''

import os
import xlrd

clean_data_dir = r'D:\work\project\卡尔蔡司AR镀膜\大清洗'
fs = os.listdir(clean_data_dir)
fs = [i for i in fs if "evt" in i]
thickness = []
steps = []
clean_start_end_time = []
count = 0
for f in fs:
    start_time = ''
    path = os.path.join(clean_data_dir, f)
    ll = len(open(path, 'r').readlines())
    data = open(path, 'r')
    tmp = []
    for index, line in enumerate(data):
        # if 'Thickness' in line:
        #     thickness.append(line.split(',')[4])
        # if 'Start Step No.' in line:
        #     steps.append(line.split(',')[2])

        if 'Production Start :' in line:
            tmp.append(line.split(',')[1])
            tmp.append(line.split(',')[2][:-1])
        if len(tmp) == 2 and index == ll - 1:
            tmp.append(line.split(',')[0])
            clean_start_end_time.append(tmp)




# y_2020 = []
# y_2021 = []
# for time in clean_start_end_time:
#     nyr = time[0]
#     year = nyr.split('/')[-1]
#     if year == "2020":
#         y_2020.append(year)
#     else:
#         y_2021.append(year)
# print(len(y_2020))
# print(len(y_2021))

nyr_start_end_pair = dict()
for time in clean_start_end_time:
    if time[0] not in nyr_start_end_pair:
        nyr_start_end_pair[time[0]] = [time[1:]]
    else:
        nyr_start_end_pair[time[0]].append(time[1:])


third_data = r'D:\work\project\卡尔蔡司AR镀膜\第三批\33机台文件'
fs = os.listdir(third_data)
fs = [i for i in fs if 'evt' in i]

# for k, v in nyr_start_end_pair.items():
#     cur_nyr = k
time_evt = dict()
for f in fs:
    evt_path = os.path.join(third_data, f)
    evt_data = open(evt_path, 'r')
    for ind, line in enumerate(evt_data):
        if 'Production Start :' in line:
            time_nyr = line.split(',')[1]
            if time_nyr not in time_evt:
                time_evt[time_nyr] = [f]
            else:
                time_evt[time_nyr].append(f)

nyr = list(nyr_start_end_pair.keys())
for k in nyr:
    if k.split('/')[-1] == '2020':
        del nyr_start_end_pair[k]
print(nyr_start_end_pair)
print('\n')

part_evts = []
for time, evt in time_evt.items():
    if time.split('/')[-1] == '2021' and time.split('/')[0] == '5' and 5 <= int(time.split('/')[1]) <= 12:
        part_evts.extend(evt)

    # for test: 12个周期的模型好了后, 用4月底的这批数据来选中一个最接近的子model
    # if time.split('/')[-1] == '2021' and time.split('/')[0] == '4' and int(time.split('/')[1]) >= 29:
    #     part_evts.extend(evt)
print(len(part_evts))

# 截至到2021/1 33条样本, 截至到2021/2月 67条样本
number_evtname = r'D:\work\project\卡尔蔡司AR镀膜\第三批\匹配关系2021.1~2021.6.xlsx'
wb = xlrd.open_workbook(number_evtname)
data = wb.sheet_by_name(r'Sheet1')
rows = data.nrows
evt_number = dict()
for i in range(1, rows):
    number = data.cell(i, 2).value
    evt = data.cell(i, 5).value
    evt_number[evt] = number
all_evt_names = list(evt_number.keys())

number33s = []
for evt in part_evts:
    tmp = evt[:-4].upper()
    if tmp in all_evt_names:
        number33s.append(evt_number[tmp])
print(len(number33s))
f = open(r'./part_clean_number.txt', 'w')
for num in number33s:
    f.write(num+',')










