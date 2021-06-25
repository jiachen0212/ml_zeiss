# coding=utf-8
import os

import xlrd

file = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#膜色文件与EVT文件对应表.xlsx'
wb = xlrd.open_workbook(file)
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
# cc_dir 背面数据对应维evt1
# print(len(evt_cc))
# count = 0
# ccs = os.listdir(r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC')
# ccs = [a for a in ccs if 'EVT' in a ]
# for i in ccs:
#     if i in evt_cc:
#         count += 1
# print(count)
# evt1是cc, evt2是cx
