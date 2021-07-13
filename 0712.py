# coding=utf-8
import xlrd
import json
import numpy as np
from util import weighted_mse

# f = r'D:\work\project\卡尔蔡司AR镀膜\第三批\0705\thick14hc3sensor64_lab.json'
# data = json.load(open(f, 'r'))
#
# def lab_okng(v, v_range):
#     V = float(v)
#     if V >= v_range[0] and V <= v_range[1]:
#         return 1
#     return 0
#
# nums = list(data.keys())
# f = r'D:\work\project\卡尔蔡司AR镀膜\第三批\33# DVS双面膜色2021.1~2021.6.xlsx'
# wb = xlrd.open_workbook(f)
# data = wb.sheet_by_name(r'Sheet1')
# rows = data.nrows
# need_title = [r'炉序号', '380', '780']
# lab_title = [r'L值', r'A值', r'B值']
# number_lab_index = dict()
# number_labokng = dict()
# number_lab = dict()
# for count in range(10):
#     try:
#         title = data.row_values(count)
#         num33_index, lab_index1, lab_index2 = title.index(need_title[0]), title.index(need_title[1]), title.index(
#             need_title[2])
#         Lind, Aind, Bind = title.index(lab_title[0]), title.index(lab_title[1]), title.index(lab_title[2])
#         a = count + 1
#         break
#     except:
#         continue
# for i in range(a, rows):
#     number33_cur = data.cell(i, num33_index).value
#     if number33_cur in nums:
#         number_lab_index[number33_cur] = number_lab_index.get(number33_cur, 0) + 1
#         if number_lab_index[number33_cur] == 4:
#             tmp = data.row_values(i)[lab_index1: lab_index2 + 1]
#             tmp = [i for i in tmp if i != '']
#             if len(tmp) == 81:
#                 L = data.cell(i, Lind).value
#                 A = data.cell(i, Aind).value
#                 B = data.cell(i, Bind).value
#                 label = lab_okng(L, [3.3, 6.8]) and lab_okng(A, [-2, 2]) and lab_okng(B, [-18, -15])
#                 number_labokng[number33_cur] = label
#                 # 只记录ng的lab曲线
#                 if label == 0:
#                     number_lab[number33_cur] = tmp
# oks = 0
# for k, v in number_labokng.items():
#     oks += v
# print("第4层镜片总数: {}, ok数目: {}, lab曲线ok合格率: {}".format(len(number_labokng), oks, oks / len(number_labokng)))
# print(len(number_lab))
# num_corr = dict()
# for num, lab in number_lab.items():
#     corr = weighted_mse(lab)
#     num_corr[num] = corr
#
#
# little_ng = dict()
# num_corr = sorted(num_corr.items(), key=lambda x: x[1], reverse=True)
# for i in num_corr[:20]:
#     little_ng[i[0]] = i[1]
# data = json.dumps(little_ng)
# with open('./large_ng', 'w') as js_file:
#     js_file.write(data)


import matplotlib.pyplot as plt
lab_s = np.load(r'./start_lab.npy')
start = np.load(r'./start_x.npy')

lab_m = np.load(r'./modified_lab.npy')
modify = np.load(r'./modified_x.npy')

f16lab = json.load(open(r'./f16lab.json', 'r'))
f1633 = json.load(open(r'./f1633.json', 'r'))
lab_num33 = dict()

for f16, lab in f16lab.items():
    lab_num33[''.join(str(i)+',' for i in lab)] = f1633[f16]

start_num33 = []
for s_lab in lab_s:
    s_lab = ''.join(str(i)+',' for i in s_lab)
    num33 = lab_num33[s_lab]
    start_num33.append(num33)
print(start_num33)



end_num33 = []
for m_lab in lab_m:
    m_lab = ''.join(str(i)+',' for i in m_lab)
    num33 = lab_num33[m_lab]
    end_num33.append(num33)
print(end_num33)


# 根据start,找modify的index
ind_ = []
for start_num in start_num33:
    ind_.append(end_num33.index(start_num))

diff_all = []
x = [i for i in range(15)]
for i in range(20):
    s = start[i]
    m = modify[ind_[i]]
    diff = [m[i]-s[i] for i in range(15)]
    diff_ = [abs((m[i]-s[i])/(s[i]+1e-5)) for i in range(15)]
    print([i for i in range(1, 15) if diff_[i] > 0.01])
    print([diff_[i]*100 for i in range(1, 15)])
    diff_all.append(diff)

