# coding=utf-8
import xlrd
import matplotlib.pyplot as plt
import json


def show_y_pred(y):
    plt.xlabel("Wave-length")
    plt.ylabel("Reflectance")
    x = [380 + 5 * i for i in range(81)]
    for i in range(70):
        single_y = y[i]
        for a in single_y:
            if a < 0:
                plt.plot(x, single_y, color='pink', label='real curve')
                # 添加网格
                plt.grid()
                plt.show()


file = r'D:\work\project\卡尔蔡司AR镀膜\正背面数据\8-27\33# Filmetrics 单面数据.csv'
# number: [z_lab, b_lab]
info_title = ['AREquipmentSortNo', 'ARTestLabelNo', '380', '780']
number_zblab = dict()
labs = []
nums = []
with open(file, 'r') as f:
    for ind, line in enumerate(f):
        line = line.split(',')
        if ind == 0:
            ind1, ind2, ind3, ind4 = line.index(info_title[0]), line.index(info_title[1]), line.index(info_title[2]), line.index(info_title[3])
        else:
            lab = line[ind3: ind4+1]
            lab = [float(lab[i*5]) for i in range(81)]
            lab = [max(0.0, a) for a in lab]
            assert len(lab) == 81
            labs.append(lab)
            nums.append(line[ind1])
lens = int(len(nums)/2)
nums = [nums[2*i] for i in range(lens)]

for i in range(lens):
    number_zblab[nums[i]] = [labs[2*i], labs[2*i+1]]
data = json.dumps(number_zblab)
with open(r'./zb_face.json', 'w') as js_file:
    js_file.write(data)













