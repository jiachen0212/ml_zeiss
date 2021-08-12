# coding=utf-8
import json
import matplotlib.pyplot as plt
import numpy as np


def show_thickness(thicks, lq_numbers, nums):
    x = [i for i in range(10)]
    for ind, thick in enumerate(thicks):
        cur_number = nums[ind]
        if cur_number not in lq_numbers:
            plt.plot(x, thick, color='black')
        # else:
        #     plt.plot(x, thick, color='pink')
    plt.show()



def show_lab(labs, lq_numbers, nums):
    best = [5.52, 3.53, 1.97, 1.28, 0.74, 0.7, 0.85, 1.05, 1.23, 1.43, 1.63, 1.82, 1.84, 1.8, 1.75, 1.73, 1.64, 1.49,
            1.39, 1.31, 1.23, 1.16, 1.03, 0.91, 0.85, 0.86, 0.84, 0.77, 0.71, 0.64, 0.61, 0.61, 0.58, 0.56, 0.53, 0.46,
            0.46, 0.44, 0.41, 0.43, 0.4, 0.39, 0.36, 0.25, 0.19, 0.17, 0.21, 0.19, 0.17, 0.17, 0.2, 0.2, 0.16, 0.20,
            0.26, 0.35, 0.41, 0.57, 0.64, 0.71, 0.9, 1.04, 1.17, 1.27, 1.43, 1.56, 1.82, 2.07, 2.4, 2.72, 3.02, 3.33,
            3.58, 3.87, 3.97, 4.34, 4.57, 4.73, 5.03, 5.45, 5.94]
    x = [i for i in range(81)]
    for ind, lab in enumerate(labs):
        lab = [float(a) for a in lab]
        cur_number = nums[ind]
        if cur_number in lq_numbers:
            plt.plot(x, lab, color='black')
        # else:
        #     plt.plot(x, thick, color='pink')
    plt.plot(x, best, color='pink')
    plt.show()



all_data = json.load(open(r'D:\work\project\卡尔蔡司AR镀膜\第三批\0705\thick14hc3sensor64_lab.json', 'r'))
print(len(all_data))
# for k in list(all_data.keys()):
#     if k == '33321021303':
#         del all_data[k]
# print(len(all_data))
# data = json.dumps(all_data)
# with open(r'D:\work\project\卡尔蔡司AR镀膜\第三批\0705\thick14hc3sensor64_lab.json', 'w') as js_file:
#     js_file.write(data)


lq_numbers = open(r'./lq_numbers.txt', 'r').readlines()[0].split(',')[:-1]
print(len(lq_numbers))
nums = []
thick10s = []
labs = []
for number, thick10std16_lab in all_data.items():
    nums.append(number)
    thick10 = thick10std16_lab[0].split(',')[:-1][:10]
    thick10s.append([float(a) for a in thick10])
    lab = thick10std16_lab[1]
    labs.append(lab)
# show_thickness(thick10s, lq_numbers, nums)
show_lab(labs, lq_numbers, nums)
for lq_n in lq_numbers:
    ind = nums.index(lq_n)
    print(thick10s[ind])

print("mean thickness: {}".format(np.mean(thick10s, axis=0)))


