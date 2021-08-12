# coding=utf-8
'''
共16个周期, 抽一个作为test, 在剩下15个里找最佳model12_{}.pth进行eval().
edited by chen-jia  2021.0804
'''
import os
import json
from online_test_ind_add1 import Model_Check   # from online_test_ind_add1_1 import Model_Check

# from online_test import Model_Check              # from online_test1 import Model_Check
# from online_test_ind1 import Model_Check         # from online_test_ind import Model_Check

from Get_Thickness import get_thickness



# 0810 0811 ycy
if __name__ == "__main__":

    mean_thick = [float(i) for i in open(r'./thickness_mean.txt', 'r').readlines()[0].split(',')[:-1]]
    std_thick = [float(i) for i in open(r'./thickness_std.txt', 'r').readlines()[0].split(',')[:-1]]
    mean_sensor = [float(i) for i in open(r'./sensor_mean.txt', 'r').readlines()[0].split(',')[:-1]]
    std_sensor = [float(i) for i in open(r'./sensor_std.txt', 'r').readlines()[0].split(',')[:-1]]
    assert len(mean_sensor) == len(std_sensor)
    assert len(mean_thick) == len(std_thick)

    # load data
    numbers = ['33338882233']
    data_ = r'D:\work\project\卡尔蔡司AR镀膜\0812\finally.json'

    # step1. 正向得到最佳model1,2
    Model_Check(data_, mean_thick, std_thick, mean_sensor, std_sensor, targ=1)

    # step2. 获取推荐膜厚设置
    f = json.load(open(data_, 'r'))
    for number, thick10std16 in f.items():
        data = thick10std16
        ind = numbers.index(number)
        # epoch lr 搜索
        thickness = get_thickness(number, ind, mean_sensor, std_sensor, mean_thick, std_thick, data, flag=0)
        # 用最佳epoch run
        thickness = get_thickness(number, ind, mean_sensor, std_sensor, mean_thick, std_thick, data, flag=1)

        # 调整正反膜厚顺序
        thickness = thickness[5:] + thickness[:5]
        print("{}: {}".format(number, [round(a, 2) for a in thickness]))



















# txts = os.listdir(r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_data_txts')
# c = 0
# for txt in txts:
#     f = open(os.path.join(r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_data_txts', txt))
#     nums = f.readlines()[0].split(',')[:-1]
#     print(len(nums))
#     c += len(nums)
# print(c)