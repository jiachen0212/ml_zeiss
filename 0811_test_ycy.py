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


if __name__ == "__main__":

    mean_thick = [float(i) for i in open(r'./thickness_mean.txt', 'r').readlines()[0].split(',')[:-1]]
    std_thick = [float(i) for i in open(r'./thickness_std.txt', 'r').readlines()[0].split(',')[:-1]]
    mean_sensor = [float(i) for i in open(r'./sensor_mean.txt', 'r').readlines()[0].split(',')[:-1]]
    std_sensor = [float(i) for i in open(r'./sensor_std.txt', 'r').readlines()[0].split(',')[:-1]]
    assert len(mean_sensor) == len(std_sensor)
    assert len(mean_thick) == len(std_thick)

    # load data
    data_ = r'D:\work\project\卡尔蔡司AR镀膜\第四批\0811\thick14hc3sensor64_lab.json'
    f = json.load(open(data_, 'r'))

    # 0811
    # 只测试10张吧先
    test10 = dict()
    c = 0
    for k, v in f.items():
        test10[k] = v
        c += 1
        if c >= 10:
            break
    print(len(test10))
    f = test10
    # 0811

    numbers = list(f.keys())

    # step1. 正向得到最佳model1,2
    Model_Check(f, mean_thick, std_thick, mean_sensor, std_sensor, targ=1)

    # step2. 获取推荐膜厚设置
    for number, thick10std16 in f.items():
        data = thick10std16
        ind = numbers.index(number)
        # epoch lr 搜索
        thickness = get_thickness(number, ind, mean_sensor, std_sensor, mean_thick, std_thick, data, flag=0)
        # 用最佳epoch run
        thickness = get_thickness(number, ind, mean_sensor, std_sensor, mean_thick, std_thick, data, flag=1)

        # 调整正反膜厚顺序
        thickness = thickness[5:] + thickness[:5]
        # print("{}: {}".format(number, [round(a, 2) for a in thickness]))

