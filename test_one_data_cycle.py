# coding=utf-8
'''
共16个周期, 抽一个作为test, 在剩下15个里找最佳model12_{}.pth进行eval().
edited by chen-jia  2021.0804
'''
import os
import json
from model_check import Model_Check

def prepare_data(data_):

    data_cycle_txts = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_data_txts'
    txts = os.listdir(data_cycle_txts)
    all_json = r'D:\work\project\卡尔蔡司AR镀膜\第三批\0705\thick14hc3sensor64_lab.json'
    all_json_ = json.load(open(all_json, 'r'))
    all_numbers = list(all_json_.keys())

    cycle_nums = []
    for ind, txt in enumerate(txts):
        f = open(os.path.join(data_cycle_txts, txt))
        numbers = f.readlines()[0].split(',')[:-1]
        numbers = list(set(numbers))
        numbers = [a for a in numbers if a in all_numbers]
        cycle_nums.append(numbers)

    # 选parte_15的作为test, 生成这个test data cycle的json
    tests_numbers = cycle_nums[7]
    test_js = dict()
    for number, thick10f16 in all_json_.items():
        if number in tests_numbers:
            test_js[number] = thick10f16
    data = json.dumps(test_js)
    with open(data_, 'w') as js_file:
        js_file.write(data)


    # # test 周期
    # txts = os.listdir(data_cycle_txts)
    # print(txts[7])
    # model1s = os.listdir(r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model1')
    # model2s = os.listdir(r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\cycle_model2')
    # print(model1s[7])


data_ = r'./test_js.json'
prepare_data(data_)

Model_Check(data_, targ=1)
Model_Check(data_, targ=2)


# f = open(r'D:\work\project\卡尔蔡司AR镀膜\第三批\0705\正背面_thickness_evtname.txt', 'r').readlines()
# for line in f:
#     print(line[:-1].split(',')[3:])
