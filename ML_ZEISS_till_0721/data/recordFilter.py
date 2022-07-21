# coding=utf-8
import os

import pandas as pd

'''
filter 各种数据筛选规则

'''


def split_CX_CC_data(all_data, cxcc):
    '''
    分别读取正背面数据, 落盘.csv

    '''
    CXCC = all_data[cxcc]
    CX_oven = CXCC.apply(lambda x: x[-2:] == "CX").reset_index(drop=True)
    CC_oven = CXCC.apply(lambda x: x[-2:] == "CC").reset_index(drop=True)
    all_CX = all_data[CX_oven].reset_index(drop=True)
    all_CC = all_data[CC_oven].reset_index(drop=True)

    if not os.path.isdir("./out"):
        os.mkdir("./out")
    all_CX.to_csv('./out/all_cx_data.csv', index=False)
    all_CC.to_csv('./out/all_cc_data.csv', index=False)

    print("正面数据量: {}, 背面数据量: {}".format(len(all_CX), len(all_CC)))

    return all_CX, all_CC


def get_single_cycle_data(all_data, cycle_index, cxcc):
    '''
    根据周期index, 读取某特定周期的数据

    '''
    indexs = all_data['clean_cycle_index']
    cycle_oven = indexs.apply(lambda x: x == cycle_index).reset_index(drop=True)
    single_cycle_data = all_data[cycle_oven].reset_index(drop=True)

    # print(single_cycle_data)
    # print("清洗周期{}, 数据量: {}".format(cycle_index, len(single_cycle_data)))

    if not os.path.isdir("./out"):
        os.mkdir("./out")

    single_cycle_data.to_csv('./out/cycle_{}_data.csv'.format(cycle_index), index=False)

    # 分别获取正背面
    CXCC = single_cycle_data[cxcc]
    cx_oven = CXCC.apply(lambda x: x[-2:] == "CX").reset_index(drop=True)
    single_cycle_cx = single_cycle_data[cx_oven].reset_index(drop=True)
    # print(single_cycle_cx)
    single_cycle_cx.to_csv('./out/cycle_{}_cx_data.csv'.format(cycle_index), index=False)

    cc_oven = CXCC.apply(lambda x: x[-2:] == "CC").reset_index(drop=True)
    single_cycle_cc = single_cycle_data[cc_oven].reset_index(drop=True)
    single_cycle_cc.to_csv('./out/cycle_{}_cc_data.csv'.format(cycle_index), index=False)

    print("cycle {}, 正面数据量: {}, 背面数据量: {}".format(cycle_index, len(single_cycle_cx), len(single_cycle_cc)))


if __name__ == "__main__":
    # load all data
    all_data = pd.read_csv(r'../resources/data/all_data.csv')
    math_title = {"炉号": "OvenNo", "创建时间": "CreationTime_MES", "正背面": "FilmCode_MES"}
    cxcc = math_title['正背面']
    # create_time = math_title['创建时间']

    # 数据拆分开正背面
    split_CX_CC_data(all_data, cxcc)

    # 获取特定清洗周期的数据
    cycle_index = 6
    get_single_cycle_data(all_data, cycle_index, cxcc)

    # 走一遍xixi的条件概率思路, 使用一个周期内的数据打通一遍. 可在pipline.py中实现.
