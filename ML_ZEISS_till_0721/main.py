from data import *

if __name__ == "__main__":
    # load all data
    all_data = pd.read_csv(r'.\resources\data\all_data.csv')
    math_title = {"炉号": "OvenNo", "创建时间": "CreationTime_MES", "正背面": "FilmCode_MES"}
    cxcc = math_title['正背面']
    # create_time = math_title['创建时间']

    # 数据拆分开正背面
    cc_data, cx_data = split_CX_CC_data(all_data, cxcc)

    # 获取特定清洗周期的数据
    cycle_index = 6
    get_single_cycle_data(all_data, cycle_index, cxcc)

    # 走一遍xixi的条件概率思路, 使用一个周期内的数据打通一遍. 可在pipline.py中实现.

    print("DONE!")
