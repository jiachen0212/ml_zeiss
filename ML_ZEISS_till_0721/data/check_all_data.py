# coding=utf-8
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import xlrd


def is_thick_modified(thick2, thick1):
    deta_thick = [abs(thick2[r] - thick1[r]) for r in range(len(thick2))]
    for deta_ in deta_thick:
        if deta_ > 0:
            return True
    return False


def show_pre_cur_deta_thick_and_lab(bad_pairs, save_dir, fig_name, deta_thick, lab1, lab2, pre_cx_or_cc, cur_cx_or_cc, flag=None):

    if flag in ['1', '0']:
        bad_pair_dir = os.path.join(save_dir, 'bad_pair')
        if not os.path.exists(bad_pair_dir):
            os.makedirs(bad_pair_dir)
        ok_pair_dir = os.path.join(save_dir, 'ok_pair')
        if not os.path.exists(ok_pair_dir):
            os.makedirs(ok_pair_dir)
    elif flag == '2':
        bad_pair_dir, ok_pair_dir = os.path.join(save_dir, 'cxcc'), os.path.join(save_dir, 'cxcc')
        if not os.path.exists(bad_pair_dir):
            os.makedirs(bad_pair_dir)

    tmp = ''
    if flag == '1':
        for ind, deta in enumerate(deta_thick):
            if abs(deta) > 0:
                tmp += 'L{}: {}, '.format(ind+1, deta)

    aa = [380+i*5 for i in range(81)]
    plt.ylim(0, 10)
    plt.plot(aa, lab1, color='pink', label='{}_pre'.format(pre_cx_or_cc))
    plt.plot(aa, lab2, color='cornflowerblue', label='{}_cur, thick2-1: {}'.format(cur_cx_or_cc, tmp))
    plt.legend()
    # plt.show()
    if fig_name in bad_pairs:
        plt.savefig(os.path.join(bad_pair_dir, "{}.png".format(fig_name)))
    else:
        plt.savefig(os.path.join(ok_pair_dir, "{}.png".format(fig_name)))
    plt.close()


def ok_lab_check(lab):
    '''
    同时判断滤波前(5频段采样的)后的值, 保留不是0的那个值作为本频段的值.
    lab: 没滤波前的curve
    '''
    lab_blue = savgol_filter(lab, 15, 5).tolist()
    new_lab = []
    for ind, v in enumerate(lab_blue):
        if v <= 0:
            if lab[ind] <= 0:
                return []
            else:
                new_lab.append(lab[ind])
        else:
            new_lab.append(v)
    assert len(new_lab) == 81

    return new_lab


def negative_check(lab):
    for a in lab:
        if a <= 0:
            return False

    return True


def run(all_data, bad_ovens):
    # 人工筛选出的: 膜厚不变, 前后炉曲线变化异常的数据对
    lines = open(r'./same_thick_bad.txt', 'r').readlines()
    bad_pairs = []
    for line in lines:
        bad_pairs.append(line[:-1])

    # 人工筛选出的: 膜厚变化, 前后炉曲线变化异常的数据对
    lines1 = open(r'bad_thick_modufued.txt', 'r').readlines()
    thick_modified_bad_pairs = []
    for line in lines1:
        thick_modified_bad_pairs.append(line[:-1])

    save_dir1 = r'./pre_cur_thick_and_lab/thick_modified'
    save_dir2 = r'./pre_cur_thick_and_lab/thick_same'
    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)

    count = 0
    lower_lab = 0
    a = [380 + i * 5 for i in range(81)]
    for data in all_data:
        ll = len(data)
        for i in range(ll-1):
            evt_name1, evt_name2 = data.iloc[i]['FileID'][3:], data.iloc[i + 1]['FileID'][3:]
            evt_name2_ = int(evt_name2[-3:])
            evt_name1_ = int(evt_name1[-3:])
            pre_oven = str(data.iloc[i]['OvenNo'])
            cur_oven = str(data.iloc[i + 1]['OvenNo'])
            if evt_name2_ - evt_name1_ < 2:
                if (pre_oven not in bad_ovens) and (cur_oven not in bad_ovens):
                    count += 1
                    thick1 = [float(a.strip("['] ")) for a in data.iloc[i]['Thickness'].split(',')]
                    thick2 = [float(a.strip("['] ")) for a in data.iloc[i + 1]['Thickness'].split(',')]
                    pre_lab = [float(a.strip("['] ")) for a in data.iloc[i]['single_lab_curve'].split(',')]
                    cur_lab = [float(a.strip("['] ")) for a in data.iloc[i + 1]['single_lab_curve'].split(',')]
                    # print(pre_lab)
                    # pre_lab1 = savgol_filter(pre_lab, 15, 5).tolist()
                    # cur_lab1 = savgol_filter(cur_lab, 15, 5).tolist()
                    # plt.plot(a, pre_lab, label='org')
                    # plt.plot(a, pre_lab1, label='blued')
                    # plt.plot(a, cur_lab, label='org')
                    # plt.plot(a, cur_lab1, label='blued')
                    # plt.legend()
                    # plt.savefig(r'./pre_cur_thick_and_lab/1/{}.png'.format(i))
                    # plt.close()
                    pre_lab = ok_lab_check(pre_lab)
                    cur_lab = ok_lab_check(cur_lab)
                    if len(pre_lab) > 0 and len(cur_lab) > 0:
                        fig_name = "{}_{}".format(evt_name1, evt_name2)
                        pre_cx_or_cc = data.iloc[i]['FilmCode_MES'].split('_')[-1]
                        cur_cx_or_cc = data.iloc[i+1]['FilmCode_MES'].split('_')[-1]
                        deta_thick = [np.round(thick2[i] - thick1[i], 2) for i in range(7)]
                        if is_thick_modified(thick2, thick1):
                            if pre_cx_or_cc == cur_cx_or_cc:
                                show_pre_cur_deta_thick_and_lab(thick_modified_bad_pairs, save_dir1, fig_name, deta_thick, pre_lab, cur_lab, pre_cx_or_cc, cur_cx_or_cc, flag='1')
                            else:
                                # 连续的正背数据对
                                show_pre_cur_deta_thick_and_lab(thick_modified_bad_pairs, save_dir1, fig_name,
                                                                deta_thick, pre_lab, cur_lab, pre_cx_or_cc,
                                                                cur_cx_or_cc, flag='2')
                        else:
                            show_pre_cur_deta_thick_and_lab(bad_pairs, save_dir2, fig_name, deta_thick, pre_lab, cur_lab, pre_cx_or_cc,
                                                    cur_cx_or_cc, flag='0')
                    else:
                        # print("pre_lab_len: {}, cur_lab_len: {}".format(len(pre_lab), len(cur_lab)))
                        lower_lab += 1
    print("all data: {}, negative_lab data: {}".format(count, lower_lab))

def oven_cx_cc_lab_thick_check():

    save_dir = r'oven_cx_cc_lab_thick_check'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ff = open(r'./oven_cc_cx_deta_thick.txt', 'w')
    all_ovens = []
    for data in all_data:
        ll = len(data)
        for i in range(ll):
            oven = str(data.iloc[i]['OvenNo'])
            if oven not in bad_ovens and oven not in all_ovens:
                all_ovens.append(oven)

    oven_cx_cc_thicks = dict()
    oven_cx_cc_labs = dict()
    for oven in all_ovens:
        print(oven)
        for data in all_data:
            ll = len(data)
            for i in range(ll):
                oven = str(data.iloc[i]['OvenNo'])
                if oven in all_ovens:
                    if oven not in oven_cx_cc_thicks:
                        oven_cx_cc_thicks[oven] = [[],[]]
                        oven_cx_cc_labs[oven] = [[], []]
                    cx_or_cc = data.iloc[i]['FilmCode_MES'].split('_')[-1]
                    lab = [float(a.strip("['] ")) for a in data.iloc[i]['single_lab_curve'].split(',')]
                    lab = savgol_filter(lab, 15, 5).tolist()
                    if ok_lab_check(lab):
                        thick = [float(a.strip("['] ")) for a in data.iloc[i]['Thickness'].split(',')]
                        if cx_or_cc == "CX":
                            oven_cx_cc_thicks[oven][0] = thick
                            oven_cx_cc_labs[oven][0] = lab
                        else:
                            oven_cx_cc_thicks[oven][1] = thick
                            oven_cx_cc_labs[oven][1] = lab
                        if len(oven_cx_cc_thicks[oven][0]) > 0 and len(oven_cx_cc_thicks[oven][1]) > 0:
                            thick_cc_minus_cx = [np.round(oven_cx_cc_thicks[oven][1][i] - oven_cx_cc_thicks[oven][0][i], 2) for i in range(7)]
                            line = "oven: {}, cc_thick-cx_thick: {}".format(oven, thick_cc_minus_cx)
                            # print(line)
                            ff.write(line+'\n')
                            plt.ylim(0, 10)
                            plt.plot([a*5+380 for a in range(81)], oven_cx_cc_labs[oven][0], color='pink', label='cx')
                            plt.plot([a * 5 + 380 for a in range(81)], oven_cx_cc_labs[oven][1], color='cornflowerblue', label='cc, cc-cx_thick:{}'.format(thick_cc_minus_cx))
                            # plt.show()
                            plt.legend()
                            save_name = os.path.join(save_dir, "{}.png".format(oven))
                            print(save_name)
                            plt.savefig(save_name)
                            plt.close()
                            continue


def find_best_blue_mothod():
    save_dir = r'./curve_step_compare'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file = r'/Users/chenjia/Downloads/Learning/SmartMore/1110_beijing/shanghai_data/膜色推优/ML_ZEISS-dev/9.18_10.8/9.18~10.8/33#单面(9.18-10.8).xlsx'
    wb = xlrd.open_workbook(file)
    data = wb.sheet_by_name(r'Sheet1')
    a = [i for i in range(380, 781)]
    a1 = [380 + 5 * i for i in range(81)]
    rows = data.nrows
    for i in range(1, rows):
        lab = data.row_values(i)[17]
        lab = [float(a) for a in lab.split(',')[190: 591]]
        lab1 = lab[::5]
        # if lab[20] < 10:
        #     lab = savgol_filter(lab, 15, 5).tolist()
        #     lab1 = savgol_filter(lab1, 15, 5).tolist()
        #     plt.plot(a, lab, label="1step_curve", color='cornflowerblue')
        #     plt.plot(a1, lab1, label="5step_curve", color='pink')  # cornflowerblue
        #     plt.legend()
        #     plt.grid()
        #     plt.savefig(os.path.join(save_dir, 'img{}_1_and_5_step.png'.format(i)))
        #     # plt.show()
        #     plt.close()

        if lab[20] < 10:
            lab2 = savgol_filter(lab1[:35], 15, 5).tolist() + savgol_filter(lab1[35: 45], 5,
                                                                            3).tolist() + savgol_filter(lab1[45:], 31,
                                                                                                        5).tolist()
            lab2_1 = savgol_filter(lab1, 15, 5).tolist()
            # plt.plot(a, lab, label="1_step_org", color='cornflowerblue')
            plt.plot(a1, lab2, label="savgol_filter_3bins", color='pink')  # cornflowerblue
            plt.plot(a1, lab2_1, label="savgol_filter_1bin", color='blue')
            plt.legend()
            plt.grid()
            # plt.savefig(os.path.join(save_dir, 'img{}_5step_blued.png'.format(i)))
            plt.show()
            # plt.close()


if __name__ == '__main__':

    data1 = r'./0910_all_data.csv'
    data2 = r'./0917_all_data.csv'
    data3 = r'./1008_all_data.csv'
    data4 = r'./1021_all_data.csv'
    data5 = r'./1105_all_data.csv'
    data6 = r'./1121_all_data.csv'

    all_data = [pd.read_csv(data1), pd.read_csv(data2), pd.read_csv(data3), pd.read_csv(data4),
                      pd.read_csv(data5), pd.read_csv(data6)]
    bad_ovens = ["33121090910", "33121110302", "33321091006", "33321090806", "33121090508"]

    run(all_data, bad_ovens)
    # oven_cx_cc_lab_thick_check()
    # find_best_blue_mothod()














'''
   else:
        diff = [abs(cur_lab[k] - pre_lab[k]) for k in range(len(pre_lab))]
        diff = np.mean(diff)
        if fig_name in bad_pairs:
            # print("lab_diff: {}".format(diff))
            tmp.append(diff)
        else:
            if diff >= 0.3339616229853126:
                print('-=-=-=-=')
                                
'''