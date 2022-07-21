# coding=utf-8
'''
FILENAME:        check_thickness_and_lab_pair_tool

AUTHORS:         jia.chen

START DATE:      2021.12.14

CONTACT:         jia.chen@smartmore.com

Description:
                check current, previous thickness and previous, current single_lab_curve

'''
import os
import matplotlib.pyplot as plt
import numpy as np
import xlrd

def get_thickness(evt_dir):
    evt_file = os.listdir(evt_dir)
    evt_file = [a for a in evt_file if ("EVT" in a) or ("evt" in a)]
    evt_thickness = dict()
    for evt_ in evt_file:
        csv_file = os.path.join(os.path.join(evt_dir, evt_))
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            thickness = []
            for line in lines:
                if 'Thickness' in line:
                    thickness.append(float(line.split(',')[4]))
                    evt_thickness[evt_[:-4]] = thickness
            # if len(thickness) != 7:
            #     print("evt file {} has less thickness data than expected {}".format(evt_file, 7))

    return evt_thickness


def process_match_file(match_file):
    wb = xlrd.open_workbook(match_file)
    data = wb.sheet_by_name(r'Sheet1')
    rows = data.nrows
    need_title = ['FileID', 'FileFilmCode', 'OvenNo']
    title = data.row_values(0)
    OvenNo_CCCX_EvtName = dict()
    EvtName_CXCC = dict()
    evtname, cx_or_cc, ovenno = title.index(need_title[0]), title.index(need_title[1]), title.index(
        need_title[2])
    for i in range(1, rows):
        EVT_name, CX_or_CX, OvenNo = data.cell(i, evtname).value, data.cell(i, cx_or_cc).value.split('_')[-1], data.cell(i, ovenno).value
        OvenNo_CCCX_EvtName["{}_{}".format(OvenNo, CX_or_CX)] = EVT_name
        EvtName_CXCC[EVT_name] = CX_or_CX

    return OvenNo_CCCX_EvtName, EvtName_CXCC


def get_single_labcurve(curve_file, OvenNo_CCCX_EvtName):
    wb = xlrd.open_workbook(curve_file)
    data = wb.sheet_by_name(r'Sheet1')
    rows = data.nrows
    need_title = ["测膜标签", "DataExpand"]
    title = data.row_values(0)
    OvenNo_CXCC, SingleLab_index = title.index(need_title[0]), title.index(need_title[1])
    evt_single_curve = dict()
    for i in range(1, rows):
        line = data.row_values(i)
        lab = line[SingleLab_index]
        lab = [float(a) for a in lab.split(',')[190: 591][::5]]
        try:
            evt_name = OvenNo_CCCX_EvtName[line[OvenNo_CXCC].upper()]
        except:
            continue
        evt_single_curve[evt_name] = lab

    return evt_single_curve


def evt_thick_and_lab(evt_thickness, evt_single_curve, EvtName_CXCC, save_dir):
    aa = [380+i*5 for i in range(81)]
    evts = list(evt_single_curve)
    evts = [a for a in evts if a in evt_thickness]
    lens = len(evts)
    for i in range(lens):
        evt_name1 = evts[i][3:]
        evt_name2 = "EVT{}".format(evt_name1[:-3] + str(int(evt_name1[-3:])+1))
        if evt_name2 in evts:
            # 连续炉数据对
            pre_thickness = [np.round(a, 2) for a in evt_thickness[evts[i]]]
            cur_thickness = [np.round(a, 2) for a in evt_thickness[evt_name2]]
            cx_cc_info1, cx_cc_info2 = EvtName_CXCC[evts[i]], EvtName_CXCC[evt_name2]
            cur_minus_pre_thickness = [np.round(cur_thickness[r]-pre_thickness[r], 2) for r in range(7)]
            plt.plot(aa, evt_single_curve[evts[i]], color='pink', label='pre_OvenNo: {}, thickness: {}'.format(cx_cc_info1, ''.join(str(a)+', ' for a in pre_thickness)))
            plt.plot(aa, evt_single_curve[evt_name2], color='cornflowerblue', label='cur_OvenNo: {}, cur-pre_thickness: {}'.format(cx_cc_info2, ''.join(str(a)+', ' for a in cur_minus_pre_thickness)))
            plt.legend()
            plt.savefig(os.path.join(save_dir, "{}_{}.png".format(evts[i], evt_name2)))
            plt.close()

if __name__ == '__main__':

    evt_file = r"C:\Users\VI8HCAR1\Desktop\33#数据查询\EVT数据"
    curve_file = r"C:\Users\VI8HCAR1\Desktop\33#数据查询\曲线数据（单面）"
    match_file = r"C:\Users\VI8HCAR1\Desktop\33#数据查询\炉号对应关系"

    # 12.14 data_check_tool for zeiss
    evt_thickness = get_thickness(evt_file)
    OvenNo_CCCX_EvtName, EvtName_CXCC = process_match_file(match_file)
    evt_single_curve = get_single_labcurve(curve_file, OvenNo_CCCX_EvtName)
    save_dir = r'./for_zeiss_check_thickness_and_lab'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    evt_thick_and_lab(evt_thickness, evt_single_curve, EvtName_CXCC, save_dir)
