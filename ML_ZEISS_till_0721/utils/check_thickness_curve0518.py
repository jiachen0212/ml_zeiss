# coding=utf-8
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import xlwt


def deta_thickness(diff_list):
    lys = ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7"]
    str_ = ''
    flag = 0
    for i, diff_ in enumerate(diff_list):
        if diff_ > 0:
            flag = 1
            str_ += "{}: {}, ".format(lys[i], diff_)
    if flag == 0:
        return "cur_thickness == pre_thickness"
    else:
        return str_[:-2]


def evt_thick_and_lab(evts, thicknesses, curevs, faces, types, save_dir):
    
    thicknesses = [[float(b) for b in a.strip("['] ").split(',')] for a in thicknesses]
    curevs = [[float(b) for b in a.strip("['] ").split(',')] for a in curevs]
    curevs = [savgol_filter(lab, 15, 5).tolist() for lab in curevs]
    # 检查连续炉膜厚和曲线变化是否符合规律
    aa = [380+i*5 for i in range(81)]
    lens = len(evts)
    for i in range(lens-1):
        evt1, evt2 = evts[i] % 1000, evts[i+1] % 1000
        type1, type2 = types[i], types[i+1]
        if abs(evt2 - evt1 == 2) <= 2:
            if (type1 == type2):
                pre_thickness = [np.round(a, 2) for a in thicknesses[i]]
                cur_thickness = [np.round(a, 2) for a in thicknesses[i+1]]
                cur_minus_pre_thickness = [np.round(cur_thickness[r]-pre_thickness[r], 2) for r in range(7)]
                plt.plot(aa, curevs[i], color='pink', label='pre_{}, thickness: {}'.format(faces[i], ''.join(str(a)+', ' for a in pre_thickness)))
                plt.plot(aa, curevs[i+1], color='cornflowerblue', label='cur_{}, cur-pre: {}'.format(faces[i+1], deta_thickness(cur_minus_pre_thickness)))
                plt.legend()
                plt.savefig(os.path.join(save_dir, "{}_{}.png".format(evts[i], evts[i+1])))
                plt.close()


def check_thick_and_curve(save_dir, compare_res_dir):
    if not os.path.isdir(compare_res_dir):
        os.mkdir(compare_res_dir)
    csvs = [a for a in os.listdir(save_dir) if '.csv' in a and 'all' not in a]
    csvs = [a for a in os.listdir(save_dir) if '.csv' in a and 'mix' not in a]
    for csv in csvs:
        print("comparing {}".format(csv))
        try:
            data = pd.read_csv(os.path.join(save_dir, csv))
        except:
            continue
        evts, thicknesses, single_curves, faces, types = [a for a in data['FileName']], [a for a in data['Thickness']], [a for a in data[
            'single_lab_curve']], [a for a in data['CCCX']], [a for a in data['Type']]
        compare_dir = os.path.join(compare_res_dir, csv[:-4])
        if not os.path.exists(compare_dir):
            os.mkdir(compare_dir)
        evt_thick_and_lab(evts, thicknesses, single_curves, faces, types, compare_dir)


def clean_zeiss_review(save_dir, zeiss_res_dir, xlsx_name):
    print('clean_zeiss_review')
    # 读取记录所有连续evt且类型一致的数列excel
    filePath = os.path.join(save_dir, xlsx_name)
    data = pd.read_excel(filePath, header=None)
    dataValue = data.values
    rowNum = len(dataValue)
    bad_evts = []
    sub_type_dirs = os.listdir(zeiss_res_dir)
    for sub_type_dir in sub_type_dirs:
        org_len = 0
        dir_ = os.path.join(zeiss_res_dir, sub_type_dir, r'差')
        try:
            ims = os.listdir(dir_)
        except:
            continue
        if len(ims) > 0:
            ims = [a[:-4] for a in ims]
            # ims = [a for a in ims if '._E' not in a]
            evts = []
            for im in ims:
                evt1, evt2 = im.split('_')
                if evt1 not in evts:
                    evts.append(evt1)
                    bad_evts.append(evt1)
                if evt2 not in evts:
                    evts.append(evt2)
                    bad_evts.append(evt2)
            csv_path = os.path.join(save_dir, "{}.csv".format(sub_type_dir))
            data = pd.read_csv(csv_path)
            org_len = len(data) // 2
            slim_evt = data.FileName.apply(lambda x: str(x) not in evts).reset_index(drop=True)
            slim_data = data[slim_evt].reset_index(drop=True)
            slim_data.to_csv(csv_path, index=False)
        csv_path = os.path.join(save_dir, "{}.csv".format(sub_type_dir))
        data1 = pd.read_csv(csv_path)
        if "mixface" not in csv_path:
            if org_len > 0:
                print("type: {}, origin: {}, slimed: {}, rate: {}".format(os.path.basename(csv_path), org_len, len(data1)//2, np.round(len(data1)/(2*org_len), 2)))

    # 高亮那些 bad evts
    book = xlwt.Workbook(encoding='utf-8')
    sheet = book.add_sheet('Sheet1')
    # 设置样式
    pattern = xlwt.Pattern()
    pattern.pattern = xlwt.Pattern.SOLID_PATTERN
    pattern.pattern_fore_colour = 5  # 背景颜色设置为5表示黄色
    style = xlwt.XFStyle()
    style.pattern = pattern
    for i in range(rowNum):
        # 13是evt_name的index
        if str(dataValue[i][13]) in bad_evts:
            for j in range(len(dataValue[i])):
                sheet.write(i, j, dataValue[i][j], style)
        else:
            for j in range(len(dataValue[i])):
                sheet.write(i, j, dataValue[i][j])
    os.remove(filePath)
    book.save(filePath)