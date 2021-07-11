# coding=utf-8
import json
import os
import random
import shutil
import copy
import numpy as np
import pandas as pd
import tsfresh as tsf
import xlrd

from check_data import rate_thickness_check
from read_csvs import get8step_sensor_feature

'''
    edited by chen-jia 2021.0625    

'''


class data_post_process():
    '''
    关联膜厚设置, lab曲线

    '''

    def __init__(self, evt33, membrane, process_data, base_data_dir, CC_dir, CX_dir, thick14_hc3_sensor16_lab_js,
                 thick14_hc3_sensor80_lab_js, feature135_lab_js, flag=0):
        '''
        :param MachineName-kinds:   ['1.56_DVS_CC', '1.56_DVS_CX', '1.6&1.67_DVS_CC', '1.6&1.67_DVS_CC_hpkf', '1.6&1.67_DVS_CX', '1.6&1.67_DVS_CX_hpkf', '1.6_DVSUN_CC']

        :param n_thick: 镜片镀膜层数, 7,5,4 等不等。模型训练时,需保证数据都是同种层数的镀膜数

        :param evt33:  33#膜色文件与EVT文件对应表.xlsx
                       note： 可将 33121052204 编号 和 EVT21050506 对应上
        :param membrane:  膜色数据.xlsx
                       note： 可将 33121060503 编号 和 lab 膜色曲线 对应上
        :param process_data:  工艺记录.xlsx
                       note： 含生产周期信息, 解决一种膜厚设置对应多条lab曲线(evt_file)问题

        '''

        self.base_data_dir = base_data_dir
        self.n_thickness = 7
        self.evt33 = evt33
        self.membrane = membrane
        self.evt_dict = dict()
        self.process_data = process_data
        self.CC_dir = CC_dir
        self.CX_dir = CX_dir
        self.thick14_hc3_sensor16_lab_js = thick14_hc3_sensor16_lab_js
        self.thick14_hc3_sensor80_lab_js = thick14_hc3_sensor80_lab_js
        self.feature135_lab_js = feature135_lab_js
        # 0711
        self.evt_pair = r'./正背面_thickness_evtname.txt'
        self.number33_thicklab_js = r'./number33_thicklab.json'
        self.face = r'背面'
        self.number33_thick14hc3lab_js = r'./number33_thiclhc.json'
        self.num_evt12 = r'./num_evt12.json'
        self.sen_list = ['ACT_O1_QCMS_THICKNESS', 'ACT_O1_QCMS_RATE', 'ACT_O1_QCMS_THICKNESS_CH1',
                         'ACT_O1_QCMS_RATE_CH1']
        self.csv_dict_js = r'./evtname_sensor_name_value.json'


    # def __call__(self, ):
    def run(self, ):

        # step1.
        # EVT302987.CSV 统一转换为小写evt302987.csv
        # lower_filename(self.base_data_dir, self.tmp_dir)
        # split出正背面 cc cx
        # split_CX_CC(self.base_data_dir)

        # step2.
        get_evtpair_info(self.CC_dir, self.CX_dir, self.evt33, self.evt_pair, self.n_thickness)
        number33_thick14lab(self.membrane, self.evt_pair, self.number33_thicklab_js, self.num_evt12)
        # 添加3个耗材维度特征
        number33_thick14hc3lab(self.process_data, self.number33_thicklab_js, self.face, self.number33_thick14hc3lab_js)
        # 添加thickness,rate等sensor列的时序特征(4*4=16dims)
        add_sensor_feature(self.base_data_dir, self.num_evt12, self.number33_thick14hc3lab_js, self.sen_list, self.thick14_hc3_sensor16_lab_js)
        # 添加8个step的时序特征(len=64 8*4*2)
        get8step_sensor_feature(self.num_evt12, self.base_data_dir, self.csv_dict_js, self.thick14_hc3_sensor16_lab_js, self.thick14_hc3_sensor80_lab_js, self.sen_list)
        # 再加入19列有意义数据的38维特征
        all_usful_sensor_except_thickness(self.base_data_dir, self.num_evt12,
                                          self.thick14_hc3_sensor80_lab_js, self.feature135_lab_js)
        # import check_data.py 中的函数实现部分数据清洗功能
        # rate_thickness_check(self.data_dir)  # 膜厚设置\实测值diff与rate*2对比

    def clean_data_machineid(self, ):
        '''
        数据清洗: 包括机台号一致,镀膜层数一致等..
        :param base_data_dir: all data path [没做任何筛选处理的所有data]
        :param new_path:  根据机器版本号等,把数据拆分到sub文件夹,保证各个子文件夹中,机器版本号都一致
        :return:

        '''

        files = os.listdir(self.base_data_dir)
        files = [i for i in files if ("evt" or "EVT") in i]
        PLC_version = []
        MachineName = []
        RecipeName = []
        RecipeName_dict = dict()
        for file in files:
            f = open(os.path.join(self.base_data_dir, file), 'r')
            for index, line in enumerate(f):
                # if index == 1:
                #     PLC_version.append(line.split(',')[-1])
                # elif index == 2:
                #     MachineName.append(line.split(',')[-1])
                if index == 3:
                    recipename = line[:-1].split(',')[-1]
                    if recipename:
                        RecipeName.append(recipename)
                        if recipename and recipename not in RecipeName_dict:
                            RecipeName_dict[recipename] = []
                            RecipeName_dict[recipename].append(file)
                        elif recipename:
                            RecipeName_dict[recipename].append(file)
        count = 0
        for k, v in RecipeName_dict.items():
            # print(len(v))
            count += len(v)
            path = os.path.join(self.base_data_dir, k)
            # isExists = os.path.exists(path)
            os.makedirs(path)
            for cvs_name in v:
                filename0 = os.path.join(self.base_data_dir, cvs_name)
                filename1 = os.path.join(path, cvs_name)
                filename0_ = os.path.join(self.base_data_dir, cvs_name[3:])
                filename1_ = os.path.join(path, cvs_name[3:])
                try:
                    shutil.move(filename0, filename1)
                except:
                    print(filename0)
                    pass
                try:
                    shutil.move(filename0_, filename1_)
                except:
                    print(filename0_)
                    pass
        print(count)

    def clean_data_nthickness(self, ):
        '''
        :param data_dir: clean_data_machineid()中生成的某一sub_dir
        :param n_thickness: 7层膜厚设置, 也可以是5、6、4等
        :param evt_thick_js: 落盘js名称, evtname:thickness
        :return:

        '''
        evt_thick = dict()
        files = os.listdir(self.data_dir)
        files = [a for a in files if ('evt' or 'EVT') in a]
        for file in files:
            thickness_list = []
            full_path = os.path.join(self.data_dir, file)
            process = open(full_path, 'r')
            for index, line in enumerate(process):
                if 'Thickness' in line:
                    thickness_list.append(line.split(',')[4])
            if len(thickness_list) == self.n_thickness:
                evt_thick[file] = thickness_list
        data = json.dumps(evt_thick)
        with open(self.evt_thick_js, 'w') as js_file:
            js_file.write(data)


def lower_filename(base_data_dir, tmp_dir):
    '''

    :param base_dir: D:\work\project\卡尔蔡司AR镀膜\第三批\33机台文件
    :param tmp_dir: 临时文件夹, 后续会被删除可随意命名
    :return:

    '''
    # base_dir = r'D:\work\project\卡尔蔡司AR镀膜\第三批\33机台文件'
    files = os.listdir(base_data_dir)
    for f in files:
        if '.7z' in f or '.zip' in f:
            os.remove(os.path.join(base_data_dir, f))
    fs = os.listdir(base_data_dir)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    for f in fs:
        path1 = os.path.join(base_data_dir, f)
        path2 = os.path.join(tmp_dir, f.lower())
        shutil.move(path1, path2)
    os.rmdir(base_data_dir)
    os.rename(tmp_dir, base_data_dir)



def split_CX_CC(base_data_dir):
    pre_dir = base_data_dir[:-1 - len(base_data_dir.split('\\')[-1])]
    cc_dir = os.path.join(pre_dir, r'cc')
    cx_dir = os.path.join(pre_dir, r'cx')
    if not os.path.exists(cc_dir):
        os.mkdir(cc_dir)
    if not os.path.exists(cx_dir):
        os.mkdir(cx_dir)
    all_file = os.listdir(base_data_dir)
    evt_files = [i for i in all_file if "evt" in i]
    cc = []
    cx = []
    for evt_f in evt_files:
        path = os.path.join(base_data_dir, evt_f)
        with open(path, 'r') as f:
            for index, line in enumerate(f):
                if 'Recipe Name :' in line:
                    cccx = line.split(',')[1]
                    if 'CC' in cccx:
                        cc.append(evt_f)
                    elif 'CX' in cccx:
                        cx.append(evt_f)
    for cc_ in cc:
        path_sensor1 = os.path.join(base_data_dir, cc_[3:])
        path = os.path.join(base_data_dir, cc_)
        path_evt2 = os.path.join(cc_dir, cc_)
        shutil.copy(path, path_evt2)
        if os.path.exists(path_sensor1):
            path_sensor2 = os.path.join(cc_dir, cc_[3:])
            shutil.copy(path_sensor1, path_sensor2)
    for cx_ in cx:
        path_sensor1 = os.path.join(base_data_dir, cx_[3:])
        path = os.path.join(base_data_dir, cx_)
        path_evt2 = os.path.join(cx_dir, cx_)
        shutil.copy(path, path_evt2)
        if os.path.exists(path_sensor1):
            path_sensor2 = os.path.join(cx_dir, cx_[3:])
            shutil.copy(path_sensor1, path_sensor2)


def get_evtpair_info(CC_dir, CX_dir, evt33, evt_pair, n_thickness):
    '''

    :param CC_dir:
    :param CX_dir:
    :param evt33:
    :param evt_pair: 落盘cc cx对 thickness14
    :param n_thickness: 7
    :return:
    '''
    wb = xlrd.open_workbook(evt33)
    info_title = ['OvenNo', 'FileID', 'FilmCode_MES']
    data = wb.sheet_by_name('Sheet1')
    for count in range(10):
        try:
            title = data.row_values(count)
            index1, index2, index3 = title.index(info_title[0]), title.index(info_title[1]), title.index(info_title[2])
            a = count + 1
            break
        except:
            continue
    # number33: [cc, cx]
    number33_cccx = dict()
    rows = data.nrows
    for i in range(a, rows):
        number33 = data.cell(i, index1).value
        if number33 not in number33_cccx:
            number33_cccx[number33] = ['', '']
        if "CC" in data.cell(i, index3).value:
            number33_cccx[number33][0] = data.cell(i, index2).value.lower()
        elif "CX" in data.cell(i, index3).value:
            number33_cccx[number33][1] = data.cell(i, index2).value.lower()
    print('number33_evt_cccx lens: ', len(number33_cccx))

    cxs = os.listdir(CX_dir)
    ccs = os.listdir(CC_dir)

    f = open(evt_pair, 'w')
    count = 0
    for num33, evt_list in number33_cccx.items():
        if evt_list[1] + '.csv' in cxs and evt_list[0] + '.csv' in ccs:
            ccpath = os.path.join(CC_dir, evt_list[0]+ '.csv')
            cxpath = os.path.join(CX_dir, evt_list[1]+ '.csv')
            if os.path.exists(ccpath) and os.path.exists(cxpath):
                evts = [ccpath, cxpath]
            elif os.path.exists(ccpath) and not os.path.exists(cxpath):
                evts = [ccpath, ccpath]
            elif not os.path.exists(ccpath) and os.path.exists(cxpath):
                evts = [cxpath, cxpath]
            else:
                continue
            thickness14 = []
            for evt in evts:
                with open(evt, 'r') as file:
                    for line in file:
                        if "Thickness" in line:
                            thickness14.append(line.split(',')[4])
            if len(thickness14) == 2 * n_thickness:
                count += 1
                f.write(num33 + ',')
                f.write(evts[0].split('\\')[-1] + ',' + evts[1].split('\\')[-1] + ',')
                f.write(''.join(str(i) + ',' for i in thickness14))
                f.write('\n')
    print("thick14 pair lens: {}".format(count))


def number33_thick14lab(membrane, evt_pair, number33_thicklab_js, num_evt12):
    '''

    处理 膜色数据.xlsx, 把33321043002编号和膜色曲线关联起来,
    同一编号对应多条曲线(炉内不同的层), 统一取第四层数据

    :param  membrane: 膜色数据.xlsx
    :param  evt_dict
    :param  base_data_dir
    :param  data_js: thickness_lab的映射dict, 落盘文件名
    param: thickness_lab_curve: [thickness]: lab_curve
    :return:

    '''
    num_evt12_data = dict()
    # number33_thicklab {number33: [thick14, lab]}
    number33_thicklab = dict()
    # number33, evt1,evt2,thick14
    number33_evt_pair_thick14 = open(evt_pair, 'r').readlines()
    for line in number33_evt_pair_thick14:
        thick14 = ''.join(i+',' for i in line[:-1].split(',')[3:-1])
        number33_thicklab[line.split(',')[0]] = [thick14, []]
        num_evt12_data[line.split(',')[0]] = [line.split(',')[1], line.split(',')[2]]

    # 读取膜厚数据
    wb = xlrd.open_workbook(membrane)
    # data = wb.sheet_by_name(r'双面膜色曲线 (第二批)')
    data = wb.sheet_by_name(r'Sheet1')
    rows = data.nrows
    need_title = [r'炉序号', '380', '780']
    # 记录number33编号, 第几层lab曲线index
    number_lab_index = dict()
    for count in range(10):
        try:
            title = data.row_values(count)
            num33_index, lab_index1, lab_index2 = title.index(need_title[0]), title.index(need_title[1]), title.index(
                need_title[2])
            a = count + 1
            break
        except:
            continue
    for i in range(a, rows):
        number33_cur = data.cell(i, num33_index).value
        if number33_cur in number33_thicklab:
            number_lab_index[number33_cur] = number_lab_index.get(number33_cur, 0) + 1
            # 我们取第四层的膜色曲线为基准
            if number_lab_index[number33_cur] == 4:
                tmp = data.row_values(i)[lab_index1: lab_index2 + 1]
                # 剔除lab曲线少于81维的lab数据
                tmp = [i for i in tmp if i != '']
                if len(tmp) == 81:
                    number33_thicklab[number33_cur][1] = tmp
    numbs = list(number33_thicklab.keys())
    for num in numbs:
        # 部分lab是 []
        if len(number33_thicklab[num][1]) < 81:
            del number33_thicklab[num]
            del num_evt12_data[num]
    print('取炉子第四层曲线, 且曲线有81维值的 number33 数目: {}'.format(len(number33_thicklab)))

    data = json.dumps(number33_thicklab)
    with open(number33_thicklab_js, 'w') as js_file:
        js_file.write(data)

    data = json.dumps(num_evt12_data)
    with open(num_evt12, 'w') as js_file:
        js_file.write(data)



def number33_thick14hc3lab(process_data, number33_thicklab_js, face, number33_thick14hc3lab_js):
    number33_thicklab = json.load(open(number33_thicklab_js, 'r'))
    number33_thick14hc3lab = copy.deepcopy(number33_thicklab)
    wb = xlrd.open_workbook(process_data)
    data = wb.sheet_by_name('Sheet1')
    rows = data.nrows
    info_title = ['炉序号', '反正面', '离子枪灯丝', '电子枪灯丝', '挡板']
    for count in range(10):
        try:
            title = data.row_values(count)
            index1, index2, index3, index4, index5 = title.index(info_title[0]), title.index(
                info_title[1]), title.index(
                info_title[2]), title.index(info_title[3]), title.index(info_title[4])
            a = count + 1
            break
        except:
            continue
    for i in range(a, rows):
        number33_cur = data.cell(i, index1).value
        if number33_cur in number33_thicklab and data.cell(i, index2).value == face:
            tmp_hc_list = [round(data.cell(i, index3).value / 8, 5), round(data.cell(i, index4).value / 120, 5),
             round(data.cell(i, index5).value / 120, 5)]
            number33_thick14hc3lab[number33_cur][0] = number33_thicklab[number33_cur][0] + ''.join(str(i)+',' for i in tmp_hc_list)

    nums = list(number33_thick14hc3lab.keys())
    for k in nums:
        if len(number33_thick14hc3lab[k][0].split(',')) != 18:
            del number33_thick14hc3lab[k]
    print("膜厚耗材数据量: {}".format(len(number33_thick14hc3lab)))

    data = json.dumps(number33_thick14hc3lab)
    with open(number33_thick14hc3lab_js, 'w') as js_file:
        js_file.write(data)

    # check data_size
    tmp_dict = dict()
    for num, thickhc_lab in number33_thick14hc3lab.items():
        tmp_dict[thickhc_lab[0]] = thickhc_lab[1]
    print("膜厚加耗材特征不同的key数量: {}, 数据重复量: {} - {} = {}".format(len(tmp_dict), len(number33_thick14hc3lab), len(tmp_dict), (len(number33_thick14hc3lab) - len(tmp_dict))))


def sensor_csv_feature(sen_list, data):
    f_sensor = []
    for sen in sen_list:
        col = [i for i in data[sen]]
        ts = pd.Series(col)
        # 时序数据的的周期性、不可预测性和波动性
        ae1 = tsf.feature_extraction.feature_calculators.ar_coefficient(ts, [{'coeff': 0, 'k': 10}])
        f_sensor.append(ae1[0][1])
        # # 时序数据的平稳性
        ae2 = tsf.feature_extraction.feature_calculators.augmented_dickey_fuller(ts, [{'attr': 'pvalue'}])
        f_sensor.append(ae2[0][1])
        mean = np.mean([i for i in data[sen]][2:])
        std = np.std([i for i in data[sen]][2:], ddof=1)
        f_sensor.append(mean)
        f_sensor.append(std)
    f_sensor = [round(i, 3) for i in f_sensor]
    # return ''.join(str(i) + ',' for i in f_sensor)
    return f_sensor


def add_sensor_feature(base_data_dir, num_evt12, number33_thick14hc3lab_js, sen_list, thick14_hc3_sensor16_lab_js):
    num_thickhclab = json.load(open(number33_thick14hc3lab_js, 'r'))
    thick14_hc3_sensor16_lab = copy.deepcopy(num_thickhclab)
    num_evt12 = json.load(open(num_evt12, 'r'))
    exists_sensor_csv = open(r'./sensor_csv.txt', 'w')
    for num, evt12 in num_evt12.items():
        path1 = os.path.join(base_data_dir, evt12[0][3:])
        path2 = os.path.join(base_data_dir, evt12[1][3:])
        for path in [path1, path2]:
            if os.path.exists(path):
                exists_sensor_csv.write(path.split('\\')[-1] + ',')
        if not os.path.exists(path1) and not os.path.exists(path2):
            continue
        elif os.path.exists(path1) and not os.path.exists(path2):
            path2 = path1
        elif not os.path.exists(path1) and os.path.exists(path2):
            path1 = path2
        data1 = pd.read_csv(path1, error_bad_lines=False)
        data2 = pd.read_csv(path2, error_bad_lines=False)
        sensor16_1 = sensor_csv_feature(sen_list, data1)
        sensor16_2 = sensor_csv_feature(sen_list, data2)
        sensor16 = [(sensor16_1[i] + sensor16_2[i]) / 2 for i in range(len(sensor16_1))]
        sensor16_f = ''.join(str(i)+',' for i in sensor16)
        try:
            pre_f = num_thickhclab[num][0]
        except:
            continue
        thick14hc3sensor16 = pre_f + sensor16_f
        assert len(thick14hc3sensor16.split(',')) == 34
        thick14_hc3_sensor16_lab[num][0] = thick14hc3sensor16
    print("膜厚耗材snesor16 数据量: {}".format(len(thick14_hc3_sensor16_lab)))

    data = json.dumps(thick14_hc3_sensor16_lab)
    with open(thick14_hc3_sensor16_lab_js, 'w') as js_file:
        js_file.write(data)


def usful_sensor_feature(sensor_csv):
    try:
        sensor_csv = pd.read_csv(sensor_csv, error_bad_lines=False)
    except:
        return ''
    tmp = open(r'./info_sensor_nothick.txt', 'r')
    ok_sen_list = tmp.readlines()[0].split(',')[:-1]
    f = []
    for sen_n in ok_sen_list:
        col = sensor_csv[sen_n]
        col_data = [i for i in col]
        ts = pd.Series(col_data)
        ae1 = tsf.feature_extraction.feature_calculators.ar_coefficient(ts, [{'coeff': 0, 'k': 10}])
        f.append(ae1[0][1])
        # ae2 = tsf.feature_extraction.feature_calculators.augmented_dickey_fuller(ts, [{'attr': 'pvalue'}])
        ae3 = tsf.feature_extraction.feature_calculators.binned_entropy(ts, 10)  # 信息熵,可以考虑加入
        f.append(ae3)
    return ''.join(str(i) + ',' for i in f)



def get_sensor38(sensor_csv, ok_sen_list):
    f = []
    for sen_n in ok_sen_list:
        col = sensor_csv[sen_n]
        col_data = [i for i in col]
        ts = pd.Series(col_data)
        ae1 = tsf.feature_extraction.feature_calculators.ar_coefficient(ts, [{'coeff': 0, 'k': 10}])
        f.append(ae1[0][1])
        ae3 = tsf.feature_extraction.feature_calculators.binned_entropy(ts, 10)  # 信息熵,可以考虑加入
        f.append(ae3)
    return f


def all_usful_sensor_except_thickness(base_data_dir, num_evt12, thick14_hc3_sensor80_lab_js, feature135_lab_js):
    '''
    提取, 整合出来已经处理的thickness、rate这四列之外的,有意义19列数据的feature
    :return: 19*2=38维特征

    '''
    num_evt12_data = json.load(open(num_evt12, 'r'))
    sensor_19 = open(r'info_sensor_nothick.txt', 'r').readlines()[0]
    sensor_19_list = sensor_19.split(',')[:-1]
    thick14_hc3_sensor80_lab = json.load(open(thick14_hc3_sensor80_lab_js, 'r'))
    number33_feature135 = copy.deepcopy(thick14_hc3_sensor80_lab)
    finall = dict()
    for num, thick14hc3sensor80 in thick14_hc3_sensor80_lab.items():
        evtcc = num_evt12_data[num][0]
        evtcx = num_evt12_data[num][1]
        path1 = os.path.join(base_data_dir, evtcc[3:])
        path2 = os.path.join(base_data_dir, evtcx[3:])
        if os.path.exists(path1) and not os.path.exists(path2):
            csv_data = pd.read_csv(path1, error_bad_lines=False)
            f = get_sensor38(csv_data, sensor_19_list)
        elif not os.path.exists(path1) and os.path.exists(path2):
            csv_data = pd.read_csv(path2, error_bad_lines=False)
            f = get_sensor38(csv_data, sensor_19_list)
        elif os.path.exists(path1) and os.path.exists(path2):
            csv_data1 = pd.read_csv(path1, error_bad_lines=False)
            f1 = get_sensor38(csv_data1, sensor_19_list)
            csv_data2 = pd.read_csv(path2, error_bad_lines=False)
            f2 = get_sensor38(csv_data2, sensor_19_list)
            f = [(f1[i]+f2[i])/2 for i in range(len(f1))]
        else:
            continue
        f = ''.join(str(i)+',' for i in f)
        number33_feature135[num][0] = thick14_hc3_sensor80_lab[num][0] + f
    nums = list(number33_feature135.keys())
    for num in nums:
        if len(number33_feature135[num][0].split(',')) != 136:
            del number33_feature135[num]
    # finally
    for num, f135lab in number33_feature135.items():
        f135 = f135lab[0]
        lab = f135lab[1]
        finall[f135] = lab

    js = json.dumps(finall)
    with open(feature135_lab_js, 'w') as js_:
        js_.write(js)
    print("135维特征数据量: {}".format(len(number33_feature135)))


if __name__ == "__main__":
    base_data_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件'
    file1 = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#膜色文件与EVT文件对应表.xlsx'
    file2 = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\膜色数据.xlsx'
    data_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\org_refine_thickness_lab_curve.json'
    evt_cc_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC'
    process_data = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\工艺记录.xlsx'
    evt_thick = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\oneone_evt_thickness.json'
    rate_thickness_check(evt_cc_dir)  # import check_data.py 中的函数实现部分数据清洗功能
