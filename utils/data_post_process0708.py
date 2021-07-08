# coding=utf-8
import json
import os
import random
import shutil

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

    def __init__(self, evt33, membrane, data_dir, data_js, process_data, refine_data_json, oneone_evt_thickness,
                 evt_33number, base_data_dir, CC_dir, CX_dir, bad_js, num33_hc_js, number33_thick_js, thick_hc_lab_js,
                 thick14_hc3_sensor16_lab_js, thick14_hc3_sensor80_lab_js, feature135_lab_js, flag=0):
        '''
        :param MachineName-kinds:   ['1.56_DVS_CC', '1.56_DVS_CX', '1.6&1.67_DVS_CC', '1.6&1.67_DVS_CC_hpkf', '1.6&1.67_DVS_CX', '1.6&1.67_DVS_CX_hpkf', '1.6_DVSUN_CC']

        :param n_thick: 镜片镀膜层数, 7,5,4 等不等。模型训练时,需保证数据都是同种层数的镀膜数

        :param evt33:  33#膜色文件与EVT文件对应表.xlsx
                       note： 可将 33121052204 编号 和 EVT21050506 对应上
        :param membrane:  膜色数据.xlsx
                       note： 可将 33121060503 编号 和 lab 膜色曲线 对应上
        :param data_dir:  33#机台文件_7dirs\1.6&1.67_DVS_CC
                       note:  evt21050506.csv文件夹, 可读取出膜厚设置值,测量值等信息.
        :param data_js:  thickness_lab_curve.json、refine_thickness_lab_curve.json
                       note： 都是要落盘的文件, 将膜厚和lab_curve对应上. 但因为一种膜厚设置对应多条lab_curve，
                       故使用data_chcek()、 data_refine()修正.
        :param process_data:  工艺记录.xlsx
                       note： 含生产周期信息, 解决一种膜厚设置对应多条lab曲线(evt_file)问题

        :param refine_data_json, oneone_evt_thickness, evt_33number 一些落盘中间文件

        :param base_data_dir: 总的数据文件夹,其中可能包含不同机台版本数据,膜厚层数不一数据, 需要最近清理此部分data

        :return refine_thickness_lab_curve.json

        '''

        self.base_data_dir = base_data_dir
        self.new_path = base_data_dir[:-1 - len(base_data_dir.split('\\')[-1])]
        self.n_thickness = 7
        self.evt33 = evt33
        self.membrane = membrane
        self.data_dir = data_dir
        self.data_js = data_js
        self.evt_dict = dict()
        self.thickness_lab_curve = dict()
        self.bad_thick_lab = dict()
        self.process_data = process_data
        self.refine_data_json = refine_data_json
        self.oneone_evt_thickness = oneone_evt_thickness
        self.evt_33number = evt_33number
        self.bad_js = bad_js
        self.CC_dir = CC_dir
        self.CX_dir = CX_dir
        self.num33_hc_js = num33_hc_js
        self.number33_thick_js = number33_thick_js
        self.thick_hc_lab_js = thick_hc_lab_js
        self.thick14_hc3_sensor16_lab_js = thick14_hc3_sensor16_lab_js
        self.thick14_hc3_sensor80_lab_js = thick14_hc3_sensor80_lab_js
        self.feature135_lab_js = feature135_lab_js
        self.thick7_lab_dict = dict()

        # tmp 落盘文件
        self.evt_thick_js = r'./evt_thick.json'
        self.evt_pair = r'正背面_thickness_evtname.txt'
        # self.thick7_lab = r'./org_thick7_lab.json'
        self.sen_list = ['ACT_O1_QCMS_THICKNESS', 'ACT_O1_QCMS_RATE', 'ACT_O1_QCMS_THICKNESS_CH1',
                         'ACT_O1_QCMS_RATE_CH1']
        self.csv_dict_js = r'./evtname_sensor_name_value.json'

    # def __call__(self, ):
    def run(self, ):

        # start data_clean
        get_evtpair_info(self.CC_dir, self.CX_dir, self.evt33, self.evt_pair, self.n_thickness)
        evt_33(self.evt33, self.evt_dict, self.data_dir, self.evt_33number)
        lab_curve33(self.membrane, self.evt_dict, self.data_js, self.data_dir, self.thickness_lab_curve)
        check_data(self.thickness_lab_curve, self.evt_dict, self.bad_thick_lab, self.bad_js)
        refine_data(self.bad_thick_lab, self.process_data, self.refine_data_json, self.oneone_evt_thickness,
                    self.number33_thick_js)

        # 添加3个耗材维度特征
        get_hc_value(self.process_data, self.num33_hc_js, '背面')
        hc_feature(self.num33_hc_js, self.number33_thick_js, self.refine_data_json, self.thick_hc_lab_js)

        # 添加膜厚,rate等sensor列的时序特征(4*4=16dims)
        add_sensor_feature(self.data_dir, self.oneone_evt_thickness, self.bad_thick_lab, self.thick_hc_lab_js,
                           self.sen_list, self.thick14_hc3_sensor16_lab_js)
        # 添加8个step的时序特征(len=128 8*4*4)
        get8step_sensor_feature(self.data_dir, self.csv_dict_js, self.thick14_hc3_sensor80_lab_js,
                                self.thick14_hc3_sensor16_lab_js, self.oneone_evt_thickness,
                                self.refine_data_json, self.sen_list)
        # 再加入19列有意义数据的38维特征
        all_usful_sensor_except_thickness(self.data_dir, self.refine_data_json, self.oneone_evt_thickness,
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


def evt_33(evt33, evt_dict, data_dir, evt_33number):
    '''
    :param   data_dir: 33#机台文件_7dirs\1.6&1.67_DVS_CC: EVT21050425.csv_s
             note:     evt文件所在路径, 需要确保每个evt文件机台编号一致, 正背面一致, 膜厚数一致
    :return  evt_dict
             note: evt_dict[EVT21050506 ] = 33121052204

             evt_33number: json落盘: EVT21050506: 33121052204

    '''
    wb = xlrd.open_workbook(evt33)
    data = wb.sheet_by_name('Sheet1')
    info_title = ['OvenNo', 'FileID']
    for count in range(10):
        try:
            title = data.row_values(count)
            index1, index2 = title.index(info_title[0]), title.index(info_title[1])
            a = count + 1
            break
        except:
            continue
    rows = data.nrows
    for i in range(a, rows):
        evt_dict[(data.cell(i, index2).value).lower() + '.csv'] = data.cell(i, index1).value  # evt_name统一成小写: evt21062916.csv
    evt_dict_keys = list(evt_dict.keys())
    evt_cc_list = os.listdir(data_dir)
    for evt in evt_dict_keys:
        if evt not in evt_cc_list:
            del evt_dict[evt]
    data = json.dumps(evt_dict)
    with open(evt_33number, 'w') as js_file:
        js_file.write(data)
    print("读取匹配关系2021.1~2021.6.xlsx 文件中, 匹配33321063002 与 EVT21070127信息, 并确保存在对应的evtxxx.csv文件")
    print('筛选结果: {}'. format(len(evt_dict)))


def lab_curve33(membrane, evt_dict, data_js, data_dir, thickness_lab_curve):
    '''

    处理 膜色数据.xlsx, 把33321043002编号和膜色曲线关联起来,
    同一编号对应多条曲线(炉内不同的层), 统一取第四层数据

    :param  membrane: 膜色数据.xlsx
    :param  evt_dict
    :param  data_dir
    :param  data_js: thickness_lab的映射dict, 落盘文件名
    param: thickness_lab_curve: [thickness]: lab_curve
    :return:

    '''
    # 读取膜厚数据
    wb = xlrd.open_workbook(membrane)
    # data = wb.sheet_by_name(r'双面膜色曲线 (第二批)')
    data = wb.sheet_by_name(r'Sheet1')
    rows = data.nrows
    need_title = [r'炉序号', '380', '780']
    numberss_dict = dict()
    number33_lab_curve = dict()
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
        numberss_dict[data.cell(i, num33_index).value] = numberss_dict.get(data.cell(i, num33_index).value, 0) + 1
        # 我们取第四层的膜色曲线为基准
        if numberss_dict[data.cell(i, num33_index).value] == 4:
            tmp = data.row_values(i)[lab_index1: lab_index2 + 1]
            # 剔除lab曲线少于81维的lab数据
            tmp = [i for i in tmp if i != '']
            if len(tmp) == 81:
                number33_lab_curve[data.cell(i, num33_index).value] = tmp
    print("读取膜色数据文件,根据33321063002编号,读取每一炉的第四条lab曲线数据.")
    print(len(number33_lab_curve), 'number33_lab_curve')
    data = json.dumps(number33_lab_curve)
    with open(r'./num33_lab.json', 'w') as js_file:
        js_file.write(data)

    # 借助 evtname:33number、33number:lab_curv,获得evtname:lab_curv关联
    # get: evtname:lab_curve
    evt_name_lab_curve = dict()
    for evt_name, number in evt_dict.items():
        if number in number33_lab_curve:
            evt_name_lab_curve[evt_name] = number33_lab_curve[number]
    print("根据evt_dict的evtname和33321063002编号匹配关系,关联evtxxx.csv文件和lab曲线")
    print(len(evt_name_lab_curve), 'evt_name_lab_curve')   # 这个lens < 2*len(number33_lab_curve) 说明有一些evt对匹配缺失

    # evtname:lab_curve, 根据evtname获取thickness,关联thickness:lab_curve
    for evt_name in evt_name_lab_curve:
        evt_path = os.path.join(data_dir, evt_name)
        thickness = []
        with open(evt_path, 'r') as file:
            for line in file:
                if "Thickness" in line:
                    thickness.append(line.split(',')[4])  # 取膜厚设置值
                    # thickness.append(line.split(',')[6])    # 取膜厚实测值(会因为时延有一定误差)
        # 由于不同的evt文件,膜厚设置值可能一致,故后面添加evt_name作为区分.
        # 这也是为什么后面要check_data, refine_data的原因.. 直接相同key value被直接覆盖,倒是也可以的..
        # thickness_lab_curve[''.join(i + ',' for i in thickness)] = evt_name_lab_curve[evt_name]
        thickness_lab_curve[''.join(i + ',' for i in thickness) + '{}'.format(evt_name[:-4])] = evt_name_lab_curve[
            evt_name]
    print("读取evtxxx.csv文件,获取膜厚设置值, 匹配lab曲线")
    print(len(thickness_lab_curve), 'thickness_lab_curve')
    data = json.dumps(thickness_lab_curve)
    with open(data_js, 'w') as js_file:
        js_file.write(data)


# check_data 0624 chenjia
def check_data(thickness_lab_curve, evt_dict, bad_thick_lab, bad_js):
    print('检查膜厚设置值和lab曲线一对多情况..')
    '''
    # 同一膜厚设置,对应不同的evtname, 也即不同的lab_curve
    :param thickness_lab_curve
    :param evt_dict
    :return:

    '''
    for thickness, lab_curve in thickness_lab_curve.items():
        thickes, evt_name = thickness.split(',')[:-1], thickness.split(',')[-1]
        thickes = ''.join(i + ',' for i in thickes)
        if thickes not in bad_thick_lab:
            bad_thick_lab[thickes] = []
        bad_thick_lab[thickes].append([evt_name, lab_curve, evt_dict[evt_name + '.csv']])
    print(len(bad_thick_lab), 'bad_thick_lab')
    # 落盘下
    data = json.dumps(bad_thick_lab)
    with open(bad_js, 'w') as js_file:
        js_file.write(data)
    # txt 输出可以给数据方(蔡司排查问题)
    if len(thickness_lab_curve) != len(bad_thick_lab):
        f = open('./data_check.txt', 'w', encoding="utf-8")
        for thick, list_ in bad_thick_lab.items():
            f.write("膜厚设置值: {}".format(thick) + '\n')
            f.write('对应不同lab曲线: ' + '\n')
            for num in list_:
                f.write(''.join(str(i) for i in num) + '\n')
            f.write('\n')
            f.write('\n')


def get_evtpair_info(CC_dir, CX_dir, evt33, evt_pair, n_thickness):
    wb = xlrd.open_workbook(evt33)
    info_title = ['OvenNo', 'FileID']
    data = wb.sheet_by_name('Sheet1')
    for count in range(10):
        try:
            title = data.row_values(count)
            index1, index2 = title.index(info_title[0]), title.index(info_title[1])
            a = count + 1
            break
        except:
            continue
    number33_evts = dict()
    rows = data.nrows
    for i in range(a, rows):
        number33 = data.cell(i, index1).value
        if number33 not in number33_evts:
            number33_evts[number33] = []
        # 这里有时候大写,有时候小写.. 统一成小写.
        number33_evts[number33].append((data.cell(i, index2).value).lower())
    print('number33_evts lens: ', len(number33_evts))

    dirs = [CC_dir, CX_dir]
    cxs = os.listdir(CX_dir)
    ccs = os.listdir(CC_dir)

    f = open(evt_pair, 'w')
    evt_cc = []
    for num33, evt_list in number33_evts.items():
        print(evt_list)
        if len(evt_list) == 2:
            if (evt_list[1] + '.csv' in cxs and evt_list[0] + '.csv' in ccs) or (
                    evt_list[0] + '.csv' in cxs and evt_list[1] + '.csv' in ccs):
                evts = []
                for dir_ in dirs:
                    for i in range(2):
                        if os.path.exists(os.path.join(dir_, evt_list[i] + '.csv')):
                            evts.append(os.path.join(dir_, evt_list[i] + '.csv'))
                thickness1 = []
                thickness2 = []
                evt1, evt2 = evts[0], evts[1]
                with open(evt1, 'r') as file:
                    for line in file:
                        if "Thickness" in line:
                            thickness1.append(line.split(',')[4])
                with open(evt2, 'r') as file:
                    for line in file:
                        if "Thickness" in line:
                            thickness2.append(line.split(',')[4])
                # if thickness1 != thickness2:
                if len(thickness2) == len(thickness1) and len(thickness2) == n_thickness:
                    f.write(num33 + ',')
                    f.write(evt1.split('\\')[-1] + ',' + evt2.split('\\')[-1] + ',')
                    f.write(''.join(str(i) + ' ' for i in thickness1) + ',')
                    f.write(''.join(str(i) + ' ' for i in thickness2) + '\n')
                    evt_cc.append(evt1.split('\\')[-1])
    print('get evt pair ~')


def evt_pair_thick(evt_name):
    '''
    :param  evt_pair 存储正背面evt_pair name.
    :param  只读取正背面均有evt文件的膜厚数据. 只有一面膜厚数据的情况,在refine_data中的处理是,直接copy背面层的数据

    :return:
    '''
    evt_pair = r'./正背面_thickness_evtname.txt'

    # #  added 0707 chenjia
    # tmp = r'D:\work\project\卡尔蔡司AR镀膜\第三批\匹配关系2021.1~2021.6.xlsx'
    # wb = xlrd.open_workbook(tmp)
    # data = wb.sheet_by_name('Sheet1')
    # rows = data.nrows
    # evt_names = []
    # for i in range(1, rows):
    #     if "CC" in data.cell(i, 4).value:
    #         evt_names.append(data.cell(i, 5).value.lower())

    lines = open(evt_pair, 'r').readlines()
    for line in lines:
        evt1, evt2, thick2, thick1 = line[:-1].split(',')[1], line[:-1].split(',')[2], line[:-1].split(',')[4], line[:-1].split(',')[3]
        if evt_name == evt1:
            return ''.join(str(i) + ',' for i in thick2.split(' '))[:-1], thick1.split(' ')

    return [], []


# refine_data 0624 chenjia
# add 正背面膜厚值, concate or mean. 0625 chenjia
def refine_data(bad_thick_lab, process_data, refine_data_json, one_evt_thickness, number33_thick_js, concate=True):
    '''
    :param concate or mean True/False
    :param process_data: 工艺记录.xlsx
    :param one_evt_thickness: evt_name: thickness
    :param refine_data_json: finall_thick_lab的落盘json名

    '''
    number33 = open(r'./33number.txt', 'w')
    wb = xlrd.open_workbook(process_data)
    data = wb.sheet_by_name('Sheet1')
    info_title = ['反正面', '炉序号', '电子枪灯丝']
    for count in range(10):
        try:
            title = data.row_values(count)
            index1, index2, index3 = title.index(info_title[0]), title.index(info_title[1]), title.index(info_title[2])
            a = count + 1
            break
        except:
            continue
    rows = data.nrows
    number_time = dict()
    time_number = dict()
    finall_thick_lab = dict()
    oneone_evt_thickness = dict()
    number33_thick = dict()
    for i in range(a, rows):
        # 正背面, 与data_dir中的 CC or CX后缀对应
        # if data.cell(i, index1).value == '背面':
        #     # 33121060707:电子枪数号
        #     number_time[data.cell(i, index2).value] = data.cell(i, index3).value  # 33121052004: time_index
        #     time_number[data.cell(i, index3).value] = data.cell(i, index2).value  # time_index: 33121052004

        # update 0707
        number_time[data.cell(i, index2).value] = data.cell(i, index3).value
        time_number[data.cell(i, index3).value] = data.cell(i, index2).value

    # tmp 落盘
    data = json.dumps(number_time)
    with open(r'/number33.json', 'w') as js_file:
        js_file.write(data)

    for thickness, list_ in bad_thick_lab.items():
        # time_index = []
        for single_list in list_:
            # single_list：[EVT21050425, [lab_curve], 33number]
            number33.write(single_list[-1] + '\n')
            if single_list[-1] in number_time:
                # time_index.append(number_time[single_list[-1]])
                # the_number_33 = time_number[min(time_index)]  # 电枪数被使用最少的那个33编号文件 会导致有些33121052004在bad_thick_lab中找不到...
                oneone_evt_thickness[single_list[0]] = thickness  # evtname: thickness
                # number33_thick[single_list[-1]] = thickness  # 获取7层膜厚数值
                # 在这里穿插,根据evtname, 找到当前evt的对应正面,并获取膜厚设置值
                pair_thick, thick1 = evt_pair_thick(single_list[0] + '.csv')  # str, list
                if pair_thick and thick1 == thickness.split(','):
                    if not concate:
                        # 取mean, 保留7层维度
                        th1 = thickness.split(',')[:-1]
                        th2 = pair_thick.split(',')[:-1]
                        len_ = len(th1)
                        final = [(float(th1[i]) + float(th2[i])) / 2 + random.uniform(1e-4, 5e-5) for i in range(len_)]
                        final = ''.join(str(i) + ',' for i in final)
                        finall_thick_lab[final] = single_list[1]
                    else:
                        # 正背concate
                        finall_thick_lab[thickness + pair_thick] = single_list[1]
                        number33_thick[single_list[-1]] = thickness + pair_thick
                else:
                    if not concate:
                        # print(pair_thick, thick1, thickness, single_list[0]+'.csv')   # 部分背面evt找不到正面的evt,没事那就copy一份膜值.
                        finall_thick_lab[thickness] = single_list[1]
                    else:  # concate
                        finall_thick_lab[thickness + thickness[:-1]] = single_list[1]  # 正背concate
                        number33_thick[single_list[-1]] = thickness + thickness[:-1]

                break  # 找到一个33number了,就不再遍历list_, break出循环
    # mean处理正背面的膜厚时, 可能出现key值重复,覆盖更新了value,故而导致finall_thick_lab 和 oneone_evt_thickness 长度不等
    assert len(finall_thick_lab) == len(oneone_evt_thickness)  # len需要一致!
    print(len(finall_thick_lab), 'finall_thick_lab')
    data = json.dumps(finall_thick_lab)
    with open(refine_data_json, 'w') as js_file:
        js_file.write(data)

    data = json.dumps(oneone_evt_thickness)
    with open(one_evt_thickness, 'w') as js_file:
        js_file.write(data)

    data = json.dumps(number33_thick)
    with open(number33_thick_js, 'w') as js_file:
        js_file.write(data)


def get_hc_value(process_data, num33_hc_js, face):
    f = open(r'./33number.txt', 'r')
    number33 = f.readlines()
    num33_list = []
    for nub33 in number33:
        num33_list.append(nub33[:-1])
    wb = xlrd.open_workbook(process_data)
    data = wb.sheet_by_name('Sheet1')
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
    rows = data.nrows
    number33_dsdzqdb = dict()
    for i in range(a, rows):
        # 依然是读取的背面行的数据,正面镀膜完后再背面..耗材去背面的更合理
        if data.cell(i, index1).value in num33_list and data.cell(i, index2).value == face:
            # print(data.row_values(i))
            number33_dsdzqdb[data.cell(i, index1).value] = [data.cell(i, index3).value, data.cell(i, index4).value,
                                                            data.cell(i, index5).value]
            # print('离子枪灯丝: {}, 电子枪灯丝: {}, 挡板: {}'.format(data.cell(i, index3).value, data.cell(i, index4).value,
            #                                                 data.cell(i, index5).value))
    data = json.dumps(number33_dsdzqdb)
    with open(num33_hc_js, 'w') as js_file:
        js_file.write(data)


def hc_feature(num33_hc_js, number33_thick_js, org_refine_data_json, thick_hc_lab_js):
    num33_hc = json.load(open(num33_hc_js, 'r'))
    num33_thick14 = json.load(open(number33_thick_js, 'r'))
    thick14_lab = json.load(open(org_refine_data_json, 'r'))
    thick_hc_lab = dict()
    for num33 in num33_hc:
        # method1.
        # # one_hot转换, 120维太大了,转成二进制试试看.每个耗材转化为7个01序列,耗材特征共21维
        # hc = ''.join(bin(int(str(int(num33_hc[num33][i])), 10))[2:].zfill(7) for i in range(3))

        # method2.
        # 简单线性norm  要对hc维度的数据*40之类的吗? 虽然后续会有数据规整化操作
        hc = str(round(num33_hc[num33][0] / 8, 2)) + ','
        hc += str(round(num33_hc[num33][1] / 120, 2)) + ','
        hc += str(round(num33_hc[num33][2] / 120, 2)) + ','
        # 14+3
        if num33_thick14[num33].endswith(','):
            thickness_hc = num33_thick14[num33] + hc
        else:
            thickness_hc = num33_thick14[num33] + ',' + hc

        # # method3. 好像不大ok..
        # # 交给sklearn直接norm, 直接将耗材数值添加进来
        # hc = ''.join(str(i)+',' for i in num33_hc[num33])
        # if num33_thick14[num33].endswith(','):
        #     thickness_hc = num33_thick14[num33]+hc
        # else:
        #     thickness_hc = num33_thick14[num33] + ',' + hc

        thick_hc_lab[thickness_hc] = thick14_lab[num33_thick14[num33]]
    print(len(thick_hc_lab), "thick_hc_lab")
    data = json.dumps(thick_hc_lab)
    with open(thick_hc_lab_js, 'w') as js_file:
        js_file.write(data)


def sensor_csv_feature(sen_list, data):
    # 一个data/evt样本输入进来,输出特征维度: 4*2+2*2=12维
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
        # rate的 mean, std,
        # if 'RATE' in sen:
        mean = np.mean([i for i in data[sen]][2:])
        std = np.std([i for i in data[sen]][2:], ddof=1)
        f_sensor.append(mean)
        f_sensor.append(std)
    # scale = StandardScaler(with_mean=True, with_std=True)
    # f_sensor = np.reshape(np.array(f_sensor), (-1, 12))
    f_sensor = [round(i, 3) for i in f_sensor]
    return ''.join(str(i) + ',' for i in f_sensor)


def add_sensor_feature(data_dir, evt_7thick_js, thick7_lab_js, thick_hc_lab_js, sen_list, thick14_hc3_sensor16_lab_js):
    evt_7thick = json.load(open(evt_7thick_js, 'r'))
    thick7_multilab = thick7_lab_js
    thick14hc3sensor16_lab = dict()
    lab_thick_hc = dict()
    thick_hc_lab = json.load(open(thick_hc_lab_js, 'r'))
    for thick_hc, lab in thick_hc_lab.items():
        lab_thick_hc[''.join(str(i) for i in lab)] = thick_hc
    for evtname, thick7 in evt_7thick.items():
        # 等待补充的17维feature
        for lab in thick7_multilab[thick7]:
            try:
                thick14_hc3 = lab_thick_hc[''.join(str(i) for i in lab[1])]
            except:
                continue
            # 读取sensor数据csv文件
            full_file_path = os.path.join(data_dir, evtname[3:] + '.csv')
            try:
                data = pd.read_csv(full_file_path, error_bad_lines=False)
            except:
                continue
            sensor16 = sensor_csv_feature(sen_list, data)
            thick14_hc3_sensor16 = thick14_hc3 + sensor16
            thick14hc3sensor16_lab[thick14_hc3_sensor16] = lab

    print(len(thick14hc3sensor16_lab), 'thick14hc3sensor16_lab')
    data = json.dumps(thick14hc3sensor16_lab)
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


def all_usful_sensor_except_thickness(csv_dir, org_refine_thick_lab, oneone_evt_thick, thick14_hc3_sensor64_lab_js,
                                      feature135_lab_js):
    '''
    提取, 整合出来已经处理的thickness、rate这四列之外的,有意义19列数据的feature
    :return: 19*2=38维特征

    '''
    thick14_lab = json.load(open(org_refine_thick_lab, 'r'))
    thick7_lab = dict()
    for k, v in thick14_lab.items():
        tmp = k.split(',')[:7]
        thick7_lab[''.join(i + ',' for i in tmp)] = v
    evt_7thick = json.load(open(oneone_evt_thick, 'r'))
    feature135_lab = dict()
    # key value 转换
    feature97_lab = json.load(open(thick14_hc3_sensor64_lab_js, 'r'))
    lab_feature97 = dict()
    for feature97, lab in feature97_lab.items():
        lab_feature97[''.join(str(i) for i in lab)] = feature97
    for evt, thick7 in evt_7thick.items():
        feature38_sensor = usful_sensor_feature(os.path.join(csv_dir, evt[3:] + '.csv'))
        try:
            lab = thick7_lab[thick7]
        except:
            continue
        try:
            old_thick_hc_sensor = lab_feature97[''.join(str(i) for i in lab)]
        except:
            continue
        if feature38_sensor != '':
            # print(old_thick_hc_sensor, thick7, evt)
            new_thick_hc_sensor = old_thick_hc_sensor + feature38_sensor
            new_thick_hc_sensor += evt  # key: feature135+evt_name
            feature135_lab[new_thick_hc_sensor] = lab

    len_ = 0
    for k, v in feature135_lab.items():
        len_ = len(k.split(','))
        break
    keys = list(feature135_lab.keys())
    for k in keys:
        l = len(k.split(','))
        if l != len_:
            del feature135_lab[k]
    # 落盘
    js = json.dumps(feature135_lab)
    with open(feature135_lab_js, 'w') as js_:
        js_.write(js)
    print("got {}!!".format(feature135_lab_js))


if __name__ == "__main__":
    base_data_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件'
    file1 = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#膜色文件与EVT文件对应表.xlsx'
    file2 = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\膜色数据.xlsx'
    data_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\org_refine_thickness_lab_curve.json'
    evt_cc_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC'
    process_data = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\工艺记录.xlsx'
    evt_thick = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\oneone_evt_thickness.json'
    rate_thickness_check(evt_cc_dir)  # import check_data.py 中的函数实现部分数据清洗功能