# coding=utf-8
'''
cd  data && python dataloader.py
'''

import os
import time
import pandas as pd
import xlrd
import collections
import sys
sys.path.append("..")
from utils.datetime_utils import check_one_day_offset
from utils.check_thickness_curve import check_thick_and_curve, clean_zeiss_review


class DataLoader():

    def __init__(self, base_dir, evt_dir, process_file, match_file, data_detail_file, math_file_title,
                 single_lab_curve_file, double_lab_curve_file,
                 single_lab_curve_file_title, data_detail_title, cycle_info_title, save_dir, types=None,
                 glasses_type=None, csv_name=None, thickness_layer=0, oven_layer_index=0):
        '''
        '''

        self.glasses_type = glasses_type
        self.csv_name = csv_name
        self.base_dir = base_dir
        self.evt_dir = evt_dir
        self.match_file = match_file
        self.math_file_title = math_file_title
        self.single_lab_curve_file = single_lab_curve_file
        self.double_lab_curve_file = double_lab_curve_file
        self.single_lab_curve_file_title = single_lab_curve_file_title
        self.data_detail_title = data_detail_title
        self.process_file = process_file
        self.cycle_info_title = cycle_info_title
        self.data_detail_file = data_detail_file
        self.thickness_layer = thickness_layer
        self.oven_layer_index = oven_layer_index
        self.types = types
        self.save_dir = save_dir
        self.titles = ["OvenNo", "FileID", "Create_Time", "CCCX", "Thickness", "Material", "Start_time",
                       "Step_start_timestamp",
                       "Consumables", "single_lab_curve", "single_lab_value", "path_sensor_data", "clean_cycle_index",
                       "Type"]

        # {炉号:[evt_cx, evtcc]}
        self.OvenNo_EvtPair = dict()
        self.Oven_Type = dict()
        self.OvenNo_CX_LabCurve = dict()
        self.OvenNo_CC_LabCurve = dict()
        self.all_OvenNo_CCCX_LabCurve = dict()
        self.double_LabCurve = dict()
        self.FileId_LabCurve = dict()
        self.bad_evts = open('./bad_evts.txt', 'w')
        self.bad_evt = []

        # data_detail_title = {"炉号": "炉序号", "老花近视1.6_1.67": "产品名"}
        self.data_type = self.data_detail_title['老花近视1.6_1.67']
        self.oven = self.data_detail_title['炉号']

        # 定义正背输出
        self.df_cx = pd.DataFrame()
        self.df_cc = pd.DataFrame()

        # 定义evt连续但正背跳转输出
        self.df_cxcc = pd.DataFrame()

    def evt_and_sensor_thickness(self, ):
        '''
        筛选有evt和sensor文件, 且同时是thickness_layer层镀膜的 evt_name-list: 22010604, 22010605

        '''

        file_list = []
        csvs_dir = os.path.join(self.base_dir, self.evt_dir)
        nns = os.listdir(csvs_dir)
        nns = [a for a in nns if a[:3] in ["evt", "EVT"]]
        for file_name in nns:
            with open(os.path.join(csvs_dir, file_name), 'r') as f:
                if f.read().count('Thickness') == self.thickness_layer:
                    file_list.append(file_name[3:-4])

        return file_list

    def get_thickness(self, evt_file):
        '''
        read evt_file, get thickness_list

        '''
        csv_file = os.path.join(self.base_dir, self.evt_dir, evt_file + '.csv')
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            thickness = []
            for line in lines:
                if 'Thickness' in line:
                    thickness.append(float(line.split(',')[4]))
            if len(thickness) != self.thickness_layer:
                print("evt file {} has less thickness data than expected {}".format(evt_file, self.thickness_layer))
        return thickness

    def get_material(self, evt_file):
        '''
        read evt_file, get material_list
        这部分写的有点... 考虑同机台的产品, 使用的镀膜材料应该是固定的? 需要每一次从evt中解析吗?

        '''
        layer_ind = [2, 3, 4, 5, 6, 7]
        csv_file = os.path.join(self.base_dir, self.evt_dir, evt_file + '.csv')
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            material = []
            for ind in layer_ind:
                for line in lines:
                    if 'Start Step No. {}'.format(ind) in line:
                        material_ = line.split(',')[4].split('_')[-1]
                        material_ = material_.split('/')[0]
                        material.append(material_)
            for line in lines:
                if 'Start Step No. 8' in line:
                    material.append(line.split(',')[4].split('_')[1])
            if len(material) != self.thickness_layer:
                print("evt file {} has less material data than expected {}".format(evt_file, self.thickness_layer))
        return material
        # return ['SiO2', 'ITO', 'Ti3O5', 'SiO2', 'Ti3O5', 'SiO2', 'COTEC1200']

    def t2t(self, t):
        '''
        2021/3/17 5：21：31 时间格式转换成数值

        '''
        timeArray = time.strptime(t, "%Y/%m/%d %H:%M:%S")

        return time.mktime(timeArray)

    def check_evts(self, lines, csv_file):
        for line in lines:
            if 'Production Start' in line:
                return
        evt_name = os.path.basename(csv_file)
        if evt_name not in self.bad_evt:
            self.bad_evt.append(evt_name)
            self.bad_evts.write(evt_name + '\n')


    def get_start_time(self, evt_file):
        '''
        read evt_file, get start_time

        '''
        csv_file = os.path.join(self.base_dir, self.evt_dir, evt_file + '.csv')
        res = 0
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            self.check_evts(lines, csv_file)
            for line in lines:
                if 'Production Start' in line:
                    dmy = line.split(',')[1].split('/')
                    m = '0' * (2 - len(dmy[0])) + dmy[0]
                    d = '0' * (2 - len(dmy[1])) + dmy[1]
                    dmy = dmy[-1] + '/' + m + '/' + d
                    start = line.split(',')[2]
                if "Start Step No." in line:
                    step = line.split(',')[0]
                    t1 = dmy + ' ' + start[:-1]
                    t2 = dmy + ' ' + step
                    res = self.t2t(t2) - self.t2t(t1)
                    break
        return res

    def get_step_timestamp(self, evt_file):
        '''
        read evt_file, get start_time for each process step

        '''
        csv_file = os.path.join(self.base_dir, self.evt_dir, evt_file + '.csv')
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            step_counter = 1
            step_timestamps = []
            Ymd = None
            privious_timestamp = None
            for line in lines:
                if 'Production Start' in line:
                    dmy = line.split(',')[1].split('/')
                    m = '0' * (2 - len(dmy[0])) + dmy[0]
                    d = '0' * (2 - len(dmy[1])) + dmy[1]
                    Ymd = dmy[-1] + '/' + m + '/' + d
                    start = line.split(',')[2]
                    privious_timestamp = Ymd + " " + start

                if "Start Step No. " in line:
                    current_step_number = int(line.split(",")[2].split("Start Step No. ")[1].split("of")[0])
                    step = line.split(',')[0]
                    if current_step_number != step_counter:
                        print("EVT file {} has process order error, at step {}".format(evt_file, step_counter))
                    current_timestamp = Ymd + " " + step
                    current_timestamp = check_one_day_offset(privious_timestamp, current_timestamp, evt_file)
                    step_timestamps.append(current_timestamp)
                    step_counter += 1

        return step_timestamps

    def get_double_curve(self, OvenNo_cx_or_cc_title, lab_value_title):
        '''
        获取每一炉的self.oven_layer_index层双面曲线, lab值

        '''
        tmp = dict()
        wb = xlrd.open_workbook(os.path.join(self.base_dir, self.double_lab_curve_file))
        data = wb.sheet_by_name(r'Sheet1')
        rows = data.nrows
        need_title = [OvenNo_cx_or_cc_title, lab_value_title]
        title = data.row_values(0)
        OvenNo_Cx_or_CC_ind, l_ind, a_ind, b_ind, curve_ind1, curve_ind2 = title.index(need_title[0]), title.index(
            need_title[1][0]), title.index(need_title[1][1]), title.index(need_title[1][2]), title.index(
            '380'), title.index('780')
        for i in range(1, rows):
            OvenNo = data.cell(i, OvenNo_Cx_or_CC_ind).value.split('_')[0]
            if OvenNo not in tmp:
                tmp[OvenNo] = []
            line = data.row_values(i)
            curve = line[curve_ind1: curve_ind2 + 1]
            try:
                curve = [float(a) for a in curve]
            except:
                continue
            for a in curve:
                if a <= 0.0:
                    curve = []
                    break
            L, A, B = line[l_ind], line[a_ind], line[b_ind]
            tmp[OvenNo].append(curve + [L, A, B])
        for oven, lab_value_list in tmp.items():
            if len(lab_value_list) >= self.oven_layer_index:
                if len(lab_value_list[self.oven_layer_index - 1]) > 3:
                    self.double_LabCurve[oven] = [lab_value_list[self.oven_layer_index - 1][:-3],
                                                  lab_value_list[self.oven_layer_index - 1][-3:]]
        # print(self.double_LabCurve)

    def get_single_curve(self, OvenNo_cx_or_cc_title, lab_value_title, single_curve_title):
        '''
        获取 cx cc lab_curve, lab_value

        '''

        wb = xlrd.open_workbook(os.path.join(self.base_dir, self.single_lab_curve_file))
        data = wb.sheet_by_name(r'Sheet1')
        rows = data.nrows
        need_title = [OvenNo_cx_or_cc_title, lab_value_title, single_curve_title]
        title = data.row_values(0)
        OvenNo_Cx_or_CC_ind, l_ind, a_ind, b_ind, curve_ind = title.index(need_title[0]), title.index(
            need_title[1][0]), title.index(need_title[1][1]), \
                                                              title.index(need_title[1][2]), title.index(need_title[2])
        for i in range(1, rows):
            OvenNo_CX_or_CC = data.cell(i, OvenNo_Cx_or_CC_ind).value
            OvenNo, CX_or_CC = OvenNo_CX_or_CC.split('_')[0], OvenNo_CX_or_CC.split('_')[1]
            line = data.row_values(i)
            curve = line[curve_ind]
            curve = [float(a) for a in curve.split(',')[190: 591][::5]]
            for a in curve:
                if a <= 0.0:
                    curve = []
                    break
            if len(curve) > 0:
                L, A, B = line[l_ind], line[a_ind], line[b_ind]
                self.all_OvenNo_CCCX_LabCurve[OvenNo_CX_or_CC] = [curve, [L, A, B]]
                if CX_or_CC.lower() == "cx":
                    self.OvenNo_CX_LabCurve[OvenNo] = [curve, [L, A, B]]
                elif CX_or_CC.lower() == "cc":
                    self.OvenNo_CC_LabCurve[OvenNo] = [curve, [L, A, B]]

    def get_lab_curve(self, ovenno_cx, ovenno_cc):
        # cx
        curve_cx = []
        selected_oven_cx = []
        LAB_cx = []
        for ovenno in ovenno_cx:
            ovenno = int(ovenno)
            try:
                curve_value = self.OvenNo_CX_LabCurve[str(ovenno)]
                curve_cx.append(curve_value[0])
                LAB_cx.append(curve_value[1])
                selected_oven_cx.append(ovenno)
            except:
                continue
        # cc
        curve_cc = []
        LAB_cc = []
        selected_oven_cc = []
        for ovenno in ovenno_cc:
            try:
                curve_value = self.OvenNo_CC_LabCurve[str(ovenno)]
                curve_cc.append(curve_value[0])
                LAB_cc.append(curve_value[1])
                selected_oven_cc.append(ovenno)
            except:
                continue

        return curve_cx, LAB_cx, selected_oven_cx, curve_cc, LAB_cc, selected_oven_cc

    def get_consumables_info(self, OvenNo, CX_or_CC):
        '''
        读取工艺记录文件, 获取耗材: 离子枪灯丝, 电子枪灯丝, 挡板, 晶振片 等耗材信息
        cycle_info_title = ['反正面', '炉序号', '电子枪灯丝', '离子枪灯丝', '挡板', '晶振片']

        '''
        consumables = []
        tmp = {"正面": "CX", "背面": "CC"}
        wb = xlrd.open_workbook(os.path.join(self.base_dir, self.process_file))
        data = wb.sheet_by_name('Sheet1')
        rows = data.nrows
        consumable_dict = dict()
        title = data.row_values(0)
        face, oven, dzq, lzq, db, jzp = title.index(self.cycle_info_title[0]), title.index(
            self.cycle_info_title[1]), title.index(
            self.cycle_info_title[2]), title.index(self.cycle_info_title[3]), title.index(
            self.cycle_info_title[4]), title.index(self.cycle_info_title[5])
        for i in range(1, rows):
            ovenno = data.cell(i, oven).value
            if str(ovenno)[:2] == str(self.glasses_type):
                cxcc = data.cell(i, face).value
                key = "{}_{}".format(ovenno, tmp[cxcc])
                # 离子枪灯丝, 电子枪灯丝, 挡板, 晶振片
                consumable_dict[key] = [data.cell(i, lzq).value, data.cell(i, dzq).value, data.cell(i, db).value,
                                        data.cell(i, jzp).value]
        for i, ovenno in enumerate(OvenNo):
            if type(CX_or_CC) == list:
                ovenno_face = "{}_{}".format(ovenno, CX_or_CC[i])
            else:
                ovenno_face = "{}_{}".format(ovenno, CX_or_CC)
            try:
                consumable = consumable_dict[ovenno_face]
                consumables.append(consumable)
            except:
                consumables.append(['', '', '', ''])

        assert len(OvenNo) == len(consumables)

        return consumables

    def get_sensor_data(self, evt_csv):
        '''
        read sensor.csv

        '''
        file_path = os.path.join(self.base_dir, self.evt_dir, evt_csv[3:] + '.csv')
        # try:
        #     file = open(file_path, 'r')
        # except:
        #     return ''
        # lines = file.readlines()

        return file_path

    def get_cycle_value(self, OvenNo_cx, OvenNo_cycle_cx):
        cycles = []
        for oven_face in OvenNo_cx:
            oven_face = '{}_{}'.format(int(oven_face.split('_')[0]), oven_face.split('_')[1])
            try:
                cycles.append(OvenNo_cycle_cx[oven_face])
            except:
                cycles.append('')
        return cycles

    def read_process_get_cycle_index(self, ):
        '''
        根据耗材文件中的电子枪信息, 获取清洗index

        '''

        process_path = os.path.join(self.base_dir, self.process_file)
        df = pd.read_excel(process_path)

        # 剔除非33编号的样本  self.glasses_type
        selected_arid = df.AR设备ID.apply(lambda x: x == self.glasses_type).reset_index(drop=True)
        df = df[selected_arid].reset_index(drop=True)
        # 逆序dataframe
        df = df.reindex(index=df.index[::-1])
        electron_gun = []
        for e_gun in df.电子枪灯丝:
            electron_gun.append(e_gun)
        lens = len(electron_gun)
        cycle_index = 0
        cycle_value = []
        for i in range(lens - 1):
            cycle_value.append(cycle_index)
            # 电子枪灯丝值从>100突然变化到<10, 则认为是进入了新的清洗周期
            if electron_gun[i] >= 100 and electron_gun[i + 1] <= 10:
                cycle_index += 1
        # 最后一炉的清洗周期单独处理
        if electron_gun[-1] <= 10:
            last_cycle_value = cycle_value[-1] + 1
        else:
            last_cycle_value = cycle_value[-1]
        cycle_value += [last_cycle_value]

        # check cycle_value
        # a = electron_gun.index(2)
        # print(cycle_value[a-1], cycle_value[a], cycle_value[a+1])

        # 把df再一次逆序, 数据复原~
        df = df.reindex(index=df.index[::-1])
        # 对应以上获得的cycle_value也逆序
        cycle_value = cycle_value[::-1]
        df['cycle_index'] = cycle_value
        assert len(cycle_value) == len(df)

        all_cxcc = df.反正面
        # 把正背面数据区分开
        df_cx = all_cxcc.apply(lambda x: x == "正面").reset_index(drop=True)
        df_cc = all_cxcc.apply(lambda x: x == "背面").reset_index(drop=True)
        df_cx = df[df_cx].reset_index(drop=True)
        df_cc = df[df_cc].reset_index(drop=True)

        # 正背分别粗暴做炉号去重..
        df_cx = df_cx.drop_duplicates(subset=[self.cycle_info_title[1]], keep='first').reset_index(drop=True)
        df_cc = df_cc.drop_duplicates(subset=[self.cycle_info_title[1]], keep='first').reset_index(drop=True)

        OvenNo_cycle_cx = dict()
        for i in range(len(df_cx)):
            OvenNo_cycle_cx["{}_{}".format(df_cx[self.cycle_info_title[1]][i], "CX")] = df_cx['cycle_index'][i]
        assert len(OvenNo_cycle_cx) == len(df_cx)
        OvenNo_cycle_cc = dict()
        for i in range(len(df_cc)):
            OvenNo_cycle_cc["{}_{}".format(df_cc[self.cycle_info_title[1]][i], "CC")] = df_cc['cycle_index'][i]
        assert len(OvenNo_cycle_cc) == len(df_cc)

        all_oven_cycle = dict()
        all_oven_cycle.update(OvenNo_cycle_cx)
        all_oven_cycle.update(OvenNo_cycle_cc)
        assert len(all_oven_cycle) == len(OvenNo_cycle_cc) + len(OvenNo_cycle_cx)

        return OvenNo_cycle_cx, OvenNo_cycle_cc, all_oven_cycle

    def get_ovenno_cxcc(self, df_match):

        ll = len(df_match)
        OvenNo_cxccs = []
        for i in range(ll):
            ovenno_cxcc = '{}_{}'.format(df_match.iloc[i]['OvenNo'], df_match.iloc[i]['FileFilmCode'].split('_')[-1])
            OvenNo_cxccs.append(ovenno_cxcc)

        return OvenNo_cxccs

    def slim_data(self, path=None):
        # 滤除一些数据缺失的行
        data = pd.read_csv(path)
        for title in self.titles:
            data = data[data[title].notna()]
        data.to_csv(path, index=False)

    def split_type(self, type_):
        if "1.67" in type_:
            if "近视" in type_:
                return "1.67近视"
            elif "老花" in type_:
                return "1.67老花"
            else:
                return ''
        elif "1.6" in type_:
            if "近视" in type_:
                return "1.6近视"
            elif "老花" in type_:
                return "1.6老花"
            else:
                return ''
        else:
            return ''

    def get_oven_type(self, ):
        # 1.6 1.67 近视 老花

        wb = xlrd.open_workbook(os.path.join(self.base_dir, self.data_detail_file))
        data = wb.sheet_by_name(r'Sheet1')
        rows = data.nrows

        need_title = [self.oven, self.data_type]
        title = data.row_values(0)
        oven_index, type_index = title.index(need_title[0]), title.index(need_title[1])
        for i in range(1, rows):
            Oven = data.cell(i, oven_index).value
            type_ = data.cell(i, type_index).value
            type_ = self.split_type(type_)
            if Oven not in self.Oven_Type:
                self.Oven_Type[Oven] = []
            self.Oven_Type[Oven].append(type_)
        # 每一炉的镜片属性为产片占比最多的那类
        ovens = list(self.Oven_Type.keys())
        for oven in ovens:
            # 统计产类型占多数的type, 赋给oven
            # print(self.Oven_Type[oven], oven)
            distribute = collections.Counter(self.Oven_Type[oven])
            sored = sorted(distribute.items(), key=lambda kv: (kv[1], kv[0]))[::-1]
            del self.Oven_Type[oven]
            self.Oven_Type[oven] = sored[0][0]
        # print(self.Oven_Type)

    def get_data_type(self, oven):

        try:
            type_ = self.Oven_Type[str(oven)]
        except:
            type_ = ''
        return type_

    def split_data_pair(self, pd_data):
        type_pair = dict()
        for key in self.types:
            type_pair[key] = []
        lens = len(pd_data)
        for i in range(lens - 1):
            evtname1, evtname2 = pd_data["FileID"][i], pd_data["FileID"][i + 1]
            evt1, evt2 = int(evtname1[-3:]), int(evtname2[-3:])
            if (evt2 - evt1 == 2) or (evt2 - evt1 == 1):
                if evt2 - evt1 == 2:
                    print('123', evtname1, evtname2)
                # 连续时间炉数据对
                if (pd_data["Type"][i] != '') and (pd_data["Type"][i] == pd_data["Type"][i + 1]):
                    type_pair[pd_data["Type"][i]].extend([pd_data["OvenNo"][i], pd_data["OvenNo"][i + 1]])
        type_pair_df = []
        for k, single_type_ovens in type_pair.items():
            slim_ovens = pd_data.OvenNo.apply(lambda x: x in single_type_ovens).reset_index(drop=True)
            slim_data = pd_data[slim_ovens].reset_index(drop=True)
            type_pair_df.append(slim_data)

        return type_pair_df

    def mix_cx_cc(self, df_match):

        lens = len(df_match)
        cx_cc_evts = []
        ovencxcc = []

        for i in range(lens - 1):
            evtname1, evtname2 = df_match["FileID"][i], df_match["FileID"][i + 1]
            evt1, evt2 = int(evtname1[-3:]), int(evtname2[-3:])
            if evt1 + 1 == evt2:
                oven1, oven2 = df_match["OvenNo"][i], df_match["OvenNo"][i + 1]
                cx_cc1, cx_cc2 = df_match["FileFilmCode"][i].split("_")[-1], df_match["FileFilmCode"][i + 1].split("_")[
                    -1]
                if (cx_cc1 + cx_cc2).lower() in ["cccx", "cxcc"]:
                    # 连续且正背面跳转:
                    if evtname1 not in cx_cc_evts:
                        cx_cc_evts.append(evtname1)
                        ovencxcc.append("{}_{}".format(oven1, cx_cc1))
                    if evtname2 not in cx_cc_evts:
                        cx_cc_evts.append(evtname2)
                        ovencxcc.append("{}_{}".format(oven2, cx_cc2))

        slim_evts = df_match.FileID.apply(lambda x: x in cx_cc_evts).reset_index(drop=True)
        slim_df = df_match[slim_evts].reset_index(drop=True)
        # 返回所有的evt, 然后以evt为index获取lab_value, lab_curve, clean_index等信息f
        lens = len(slim_df)
        for i in range(lens):
            oven, evt, face = slim_df["OvenNo"][i], slim_df["FileID"][i], slim_df["FileFilmCode"][i].split("_")[-1]
            if oven not in self.OvenNo_EvtPair:
                self.OvenNo_EvtPair[oven] = ['', '']
            if face.lower() == "cx":
                self.OvenNo_EvtPair[oven][0] = evt
            elif face.lower() == "cc":
                self.OvenNo_EvtPair[oven][1] = evt
        return ovencxcc, slim_df

    def get_mix_face_lab_curve(self, ovencxcc_list):
        curves, lab_values = [], []
        for a in ovencxcc_list:
            try:
                curves.append(self.all_OvenNo_CCCX_LabCurve[a][0])
                lab_values.append(self.all_OvenNo_CCCX_LabCurve[a][1])
            except:
                curves.append('')
                lab_values.append('')

        return curves, lab_values

    def get_double_lab_curve(self, oven):
        try:
            curve = self.double_LabCurve[str(oven)][0]
        except:
            curve = ''

        return curve

    def get_double_lab_value(self, oven):
        try:
            value = self.double_LabCurve[str(oven)][1]
        except:
            value = ''

        return value

    def evt_continue(self, pd_data):
        evts = []
        lens = len(pd_data)
        for i in range(lens - 1):
            evtname1, evtname2 = pd_data["FileID"][i], pd_data["FileID"][i + 1]
            evt1, evt2 = int(evtname1[-3:]), int(evtname2[-3:])
            # oven_face1, oven_face2 = "{}_{}".format(pd_data["OvenNo"][i], pd_data["FilmCode_MES"][i].split("_")[-1]), "{}_{}".format(pd_data["OvenNo"][i+1], pd_data["FilmCode_MES"][i+1].split("_")[-1])
            if (evt2 - evt1 == 2) or (evt2 - evt1 == 1):
                if evt2 - evt1 == 2:
                    # print('456', evtname1, evtname2)
                    pass
                if evt1 not in evts:
                    evts.append(evtname1)
                if evt2 not in evts:
                    evts.append(evtname2)

        return evts

    def change_time(self, time):
        # 2022/1/13 15:29:07 -> 2022:1:13-15:29:07
        try:
            t1 = time.split(' ')[0]
            y, m, d, a = t1.split("-")[0], t1.split("-")[1], t1.split("-")[2], time.split(' ')[1]
        except:
            y, m, d, a = '', '', '', ''

        return "{}/{}/{}-{}".format(y, m, d, a)

    def mix_face_post_process(self, data_path):
        slim_evts = []
        # 删除空格后, 还需要确保连续的cx cc的镜片类型一致
        data = pd.read_csv(data_path)
        evts, faces, types = data["FileID"], data["CCCX"], data["Type"]
        lens = len(data)
        for i in range(lens - 1):
            evtname1, evtname2 = evts[i], evts[i + 1]
            cx_cc1, cx_cc2 = faces[i], faces[i + 1]
            type1, type2 = types[i], types[i + 1]
            evt1, evt2 = int(evtname1[-3:]), int(evtname2[-3:])
            if evt1 + 1 == evt2:
                if (cx_cc1 + cx_cc2).lower() == "cccx" or (cx_cc1 + cx_cc2).lower() == "cxcc":
                    if type1 == type2:
                        if evtname1 not in slim_evts:
                            slim_evts.append(evtname1)
                        if evtname2 not in slim_evts:
                            slim_evts.append(evtname2)
        slim_evt = data.FileID.apply(lambda x: x in slim_evts).reset_index(drop=True)
        slim_data = data[slim_evt].reset_index(drop=True)
        slim_data.to_csv(data_path, index=False)

    def int_ovens(self, oven):

        if type(oven) in [float, int]:
            oven = str(oven)

        return int(oven.split('.')[0])

    def run_load_data(self, ):
        '''
            main

            math_file_title = {"炉号": "OvenNo", "创建时间": "CreationTime_MES", "正背面": "FilmCode_MES"}
            single_lab_curve_file_title = {"膜色曲线": "DataExpand", "OvenNo_CXCC": "测膜标签", "LAB_value": ["L值", "A值", "B值"]}

        '''

        # 一些文件的 title_key
        cx_or_cc = self.math_file_title["正背面"]
        create_time = self.math_file_title["创建时间"]
        file_id = self.math_file_title["evt_name"]
        OvenNo_cx_or_cc_title = self.single_lab_curve_file_title['OvenNo_CXCC']
        lab_value_title = self.single_lab_curve_file_title['LAB_value']
        single_curve_title = self.single_lab_curve_file_title['膜色曲线']

        # 获取炉号的双面曲线
        self.get_double_curve(OvenNo_cx_or_cc_title, lab_value_title)
        # 分别获取正背面lab曲线
        self.get_single_curve(OvenNo_cx_or_cc_title, lab_value_title, single_curve_title)
        # 根据耗材文件, 获取cycle index 信息
        OvenNo_cycle_cx, OvenNo_cycle_cc, all_OvenNo_cycle = self.read_process_get_cycle_index()
        # 读取每炉明细获取老花近视1.61.67信息
        self.get_oven_type()

        # 读取炉号evt对应关系文件
        df_match = pd.read_excel(os.path.join(self.base_dir, self.match_file))
        df_match = df_match[df_match.IsFinished == 1][df_match.IsRight == 1].reset_index(drop=True)

        # 33炉号筛选
        slim_ovens = df_match.OvenNo.apply(lambda x: str(x)[:2] == str(self.glasses_type)).reset_index(drop=True)
        df_match = df_match[slim_ovens].reset_index(drop=True)
        # evtname 去重
        df_match = df_match.drop_duplicates(subset=[file_id], keep='first').reset_index(drop=True)

        # 筛选具有evt和sensor, 且同时是thickness_layer层镀膜的文件
        file_list = self.evt_and_sensor_thickness()
        selected_oven = df_match.FileID.apply(lambda x: x[3:] in file_list).reset_index(drop=True)
        df_match = df_match[selected_oven].reset_index(drop=True)
        # 创建时间去重
        df_match = df_match.drop_duplicates(subset=[create_time], keep='first').reset_index(drop=True)
        # 炉号全部int处理
        df_match["OvenNo"] = df_match.OvenNo.apply(self.int_ovens)

        # 不区分 1.671.6 近视老花类型, 只做evt连续筛选, 方便后面排查"差"数据是否是与清洗周期有关系
        continue_evts = self.evt_continue(df_match)
        continue_ovens = df_match.FileID.apply(lambda x: x in continue_evts).reset_index(drop=True)
        all_continue_df = df_match[continue_ovens].reset_index(drop=True)
        ovens, faces = all_continue_df.OvenNo, all_continue_df.FilmCode_MES
        all_ovencxcc_list = ["{}_{}".format(ovens[i], faces[i].split("_")[-1]) for i in range(len(all_continue_df))]
        all_continue_df['clean_cycle_index'] = self.get_cycle_value(all_ovencxcc_list, all_OvenNo_cycle)

        # 添加type信息
        all_continue_df['Type'] = all_continue_df.OvenNo.apply(self.get_data_type)
        # 修改时间显示: CreationTime_MES, FileCreateTime, CreateDate
        all_continue_df['CreationTime_MES'] = all_continue_df.CreationTime_MES.apply(self.change_time)
        all_continue_df['FileCreateTime'] = all_continue_df.FileCreateTime.apply(self.change_time)
        all_continue_df['CreateDate'] = all_continue_df.CreateDate.apply(self.change_time)
        path_ = os.path.join(self.save_dir, '{}_all.xlsx'.format(self.csv_name))
        writer = pd.ExcelWriter(path_, encoding="utf-8-sig")
        all_continue_df.to_excel(writer, "sheet1")
        writer.save()

        # 场景1, 连续且正背跳转
        ovencxcc_list, slim_df = self.mix_cx_cc(df_match)  # ["33322010705_CC", "33322010705_CX", ...]
        ovens, faces = [a.split("_")[0] for a in ovencxcc_list], [a.split("_")[1] for a in ovencxcc_list]
        self.df_cxcc['OvenNo'] = slim_df.OvenNo
        self.df_cxcc['FileID'] = slim_df.FileID
        self.df_cxcc["CCCX"] = faces
        self.df_cxcc['Create_Time'] = slim_df.CreateDate
        self.df_cxcc['Create_Time'] = self.df_cxcc.Create_Time.apply(self.change_time)
        curve, lab_value = self.get_mix_face_lab_curve(ovencxcc_list)
        self.df_cxcc['single_lab_curve'] = curve
        self.df_cxcc['single_lab_value'] = lab_value
        self.df_cxcc['double_lab_curve'] = self.df_cxcc.OvenNo.apply(self.get_double_lab_curve)
        self.df_cxcc['double_lab_value'] = self.df_cxcc.OvenNo.apply(self.get_double_lab_value)
        self.df_cxcc['Thickness'] = self.df_cxcc.FileID.apply(self.get_thickness)
        self.df_cxcc['Material'] = self.df_cxcc.FileID.apply(self.get_material)
        self.df_cxcc['Start_time'] = self.df_cxcc.FileID.apply(self.get_start_time)
        self.df_cxcc['Step_start_timestamp'] = self.df_cxcc.FileID.apply(self.get_step_timestamp)
        self.df_cxcc['Consumables'] = self.get_consumables_info(ovens, faces)
        self.df_cxcc['path_sensor_data'] = self.df_cxcc.FileID.apply(self.get_sensor_data)
        self.df_cxcc['clean_cycle_index'] = self.get_cycle_value(ovencxcc_list, all_OvenNo_cycle)
        # 添加老花近视信息
        self.df_cxcc['Type'] = self.df_cxcc.OvenNo.apply(self.get_data_type)

        # 场景2, 拆分正背面数据
        CXCC = df_match[cx_or_cc]
        CX_oven = CXCC.apply(lambda x: x[-2:] in ["CX", 'cX', 'Cx', 'cx']).reset_index(drop=True)
        CC_oven = CXCC.apply(lambda x: x[-2:] in ['cC', 'cc', 'Cc', 'CC']).reset_index(drop=True)
        all_CX = df_match[CX_oven].reset_index(drop=True)
        all_CC = df_match[CC_oven].reset_index(drop=True)
        # 正背面匹配 lab_curve, lab_value
        curve_cx, LAB_cx, selected_oven_cx, curve_cc, LAB_cc, selected_oven_cc = self.get_lab_curve(all_CX.OvenNo,
                                                                                             all_CC.OvenNo)
        selected_cx = all_CX.OvenNo.apply(lambda x: x in selected_oven_cx).reset_index(drop=True)
        df_match_cx = all_CX[selected_cx].reset_index(drop=True)
        selected_cc = all_CC.OvenNo.apply(lambda x: x in selected_oven_cc).reset_index(drop=True)
        df_match_cc = all_CC[selected_cc].reset_index(drop=True)

        # 分别添加炉号, EvtName FileID, Create_Time, FilmCode_MES CX-CC 信息
        # cx
        self.df_cx['OvenNo'] = df_match_cx.OvenNo
        self.df_cx['FileID'] = df_match_cx['FileID']
        self.df_cx['Create_Time'] = df_match_cx[create_time]
        self.df_cx['Create_Time'] = self.df_cx.Create_Time.apply(self.change_time)
        self.df_cx["CCCX"] = 'CX'
        # cc
        self.df_cc['OvenNo'] = df_match_cc.OvenNo
        self.df_cc['FileID'] = df_match_cc['FileID']
        self.df_cc['Create_Time'] = df_match_cc[create_time]
        self.df_cc['Create_Time'] = self.df_cc.Create_Time.apply(self.change_time)
        self.df_cc["CCCX"] = 'CC'

        # 添加 evt_file 的 thickness_list 信息
        self.df_cx['Thickness'] = df_match_cx.FileID.apply(self.get_thickness)
        self.df_cc['Thickness'] = df_match_cc.FileID.apply(self.get_thickness)

        # 添加 evt_file 的 material_list 信息
        self.df_cx['Material'] = df_match_cx.FileID.apply(self.get_material)
        self.df_cc['Material'] = df_match_cc.FileID.apply(self.get_material)

        # 添加机器启动耗时信息
        self.df_cx['Start_time'] = df_match_cx.FileID.apply(self.get_start_time)
        self.df_cc['Start_time'] = df_match_cc.FileID.apply(self.get_start_time)
        self.df_cx['Step_start_timestamp'] = df_match_cx.FileID.apply(self.get_step_timestamp)
        self.df_cc['Step_start_timestamp'] = df_match_cc.FileID.apply(self.get_step_timestamp)

        # 添加正背耗材信息
        self.df_cx['Consumables'] = self.get_consumables_info(df_match_cx.OvenNo, "CX")
        self.df_cc['Consumables'] = self.get_consumables_info(df_match_cc.OvenNo, "CC")

        # 添加lab_curve, lab_value
        self.df_cx['single_lab_curve'] = curve_cx
        self.df_cc['single_lab_curve'] = curve_cc
        self.df_cx['single_lab_value'] = LAB_cx
        self.df_cc['single_lab_value'] = LAB_cc
        self.df_cc['double_lab_curve'] = self.df_cc.OvenNo.apply(self.get_double_lab_curve)
        self.df_cc['double_lab_value'] = self.df_cc.OvenNo.apply(self.get_double_lab_value)
        self.df_cx['double_lab_curve'] = self.df_cx.OvenNo.apply(self.get_double_lab_curve)
        self.df_cx['double_lab_value'] = self.df_cx.OvenNo.apply(self.get_double_lab_value)

        # 添加 sensor_data, 直接读入.csv文件, 后续特征工程放在features.py中处理
        self.df_cx['path_sensor_data'] = df_match_cx.FileID.apply(self.get_sensor_data)
        self.df_cc['path_sensor_data'] = df_match_cc.FileID.apply(self.get_sensor_data)

        Oven_CX, OvenCC = ["{}_{}".format(a, "CX") for a in df_match_cx.OvenNo], ["{}_{}".format(a, "CC") for a in
                                                                                  df_match_cc.OvenNo]
        self.df_cx['clean_cycle_index'] = self.get_cycle_value(Oven_CX, OvenNo_cycle_cx)
        self.df_cc['clean_cycle_index'] = self.get_cycle_value(OvenCC, OvenNo_cycle_cc)

        self.df_cx['Type'] = df_match_cx.OvenNo.apply(self.get_data_type)
        self.df_cc['Type'] = df_match_cc.OvenNo.apply(self.get_data_type)
        # print(len(self.df_cx), len(self.df_cc))
        # cc cx 数据分别拆分出: 1.67近视 1.67老花 1.6近视 1.6老花 实验片
        cc_type_lists = self.split_data_pair(self.df_cc)
        cx_type_lists = self.split_data_pair(self.df_cx)

        # slim_and_save .csv
        for ind, df in enumerate(cx_type_lists):
            if len(df) > 0:
                path_ = os.path.join(self.save_dir, '{}_{}_cx.csv'.format(self.csv_name, self.types[ind]))
                df.to_csv(path_, index=False)
                self.slim_data(path_)

        for ind, df in enumerate(cc_type_lists):
            if len(df) > 0:
                path_ = os.path.join(self.save_dir, '{}_{}_cc.csv'.format(self.csv_name, self.types[ind]))
                df.to_csv(path_, index=False)

        path_ = os.path.join(self.save_dir, '{}_mixface.csv'.format(self.csv_name))
        self.df_cxcc.to_csv(path_, index=False)
        # 剔除空格行
        self.slim_data(path_)
        # 确保连续的cx_cc的镜片类型一致
        self.mix_face_post_process(path_)

        print("Done!")


if __name__ == "__main__":

    base_dir = '/Users/chenjia/Downloads/Learning/SmartMore/1110_beijing/shanghai_data/膜厚推优/ML_ZEISS-dev/4.11data'
    evt_dir = 'EVT'
    single_lab_curve_file = r'单面数据.xlsx'
    double_lab_curve_file = r'双面数据.xlsx'
    process_file = r'工艺记录.xlsx'
    match_file = r'对应关系.xlsx'
    data_detail_file = r'每炉明细.xlsx'

    # 先清洗出正背 老花近视 1.6-1.67
    # windows
    # csv_name = base_dir.split('\\')[-1]
    # mac, linux
    csv_name = base_dir.split('/')[-1]
    save_dir = r'./out/{}'.format(csv_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    math_file_title = {"炉号": "OvenNo", "创建时间": "CreationTime_MES", "正背面": "FilmCode_MES", "evt_name": "FileID"}
    single_lab_curve_file_title = {"膜色曲线": "DataExpand", "OvenNo_CXCC": "测膜标签", "LAB_value": ["L值", "A值", "B值"]}
    data_detail_title = {"炉号": "炉序号", "老花近视1.6_1.67": "产品名"}

    # 根据耗材文件中的 电子枪灯丝==1, 计算清洗周期
    cycle_info_title = ['反正面', '炉序号', '电子枪灯丝', '离子枪灯丝', '挡板', '晶振片']

    thickness_layer = 7
    oven_layer_index = 4
    glasses_type = 33
    types = ["1.67近视", "1.67老花", "1.6近视", "1.6老花", "实验片"]

    dl = DataLoader(base_dir, evt_dir, process_file, match_file, data_detail_file, math_file_title,
                    single_lab_curve_file, double_lab_curve_file,
                    single_lab_curve_file_title, data_detail_title, cycle_info_title, save_dir, types=types,
                    glasses_type=glasses_type, csv_name=csv_name,
                    thickness_layer=thickness_layer, oven_layer_index=oven_layer_index)

    dl.run_load_data()

    # 检查膜厚和曲线的变化规律
    compare_res_dir = r'./compare_thickness_curve'
    check_thick_and_curve(save_dir, compare_res_dir)

    # 根据李工review的文件夹, 删除那些"差"数据对， 并且高亮查的数据输出至xls
    zeiss_res_dir = os.path.join(save_dir, 'compare_thickness_curve')
    xlsx_name = '{}_all.xlsx'.format(csv_name)
    clean_zeiss_review(save_dir, zeiss_res_dir, xlsx_name)
