# coding=utf-8
import json
import os
import shutil

import xlrd


# edited by chen-jia 2021.0625


class data_post_process():
    '''
    关联膜厚设置, lab曲线

    '''

    def __init__(self, evt33, membrane, data_dir, data_js, process_data, refine_data_json, oneone_evt_thickness,
                 evt_33number, base_data_dir):
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

        # tmp 落盘文件
        self.evt_thick_js = r'./evt_thick.json'

    # def __call__(self, ):
    def run(self, ):

        if 'CC' in self.data_dir:
            self.face = '背面'
        elif 'CX' in self.data_dir:
            self.face = '正面'

        # 数据清洗: 包括机台号一致,镀膜层数一致等..
        # clean_data_machineid(self.base_data_dir, self.new_path)
        # clean_data_nthickness(self.data_dir, self.n_thickness, self.evt_thick_js)

        evt_33(self.evt33, self.evt_dict, self.data_dir, self.evt_33number)
        lab_curve33(self.membrane, self.evt_dict, self.data_js, self.data_dir, self.thickness_lab_curve)
        check_data(self.thickness_lab_curve, self.evt_dict, self.bad_thick_lab)
        refine_data(self.bad_thick_lab, self.process_data, self.refine_data_json, self.oneone_evt_thickness, self.face)


def clean_data_machineid(base_data_dir, new_path):
    '''
    :param base_data_dir: all data path [没做任何筛选处理的所有data]
    :param new_path:  根据机器版本号等,把数据拆分到sub文件夹,保证各个子文件夹中,机器版本号都一致
    :return:

    '''
    files = os.listdir(base_data_dir)
    files = [i for i in files if "EVT" in i]
    # PLC_version = []
    # MachineName = []
    RecipeName = []
    RecipeName_dict = dict()
    for file in files:
        f = open(os.path.join(base_data_dir, file), 'r')
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

    for k, v in RecipeName_dict.items():
        path = os.path.join(new_path, r'33#机台文件_7dirs', k)
        if not os.path.exists(path):
            os.makedirs(path)
        for cvs_name in v:
            filename0 = os.path.join(base_data_dir, cvs_name)
            filename1 = os.path.join(path, cvs_name)
            filename0_ = os.path.join(base_data_dir, cvs_name[3:])
            filename1_ = os.path.join(path, cvs_name[3:])
            shutil.move(filename0, filename1)
            shutil.move(filename0_, filename1_)

    f = open('kinds_of_data.txt', 'w')
    sub_data_path = os.path.join(new_path, r'33#机台文件_7dirs')
    print('MachineName kinds: {}'.format(os.listdir(sub_data_path)))
    for dir_ in os.listdir(sub_data_path):
        if '.txt' not in dir_:
            f.write(dir_ + '\n')


def clean_data_nthickness(data_dir, n_thickness, evt_thick_js):
    '''
    :param data_dir: clean_data_machineid()中生成的某一sub_dir
    :param n_thickness: 7层膜厚设置, 也可以是5、6、4等
    :param evt_thick_js: 落盘js名称, evtname:thickness
    :return:

    '''
    evt_thick = dict()
    files = os.listdir(data_dir)
    files = [a for a in files if 'EVT' in a]
    for file in files:
        thickness_list = []
        full_path = os.path.join(data_dir, file)
        process = open(full_path, 'r')
        for index, line in enumerate(process):
            if 'Thickness' in line:
                thickness_list.append(line.split(',')[4])
        if len(thickness_list) == n_thickness:
            evt_thick[file] = thickness_list

    data = json.dumps(evt_thick)
    with open(evt_thick_js, 'w') as js_file:
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
    rows = data.nrows
    for i in range(1, rows):
        # print(data.row_values(i))  # 每一行都是一个list
        evt_dict[data.cell(i, 5).value + '.CSV'] = data.cell(i, 2).value
    evt_dict_keys = list(evt_dict.keys())
    evt_cc_list = os.listdir(data_dir)
    for evt in evt_dict_keys:
        if evt not in evt_cc_list:
            del evt_dict[evt]
    # 做一下 evt_dict[EVT21050506 ] = 33121052204 落盘
    data = json.dumps(evt_dict)
    with open(evt_33number, 'w') as js_file:
        js_file.write(data)


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
    data = wb.sheet_by_name('Sheet1')
    rows = data.nrows
    numberss_dict = dict()
    number33_lab_curve = dict()
    for i in range(1, rows):
        numberss_dict[data.cell(i, 2).value] = numberss_dict.get(data.cell(i, 2).value, 0) + 1
        # 我们取第四层的膜色曲线为基准
        if numberss_dict[data.cell(i, 2).value] == 4:
            # print(data.cell(i, 2).value, data.row_values(i)[19:-1])
            number33_lab_curve[data.cell(i, 2).value] = data.row_values(i)[19:-1]

    # 借助 evtname:33number、33number:lab_curv,获得evtname:lab_curv关联
    # get: evtname:lab_curve
    evt_name_lab_curve = dict()
    for evt_name, number in evt_dict.items():
        if number in number33_lab_curve:
            evt_name_lab_curve[evt_name] = number33_lab_curve[number]

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
    # print(len(thickness_lab_curve))

    # 做一个thickness_lab_curve 落盘
    data = json.dumps(thickness_lab_curve)
    with open(data_js, 'w') as js_file:
        js_file.write(data)


# check_data 0624 chenjia
def check_data(thickness_lab_curve, evt_dict, bad_thick_lab):
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
        bad_thick_lab[thickes].append([evt_name, lab_curve, evt_dict[evt_name + '.CSV']])

    # 落盘下
    data = json.dumps(bad_thick_lab)
    with open(r'./bad_thick_lab.json', 'w') as js_file:
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


# refine_data 0624 chenjia
def refine_data(bad_thick_lab, process_data, refine_data_json, one_evt_thickness, face):
    '''
    :param process_data: 工艺记录.xlsx
    :param one_evt_thickness: evt_name: thickness
    :param refine_data_json: finall_thick_lab的落盘json名

    '''
    wb = xlrd.open_workbook(process_data)
    data = wb.sheet_by_name('Sheet1')
    rows = data.nrows
    number_time = dict()
    time_number = dict()
    finall_thick_lab = dict()
    oneone_evt_thickness = dict()
    for i in range(1, rows):
        # 正背面, 与data_dir中的 CC or CX后缀对应
        if data.cell(i, 2).value == face:
            # 33121060707:电子枪数号
            number_time[data.cell(i, 1).value] = data.cell(i, 14).value  # 33121052004: time_index
            time_number[data.cell(i, 14).value] = data.cell(i, 1).value  # time_index: 33121052004
    for thickness, list_ in bad_thick_lab.items():
        # time_index = []
        for single_list in list_:
            # single_list：[evt_name, [lab_curv], 33number]
            if single_list[-1] in number_time:
                # time_index.append(number_time[single_list[-1]])
                # the_number_33 = time_number[min(time_index)]  # 电枪数被使用最少的那个33编号文件 会导致有些33121052004在bad_thick_lab中找不到...
                finall_thick_lab[thickness] = single_list[1]
                oneone_evt_thickness[single_list[0]] = thickness  # evtname:thickness
                break  # 找到一个33number了,就不再遍历list_, break出循环
    assert len(finall_thick_lab) == len(oneone_evt_thickness)  # len需要一致!

    data = json.dumps(finall_thick_lab)
    with open(refine_data_json, 'w') as js_file:
        js_file.write(data)

    data = json.dumps(oneone_evt_thickness)
    with open(one_evt_thickness, 'w') as js_file:
        js_file.write(data)


if __name__ == "__main__":
    base_data_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件'
    file1 = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#膜色文件与EVT文件对应表.xlsx'
    file2 = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\膜色数据.xlsx'
    data_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\thickness_lab_curve.json'
    evt_cc_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC'
    process_data = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\工艺记录.xlsx'
