# coding=utf-8
import json
import os

import xlrd


class data_post_process():
    '''
    处理膜色数据. evt33编号对应数据等. 关联起 膜厚 和 膜色lab_curve值

    '''

    def __init__(self, evt33, membrane, data_dir, data_js, process_data, refine_data_json, oneone_evt_thickness,
                 evt_33number):
        '''
        :param evt33:  33#膜色文件与EVT文件对应表.xlsx
        :param membrane:  膜色数据.xlsx
        :param data_dir:  33#机台文件_7dirs\1.6&1.67_DVS_CC
        :param data_js:  thickness_lab_curve.json
        :param process_data: 含生产周期信息, 解决一种膜厚设置对应多条lab曲线(evt_file)问题

        '''

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

    # def __call__(self, ):
    def run(self, ):
        evt_33(self.evt33, self.evt_dict, self.data_dir, self.evt_33number)
        lab_curve33(self.membrane, self.evt_dict, self.data_js, self.data_dir, self.thickness_lab_curve)
        data_check(self.thickness_lab_curve, self.evt_dict, self.bad_thick_lab)
        refine_data(self.bad_thick_lab, self.process_data, self.refine_data_json, self.oneone_evt_thickness)


def evt_33(evt33, evt_dict, data_dir, evt_33number):
    '''
    :return:
    获取到33#膜色文件与EVT文件对应表.xlsx文件中的evt文件列表;
    以上evt文件列表, 包含cx和cc. 即正反面, 在1.6&1.67_DVS_CC中挑选出存在的evt_files

    '''
    # step1.
    wb = xlrd.open_workbook(evt33)
    data = wb.sheet_by_name('Sheet1')
    rows = data.nrows
    # evt_name: 33321043002编号
    for i in range(1, rows):
        # print(data.row_values(i))  # 每一行都是一个list
        evt_dict[data.cell(i, 5).value + '.CSV'] = data.cell(i, 2).value
    # step2.
    evt_dict_keys = list(evt_dict.keys())
    evt_cc_list = os.listdir(data_dir)
    for evt in evt_dict_keys:
        if evt not in evt_cc_list:
            del evt_dict[evt]
    # print(evt_dict)   # 167个
    # evt_dict：可在1.6&1.67_DVS_CC下找到膜厚值,且可根据33#膜色文件与EVT文件对应表.xlsx找到对应的33321043002编号
    # return evt_dict   # 全局变量, 可不return
    # print(evt_dict, 'evt_dict')

    # 做一下evt:33number js的落盘
    data = json.dumps(evt_dict)
    with open(evt_33number, 'w') as js_file:
        js_file.write(data)


def lab_curve33(membrane, evt_dict, data_js, data_dir, thickness_lab_curve):
    '''
    :return:
    处理下膜色数据.xlsx, 把33321043002编号和膜色曲线list_value关联起来, 并且直接取第四层数据
    处理evt_dict 和 lab_curve, 关联起膜厚值和膜色曲线

    '''
    # step3.
    wb = xlrd.open_workbook(membrane)
    data = wb.sheet_by_name('Sheet1')
    rows = data.nrows
    numberss_dict = dict()
    lab_curve = dict()
    for i in range(1, rows):
        numberss_dict[data.cell(i, 2).value] = numberss_dict.get(data.cell(i, 2).value, 0) + 1
        # 我们取第四层的膜色曲线为基准
        if numberss_dict[data.cell(i, 2).value] == 4:
            # print(data.cell(i, 2).value, data.row_values(i)[19:-1])
            lab_curve[data.cell(i, 2).value] = data.row_values(i)[19:-1]
    # print(lab_curve, '=====')    # 33121060503: [膜色曲线list value]

    # step4.
    evt_name_lab_curve = dict()
    for evt_name, number in evt_dict.items():
        if number in lab_curve:
            evt_name_lab_curve[evt_name] = lab_curve[number]
    # print(len(evt_dict), len(lab_curve), len(evt_name_lab_curve))
    # 确认132个可关联文件
    # num33 = []
    # for a, b in evt_dict.items():
    #     num33.append(b)
    # num33_1 = list(lab_curve.keys())
    # print(len(num33))
    # print(len(num33_1))
    # tmp = []
    # for a in num33:
    #     if a not in num33_1:
    #         tmp.append(a)
    # print(tmp)

    for evt_name in evt_name_lab_curve:
        evt_path = os.path.join(data_dir, evt_name)
        thickness = []
        with open(evt_path, 'r') as file:
            for line in file:
                if "Thickness" in line:
                    thickness.append(line.split(',')[4])  # 膜厚设置值
                    # thickness.append(line.split(',')[6])    # 膜厚实测值(会因为时延有一定误差)
        thickness_lab_curve[''.join(i + ',' for i in thickness) + '{}'.format(evt_name[:-4])] = evt_name_lab_curve[
            evt_name]
    # Y = []
    # for thickness, lab_curve in thickness_lab_curve.items():
    #     Y.append(thickness_lab_curve[thickness])
    # 做一个thickness_lab_curve 的落盘
    data = json.dumps(thickness_lab_curve)
    with open(data_js, 'w') as js_file:
        js_file.write(data)


# data check 0624 chenjia
def data_check(thickness_lab_curve, evt_dict, bad_thick_lab):
    '''
    # 同一膜色设置,对应不同lab_curve
    :param thickness_lab_curve:
    :param evt_dict:
    :return:
    '''
    for thickness, lab_curve in thickness_lab_curve.items():
        thickes, evt_name = thickness.split(',')[:-1], thickness.split(',')[-1]
        thickes = ''.join(i + ',' for i in thickes)
        if thickes not in bad_thick_lab:
            bad_thick_lab[thickes] = []
        bad_thick_lab[thickes].append([evt_name, lab_curve, evt_dict[evt_name + '.CSV']])
    data = json.dumps(bad_thick_lab)
    with open(r'./data_check.json', 'w') as js_file:
        js_file.write(data)
    print('data right? : {}'.format(len(thickness_lab_curve) == len(bad_thick_lab)))
    # 解析json,信息写入txt,反馈给蔡司
    f = open('./data_check.txt', 'w', encoding="utf-8")
    for thick, list_ in bad_thick_lab.items():
        f.write("膜厚设置值: {}".format(thick) + '\n')
        f.write('对应不同lab曲线: ' + '\n')
        for num in list_:
            f.write(''.join(str(i) for i in num) + '\n')
        f.write('\n')
        f.write('\n')


# data refine 0624 chenjia
# process_data: 工艺记录.xlsx
def refine_data(bad_thick_lab, process_data, refine_data_json, one_evt_thickness):
    wb = xlrd.open_workbook(process_data)
    data = wb.sheet_by_name('Sheet1')

    # check 33121052004 等编号个数
    # number33_list = data.col_values(1)[2:]  # 工艺记录中的33标号list
    # nums = []
    # finall_33_number = []
    # for thickness, list_ in bad_thick_lab.items():
    #     for list in list_:
    #         nums.append(list[-1])
    # for i in nums:
    #     if i in number33_list:
    #         finall_33_number.append(i)
    # print(finall_33_number)
    # print(len(finall_33_number))

    rows = data.nrows  # 893
    number_time = dict()
    time_number = dict()
    finall_thick_lab = dict()
    oneone_evt_thickness = dict()
    for i in range(1, rows):
        if data.cell(i, 2).value == "背面":
            # [33121060707, 电子枪数号]
            # print(data.row_values(i))   # 读取整行数据
            number_time[data.cell(i, 1).value] = data.cell(i, 14).value  # 33121052004: time_index
            time_number[data.cell(i, 14).value] = data.cell(i, 1).value  # time_index: 33121052004
    for thickness, list_ in bad_thick_lab.items():
        # time_index = []
        for single_list in list_:
            # single_list：[evt_name, [lab_curv], 33number]
            if single_list[-1] in number_time:
                # time_index.append(number_time[single_list[-1]])
                # print(single_list[-1])
                # the_number_33 = time_number[min(time_index)]  # 电枪数被使用最少的那个33编号文件 会导致有些33121052004在bad_thick_lab中找不到...
                finall_thick_lab[thickness] = single_list[1]
                oneone_evt_thickness[single_list[0]] = thickness  # evtname:thickness
                break  # 找到一个33number了,就不再遍历list_了.
    assert len(finall_thick_lab) == len(oneone_evt_thickness)  # len需要一致!

    data = json.dumps(finall_thick_lab)
    with open(refine_data_json, 'w') as js_file:
        js_file.write(data)

    data = json.dumps(oneone_evt_thickness)
    with open(one_evt_thickness, 'w') as js_file:
        js_file.write(data)


if __name__ == "__main__":
    file1 = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#膜色文件与EVT文件对应表.xlsx'
    file2 = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\膜色数据.xlsx'
    data_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\thickness_lab_curve.json'
    evt_cc_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC'
    process_data = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\工艺记录.xlsx'
