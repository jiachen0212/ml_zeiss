# coding=utf-8
import json
import os
import shutil
import random
import xlrd
from check_data import rate_thickness_check, bad_sample_clean

# edited by chen-jia 2021.0625


class data_post_process():
    '''
    关联膜厚设置, lab曲线

    '''

    def __init__(self, evt33, membrane, data_dir, data_js, process_data, refine_data_json, oneone_evt_thickness,
                 evt_33number, base_data_dir, CC_dir, CX_dir, num33_hc_js, number33_thick_js, thick_hc_lab_js):
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
        self.CC_dir = CC_dir
        self.CX_dir = CX_dir
        self.num33_hc_js = num33_hc_js
        self.number33_thick_js = number33_thick_js
        self.thick_hc_lab_js = thick_hc_lab_js

        # tmp 落盘文件
        self.evt_thick_js = r'./evt_thick.json'
        self.evt_pair = r'正背面_thickness_evtname.txt'

    # def __call__(self, ):
    def run(self, ):

        # 数据清洗: 包括机台号一致,镀膜层数一致等..
        # clean_data_machineid(self.base_data_dir, self.new_path)
        # clean_data_nthickness(self.data_dir, self.n_thickness, self.evt_thick_js)

        # start data_clean
        get_evtpair_info(self.CC_dir, self.CX_dir, self.evt33, self.evt_pair)
        evt_33(self.evt33, self.evt_dict, self.data_dir, self.evt_33number)
        lab_curve33(self.membrane, self.evt_dict, self.data_js, self.data_dir, self.thickness_lab_curve)
        check_data(self.thickness_lab_curve, self.evt_dict, self.bad_thick_lab)
        refine_data(self.bad_thick_lab, self.process_data, self.refine_data_json, self.oneone_evt_thickness)

        # 添加3个耗材维度特征
        get_hc_value(self.process_data, self.num33_hc_js, '背面')
        hc_feature(num33_hc_js, self.number33_thick_js, self.refine_data_json, self.thick_hc_lab_js)

        # import check_data.py 中的函数实现部分数据清洗功能
        # rate_thickness_check(self.data_dir)   # 膜厚设置\实测值diff与rate*2对比
        # [no!]
        # bad_sample_clean(self.refine_data_json, self.oneone_evt_thickness)   # 离群样本剔除,第一步拟合模型时不适用这些样本,第二步thickness微调再加入




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



def get_evtpair_info(CC_dir, CX_dir, evt33, evt_pair):
    wb = xlrd.open_workbook(evt33)
    data = wb.sheet_by_name('Sheet1')
    number33_evts = dict()
    rows = data.nrows
    for i in range(1, rows):
        number33 = data.cell(i, 2).value
        if number33 not in number33_evts:
            number33_evts[number33] = []
        number33_evts[number33].append(data.cell(i, 5).value)

    dirs = [CC_dir, CX_dir]
    cxs = os.listdir(CX_dir)
    ccs = os.listdir(CC_dir)

    f = open(evt_pair, 'w')
    evt_cc = []
    for num33, evt_list in number33_evts.items():
        if len(evt_list) == 2:
            if (evt_list[1] + '.CSV' in cxs and evt_list[0] + '.CSV' in ccs) or (
                    evt_list[0] + '.CSV' in cxs and evt_list[1] + '.CSV' in ccs):
                evts = []
                for dir_ in dirs:
                    for i in range(2):
                        if os.path.exists(os.path.join(dir_, evt_list[i] + '.CSV')):
                            evts.append(os.path.join(dir_, evt_list[i] + '.CSV'))
                thickness1 = []
                thickness2 = []
                # print(evts)
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
    lines = open(evt_pair, 'r').readlines()
    for line in lines:
        evt1, thick2, thick1 = line[:-1].split(',')[1], line[:-1].split(',')[4], line[:-1].split(',')[3]
        if evt_name == evt1:
            return ''.join(str(i)+',' for i in thick2.split(' '))[:-1], thick1.split(' ')

    return [], []


# refine_data 0624 chenjia
# add 正背面膜厚值, concate or mean. 0625 chenjia
def refine_data(bad_thick_lab, process_data, refine_data_json, one_evt_thickness, concate=True):
    '''
    :param concate or mean True/False
    :param process_data: 工艺记录.xlsx
    :param one_evt_thickness: evt_name: thickness
    :param refine_data_json: finall_thick_lab的落盘json名

    '''
    number33 = open(r'./33number.txt', 'w')
    wb = xlrd.open_workbook(process_data)
    data = wb.sheet_by_name('Sheet1')
    rows = data.nrows
    number_time = dict()
    time_number = dict()
    finall_thick_lab = dict()
    oneone_evt_thickness = dict()
    number33_thick = dict()

    for i in range(1, rows):
        # 正背面, 与data_dir中的 CC or CX后缀对应
        if data.cell(i, 2).value == '背面':
            # 33121060707:电子枪数号
            number_time[data.cell(i, 1).value] = data.cell(i, 14).value  # 33121052004: time_index
            time_number[data.cell(i, 14).value] = data.cell(i, 1).value  # time_index: 33121052004
    for thickness, list_ in bad_thick_lab.items():
        # time_index = []
        for single_list in list_:
            # single_list：[EVT21050425, [lab_curv], 33number]
            number33.write(single_list[-1]+'\n')
            if single_list[-1] in number_time:
                # time_index.append(number_time[single_list[-1]])
                # the_number_33 = time_number[min(time_index)]  # 电枪数被使用最少的那个33编号文件 会导致有些33121052004在bad_thick_lab中找不到...
                oneone_evt_thickness[single_list[0]] = thickness  # evtname:thickness
                # number33_thick[single_list[-1]] = thickness  # 获取7层膜厚数值
                # 在这里穿插,根据evtname, 找到当前evt的对应正面,并获取膜厚设置值
                pair_thick, thick1 = evt_pair_thick(single_list[0]+'.CSV')  # str, list
                if pair_thick and thick1 == thickness.split(','):
                    if not concate:
                        # 取mean, 保留7层维度
                        th1 = thickness.split(',')[:-1]
                        th2 = pair_thick.split(',')[:-1]
                        len_ = len(th1)
                        final = [(float(th1[i]) + float(th2[i]))/2+random.uniform(1e-4,5e-5) for i in range(len_)]
                        final = ''.join(str(i) + ',' for i in final)
                        finall_thick_lab[final] = single_list[1]
                    else:
                        # 正背concate
                        finall_thick_lab[thickness+pair_thick] = single_list[1]
                        number33_thick[single_list[-1]] = thickness+pair_thick
                else:
                    if not concate:
                        # print(pair_thick, thick1, thickness, single_list[0]+'.CSV')   # 部分背面evt找不到正面的evt,没事那就copy一份膜值.
                        finall_thick_lab[thickness] = single_list[1]
                    else:  # concate
                        finall_thick_lab[thickness+thickness[:-1]] = single_list[1]    # 正背concate
                        number33_thick[single_list[-1]] = thickness+thickness[:-1]

                break  # 找到一个33number了,就不再遍历list_, break出循环
    # mean处理正背面的膜厚时, 可能出现key值重复,覆盖更新了value,故而导致finall_thick_lab 和 oneone_evt_thickness 长度不等
    assert len(finall_thick_lab) == len(oneone_evt_thickness)  # len需要一致!

    data = json.dumps(finall_thick_lab)
    with open(refine_data_json, 'w') as js_file:
        js_file.write(data)

    data = json.dumps(oneone_evt_thickness)
    with open(one_evt_thickness, 'w') as js_file:
        js_file.write(data)

    data = json.dumps(number33_thick)
    with open(r'./33number_thickness.json', 'w') as js_file:
        js_file.write(data)


def get_hc_value(process_data, num33_hc_js, face):
    f = open(r'./33number.txt', 'r')
    number33 = f.readlines()
    num33_list = []
    for nub33 in number33:
        num33_list.append(nub33[:-1])
    wb = xlrd.open_workbook(process_data)
    data = wb.sheet_by_name('Sheet1')
    rows = data.nrows
    number33_dsdzqdb = dict()
    for i in range(1, rows):
        # 依然是读取的背面行的数据,正面镀膜完后再背面..耗材去背面的更合理
        if data.cell(i, 1).value in num33_list and data.cell(i, 2).value == face:
            # print(data.row_values(i))
            number33_dsdzqdb[data.cell(i, 1).value] = [data.cell(i, 13).value, data.cell(i, 14).value, data.cell(i, 16).value]
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
        hc = str(round(num33_hc[num33][0]/8, 2))+','
        hc += str(round(num33_hc[num33][1]/120, 2))+','
        hc += str(round(num33_hc[num33][2]/120, 2))+','
        # 14+3
        if num33_thick14[num33].endswith(','):
            thickness_hc = num33_thick14[num33]+hc
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
    data = json.dumps(thick_hc_lab)
    with open(thick_hc_lab_js, 'w') as js_file:
        js_file.write(data)





if __name__ == "__main__":
    base_data_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件'
    file1 = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#膜色文件与EVT文件对应表.xlsx'
    file2 = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\膜色数据.xlsx'
    data_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\org_refine_thickness_lab_curve.json'
    evt_cc_dir = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\33#机台文件_7dirs\1.6&1.67_DVS_CC'
    process_data = r'D:\work\project\卡尔蔡司AR镀膜\文档s\蔡司资料0615\工艺记录.xlsx'
    evt_thick = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\oneone_evt_thickness.json'
    # rate_thickness_check(evt_cc_dir)  # import check_data.py 中的函数实现部分数据清洗功能
    # bad_sample_clean(data_js, evt_thick)
    # get_hc_value(process_data, '背面')
    num33_hc_js = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\33_hc.json'
    number33_thick_js = r'D:\work\project\卡尔蔡司AR镀膜\ML_ZEISS\33number_thickness.json'
    refine_data_json = r'D:\work\project\卡尔蔡司AR镀膜\卡尔蔡司AR模色推优数据_20210610\0619\refine_thickness_lab_curve.json'
    hc_feature(num33_hc_js, number33_thick_js, refine_data_json)



